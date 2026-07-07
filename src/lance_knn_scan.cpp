#include "lance_knn_scan.hpp"

#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include "lance_arrow_compat.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"

#include <atomic>
#include <cstring>
#include <mutex>

namespace duckdb {

namespace {

struct LanceKnnScanGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceKnnScanLocalState : public ArrowScanLocalState {
  LanceKnnScanLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                         ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context) {}

  void *stream = nullptr;
  LanceKnnScanGlobalState *global_state = nullptr;

  ~LanceKnnScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

void PopulateSchemaFromKnn(ClientContext &context,
                           LanceKnnScanBindData &bind_data) {
  auto *schema_handle = lance_get_knn_schema(
      bind_data.dataset_entry->Handle(), bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k,
      bind_data.nprobes, bind_data.refine_factor, bind_data.ef,
      bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance KNN schema: " + bind_data.file_path +
                      LanceFormatErrorSuffix());
  }
  memset(&bind_data.schema_root.arrow_schema, 0,
         sizeof(bind_data.schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle,
                            &bind_data.schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance KNN schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);
  LanceCoerceArrowSchemaForDuckDB(&bind_data.schema_root.arrow_schema);
  ArrowTableFunction::PopulateArrowTableSchema(
      context, bind_data.arrow_table, bind_data.schema_root.arrow_schema);
  bind_data.names = bind_data.arrow_table.GetNames();
  bind_data.types = bind_data.arrow_table.GetTypes();
}

// Upper bound on the rebound query vector blob. 1 MiB / 4 = 262 144 floats,
// far above any real embedding dimension; meant to keep a typo'd or hostile
// __lance_knn_scan(..., <huge blob>, ...) call from allocating eagerly.
constexpr idx_t LANCE_KNN_REBIND_MAX_QUERY_BYTES = 1ULL << 20;

// Symmetric cap on the rebound Filter IR blob. The optimizer-built path
// produces filter IR proportional to the WHERE clause complexity (a few
// hundred bytes is typical, a few KB is generous); 1 MiB is well above
// anything a normal predicate could encode and prevents a hostile call
// from passing __lance_knn_scan an arbitrarily large blob that Lance
// would then have to parse.
constexpr idx_t LANCE_KNN_REBIND_MAX_FILTER_IR_BYTES = 1ULL << 20;

unique_ptr<FunctionData> LanceKnnScanBind(ClientContext &context,
                                          TableFunctionBindInput &input,
                                          vector<LogicalType> &return_types,
                                          vector<string> &names) {
  // The optimizer-built path attaches bind_data via the LogicalGet
  // directly; that bind_data is used at execution time. This Bind callback
  // is only invoked when DuckDB rebinds (prepared statement re-bind,
  // EXPLAIN ANALYZE replanning). We reconstruct from the BLOB parameters.
  if (input.inputs.size() < 11) {
    throw InvalidInputException(
        "__lance_knn_scan requires 11 parameters (internal function)");
  }

  auto result = make_uniq<LanceKnnScanBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  result->vector_column = input.inputs[1].GetValue<string>();

  auto query_blob = input.inputs[2].GetValueUnsafe<string_t>();
  auto query_bytes = query_blob.GetString();
  if (query_bytes.size() % sizeof(float) != 0) {
    throw InvalidInputException(
        "__lance_knn_scan query blob must be a multiple of 4 bytes");
  }
  if (query_bytes.size() > LANCE_KNN_REBIND_MAX_QUERY_BYTES) {
    throw InvalidInputException(
        "__lance_knn_scan query blob is too large (" +
        to_string(query_bytes.size()) + " bytes, max " +
        to_string(LANCE_KNN_REBIND_MAX_QUERY_BYTES) +
        "); this is an internal function — use lance_vector_search instead");
  }
  auto query_len = query_bytes.size() / sizeof(float);
  result->query.resize(query_len);
  if (query_len > 0) {
    memcpy(result->query.data(), query_bytes.data(), query_bytes.size());
  }

  auto k_val = input.inputs[3].GetValue<int64_t>();
  if (k_val <= 0) {
    throw InvalidInputException("__lance_knn_scan requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  // input.inputs[4] is the metric tag (informational; semantics are
  // already baked into the chosen Lance index — we pass it through for
  // EXPLAIN clarity but Lance itself reads metric from the index).
  result->metric = input.inputs[4].GetValue<string>();

  auto filter_blob = input.inputs[5].GetValueUnsafe<string_t>();
  if (filter_blob.GetSize() > LANCE_KNN_REBIND_MAX_FILTER_IR_BYTES) {
    throw InvalidInputException(
        "__lance_knn_scan filter IR blob is too large (" +
        to_string(filter_blob.GetSize()) + " bytes, max " +
        to_string(LANCE_KNN_REBIND_MAX_FILTER_IR_BYTES) +
        "); this is an internal function — use lance_vector_search instead");
  }
  result->filter_ir_msg.assign(filter_blob.GetData(), filter_blob.GetSize());

  auto nprobes_val = input.inputs[6].GetValue<int64_t>();
  result->nprobes = nprobes_val > 0 ? NumericCast<uint64_t>(nprobes_val) : 0;
  auto refine_val = input.inputs[7].GetValue<int64_t>();
  result->refine_factor =
      refine_val > 0 ? NumericCast<uint64_t>(refine_val) : 0;
  result->prefilter = input.inputs[8].GetValue<bool>();
  result->use_index = input.inputs[9].GetValue<bool>();
  auto ef_val = input.inputs[10].GetValue<int64_t>();
  result->ef = ef_val > 0 ? NumericCast<uint64_t>(ef_val) : 0;

  result->dataset_entry = LanceGetOrOpenDatasetEntry(
      context, result->file_path, &result->dataset_cache_hit);
  if (!result->dataset_entry) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  // Re-verify the vector index still exists with the expected metric. This
  // catches PREPARE / EXECUTE windows where the index was dropped or
  // recreated with a different metric — without this check the user would
  // hit an opaque "Failed to create Lance KNN stream" downstream.
  //
  // Limitation: GetOrLookupVectorIndexMetric is backed by the per-entry
  // cache populated at PREPARE time. Same-context DDL invalidates the
  // dataset cache entry (and thus this cache) precisely, so the check
  // catches it. Cross-context drops (another connection / process) leave
  // the cache stale; this rebind will still let the query proceed and
  // Lance itself surfaces the error a moment later. Strictly better than
  // the pre-fix path, just not perfect.
  if (result->use_index && !result->metric.empty()) {
    auto lookup = result->dataset_entry->GetOrLookupVectorIndexMetric(
        result->vector_column);
    if (lookup.status == 1) {
      throw IOException(
          "Lance vector index for column '" + result->vector_column +
          "' no longer exists on dataset " + result->file_path +
          "; re-prepare the statement so the optimizer can re-plan" +
          LanceFormatErrorSuffix());
    }
    if (lookup.status != 0) {
      throw IOException("Lance vector index metric lookup failed: " +
                        lookup.error_message + LanceFormatErrorSuffix());
    }
    if (lookup.metric != result->metric) {
      throw IOException("Lance vector index metric for column '" +
                        result->vector_column + "' changed from '" +
                        result->metric + "' to '" + lookup.metric +
                        "' since the statement was prepared; re-prepare it" +
                        LanceFormatErrorSuffix());
    }
  }

  PopulateSchemaFromKnn(context, *result);
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

unique_ptr<GlobalTableFunctionState>
LanceKnnScanInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceKnnScanBindData>();
  auto state =
      make_uniq_base<GlobalTableFunctionState, LanceKnnScanGlobalState>();
  auto &global = state->Cast<LanceKnnScanGlobalState>();

  global.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    global.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      global.scanned_types.push_back(bind_data.types[col_id]);
    }
  }
  return state;
}

unique_ptr<LocalTableFunctionState>
LanceKnnScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                      GlobalTableFunctionState *global_state) {
  auto &bind_data = input.bind_data->Cast<LanceKnnScanBindData>();
  auto &global = global_state->Cast<LanceKnnScanGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceKnnScanLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }

  const uint8_t *filter_ir =
      bind_data.filter_ir_msg.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(bind_data.filter_ir_msg.data());
  auto filter_ir_len = bind_data.filter_ir_msg.size();
  result->stream = lance_create_knn_stream_ir(
      bind_data.dataset_entry->Handle(), bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k,
      bind_data.nprobes, bind_data.refine_factor, bind_data.ef, filter_ir,
      filter_ir_len, bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  if (!result->stream) {
    throw IOException("Failed to create Lance KNN stream" +
                      LanceFormatErrorSuffix());
  }
  return std::move(result);
}

bool LanceKnnScanLoadNextBatch(LanceKnnScanLocalState &local_state) {
  if (!local_state.stream) {
    return false;
  }

  void *batch = nullptr;
  auto rc = lance_stream_next(local_state.stream, &batch);
  if (rc == 1) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }
  if (rc != 0) {
    throw IOException("Failed to read next Lance RecordBatch" +
                      LanceFormatErrorSuffix());
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  memset(&new_chunk->arrow_array, 0, sizeof(new_chunk->arrow_array));
  ArrowSchema tmp_schema;
  memset(&tmp_schema, 0, sizeof(tmp_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array, &tmp_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_batch(batch);
  LanceCoerceArrowArrayForDuckDB(&tmp_schema, &new_chunk->arrow_array);

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

void LanceKnnScanFunc(ClientContext &, TableFunctionInput &data,
                      DataChunk &output) {
  // LocalInit always provides a local_state; assert in debug so a future
  // refactor that breaks the contract is caught loudly.
  D_ASSERT(data.local_state);
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceKnnScanBindData>();
  auto &global_state = data.global_state->Cast<LanceKnnScanGlobalState>();
  auto &local_state = data.local_state->Cast<LanceKnnScanLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceKnnScanLoadNextBatch(local_state)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, false);
      local_state.chunk_offset += output_size;
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(
          local_state, bind_data.arrow_table.GetColumns(), output, false);
      local_state.chunk_offset += output_size;
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

bool TryLanceExplainKnnScan(const LanceKnnScanBindData &bind_data,
                            string &out_plan, string &out_error) {
  out_plan.clear();
  out_error.clear();
  if (!bind_data.dataset_entry) {
    out_error = "dataset is null";
    return false;
  }
  const uint8_t *filter_ptr =
      bind_data.filter_ir_msg.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(bind_data.filter_ir_msg.data());
  auto filter_len = bind_data.filter_ir_msg.size();

  auto *plan_ptr = lance_explain_knn_scan_ir(
      bind_data.dataset_entry->Handle(), bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k,
      bind_data.nprobes, bind_data.refine_factor, bind_data.ef, filter_ptr,
      filter_len, bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0, 0);
  if (!plan_ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  out_plan = plan_ptr;
  lance_free_string(plan_ptr);
  return true;
}

InsertionOrderPreservingMap<string>
LanceKnnScanToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnScanBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Metric"] = bind_data.metric;
  result["Lance Nprobes"] = to_string(bind_data.nprobes);
  result["Lance Refine Factor"] = to_string(bind_data.refine_factor);
  result["Lance Ef Search"] = to_string(bind_data.ef);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  result["Lance Filter IR Bytes (Bind)"] =
      to_string(bind_data.filter_ir_msg.size());

  string plan;
  string error;
  if (TryLanceExplainKnnScan(bind_data, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }
  return result;
}

InsertionOrderPreservingMap<string>
LanceKnnScanDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnScanBindData>();
  auto &global_state = input.global_state->Cast<LanceKnnScanGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Metric"] = bind_data.metric;
  result["Lance Nprobes"] = to_string(bind_data.nprobes);
  result["Lance Refine Factor"] = to_string(bind_data.refine_factor);
  result["Lance Ef Search"] = to_string(bind_data.ef);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Dataset Cache Hit"] =
      bind_data.dataset_cache_hit ? "true" : "false";
  result["Lance Filter IR Bytes"] = to_string(bind_data.filter_ir_msg.size());

  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  // Double-checked lazy init for the explain plan. Memory ordering chain:
  //   writer: explain_plan = ... ; mutex.unlock() ;
  //           explain_computed.store(true, release)
  //   reader: explain_computed.load(acquire) sees true →
  //           subsequent reads of explain_plan happen-after the writer's
  //           assignment.
  // Inner load can be relaxed since the mutex itself provides
  // synchronization for the slow path. Do not weaken the outer load to
  // relaxed — the fast path skips the mutex and relies on this acquire to
  // publish the plan string.
  if (!global_state.explain_computed.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load(std::memory_order_relaxed)) {
      string plan;
      string error;
      auto ok = TryLanceExplainKnnScan(bind_data, plan, error);
      if (ok) {
        global_state.explain_plan = std::move(plan);
      } else {
        global_state.explain_error = std::move(error);
      }
      global_state.explain_computed.store(true, std::memory_order_release);
    }
  }
  if (!global_state.explain_plan.empty()) {
    result["Lance Plan"] = global_state.explain_plan;
  } else if (!global_state.explain_error.empty()) {
    result["Lance Plan Error"] = global_state.explain_error;
  }
  return result;
}

} // namespace

void LanceKnnScanPopulateSchema(ClientContext &context,
                                LanceKnnScanBindData &bind_data) {
  PopulateSchemaFromKnn(context, bind_data);
}

TableFunction LanceKnnScanFunction() {
  TableFunction function(
      "__lance_knn_scan",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BLOB,
       LogicalType::BIGINT, LogicalType::VARCHAR, LogicalType::BLOB,
       LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BOOLEAN,
       LogicalType::BOOLEAN, LogicalType::BIGINT},
      LanceKnnScanFunc, LanceKnnScanBind, LanceKnnScanInitGlobal,
      LanceKnnScanLocalInit);
  function.projection_pushdown = true;
  // filter_pushdown is intentionally disabled: the KNN optimizer absorbs
  // every WHERE predicate it can encode into filter_ir_msg at plan time, so
  // there is nothing left for DuckDB to push down at execution. If a future
  // post-optimization pass ever re-introduces a filter above the KNN scan,
  // it will run in DuckDB row-by-row — promote filter_pushdown then.
  function.filter_pushdown = false;
  function.to_string = LanceKnnScanToString;
  function.dynamic_to_string = LanceKnnScanDynamicToString;
  return function;
}

void RegisterLanceKNNScan(ExtensionLoader &loader) {
  loader.RegisterFunction(LanceKnnScanFunction());
}

} // namespace duckdb
