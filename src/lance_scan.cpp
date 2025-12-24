#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

// FFI ownership contract (Arrow C Data Interface):
// `lance_get_schema` returns an opaque schema handle; caller frees it via
// `lance_free_schema` exactly once.
// `lance_schema_to_arrow` populates `out_schema` on success (return 0) and
// transfers ownership of the ArrowSchema to the caller, who must call
// `out_schema->release(out_schema)` exactly once (or wrap it in RAII).
// `lance_create_fragment_stream_ir` returns an opaque stream handle; caller
// closes it via `lance_close_stream` exactly once.
// `lance_stream_next` returns an opaque RecordBatch handle; caller frees it via
// `lance_free_batch` exactly once after use.
// `lance_batch_to_arrow` populates `out_array` and `out_schema` on success
// (return 0) and transfers ownership of both to the caller, who must call
// `release` exactly once on each.
// On error, the callee leaves output `ArrowSchema` / `ArrowArray` untouched; do
// not call `release` unless the caller initialized them to a valid value.
namespace duckdb {

static bool TryLanceExplainDatasetScan(void *dataset,
                                       const vector<string> *columns,
                                       const string *filter_ir, bool verbose,
                                       string &out_plan, string &out_error) {
  out_plan.clear();
  out_error.clear();

  if (!dataset) {
    out_error = "dataset is null";
    return false;
  }

  vector<const char *> col_ptrs;
  if (columns) {
    col_ptrs.reserve(columns->size());
    for (auto &col : *columns) {
      col_ptrs.push_back(col.c_str());
    }
  }

  const uint8_t *filter_ptr = nullptr;
  size_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data()); // NOLINT
    filter_len = filter_ir->size();
  }

  auto *plan_ptr = lance_explain_dataset_scan_ir(
      dataset, col_ptrs.empty() ? nullptr : col_ptrs.data(), col_ptrs.size(),
      filter_ptr, filter_len, verbose ? 1 : 0);
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

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  bool explain_verbose = false;
  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;

  ~LanceScanBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceScanGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> next_fragment_idx{0};
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  std::atomic<idx_t> streams_opened{0};
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<uint64_t> fragment_ids;
  idx_t max_threads = 1;

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  vector<string> scan_column_names;
  string lance_filter_ir;

  bool count_only = false;
  idx_t count_only_total_rows = 0;
  std::atomic<idx_t> count_only_offset{0};

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return max_threads; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceScanLocalState : public ArrowScanLocalState {
  explicit LanceScanLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                               ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceScanGlobalState *global_state = nullptr;
  idx_t fragment_pos = 0;
  SelectionVector filter_sel;

  ~LanceScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static bool LanceSupportsPushdownType(const FunctionData &bind_data,
                                      idx_t col_idx) {
  auto &scan_bind = bind_data.Cast<LanceScanBindData>();
  if (col_idx >= scan_bind.types.size()) {
    return false;
  }
  auto &type = scan_bind.types[col_idx];
  if (LanceFilterIRSupportsLogicalType(type)) {
    return true;
  }
  // Even if a type is not supported by our Lance filter IR, we can still
  // safely evaluate DuckDB table filters inside the scan (and potentially
  // prune filter-only columns).
  switch (type.id()) {
  case LogicalTypeId::DATE:
    return true;
  default:
    return false;
  }
}

static void
LancePushdownComplexFilter(ClientContext &context, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceScanBindData>();

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile() ||
        expr->CanThrow()) {
      continue;
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types, false,
                                   *expr, filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }
}

static unique_ptr<FunctionData> LanceScanBind(ClientContext &context,
                                              TableFunctionBindInput &input,
                                              vector<LogicalType> &return_types,
                                              vector<string> &names) {
  if (input.inputs.empty() || input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_scan requires a dataset root path");
  }

  auto result = make_uniq<LanceScanBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  result->dataset = LanceOpenDataset(context, result->file_path);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(result->dataset);
  if (!schema_handle) {
    throw IOException("Failed to get schema from Lance dataset: " +
                      result->file_path + LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceScanGlobalState>();
  auto &scan_state = state->Cast<LanceScanGlobalState>();

  scan_state.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    scan_state.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      scan_state.scanned_types.push_back(bind_data.types[col_id]);
    }
  }

  scan_state.scan_column_names.reserve(input.column_ids.size());
  for (auto col_id : input.column_ids) {
    if (col_id >= bind_data.names.size()) {
      throw IOException("Invalid column id in projection");
    }
    scan_state.scan_column_names.push_back(bind_data.names[col_id]);
  }

  vector<string> filter_parts;

  auto table_filters = BuildLanceTableFilterIRParts(
      bind_data.names, bind_data.types, input, false);
  filter_parts = std::move(table_filters.parts);

  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    filter_parts.reserve(filter_parts.size() +
                         bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      filter_parts.push_back(part);
    }
  }

  string filter_ir_msg;
  if (!filter_parts.empty() &&
      TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
    scan_state.lance_filter_ir = std::move(filter_ir_msg);
  }

  if (scan_state.scan_column_names.empty() &&
      scan_state.lance_filter_ir.empty()) {
    auto rows = lance_dataset_count_rows(bind_data.dataset);
    if (rows < 0) {
      throw IOException("Failed to count Lance rows" +
                        LanceFormatErrorSuffix());
    }
    scan_state.count_only = true;
    scan_state.count_only_total_rows = NumericCast<idx_t>(rows);
    scan_state.max_threads = 1;
    return state;
  }

  size_t fragment_count = 0;
  auto fragments_ptr =
      lance_dataset_list_fragments(bind_data.dataset, &fragment_count);
  if (!fragments_ptr) {
    throw IOException("Failed to list Lance fragments" +
                      LanceFormatErrorSuffix());
  }
  scan_state.fragment_ids.assign(fragments_ptr, fragments_ptr + fragment_count);
  lance_free_fragment_list(fragments_ptr, fragment_count);

  auto threads = context.db->NumberOfThreads();
  scan_state.max_threads = MaxValue<idx_t>(
      1, MinValue<idx_t>(threads, scan_state.fragment_ids.size()));

  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                   GlobalTableFunctionState *global_state) {
  auto &scan_global = global_state->Cast<LanceScanGlobalState>();
  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceScanLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &scan_global;
  if (scan_global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, scan_global.scanned_types);
  }
  if (scan_global.count_only) {
    return std::move(result);
  }
  // Early stop: no fragments left for this thread.
  auto fragment_pos = scan_global.next_fragment_idx.fetch_add(1);
  if (fragment_pos >= scan_global.fragment_ids.size()) {
    return nullptr;
  }
  result->fragment_pos = fragment_pos;
  return std::move(result);
}

static bool LanceScanOpenStream(ClientContext &context,
                                const LanceScanBindData &bind_data,
                                LanceScanGlobalState &global_state,
                                LanceScanLocalState &local_state) {
  if (local_state.stream) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
  }

  if (local_state.fragment_pos >= global_state.fragment_ids.size()) {
    return false;
  }
  auto fragment_id = global_state.fragment_ids[local_state.fragment_pos];

  vector<const char *> columns;
  columns.reserve(global_state.scan_column_names.size());
  for (auto &name : global_state.scan_column_names) {
    columns.push_back(name.c_str());
  }

  const uint8_t *filter_ir = global_state.lance_filter_ir.empty()
                                 ? nullptr
                                 : reinterpret_cast<const uint8_t *>(
                                       global_state.lance_filter_ir.data());
  auto filter_ir_len = global_state.lance_filter_ir.size();

  auto stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                                columns.data(), columns.size(),
                                                filter_ir, filter_ir_len);
  if (!stream && filter_ir) {
    // Best-effort: if filter pushdown failed, retry without it and rely on
    // DuckDB-side filter execution for correctness.
    global_state.filter_pushdown_fallbacks.fetch_add(1);
    stream = lance_create_fragment_stream_ir(bind_data.dataset, fragment_id,
                                             columns.data(), columns.size(),
                                             nullptr, 0);
  }
  if (!stream) {
    throw IOException("Failed to create Lance fragment stream" +
                      LanceFormatErrorSuffix());
  }
  global_state.streams_opened.fetch_add(1);
  local_state.stream = stream;
  return true;
}

static bool LanceScanLoadNextBatch(LanceScanLocalState &local_state) {
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

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
  auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
  auto &local_state = data.local_state->Cast<LanceScanLocalState>();

  if (global_state.count_only) {
    auto start = global_state.count_only_offset.fetch_add(STANDARD_VECTOR_SIZE);
    if (start >= global_state.count_only_total_rows) {
      return;
    }
    auto output_size = MinValue<idx_t>(
        STANDARD_VECTOR_SIZE, global_state.count_only_total_rows - start);
    output.SetCardinality(output_size);
    output.Verify();
    return;
  }

  while (true) {
    if (!local_state.stream) {
      if (!LanceScanOpenStream(context, bind_data, global_state, local_state)) {
        return;
      }
    }

    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceScanLoadNextBatch(local_state)) {
        // Stream finished, try next fragment.
        local_state.fragment_pos = global_state.next_fragment_idx.fetch_add(1);
        continue;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    auto start = global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, start);
      local_state.chunk_offset += output_size;
      if (local_state.filters) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(
          local_state, bind_data.arrow_table.GetColumns(), output, start);
      local_state.chunk_offset += output_size;
      if (local_state.filters) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceScanToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Pushed Filter Parts"] =
      to_string(bind_data.lance_pushed_filter_ir_parts.size());

  string filter_ir_msg;
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    TryEncodeLanceFilterIRMessage(bind_data.lance_pushed_filter_ir_parts,
                                  filter_ir_msg);
  }

  result["Lance Filter IR Bytes (Bind)"] = to_string(filter_ir_msg.size());

  string plan;
  string error;
  if (TryLanceExplainDatasetScan(bind_data.dataset, nullptr,
                                 filter_ir_msg.empty() ? nullptr
                                                       : &filter_ir_msg,
                                 bind_data.explain_verbose, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceScanDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceScanBindData>();
  auto &global_state = input.global_state->Cast<LanceScanGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";
  result["Lance Fragments"] = to_string(global_state.fragment_ids.size());
  result["Lance Max Threads"] = to_string(global_state.max_threads);
  result["Lance Streams Opened"] =
      to_string(global_state.streams_opened.load());
  result["Lance Filter Pushdown Fallbacks"] =
      to_string(global_state.filter_pushdown_fallbacks.load());
  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  if (global_state.count_only) {
    result["Lance Count Only"] = "true";
    result["Lance Count Total Rows"] =
        to_string(global_state.count_only_total_rows);
    return result;
  }

  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());
  if (!global_state.scan_column_names.empty()) {
    result["Lance Projection"] =
        StringUtil::Join(global_state.scan_column_names, "\n");
  }

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainDatasetScan(
          bind_data.dataset, &global_state.scan_column_names,
          global_state.lance_filter_ir.empty() ? nullptr
                                               : &global_state.lance_filter_ir,
          bind_data.explain_verbose, plan, error);
      if (ok) {
        global_state.explain_plan = std::move(plan);
      } else {
        global_state.explain_error = std::move(error);
      }
      global_state.explain_computed.store(true);
    }
  }

  if (!global_state.explain_plan.empty()) {
    result["Lance Plan"] = global_state.explain_plan;
  } else if (!global_state.explain_error.empty()) {
    result["Lance Plan Error"] = global_state.explain_error;
  }

  return result;
}

void RegisterLanceScan(ExtensionLoader &loader) {
  TableFunction lance_scan("lance_scan", {LogicalType::VARCHAR}, LanceScanFunc,
                           LanceScanBind, LanceScanInitGlobal,
                           LanceScanLocalInit);
  lance_scan.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  lance_scan.projection_pushdown = true;
  lance_scan.filter_pushdown = true;
  lance_scan.filter_prune = true;
  lance_scan.supports_pushdown_type = LanceSupportsPushdownType;
  lance_scan.pushdown_complex_filter = LancePushdownComplexFilter;
  lance_scan.to_string = LanceScanToString;
  lance_scan.dynamic_to_string = LanceScanDynamicToString;
  loader.RegisterFunction(lance_scan);
}

} // namespace duckdb
