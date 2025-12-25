#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
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

namespace duckdb {

static bool TryLanceExplainKnn(void *dataset, const string &vector_column,
                               const vector<float> &query, uint64_t k,
                               const string *filter_ir, bool prefilter,
                               bool use_index, bool verbose, string &out_plan,
                               string &out_error) {
  out_plan.clear();
  out_error.clear();

  if (!dataset) {
    out_error = "dataset is null";
    return false;
  }
  if (query.empty()) {
    out_error = "query is empty";
    return false;
  }

  const uint8_t *filter_ptr = nullptr;
  size_t filter_len = 0;
  if (filter_ir && !filter_ir->empty()) {
    filter_ptr = reinterpret_cast<const uint8_t *>(filter_ir->data());
    filter_len = filter_ir->size();
  }

  auto *plan_ptr = lance_explain_knn_scan_ir(
      dataset, vector_column.c_str(), query.data(), query.size(), k, filter_ptr,
      filter_len, prefilter ? 1 : 0, use_index ? 1 : 0, verbose ? 1 : 0);
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

static vector<float> ParseQueryVector(const Value &value) {
  if (value.IsNull()) {
    throw InvalidInputException(
        "lance_search requires a non-null query vector");
  }
  if (value.type().id() != LogicalTypeId::LIST) {
    throw InvalidInputException(
        "lance_search requires query vector to be a LIST");
  }
  auto children = ListValue::GetChildren(value);
  if (children.empty()) {
    throw InvalidInputException(
        "lance_search requires a non-empty query vector");
  }

  auto cast_f32 = [](double v) {
    if (!std::isfinite(v)) {
      throw InvalidInputException(
          "lance_search query vector contains non-finite value");
    }
    auto max_v = static_cast<double>(std::numeric_limits<float>::max());
    if (v > max_v || v < -max_v) {
      throw InvalidInputException(
          "lance_search query vector value is out of float32 range");
    }
    return static_cast<float>(v);
  };

  vector<float> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child.IsNull()) {
      throw InvalidInputException("lance_search query vector contains NULL");
    }
    switch (child.type().id()) {
    case LogicalTypeId::FLOAT:
      out.push_back(cast_f32(child.GetValue<float>()));
      break;
    case LogicalTypeId::DOUBLE:
      out.push_back(cast_f32(child.GetValue<double>()));
      break;
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<int64_t>())));
      break;
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<uint64_t>())));
      break;
    default:
      try {
        auto dbl = child.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
        out.push_back(cast_f32(dbl));
      } catch (Exception &) {
        throw InvalidInputException(
            "lance_search query vector elements must be numeric");
      }
    }
  }
  return out;
}

struct LanceKnnBindData : public TableFunctionData {
  string file_path;
  string vector_column;
  vector<float> query;
  uint64_t k = 0;
  bool prefilter = true;
  bool use_index = true;
  bool explain_verbose = false;

  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  vector<string> lance_pushed_filter_ir_parts;

  ~LanceKnnBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceKnnGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  string lance_filter_ir;
  bool filter_pushed_down = false;
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceKnnLocalState : public ArrowScanLocalState {
  explicit LanceKnnLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                              ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceKnnGlobalState *global_state = nullptr;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceKnnLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static void
LancePushdownComplexFilter(ClientContext &, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceKnnBindData>();

  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile() ||
        expr->CanThrow()) {
      continue;
    }
    string filter_ir;
    if (!TryBuildLanceExprFilterIR(get, scan_bind.names, scan_bind.types, true,
                                   *expr, filter_ir)) {
      continue;
    }
    scan_bind.lance_pushed_filter_ir_parts.push_back(std::move(filter_ir));
  }
}

static string InferDefaultVectorColumn(ClientContext &context, void *dataset,
                                       const string &path) {
  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    throw IOException("Failed to get Lance schema: " + path +
                      LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  ArrowTableSchema arrow_table;
  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(config, arrow_table,
                                               schema_root.arrow_schema);
  auto col_names = arrow_table.GetNames();
  auto col_types = arrow_table.GetTypes();

  vector<string> candidates;
  candidates.reserve(col_names.size());
  for (idx_t i = 0; i < col_names.size() && i < col_types.size(); i++) {
    auto &t = col_types[i];
    if (t.id() == LogicalTypeId::LIST) {
      auto &child = ListType::GetChildType(t);
      if (child.id() == LogicalTypeId::FLOAT) {
        candidates.push_back(col_names[i]);
      }
    } else if (t.id() == LogicalTypeId::ARRAY) {
      auto &child = ArrayType::GetChildType(t);
      if (child.id() == LogicalTypeId::FLOAT) {
        candidates.push_back(col_names[i]);
      }
    }
  }

  if (candidates.empty()) {
    throw InvalidInputException(
        "lance_search requires parameter vector_column = '<vector_column>'");
  }
  if (candidates.size() != 1) {
    throw InvalidInputException(
        "lance_search requires parameter vector_column = '<vector_column>' "
        "(multiple vector columns found: " +
        StringUtil::Join(candidates, ", ") + ")");
  }
  return candidates[0];
}

static unique_ptr<FunctionData>
LanceSearchVectorBind(ClientContext &context, TableFunctionBindInput &input,
                      vector<LogicalType> &return_types,
                      vector<string> &names) {
  if (input.inputs.size() < 2) {
    throw InvalidInputException("lance_search requires (path, query)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_search requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException(
        "lance_search requires a non-null query vector");
  }

  auto result = make_uniq<LanceKnnBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  result->query = ParseQueryVector(input.inputs[1]);
  result->prefilter = false;

  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  int64_t k_val = 10;
  bool has_positional_k = false;
  if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
    k_val = input.inputs[2].GetValue<int64_t>();
    has_positional_k = true;
  }
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    if (has_positional_k) {
      throw InvalidInputException(
          "lance_search requires k to be specified either positionally or as "
          "named parameter k");
    }
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_search requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }
  auto use_index_named = input.named_parameters.find("use_index");
  if (use_index_named != input.named_parameters.end() &&
      !use_index_named->second.IsNull()) {
    result->use_index =
        use_index_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  result->dataset = LanceOpenDataset(context, result->file_path);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto column_named = input.named_parameters.find("vector_column");
  if (column_named == input.named_parameters.end()) {
    column_named = input.named_parameters.find("column");
  }
  if (column_named != input.named_parameters.end() &&
      !column_named->second.IsNull()) {
    result->vector_column = column_named->second.GetValue<string>();
  } else {
    result->vector_column =
        InferDefaultVectorColumn(context, result->dataset, result->file_path);
  }

  auto *schema_handle = lance_get_knn_schema(
      result->dataset, result->vector_column.c_str(), result->query.data(),
      result->query.size(), result->k, result->prefilter ? 1 : 0,
      result->use_index ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance KNN schema: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance KNN schema to Arrow C Data Interface" +
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
LanceKnnInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceKnnGlobalState>();
  auto &global = state->Cast<LanceKnnGlobalState>();

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

  auto table_filters = BuildLanceTableFilterIRParts(
      bind_data.names, bind_data.types, input, true);
  if (bind_data.prefilter && !table_filters.all_prefilterable_filters_pushed) {
    throw InvalidInputException(
        "lance_search requires filter pushdown for prefilterable columns when "
        "prefilter=true");
  }

  bool has_table_filter_parts = !table_filters.parts.empty();
  auto filter_parts = std::move(table_filters.parts);
  if (!bind_data.lance_pushed_filter_ir_parts.empty()) {
    filter_parts.reserve(filter_parts.size() +
                         bind_data.lance_pushed_filter_ir_parts.size());
    for (auto &part : bind_data.lance_pushed_filter_ir_parts) {
      filter_parts.push_back(part);
    }
  }

  string filter_ir_msg;
  if (!filter_parts.empty()) {
    if (!TryEncodeLanceFilterIRMessage(filter_parts, filter_ir_msg)) {
      filter_ir_msg.clear();
    }
    global.lance_filter_ir = std::move(filter_ir_msg);
  }
  if (bind_data.prefilter && has_table_filter_parts &&
      global.lance_filter_ir.empty()) {
    throw IOException("Failed to encode Lance filter IR");
  }
  global.filter_pushed_down =
      table_filters.all_filters_pushed && !global.lance_filter_ir.empty();
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceKnnLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                  GlobalTableFunctionState *global_state) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global = global_state->Cast<LanceKnnGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result = make_uniq<LanceKnnLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  result->filter_pushed_down = global.filter_pushed_down;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }

  const uint8_t *filter_ir =
      global.lance_filter_ir.empty()
          ? nullptr
          : reinterpret_cast<const uint8_t *>(global.lance_filter_ir.data());
  auto filter_ir_len = global.lance_filter_ir.size();
  result->stream = lance_create_knn_stream_ir(
      bind_data.dataset, bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k, filter_ir,
      filter_ir_len, bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  if (!result->stream && filter_ir && !bind_data.prefilter) {
    // Best-effort: if filter pushdown failed, retry without it and rely on
    // DuckDB-side filter execution for correctness.
    global.filter_pushdown_fallbacks.fetch_add(1);
    global.filter_pushed_down = false;
    result->filter_pushed_down = false;
    result->stream = lance_create_knn_stream_ir(
        bind_data.dataset, bind_data.vector_column.c_str(),
        bind_data.query.data(), bind_data.query.size(), bind_data.k, nullptr, 0,
        bind_data.prefilter ? 1 : 0, bind_data.use_index ? 1 : 0);
  }
  if (!result->stream) {
    throw IOException("Failed to create Lance KNN stream" +
                      LanceFormatErrorSuffix());
  }

  return std::move(result);
}

static bool LanceKnnLoadNextBatch(LanceKnnLocalState &local_state) {
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

static void LanceKnnFunc(ClientContext &context, TableFunctionInput &data,
                         DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceKnnBindData>();
  auto &global_state = data.global_state->Cast<LanceKnnGlobalState>();
  auto &local_state = data.local_state->Cast<LanceKnnLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceKnnLoadNextBatch(local_state)) {
        return;
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
                                        local_state.all_columns, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        output, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
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
LanceKnnToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
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
  if (TryLanceExplainKnn(bind_data.dataset, bind_data.vector_column,
                         bind_data.query, bind_data.k,
                         filter_ir_msg.empty() ? nullptr : &filter_ir_msg,
                         bind_data.prefilter, bind_data.use_index,
                         bind_data.explain_verbose, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceKnnDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global_state = input.global_state->Cast<LanceKnnGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";

  result["Lance Filter Pushed Down"] =
      global_state.filter_pushed_down ? "true" : "false";
  result["Lance Filter Pushdown Fallbacks"] =
      to_string(global_state.filter_pushdown_fallbacks.load());
  result["Lance Filter IR Bytes"] =
      to_string(global_state.lance_filter_ir.size());

  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainKnn(bind_data.dataset, bind_data.vector_column,
                                   bind_data.query, bind_data.k,
                                   global_state.lance_filter_ir.empty()
                                       ? nullptr
                                       : &global_state.lance_filter_ir,
                                   bind_data.prefilter, bind_data.use_index,
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

static void RegisterLanceVectorSearch(ExtensionLoader &loader) {
  TableFunction search2(
      "lance_search",
      {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::FLOAT)},
      LanceKnnFunc, LanceSearchVectorBind, LanceKnnInitGlobal,
      LanceKnnLocalInit);
  search2.named_parameters["vector_column"] = LogicalType::VARCHAR;
  search2.named_parameters["k"] = LogicalType::BIGINT;
  search2.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  search2.named_parameters["use_index"] = LogicalType::BOOLEAN;
  search2.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  search2.projection_pushdown = true;
  search2.filter_pushdown = true;
  search2.filter_prune = true;
  search2.pushdown_complex_filter = LancePushdownComplexFilter;
  search2.to_string = LanceKnnToString;
  search2.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(search2);

  TableFunction search3("lance_search",
                        {LogicalType::VARCHAR,
                         LogicalType::LIST(LogicalType::FLOAT),
                         LogicalType::BIGINT},
                        LanceKnnFunc, LanceSearchVectorBind, LanceKnnInitGlobal,
                        LanceKnnLocalInit);
  search3.named_parameters["vector_column"] = LogicalType::VARCHAR;
  search3.named_parameters["k"] = LogicalType::BIGINT;
  search3.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  search3.named_parameters["use_index"] = LogicalType::BOOLEAN;
  search3.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  search3.projection_pushdown = true;
  search3.filter_pushdown = true;
  search3.filter_prune = true;
  search3.pushdown_complex_filter = LancePushdownComplexFilter;
  search3.to_string = LanceKnnToString;
  search3.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(search3);

  TableFunction search2_f64(
      "lance_search",
      {LogicalType::VARCHAR, LogicalType::LIST(LogicalType::DOUBLE)},
      LanceKnnFunc, LanceSearchVectorBind, LanceKnnInitGlobal,
      LanceKnnLocalInit);
  search2_f64.named_parameters["vector_column"] = LogicalType::VARCHAR;
  search2_f64.named_parameters["k"] = LogicalType::BIGINT;
  search2_f64.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  search2_f64.named_parameters["use_index"] = LogicalType::BOOLEAN;
  search2_f64.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  search2_f64.projection_pushdown = true;
  search2_f64.filter_pushdown = true;
  search2_f64.filter_prune = true;
  search2_f64.pushdown_complex_filter = LancePushdownComplexFilter;
  search2_f64.to_string = LanceKnnToString;
  search2_f64.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(search2_f64);

  TableFunction search3_f64("lance_search",
                            {LogicalType::VARCHAR,
                             LogicalType::LIST(LogicalType::DOUBLE),
                             LogicalType::BIGINT},
                            LanceKnnFunc, LanceSearchVectorBind,
                            LanceKnnInitGlobal, LanceKnnLocalInit);
  search3_f64.named_parameters["vector_column"] = LogicalType::VARCHAR;
  search3_f64.named_parameters["k"] = LogicalType::BIGINT;
  search3_f64.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  search3_f64.named_parameters["use_index"] = LogicalType::BOOLEAN;
  search3_f64.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  search3_f64.projection_pushdown = true;
  search3_f64.filter_pushdown = true;
  search3_f64.filter_prune = true;
  search3_f64.pushdown_complex_filter = LancePushdownComplexFilter;
  search3_f64.to_string = LanceKnnToString;
  search3_f64.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(search3_f64);
}

// --- FTS / hybrid search (`lance_search`) ---

static vector<float> ParseHybridQueryVector(const Value &value) {
  if (value.IsNull()) {
    throw InvalidInputException(
        "lance_search(hybrid) requires a non-null query vector");
  }
  if (value.type().id() != LogicalTypeId::LIST) {
    throw InvalidInputException(
        "lance_search(hybrid) requires query vector to be a LIST");
  }
  auto children = ListValue::GetChildren(value);
  if (children.empty()) {
    throw InvalidInputException(
        "lance_search(hybrid) requires a non-empty query vector");
  }

  auto cast_f32 = [](double v) {
    if (!std::isfinite(v)) {
      throw InvalidInputException(
          "lance_search(hybrid) query vector contains non-finite value");
    }
    auto max_v = static_cast<double>(std::numeric_limits<float>::max());
    if (v > max_v || v < -max_v) {
      throw InvalidInputException(
          "lance_search(hybrid) query vector value is out of float32 range");
    }
    return static_cast<float>(v);
  };

  vector<float> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child.IsNull()) {
      throw InvalidInputException(
          "lance_search(hybrid) query vector contains NULL");
    }
    switch (child.type().id()) {
    case LogicalTypeId::FLOAT:
      out.push_back(cast_f32(child.GetValue<float>()));
      break;
    case LogicalTypeId::DOUBLE:
      out.push_back(cast_f32(child.GetValue<double>()));
      break;
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<int64_t>())));
      break;
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<uint64_t>())));
      break;
    default:
      try {
        auto dbl = child.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
        out.push_back(cast_f32(dbl));
      } catch (Exception &) {
        throw InvalidInputException(
            "lance_search(hybrid) query vector elements must be numeric");
      }
    }
  }
  return out;
}

static LogicalType LanceSearchQueryType() {
  child_list_t<LogicalType> children;
  children.push_back({"type", LogicalType::VARCHAR});
  children.push_back({"query", LogicalType::VARCHAR});
  children.push_back({"column", LogicalType::VARCHAR});
  children.push_back({"vector", LogicalType::LIST(LogicalType::FLOAT)});
  children.push_back({"text", LogicalType::VARCHAR});
  children.push_back({"vector_column", LogicalType::VARCHAR});
  children.push_back({"text_column", LogicalType::VARCHAR});
  children.push_back({"alpha", LogicalType::FLOAT});
  children.push_back({"oversample_factor", LogicalType::INTEGER});
  return LogicalType::STRUCT(std::move(children));
}

static idx_t StructFieldIndex(const LogicalType &type, const string &name) {
  auto &children = StructType::GetChildTypes(type);
  for (idx_t i = 0; i < children.size(); i++) {
    if (children[i].first == name) {
      return i;
    }
  }
  throw InvalidInputException("missing field '" + name + "'");
}

static Value GetStructField(const Value &val, const string &name) {
  auto &type = val.type();
  if (type.id() != LogicalTypeId::STRUCT) {
    throw InvalidInputException("expected STRUCT");
  }
  auto idx = StructFieldIndex(type, name);
  auto children = StructValue::GetChildren(val);
  if (idx >= children.size()) {
    return Value();
  }
  return children[idx];
}

static string InferDefaultTextColumn(ClientContext &context, void *dataset,
                                     const string &path) {
  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    throw IOException("Failed to get Lance schema: " + path +
                      LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  ArrowTableSchema arrow_table;
  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(config, arrow_table,
                                               schema_root.arrow_schema);
  auto col_names = arrow_table.GetNames();
  auto col_types = arrow_table.GetTypes();

  vector<string> candidates;
  candidates.reserve(col_names.size());
  for (idx_t i = 0; i < col_names.size() && i < col_types.size(); i++) {
    if (col_types[i].id() == LogicalTypeId::VARCHAR) {
      candidates.push_back(col_names[i]);
    }
  }

  if (candidates.empty()) {
    throw InvalidInputException(
        "lance_search requires parameter text_column = '<text_column>'");
  }
  if (candidates.size() != 1) {
    throw InvalidInputException(
        "lance_search requires parameter text_column = '<text_column>' "
        "(multiple VARCHAR columns found: " +
        StringUtil::Join(candidates, ", ") + ")");
  }
  return candidates[0];
}

enum class LanceSearchMode : uint8_t { Fts = 0, Hybrid = 1 };

struct LanceSearchBindData : public TableFunctionData {
  LanceSearchMode mode = LanceSearchMode::Fts;

  string file_path;
  bool prefilter = false;

  // FTS mode
  string text_column;
  string query;

  // Hybrid mode
  string vector_column;
  vector<float> vector_query;
  string text_query;
  float alpha = 0.5F;
  uint32_t oversample_factor = 4;

  uint64_t k = 10;

  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  ~LanceSearchBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceSearchGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  string lance_filter_ir;
  bool filter_pushed_down = false;
  std::atomic<idx_t> filter_pushdown_fallbacks{0};

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceSearchLocalState : public ArrowScanLocalState {
  explicit LanceSearchLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                                 ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceSearchGlobalState *global_state = nullptr;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceSearchLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static bool LanceSearchLoadNextBatch(LanceSearchLocalState &local_state,
                                     const LanceSearchBindData &bind_data,
                                     LanceSearchGlobalState &global) {
  if (!local_state.stream) {
    const uint8_t *filter_ir =
        global.lance_filter_ir.empty()
            ? nullptr
            : reinterpret_cast<const uint8_t *>(global.lance_filter_ir.data());
    auto filter_ir_len = global.lance_filter_ir.size();

    auto create_stream = [&](const uint8_t *ir, size_t ir_len) -> void * {
      if (bind_data.mode == LanceSearchMode::Fts) {
        return lance_create_fts_stream_ir(
            bind_data.dataset, bind_data.text_column.c_str(),
            bind_data.query.c_str(), bind_data.k, ir, ir_len,
            bind_data.prefilter ? 1 : 0);
      }
      return lance_create_hybrid_stream_ir(
          bind_data.dataset, bind_data.vector_column.c_str(),
          bind_data.vector_query.data(), bind_data.vector_query.size(),
          bind_data.text_column.c_str(), bind_data.text_query.c_str(),
          bind_data.k, ir, ir_len, bind_data.prefilter ? 1 : 0, bind_data.alpha,
          bind_data.oversample_factor);
    };

    local_state.stream = create_stream(filter_ir, filter_ir_len);
    if (!local_state.stream && filter_ir && !bind_data.prefilter) {
      // Best-effort: if filter pushdown failed, retry without it and rely on
      // DuckDB-side filter execution for correctness.
      global.filter_pushdown_fallbacks.fetch_add(1);
      global.filter_pushed_down = false;
      local_state.filter_pushed_down = false;
      local_state.stream = create_stream(nullptr, 0);
    }
    if (!local_state.stream) {
      throw IOException("Failed to create Lance search stream" +
                        LanceFormatErrorSuffix());
    }
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

  local_state.global_state->record_batches.fetch_add(1);
  auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
  local_state.global_state->record_batch_rows.fetch_add(rows);

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static unique_ptr<FunctionData>
LanceSearchBindFts(ClientContext &context, TableFunctionBindInput &input,
                   vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() < 2) {
    throw InvalidInputException("lance_search requires (path, query)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_search requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException("lance_search requires a non-null query");
  }

  auto result = make_uniq<LanceSearchBindData>();
  result->mode = LanceSearchMode::Fts;
  result->file_path = input.inputs[0].GetValue<string>();
  result->query = input.inputs[1].GetValue<string>();

  int64_t k_val = 10;
  if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
    k_val = input.inputs[2].GetValue<int64_t>();
  }
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_search requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  result->dataset = LanceOpenDataset(context, result->file_path);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto column_named = input.named_parameters.find("text_column");
  if (column_named == input.named_parameters.end()) {
    column_named = input.named_parameters.find("column");
  }
  if (column_named != input.named_parameters.end() &&
      !column_named->second.IsNull()) {
    result->text_column = column_named->second.GetValue<string>();
  } else {
    result->text_column =
        InferDefaultTextColumn(context, result->dataset, result->file_path);
  }

  auto *schema_handle = lance_get_fts_schema(
      result->dataset, result->text_column.c_str(), result->query.c_str(),
      result->k, result->prefilter ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance FTS schema: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance FTS schema to Arrow C Data Interface" +
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

static unique_ptr<FunctionData>
LanceSearchBindStruct(ClientContext &context, TableFunctionBindInput &input,
                      vector<LogicalType> &return_types,
                      vector<string> &names) {
  if (input.inputs.size() < 2) {
    throw InvalidInputException("lance_search requires (path, query)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_search requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException("lance_search requires a non-null query");
  }

  auto query_val = input.inputs[1];
  auto type_val = GetStructField(query_val, "type");
  if (type_val.IsNull()) {
    throw InvalidInputException("lance_search query struct requires type");
  }
  auto type = type_val.GetValue<string>();

  auto result = make_uniq<LanceSearchBindData>();
  result->file_path = input.inputs[0].GetValue<string>();

  int64_t k_val = 10;
  if (input.inputs.size() >= 3 && !input.inputs[2].IsNull()) {
    k_val = input.inputs[2].GetValue<int64_t>();
  }
  auto k_named = input.named_parameters.find("k");
  if (k_named != input.named_parameters.end() && !k_named->second.IsNull()) {
    k_val =
        k_named->second.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
  }
  if (k_val <= 0) {
    throw InvalidInputException("lance_search requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  auto prefilter_named = input.named_parameters.find("prefilter");
  if (prefilter_named != input.named_parameters.end() &&
      !prefilter_named->second.IsNull()) {
    result->prefilter =
        prefilter_named->second.DefaultCastAs(LogicalType::BOOLEAN)
            .GetValue<bool>();
  }

  result->dataset = LanceOpenDataset(context, result->file_path);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  if (StringUtil::CIEquals(type, "match")) {
    result->mode = LanceSearchMode::Fts;
    auto q = GetStructField(query_val, "query");
    auto c = GetStructField(query_val, "column");
    if (q.IsNull() || c.IsNull()) {
      throw InvalidInputException(
          "lance_search(match) requires query and column");
    }
    result->query = q.GetValue<string>();
    result->text_column = c.GetValue<string>();

    auto *schema_handle = lance_get_fts_schema(
        result->dataset, result->text_column.c_str(), result->query.c_str(),
        result->k, result->prefilter ? 1 : 0);
    if (!schema_handle) {
      throw IOException("Failed to get Lance FTS schema: " + result->file_path +
                        LanceFormatErrorSuffix());
    }

    memset(&result->schema_root.arrow_schema, 0,
           sizeof(result->schema_root.arrow_schema));
    if (lance_schema_to_arrow(schema_handle,
                              &result->schema_root.arrow_schema) != 0) {
      lance_free_schema(schema_handle);
      throw IOException(
          "Failed to export Lance FTS schema to Arrow C Data Interface" +
          LanceFormatErrorSuffix());
    }
    lance_free_schema(schema_handle);
  } else if (StringUtil::CIEquals(type, "hybrid")) {
    result->mode = LanceSearchMode::Hybrid;

    auto vec = GetStructField(query_val, "vector");
    auto text = GetStructField(query_val, "text");
    auto vcol = GetStructField(query_val, "vector_column");
    auto tcol = GetStructField(query_val, "text_column");
    if (vec.IsNull() || text.IsNull() || vcol.IsNull() || tcol.IsNull()) {
      throw InvalidInputException(
          "lance_search(hybrid) requires vector, text, vector_column, "
          "text_column");
    }
    result->vector_query = ParseHybridQueryVector(vec);
    result->text_query = text.GetValue<string>();
    result->vector_column = vcol.GetValue<string>();
    result->text_column = tcol.GetValue<string>();

    auto alpha_val = GetStructField(query_val, "alpha");
    if (!alpha_val.IsNull()) {
      result->alpha =
          alpha_val.DefaultCastAs(LogicalType::FLOAT).GetValue<float>();
    }
    auto oversample_val = GetStructField(query_val, "oversample_factor");
    if (!oversample_val.IsNull()) {
      auto v = oversample_val.DefaultCastAs(LogicalType::INTEGER)
                   .GetValue<int32_t>();
      if (v > 0) {
        result->oversample_factor = NumericCast<uint32_t>(v);
      }
    }

    auto *schema_handle = lance_get_hybrid_schema(result->dataset);
    if (!schema_handle) {
      throw IOException("Failed to get Lance hybrid schema: " +
                        result->file_path + LanceFormatErrorSuffix());
    }

    memset(&result->schema_root.arrow_schema, 0,
           sizeof(result->schema_root.arrow_schema));
    if (lance_schema_to_arrow(schema_handle,
                              &result->schema_root.arrow_schema) != 0) {
      lance_free_schema(schema_handle);
      throw IOException(
          "Failed to export Lance hybrid schema to Arrow C Data Interface" +
          LanceFormatErrorSuffix());
    }
    lance_free_schema(schema_handle);
  } else {
    throw InvalidInputException("lance_search query type is not supported: " +
                                type);
  }

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
LanceSearchInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceSearchBindData>();
  auto state =
      make_uniq_base<GlobalTableFunctionState, LanceSearchGlobalState>();
  auto &global = state->Cast<LanceSearchGlobalState>();

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

  auto table_filters = BuildLanceTableFilterIRParts(
      bind_data.names, bind_data.types, input, true);
  if (bind_data.prefilter && !table_filters.all_prefilterable_filters_pushed) {
    throw InvalidInputException(
        "lance_search requires filter pushdown for prefilterable columns when "
        "prefilter=true");
  }

  bool has_table_filter_parts = !table_filters.parts.empty();
  string filter_ir_msg;
  if (!table_filters.parts.empty()) {
    if (!TryEncodeLanceFilterIRMessage(table_filters.parts, filter_ir_msg)) {
      filter_ir_msg.clear();
    }
    global.lance_filter_ir = std::move(filter_ir_msg);
  }
  if (bind_data.prefilter && has_table_filter_parts &&
      global.lance_filter_ir.empty()) {
    throw IOException("Failed to encode Lance filter IR");
  }
  global.filter_pushed_down =
      table_filters.all_filters_pushed && !global.lance_filter_ir.empty();
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceSearchLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                     GlobalTableFunctionState *global_state) {
  auto &global = global_state->Cast<LanceSearchGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceSearchLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  result->filter_pushed_down = global.filter_pushed_down;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }
  return std::move(result);
}

static void LanceSearchFunc(ClientContext &context, TableFunctionInput &data,
                            DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceSearchBindData>();
  auto &global_state = data.global_state->Cast<LanceSearchGlobalState>();
  auto &local_state = data.local_state->Cast<LanceSearchLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceSearchLoadNextBatch(local_state, bind_data, global_state)) {
        return;
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
                                        local_state.all_columns, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        output, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
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

static void LanceMatchQueryFunc(DataChunk &args, ExpressionState &,
                                Vector &result) {
  auto count = args.size();
  auto type = LanceSearchQueryType();
  result.SetVectorType(VectorType::FLAT_VECTOR);
  for (idx_t i = 0; i < count; i++) {
    auto query = args.data[0].GetValue(i);
    auto column = args.data[1].GetValue(i);
    if (query.IsNull() || column.IsNull()) {
      FlatVector::SetNull(result, i, true);
      continue;
    }

    std::vector<Value> children;
    children.reserve(9);
    children.push_back(Value("match"));
    children.push_back(query);
    children.push_back(column);
    children.push_back(Value(LogicalType::LIST(LogicalType::FLOAT)));
    children.push_back(Value(LogicalType::VARCHAR));
    children.push_back(Value(LogicalType::VARCHAR));
    children.push_back(Value(LogicalType::VARCHAR));
    children.push_back(Value(LogicalType::FLOAT));
    children.push_back(Value(LogicalType::INTEGER));
    result.SetValue(i, Value::STRUCT(type, std::move(children)));
  }
}

static void LanceHybridQueryFunc(DataChunk &args, ExpressionState &,
                                 Vector &result) {
  auto count = args.size();
  auto type = LanceSearchQueryType();
  result.SetVectorType(VectorType::FLAT_VECTOR);
  for (idx_t i = 0; i < count; i++) {
    auto vector_value = args.data[0].GetValue(i);
    auto text = args.data[1].GetValue(i);
    auto vector_column = args.data[2].GetValue(i);
    auto text_column = args.data[3].GetValue(i);

    if (vector_value.IsNull() || text.IsNull() || vector_column.IsNull() ||
        text_column.IsNull()) {
      FlatVector::SetNull(result, i, true);
      continue;
    }

    Value alpha(LogicalType::FLOAT);
    if (args.ColumnCount() >= 5) {
      alpha = args.data[4].GetValue(i);
    }
    Value oversample(LogicalType::INTEGER);
    if (args.ColumnCount() >= 6) {
      oversample = args.data[5].GetValue(i);
    }

    std::vector<Value> children;
    children.reserve(9);
    children.push_back(Value("hybrid"));
    children.push_back(Value(LogicalType::VARCHAR));
    children.push_back(Value(LogicalType::VARCHAR));
    children.push_back(vector_value);
    children.push_back(text);
    children.push_back(vector_column);
    children.push_back(text_column);
    children.push_back(alpha);
    children.push_back(oversample);

    result.SetValue(i, Value::STRUCT(type, std::move(children)));
  }
}

static void RegisterLanceFts(ExtensionLoader &loader) {
  TableFunction fts2("lance_search",
                     {LogicalType::VARCHAR, LogicalType::VARCHAR},
                     LanceSearchFunc, LanceSearchBindFts, LanceSearchInitGlobal,
                     LanceSearchLocalInit);
  fts2.named_parameters["text_column"] = LogicalType::VARCHAR;
  fts2.named_parameters["k"] = LogicalType::BIGINT;
  fts2.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  fts2.projection_pushdown = true;
  fts2.filter_pushdown = true;
  fts2.filter_prune = true;
  loader.RegisterFunction(fts2);

  TableFunction fts3(
      "lance_search",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BIGINT},
      LanceSearchFunc, LanceSearchBindFts, LanceSearchInitGlobal,
      LanceSearchLocalInit);
  fts3.named_parameters["text_column"] = LogicalType::VARCHAR;
  fts3.named_parameters["k"] = LogicalType::BIGINT;
  fts3.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  fts3.projection_pushdown = true;
  fts3.filter_pushdown = true;
  fts3.filter_prune = true;
  loader.RegisterFunction(fts3);

  TableFunction fts_struct("lance_search",
                           {LogicalType::VARCHAR, LanceSearchQueryType()},
                           LanceSearchFunc, LanceSearchBindStruct,
                           LanceSearchInitGlobal, LanceSearchLocalInit);
  fts_struct.named_parameters["k"] = LogicalType::BIGINT;
  fts_struct.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  fts_struct.projection_pushdown = true;
  fts_struct.filter_pushdown = true;
  fts_struct.filter_prune = true;
  loader.RegisterFunction(fts_struct);

  TableFunction fts_struct_k(
      "lance_search",
      {LogicalType::VARCHAR, LanceSearchQueryType(), LogicalType::BIGINT},
      LanceSearchFunc, LanceSearchBindStruct, LanceSearchInitGlobal,
      LanceSearchLocalInit);
  fts_struct_k.named_parameters["k"] = LogicalType::BIGINT;
  fts_struct_k.named_parameters["prefilter"] = LogicalType::BOOLEAN;
  fts_struct_k.projection_pushdown = true;
  fts_struct_k.filter_pushdown = true;
  fts_struct_k.filter_prune = true;
  loader.RegisterFunction(fts_struct_k);

  ScalarFunction match_query("lance_match_query",
                             {LogicalType::VARCHAR, LogicalType::VARCHAR},
                             LanceSearchQueryType(), LanceMatchQueryFunc);
  loader.RegisterFunction(match_query);

  ScalarFunctionSet hybrid_query_set("lance_hybrid_query");
  hybrid_query_set.AddFunction(ScalarFunction(
      {LogicalType::LIST(LogicalType::FLOAT), LogicalType::VARCHAR,
       LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceSearchQueryType(), LanceHybridQueryFunc));
  hybrid_query_set.AddFunction(ScalarFunction(
      {LogicalType::LIST(LogicalType::FLOAT), LogicalType::VARCHAR,
       LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::FLOAT},
      LanceSearchQueryType(), LanceHybridQueryFunc));
  hybrid_query_set.AddFunction(ScalarFunction(
      {LogicalType::LIST(LogicalType::FLOAT), LogicalType::VARCHAR,
       LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::FLOAT,
       LogicalType::INTEGER},
      LanceSearchQueryType(), LanceHybridQueryFunc));
  loader.RegisterFunction(hybrid_query_set);
}

void RegisterLanceSearch(ExtensionLoader &loader) {
  RegisterLanceVectorSearch(loader);
  RegisterLanceFts(loader);
}

} // namespace duckdb
