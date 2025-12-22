#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>

extern "C" {
void *lance_open_dataset(const char *path);
void lance_close_dataset(void *dataset);

void *lance_get_schema(void *dataset);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

void *lance_create_stream(void *dataset);
void *lance_stream_next(void *stream);
void lance_close_stream(void *stream);

uint64_t *lance_dataset_list_fragments(void *dataset, size_t *out_len);
void lance_free_fragment_list(uint64_t *ptr, size_t len);
void *lance_create_fragment_stream(void *dataset, uint64_t fragment_id,
                                   const char **columns, size_t columns_len,
                                   const char *filter_sql);

void lance_free_batch(void *batch);
int64_t lance_batch_num_rows(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}

namespace duckdb {

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  ~LanceScanBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceScanGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> next_fragment_idx{0};
  std::atomic<idx_t> lines_read{0};

  vector<uint64_t> fragment_ids;
  idx_t max_threads = 1;

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  vector<string> scan_column_names;
  string lance_filter_sql;

  idx_t MaxThreads() const override { return max_threads; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceScanLocalState : public ArrowScanLocalState {
  explicit LanceScanLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                               ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  idx_t fragment_pos = 0;
  SelectionVector filter_sel;

  ~LanceScanLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static string EscapeLanceColumnName(const string &name) {
  // Match Lance's backtick escaping strategy for nested column names.
  // Split by '.' and escape each segment.
  string result;
  idx_t start = 0;
  for (idx_t i = 0; i <= name.size(); i++) {
    if (i == name.size() || name[i] == '.') {
      auto segment = name.substr(start, i - start);
      if (!result.empty()) {
        result += ".";
      }
      result += "`";
      result += StringUtil::Replace(segment, "`", "``");
      result += "`";
      start = i + 1;
    }
  }
  return result;
}

static bool LanceFilterPushdownSupported(const TableFilter &filter) {
  switch (filter.filter_type) {
  case TableFilterType::CONSTANT_COMPARISON:
  case TableFilterType::IS_NULL:
  case TableFilterType::IS_NOT_NULL:
  case TableFilterType::IN_FILTER:
    return true;
  case TableFilterType::CONJUNCTION_AND: {
    auto &f = filter.Cast<ConjunctionAndFilter>();
    for (auto &child : f.child_filters) {
      if (!LanceFilterPushdownSupported(*child)) {
        return false;
      }
    }
    return true;
  }
  case TableFilterType::CONJUNCTION_OR: {
    auto &f = filter.Cast<ConjunctionOrFilter>();
    for (auto &child : f.child_filters) {
      if (!LanceFilterPushdownSupported(*child)) {
        return false;
      }
    }
    return true;
  }
  default:
    // Exclude DYNAMIC_FILTER, STRUCT_EXTRACT, EXPRESSION_FILTER,
    // OPTIONAL_FILTER, etc.
    return false;
  }
}

static string BuildLanceFilterSQL(const LanceScanBindData &bind_data,
                                  const TableFunctionInitInput &input) {
  if (!input.filters) {
    return "";
  }
  vector<string> predicates;
  predicates.reserve(input.filters->filters.size());

  for (auto &it : input.filters->filters) {
    auto scan_col_idx = it.first;
    auto &filter = *it.second;
    if (scan_col_idx >= input.column_ids.size()) {
      continue;
    }
    if (!LanceFilterPushdownSupported(filter)) {
      continue;
    }
    auto col_id = input.column_ids[scan_col_idx];
    if (col_id >= bind_data.names.size()) {
      continue;
    }
    auto col_name = EscapeLanceColumnName(bind_data.names[col_id]);
    predicates.push_back(filter.ToString(col_name));
  }

  if (predicates.empty()) {
    return "";
  }
  return StringUtil::Join(predicates, " AND ");
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

  result->dataset = lance_open_dataset(result->file_path.c_str());
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path);
  }

  auto *schema_handle = lance_get_schema(result->dataset);
  if (!schema_handle) {
    throw IOException("Failed to get schema from Lance dataset: " +
                      result->file_path);
  }

  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface");
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

  size_t fragment_count = 0;
  auto fragments_ptr =
      lance_dataset_list_fragments(bind_data.dataset, &fragment_count);
  if (!fragments_ptr && fragment_count != 0) {
    throw IOException("Failed to list Lance fragments");
  }
  if (fragment_count > 0) {
    if (!fragments_ptr) {
      throw IOException("Failed to list Lance fragments");
    }
    scan_state.fragment_ids.assign(fragments_ptr,
                                   fragments_ptr + fragment_count);
  }
  lance_free_fragment_list(fragments_ptr, fragment_count);

  auto threads = context.db->NumberOfThreads();
  scan_state.max_threads = MaxValue<idx_t>(
      1, MinValue<idx_t>(threads, scan_state.fragment_ids.size()));

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

  scan_state.lance_filter_sql = BuildLanceFilterSQL(bind_data, input);
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
  if (scan_global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, scan_global.scanned_types);
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

  const char *filter_sql = global_state.lance_filter_sql.empty()
                               ? nullptr
                               : global_state.lance_filter_sql.c_str();
  auto stream =
      lance_create_fragment_stream(bind_data.dataset, fragment_id,
                                   columns.data(), columns.size(), filter_sql);
  if (!stream && filter_sql) {
    // Best-effort: if filter pushdown failed, retry without it and rely on
    // DuckDB-side filter execution for correctness.
    stream =
        lance_create_fragment_stream(bind_data.dataset, fragment_id,
                                     columns.data(), columns.size(), nullptr);
  }
  if (!stream) {
    throw IOException("Failed to create Lance fragment stream");
  }
  local_state.stream = stream;
  return true;
}

static bool LanceScanLoadNextBatch(LanceScanLocalState &local_state) {
  if (!local_state.stream) {
    return false;
  }
  auto *batch = lance_stream_next(local_state.stream);
  if (!batch) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  ArrowSchema tmp_schema;
  memset(&tmp_schema, 0, sizeof(tmp_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array, &tmp_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface");
  }

  lance_free_batch(batch);

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                               DataChunk &chunk, SelectionVector &sel) {
  if (chunk.size() == 0) {
    return;
  }
  unique_ptr<Expression> combined;
  for (auto &it : filters.filters) {
    auto scan_col_idx = it.first;
    if (scan_col_idx >= chunk.ColumnCount()) {
      continue;
    }
    BoundReferenceExpression col_expr(chunk.data[scan_col_idx].GetType(),
                                      NumericCast<storage_t>(scan_col_idx));
    auto expr = it.second->ToExpression(col_expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conj = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conj->children.push_back(std::move(combined));
      conj->children.push_back(std::move(expr));
      combined = std::move(conj);
    }
  }
  if (!combined) {
    return;
  }
  ExpressionExecutor executor(context, *combined);
  auto selected = executor.SelectExpression(chunk, sel);
  if (selected != chunk.size()) {
    chunk.Slice(sel, selected);
  }
}

static void LanceScanFunc(ClientContext &context, TableFunctionInput &data,
                          DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceScanBindData>();
  auto &global_state = data.global_state->Cast<LanceScanGlobalState>();
  auto &local_state = data.local_state->Cast<LanceScanLocalState>();

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

void RegisterLanceScan(ExtensionLoader &loader) {
  TableFunction lance_scan("lance_scan", {LogicalType::VARCHAR}, LanceScanFunc,
                           LanceScanBind, LanceScanInitGlobal,
                           LanceScanLocalInit);
  lance_scan.projection_pushdown = true;
  lance_scan.filter_pushdown = true;
  lance_scan.filter_prune = true;
  loader.RegisterFunction(lance_scan);
}

} // namespace duckdb
