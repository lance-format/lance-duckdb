#include "duckdb/parser/query_node/merge_query_node.hpp"
#include "duckdb.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/statement/merge_into_statement.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_default_expression.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "lance_blob.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_arrow_compat.hpp"
#include "lance_insert.hpp"
#include "lance_merge.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include <cstdint>
#include <set>

namespace duckdb {

struct LanceMergeAction {
  MergeActionType action_type;
  unique_ptr<Expression> condition;
  vector<PhysicalIndex> update_columns;
  vector<string> update_column_names;
  vector<unique_ptr<Expression>> expressions;
};

struct LanceMergeActionRange {
  MergeActionCondition condition;
  idx_t start = 0;
  idx_t end = 0;
};

static vector<unique_ptr<Expression>>
BuildMergeUpdateExpressions(LogicalMergeInto &op,
                            BoundMergeIntoAction &action) {
  if (action.columns.size() != action.expressions.size()) {
    throw InternalException("Lance MERGE UPDATE action columns/expressions "
                            "size mismatch");
  }

  vector<unique_ptr<Expression>> result;
  result.reserve(action.expressions.size());

  for (idx_t i = 0; i < action.expressions.size(); i++) {
    if (!action.expressions[i]) {
      throw InternalException("Lance MERGE UPDATE action has null expression");
    }

    if (action.expressions[i]->GetExpressionClass() ==
            ExpressionClass::BOUND_DEFAULT ||
        action.expressions[i]->GetExpressionType() ==
            ExpressionType::VALUE_DEFAULT) {
      auto &target_col = op.table.GetColumns().GetColumn(action.columns[i]);
      auto storage_idx = target_col.StorageOid();
      if (storage_idx >= op.bound_defaults.size() ||
          !op.bound_defaults[storage_idx]) {
        throw InternalException(
            "Lance MERGE UPDATE default expression is missing for target "
            "column");
      }
      result.push_back(op.bound_defaults[storage_idx]->Copy());
    } else {
      result.push_back(std::move(action.expressions[i]));
    }
  }

  return result;
}

static vector<unique_ptr<Expression>>
BuildMergeInsertExpressions(LogicalMergeInto &op,
                            BoundMergeIntoAction &action) {
  if (action.column_index_map.empty()) {
    return std::move(action.expressions);
  }

  vector<unique_ptr<Expression>> result;
  result.reserve(op.table.GetColumns().PhysicalColumnCount());

  for (auto &col : op.table.GetColumns().Physical()) {
    auto mapped_index = action.column_index_map[col.Physical()];
    if (mapped_index == DConstants::INVALID_INDEX) {
      auto storage_idx = col.StorageOid();
      if (storage_idx >= op.bound_defaults.size() ||
          !op.bound_defaults[storage_idx]) {
        throw InternalException(
            "Lance MERGE INSERT default expression is missing for target "
            "column");
      }
      result.push_back(op.bound_defaults[storage_idx]->Copy());
    } else {
      if (mapped_index >= action.expressions.size() ||
          !action.expressions[mapped_index]) {
        throw InternalException(
            "Lance MERGE INSERT action expression index is out of range");
      }
      result.push_back(std::move(action.expressions[mapped_index]));
    }
  }

  return result;
}

static bool TryExtractRowIdsFromChunk(Vector &rowid_vec, idx_t count,
                                      vector<uint64_t> &out_row_ids,
                                      string &out_error) {
  out_row_ids.clear();
  out_error.clear();

  UnifiedVectorFormat format;
  rowid_vec.ToUnifiedFormat(format);
  auto row_ids = UnifiedVectorFormat::GetData<int64_t>(format);

  out_row_ids.reserve(count);
  for (idx_t i = 0; i < count; i++) {
    auto idx = format.sel->get_index(i);
    if (!format.validity.RowIsValid(idx)) {
      out_error = "MERGE target rowid is NULL for a mutating action";
      return false;
    }
    auto rowid = row_ids[idx];
    if (rowid < 0) {
      out_error = "MERGE target rowid must be non-negative";
      return false;
    }
    out_row_ids.push_back(NumericCast<uint64_t>(rowid));
  }

  return true;
}

static void
RegisterMutatedTargetRows(const vector<uint64_t> &row_ids,
                          unordered_set<uint64_t> &mutated_target_rowids) {
  for (auto row_id : row_ids) {
    if (!mutated_target_rowids.insert(row_id).second) {
      throw InvalidInputException(
          "MERGE INTO command cannot affect the same target row more than "
          "once. "
          "A target row matched more than one source row; ensure the source "
          "rows "
          "are deduplicated or the ON condition is unique.");
    }
  }
}

static void AppendMergeReturningRows(ClientContext &context,
                                     ColumnDataCollection &collection,
                                     DataChunk &rows,
                                     const string &merge_action) {
  if (rows.size() == 0) {
    return;
  }

  vector<LogicalType> output_types = rows.GetTypes();
  output_types.push_back(LogicalType::VARCHAR);

  DataChunk out;
  out.Initialize(context, output_types);
  out.SetChildCardinality(rows.size());

  for (idx_t c = 0; c < rows.ColumnCount(); c++) {
    out.data[c].Reference(rows.data[c]);
  }
  out.data[rows.ColumnCount()].Reference(Value(merge_action),
                                         count_t(rows.size()));

  collection.Append(out);
}

static void ConvertArrowArrayToDuckDataChunk(ClientContext &context,
                                             ArrowArrayWrapper &arrow_chunk,
                                             ArrowSchema &schema,
                                             DataChunk &out,
                                             const vector<LogicalType> &types);

static void
ConvertBlobInputChunkToBinary(ClientContext &context, DataChunk &chunk,
                              const vector<string> &column_names,
                              const vector<LogicalType> &output_types,
                              DataChunk &out) {
  if (chunk.ColumnCount() == 0) {
    out.Initialize(context, vector<LogicalType>());
    out.SetChildCardinality(chunk.size());
    return;
  }
  auto props = context.GetClientProperties();
  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(chunk, &array, props, extension_type_cast);

  ArrowSchemaWrapper input_schema;
  input_schema.arrow_schema = {};
  ArrowConverter::ToArrowSchema(&input_schema.arrow_schema, chunk.GetTypes(),
                                column_names, props);

  LanceBatchSourceOptions source_options;
  LanceCollectIngestSourceOptions(context, chunk, source_options);
  LanceBlobSourceOptionArrays option_arrays;
  BuildLanceBlobSourceOptionArrays(source_options, option_arrays);

  ArrowSchema output_schema;
  memset(&output_schema, 0, sizeof(output_schema));
  ArrowArray output_array;
  memset(&output_array, 0, sizeof(output_array));
  auto rc = lance_blob_inputs_to_binary(
      &input_schema.arrow_schema, &array, option_arrays.Uris(),
      option_arrays.KeyOffsets(), option_arrays.ValueOffsets(),
      option_arrays.Counts(), option_arrays.counts.size(),
      NumericCast<uint64_t>(LanceBlobIngestBatchBudget(context)),
      LanceGetSessionHandle(context), &output_schema, &output_array);
  if (array.release) {
    array.release(&array);
  }
  if (rc != 0) {
    throw IOException("Failed to materialize Blob input values for RETURNING" +
                      LanceFormatErrorSuffix());
  }

  ArrowArrayWrapper array_wrapper;
  array_wrapper.arrow_array = output_array;
  output_array.release = nullptr;
  ArrowSchemaWrapper schema_wrapper;
  schema_wrapper.arrow_schema = output_schema;
  output_schema.release = nullptr;
  ConvertArrowArrayToDuckDataChunk(
      context, array_wrapper, schema_wrapper.arrow_schema, out, output_types);
}

static void AddInsertChunkToMergeHandle(void *merge_handle, DataChunk &chunk,
                                        const vector<string> &column_names,
                                        ClientContext &context) {
  if (chunk.size() == 0) {
    return;
  }

  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;
  auto props = context.GetClientProperties();

  LanceBatchSourceOptions source_options;
  LanceCollectIngestSourceOptions(context, chunk, source_options);
  LanceBlobSourceOptionArrays option_arrays;
  BuildLanceBlobSourceOptionArrays(source_options, option_arrays);

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(chunk, &array, props, extension_type_cast);

  ArrowSchemaWrapper schema;
  schema.arrow_schema = {};
  ArrowConverter::ToArrowSchema(&schema.arrow_schema, chunk.GetTypes(),
                                column_names, props);

  auto rc = lance_merge_add_insert_batch(
      merge_handle, &schema.arrow_schema, &array, option_arrays.Uris(),
      option_arrays.KeyOffsets(), option_arrays.ValueOffsets(),
      option_arrays.Counts(), option_arrays.counts.size());
  if (array.release) {
    array.release(&array);
  }

  if (rc != 0) {
    throw IOException("Failed to add MERGE insert batch to Lance transaction" +
                      LanceFormatErrorSuffix());
  }
}

static void AddUpdateChunkToMergeHandle(void *merge_handle, DataChunk &chunk,
                                        const vector<string> &column_names,
                                        ClientContext &context) {
  if (chunk.size() == 0) {
    return;
  }
  if (column_names.size() + 1 != chunk.ColumnCount()) {
    throw InternalException(
        "Lance MERGE UPDATE column names do not match value chunk");
  }

  auto props = context.GetClientProperties();
  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(chunk, &array, props, extension_type_cast);

  vector<string> names = column_names;
  names.push_back("_rowid");
  ArrowSchemaWrapper schema;
  schema.arrow_schema = {};
  ArrowConverter::ToArrowSchema(&schema.arrow_schema, chunk.GetTypes(), names,
                                props);

  LanceBatchSourceOptions source_options;
  LanceCollectIngestSourceOptions(context, chunk, source_options);
  LanceBlobSourceOptionArrays option_arrays;
  BuildLanceBlobSourceOptionArrays(source_options, option_arrays);

  auto rc = lance_merge_add_update_batch(
      merge_handle, &schema.arrow_schema, &array, option_arrays.Uris(),
      option_arrays.KeyOffsets(), option_arrays.ValueOffsets(),
      option_arrays.Counts(), option_arrays.counts.size());
  if (array.release) {
    array.release(&array);
  }
  if (rc != 0) {
    throw IOException("Failed to add MERGE UPDATE values to Lance transaction" +
                      LanceFormatErrorSuffix());
  }
}

static void ConvertArrowArrayToDuckDataChunk(ClientContext &context,
                                             ArrowArrayWrapper &arrow_chunk,
                                             ArrowSchema &schema,
                                             DataChunk &out,
                                             const vector<LogicalType> &types) {
  ArrowTableSchema arrow_table;
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table, schema);

  auto owned = make_uniq<ArrowArrayWrapper>();
  owned->arrow_array = arrow_chunk.arrow_array;
  arrow_chunk.arrow_array.release = nullptr;

  ArrowScanLocalState scan_state(std::move(owned), context);

  out.Reset();
  out.Initialize(context, types);
  out.SetChildCardinality(
      NumericCast<idx_t>(scan_state.chunk->arrow_array.length));
  ArrowTableFunction::ArrowToDuckDB(scan_state, arrow_table.GetColumns(), out,
                                    true, DConstants::INVALID_INDEX);
}

class PhysicalLanceMergeInto;

class LanceMergeGlobalState : public GlobalSinkState {
public:
  LanceMergeGlobalState(ClientContext &context_p,
                        const PhysicalLanceMergeInto &op_p);

  ~LanceMergeGlobalState() override;

  void EnsureMergeHandle(ClientContext &context);
  void EnsureTakeDataset(ClientContext &context);
  void AddDeleteRowIds(const vector<uint64_t> &row_ids);

  template <class FUNC>
  void ForEachTakenChunkSubset(ClientContext &context,
                               const vector<uint64_t> &row_ids,
                               const vector<string> &column_names,
                               const vector<LogicalType> &column_types,
                               FUNC &&fn);

public:
  ClientContext &context;
  const PhysicalLanceMergeInto &op;

  idx_t changed_rows = 0;
  unordered_set<uint64_t> mutated_target_rowids;

  string display_uri;
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;

  void *merge_handle = nullptr;
  void *take_dataset = nullptr;

  ColumnDataCollection return_collection;
};

struct LanceMergeActionLocalState {
  unique_ptr<ExpressionExecutor> condition_executor;
  unique_ptr<ExpressionExecutor> value_executor;
  unique_ptr<DataChunk> value_chunk;
};

class LanceMergeLocalState : public LocalSinkState {
public:
  LanceMergeLocalState(ExecutionContext &context,
                       const PhysicalLanceMergeInto &op);

public:
  vector<LanceMergeActionLocalState> action_states;

  SelectionVector matched_sel;
  SelectionVector not_matched_by_target_sel;
  SelectionVector not_matched_by_source_sel;

  SelectionVector current_sel;
  SelectionVector selected_sel;
  SelectionVector remaining_sel;

  idx_t matched_count = 0;
  idx_t not_matched_by_target_count = 0;
  idx_t not_matched_by_source_count = 0;

  unique_ptr<DataChunk> sliced_chunk;
};

class LanceMergeSourceState : public GlobalSourceState {
public:
  explicit LanceMergeSourceState(LanceMergeGlobalState &sink_state,
                                 bool return_chunk_p)
      : return_chunk(return_chunk_p) {
    if (return_chunk) {
      sink_state.return_collection.InitializeScan(scan_state);
    }
  }

  bool return_chunk = false;
  bool emitted = false;
  ColumnDataScanState scan_state;
};

class PhysicalLanceMergeInto final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceMergeInto(PhysicalPlan &physical_plan,
                         vector<LogicalType> types_p, LanceTableEntry &table_p,
                         vector<string> table_column_names_p,
                         vector<LogicalType> table_column_types_p,
                         vector<unique_ptr<LanceMergeAction>> actions_p,
                         vector<LanceMergeActionRange> action_ranges_p,
                         idx_t row_id_index_p, optional_idx source_marker_p,
                         bool return_chunk_p, vector<bool> returning_needed_p,
                         idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), table_column_names(std::move(table_column_names_p)),
        table_column_types(std::move(table_column_types_p)),
        actions(std::move(actions_p)),
        action_ranges(std::move(action_ranges_p)), row_id_index(row_id_index_p),
        source_marker(source_marker_p), return_chunk(return_chunk_p),
        returning_needed(std::move(returning_needed_p)) {}

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &context) const override {
    return make_uniq<LanceMergeGlobalState>(context, *this);
  }

  unique_ptr<LocalSinkState>
  GetLocalSinkState(ExecutionContext &context) const override {
    return make_uniq<LanceMergeLocalState>(context, *this);
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override;

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override;

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    auto &sink = sink_state->Cast<LanceMergeGlobalState>();
    return make_uniq<LanceMergeSourceState>(sink, return_chunk);
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override;

  string GetName() const override { return "LanceMergeInto"; }

  void SetReturningNeeded(vector<bool> needed) {
    returning_needed = std::move(needed);
  }

  idx_t GetRangeIndex(MergeActionCondition condition) const {
    for (idx_t i = 0; i < action_ranges.size(); i++) {
      if (action_ranges[i].condition == condition) {
        return i;
      }
    }
    throw InternalException("Lance MERGE missing action range");
  }

private:
  void ComputeMatches(LanceMergeLocalState &local_state,
                      DataChunk &chunk) const;

  optional_ptr<DataChunk>
  SliceActionInput(ClientContext &context, DataChunk &chunk,
                   const SelectionVector &sel, idx_t count,
                   LanceMergeLocalState &local_state) const;

  void ProcessMatchedRows(ExecutionContext &context, DataChunk &chunk,
                          const SelectionVector &initial_sel,
                          idx_t initial_count, MergeActionCondition condition,
                          idx_t range_index, LanceMergeGlobalState &gstate,
                          LanceMergeLocalState &lstate) const;

  void HandleActionRows(ExecutionContext &context, DataChunk &input_chunk,
                        const LanceMergeAction &action,
                        LanceMergeActionLocalState &action_state,
                        MergeActionCondition condition,
                        LanceMergeGlobalState &gstate) const;

  void HandleMergeUpdateRows(ExecutionContext &context, DataChunk &input_chunk,
                             DataChunk &update_values,
                             const LanceMergeAction &action,
                             LanceMergeGlobalState &gstate) const;

  void HandleMergeInsertRows(ExecutionContext &context,
                             DataChunk &insert_values,
                             LanceMergeGlobalState &gstate) const;

  void HandleMergeDeleteRows(ExecutionContext &context, DataChunk &input_chunk,
                             LanceMergeGlobalState &gstate) const;

  void HandleMergeErrorRows(ExecutionContext &context, DataChunk &input_chunk,
                            const LanceMergeAction &action,
                            LanceMergeActionLocalState &action_state,
                            MergeActionCondition condition) const;

public:
  LanceTableEntry &table;
  vector<string> table_column_names;
  vector<LogicalType> table_column_types;
  vector<unique_ptr<LanceMergeAction>> actions;
  vector<LanceMergeActionRange> action_ranges;
  idx_t row_id_index;
  optional_idx source_marker;
  bool return_chunk;
  // Columns the SQL RETURNING projection references. Unreferenced columns are
  // emitted as NULL and never read from storage.
  vector<bool> returning_needed;
};

LanceMergeGlobalState::LanceMergeGlobalState(ClientContext &context_p,
                                             const PhysicalLanceMergeInto &op_p)
    : context(context_p), op(op_p),
      return_collection(context_p,
                        op_p.return_chunk
                            ? op_p.types
                            : vector<LogicalType>{LogicalType::BIGINT}) {}

LanceMergeGlobalState::~LanceMergeGlobalState() {
  if (merge_handle) {
    lance_merge_abort(merge_handle);
    merge_handle = nullptr;
  }
  if (take_dataset) {
    lance_close_dataset(take_dataset);
    take_dataset = nullptr;
  }
}

void LanceMergeGlobalState::EnsureMergeHandle(ClientContext &context) {
  if (merge_handle) {
    return;
  }

  ResolveLanceStorageOptionsForTable(context, op.table, open_path, option_keys,
                                     option_values, display_uri);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

  auto rc = lance_merge_begin_with_storage_options(
      open_path.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      LANCE_DEFAULT_MAX_ROWS_PER_FILE, LANCE_DEFAULT_MAX_ROWS_PER_GROUP,
      LANCE_DEFAULT_MAX_BYTES_PER_FILE,
      NumericCast<uint64_t>(LanceBlobIngestBatchBudget(context)),
      LanceGetSessionHandle(context), &merge_handle);
  if (rc != 0 || !merge_handle) {
    throw IOException("Failed to start Lance MERGE transaction for '" +
                      open_path + "'" + LanceFormatErrorSuffix());
  }
}

void LanceMergeGlobalState::EnsureTakeDataset(ClientContext &context) {
  if (take_dataset) {
    return;
  }

  take_dataset = LanceOpenDatasetForTable(context, op.table, display_uri);
  if (!take_dataset) {
    throw IOException("Failed to open Lance dataset for MERGE row fetch: " +
                      display_uri + LanceFormatErrorSuffix());
  }
}

void LanceMergeGlobalState::AddDeleteRowIds(const vector<uint64_t> &row_ids) {
  if (row_ids.empty()) {
    return;
  }
  auto rc = lance_merge_add_delete_rowids(merge_handle, row_ids.data(),
                                          row_ids.size());
  if (rc != 0) {
    throw IOException("Failed to add MERGE delete rowids for '" + open_path +
                      "'" + LanceFormatErrorSuffix());
  }
}

template <class FUNC>
void LanceMergeGlobalState::ForEachTakenChunkSubset(
    ClientContext &context, const vector<uint64_t> &row_ids,
    const vector<string> &column_names, const vector<LogicalType> &column_types,
    FUNC &&fn) {
  if (row_ids.empty()) {
    return;
  }
  if (column_names.empty()) {
    // No unassigned columns need fetching; still emit the requested row count
    // so the caller can build assigned-only row images.
    return;
  }

  EnsureTakeDataset(context);

  vector<const char *> column_ptrs;
  column_ptrs.reserve(column_names.size());
  for (auto &name : column_names) {
    column_ptrs.push_back(name.c_str());
  }

  auto *stream = lance_create_dataset_take_stream_unfiltered(
      take_dataset, row_ids.data(), row_ids.size(),
      column_ptrs.empty() ? nullptr : column_ptrs.data(), column_ptrs.size());
  if (!stream) {
    throw IOException("Failed to create Lance take stream for MERGE" +
                      LanceFormatErrorSuffix());
  }

  std::unique_ptr<void, decltype(&lance_close_stream)> stream_owner(
      stream, lance_close_stream);

  idx_t taken_rows = 0;
  while (true) {
    void *batch = nullptr;
    auto rc = lance_stream_next(stream, &batch);
    if (rc == 1) {
      break;
    }
    if (rc != 0) {
      throw IOException("Failed to read Lance take stream for MERGE" +
                        LanceFormatErrorSuffix());
    }

    LanceArrowBatch exported(
        batch, "Failed to export Lance MERGE take batch to Arrow");
    auto &arrow_chunk = exported.array;
    auto &schema = exported.schema.arrow_schema;
    LanceCoerceArrowSchemaForDuckDB(&schema);

    DataChunk taken;
    ConvertArrowArrayToDuckDataChunk(context, *arrow_chunk, schema, taken,
                                     column_types);

    fn(taken, taken_rows);
    taken_rows += taken.size();
  }

  stream_owner.reset();

  if (taken_rows != row_ids.size()) {
    throw IOException("Lance MERGE take returned unexpected row count: " +
                      to_string(taken_rows) + " rows for " +
                      to_string(row_ids.size()) + " requested rowids");
  }
}

LanceMergeLocalState::LanceMergeLocalState(ExecutionContext &context,
                                           const PhysicalLanceMergeInto &op)
    : matched_sel(STANDARD_VECTOR_SIZE),
      not_matched_by_target_sel(STANDARD_VECTOR_SIZE),
      not_matched_by_source_sel(STANDARD_VECTOR_SIZE),
      current_sel(STANDARD_VECTOR_SIZE), selected_sel(STANDARD_VECTOR_SIZE),
      remaining_sel(STANDARD_VECTOR_SIZE) {
  for (auto &action : op.actions) {
    LanceMergeActionLocalState action_state;
    if (action->condition) {
      action_state.condition_executor =
          make_uniq<ExpressionExecutor>(context.client, *action->condition);
    }
    if (!action->expressions.empty()) {
      action_state.value_executor =
          make_uniq<ExpressionExecutor>(context.client, action->expressions);
      vector<LogicalType> value_types;
      value_types.reserve(action->expressions.size());
      for (auto &expr : action->expressions) {
        value_types.push_back(expr->GetReturnType());
      }
      action_state.value_chunk = make_uniq<DataChunk>();
      action_state.value_chunk->Initialize(context.client, value_types);
    }
    action_states.push_back(std::move(action_state));
  }

  sliced_chunk = make_uniq<DataChunk>();
  sliced_chunk->Initialize(context.client, op.children[0].get().types);
}

void PhysicalLanceMergeInto::ComputeMatches(LanceMergeLocalState &local_state,
                                            DataChunk &chunk) const {
  local_state.matched_count = 0;
  local_state.not_matched_by_target_count = 0;
  local_state.not_matched_by_source_count = 0;

  UnifiedVectorFormat rowid_data;
  chunk.data[row_id_index].ToUnifiedFormat(rowid_data);

  if (source_marker.IsValid()) {
    UnifiedVectorFormat source_data;
    chunk.data[source_marker.GetIndex()].ToUnifiedFormat(source_data);
    for (idx_t i = 0; i < chunk.size(); i++) {
      auto source_idx = source_data.sel->get_index(i);
      auto rowid_idx = rowid_data.sel->get_index(i);

      if (!source_data.validity.RowIsValid(source_idx)) {
        local_state.not_matched_by_source_sel.set_index(
            local_state.not_matched_by_source_count++, i);
      } else if (!rowid_data.validity.RowIsValid(rowid_idx)) {
        local_state.not_matched_by_target_sel.set_index(
            local_state.not_matched_by_target_count++, i);
      } else {
        local_state.matched_sel.set_index(local_state.matched_count++, i);
      }
    }
  } else {
    for (idx_t i = 0; i < chunk.size(); i++) {
      auto rowid_idx = rowid_data.sel->get_index(i);
      if (rowid_data.validity.RowIsValid(rowid_idx)) {
        local_state.matched_sel.set_index(local_state.matched_count++, i);
      } else {
        local_state.not_matched_by_target_sel.set_index(
            local_state.not_matched_by_target_count++, i);
      }
    }
  }
}

optional_ptr<DataChunk> PhysicalLanceMergeInto::SliceActionInput(
    ClientContext &, DataChunk &chunk, const SelectionVector &sel, idx_t count,
    LanceMergeLocalState &local_state) const {
  if (count == 0) {
    return nullptr;
  }
  local_state.sliced_chunk->Reset();
  local_state.sliced_chunk->Slice(chunk, sel, count);
  return local_state.sliced_chunk.get();
}

void PhysicalLanceMergeInto::HandleMergeUpdateRows(
    ExecutionContext &context, DataChunk &input_chunk, DataChunk &update_values,
    const LanceMergeAction &action, LanceMergeGlobalState &gstate) const {
  if (action.update_columns.size() != update_values.ColumnCount()) {
    throw InternalException(
        "Lance MERGE UPDATE values column count does not match target columns");
  }

  NormalizeIngestRowsInChunk(context.client, update_values);

  vector<uint64_t> row_ids;
  string rowid_error;
  if (!TryExtractRowIdsFromChunk(input_chunk.data[row_id_index],
                                 input_chunk.size(), row_ids, rowid_error)) {
    throw ConstraintException(rowid_error);
  }
  RegisterMutatedTargetRows(row_ids, gstate.mutated_target_rowids);

  if (return_chunk) {
    // The row image is assembled from the newly assigned values plus only the
    // old columns the action does not assign. Columns that the SQL RETURNING
    // projection never references stay NULL and are never read.
    vector<idx_t> assigned_columns;
    vector<idx_t> needed_assigned_columns;
    vector<LogicalType> needed_assigned_types;
    vector<string> needed_assigned_names;
    assigned_columns.reserve(action.update_columns.size());
    for (idx_t update_col_idx = 0;
         update_col_idx < action.update_columns.size(); update_col_idx++) {
      auto column_idx = action.update_columns[update_col_idx].index;
      if (column_idx >= table_column_types.size()) {
        throw InternalException(
            "Lance MERGE UPDATE target column index is out of range");
      }
      assigned_columns.push_back(column_idx);
      if (returning_needed[column_idx]) {
        needed_assigned_columns.push_back(column_idx);
        needed_assigned_types.push_back(table_column_types[column_idx]);
        needed_assigned_names.push_back(
            action.update_column_names[update_col_idx]);
      }
    }
    vector<bool> is_assigned(table_column_types.size(), false);
    for (auto column_idx : assigned_columns) {
      is_assigned[column_idx] = true;
    }
    vector<string> taken_names;
    vector<LogicalType> taken_types;
    vector<idx_t> taken_to_table;
    for (idx_t column_idx = 0; column_idx < table_column_types.size();
         column_idx++) {
      if (is_assigned[column_idx] || !returning_needed[column_idx]) {
        continue;
      }
      taken_names.push_back(table_column_names[column_idx]);
      taken_types.push_back(table_column_types[column_idx]);
      taken_to_table.push_back(column_idx);
    }

    DataChunk needed_values;
    needed_values.Initialize(context.client, [&]() {
      vector<LogicalType> types;
      types.reserve(needed_assigned_columns.size());
      for (auto column_idx : needed_assigned_columns) {
        auto input_idx = std::find(assigned_columns.begin(),
                                   assigned_columns.end(), column_idx) -
                         assigned_columns.begin();
        types.push_back(update_values.data[input_idx].GetType());
      }
      return types;
    }());
    needed_values.SetChildCardinality(update_values.size());
    for (idx_t i = 0; i < needed_assigned_columns.size(); i++) {
      auto input_idx =
          std::find(assigned_columns.begin(), assigned_columns.end(),
                    needed_assigned_columns[i]) -
          assigned_columns.begin();
      needed_values.data[i].Reference(update_values.data[input_idx]);
    }
    needed_values.Flatten();

    DataChunk returning_values;
    ConvertBlobInputChunkToBinary(context.client, needed_values,
                                  needed_assigned_names, needed_assigned_types,
                                  returning_values);

    auto patch_rows = [&](DataChunk &taken, idx_t row_offset) {
      DataChunk patched;
      patched.Initialize(context.client, table_column_types);
      patched.SetChildCardinality(taken.size());
      for (idx_t column_idx = 0; column_idx < table_column_types.size();
           column_idx++) {
        if (!returning_needed[column_idx]) {
          Value null_value(table_column_types[column_idx]);
          patched.data[column_idx].Reference(null_value,
                                             count_t(patched.size()));
        }
      }
      for (idx_t column_idx = 0; column_idx < taken_to_table.size();
           column_idx++) {
        patched.data[taken_to_table[column_idx]].Reference(
            taken.data[column_idx]);
      }
      for (idx_t i = 0; i < needed_assigned_columns.size(); i++) {
        for (idx_t row_idx = 0; row_idx < taken.size(); row_idx++) {
          patched.data[needed_assigned_columns[i]].SetValue(
              row_idx, returning_values.data[i].GetValue(row_offset + row_idx));
        }
      }
      AppendMergeReturningRows(context.client, gstate.return_collection,
                               patched, "UPDATE");
    };

    if (taken_names.empty()) {
      DataChunk empty_taken;
      empty_taken.Initialize(context.client, {LogicalType::BOOLEAN});
      empty_taken.SetChildCardinality(row_ids.size());
      patch_rows(empty_taken, 0);
    } else {
      gstate.ForEachTakenChunkSubset(context.client, row_ids, taken_names,
                                     taken_types,
                                     [&](DataChunk &taken, idx_t row_offset) {
                                       patch_rows(taken, row_offset);
                                     });
    }
  }

  auto rowid_type = input_chunk.data[row_id_index].GetType();
  DataChunk value_batch;
  auto value_batch_types = update_values.GetTypes();
  value_batch_types.push_back(rowid_type);
  value_batch.Initialize(context.client, value_batch_types);
  value_batch.SetChildCardinality(row_ids.size());
  for (idx_t c = 0; c < update_values.ColumnCount(); c++) {
    value_batch.data[c].Reference(update_values.data[c]);
  }
  value_batch.data[update_values.ColumnCount()].Reference(
      input_chunk.data[row_id_index]);
  value_batch.Flatten();

  gstate.EnsureMergeHandle(context.client);
  AddUpdateChunkToMergeHandle(gstate.merge_handle, value_batch,
                              action.update_column_names, context.client);

  gstate.changed_rows += input_chunk.size();
}

void PhysicalLanceMergeInto::HandleMergeInsertRows(
    ExecutionContext &context, DataChunk &insert_values,
    LanceMergeGlobalState &gstate) const {
  if (insert_values.ColumnCount() != table_column_types.size()) {
    throw InternalException(
        "Lance MERGE INSERT values column count does not match table schema");
  }

  NormalizeIngestRowsInChunk(context.client, insert_values);

  gstate.EnsureMergeHandle(context.client);
  AddInsertChunkToMergeHandle(gstate.merge_handle, insert_values,
                              table_column_names, context.client);

  if (return_chunk) {
    vector<string> needed_names;
    vector<LogicalType> needed_types;
    vector<idx_t> needed_to_table;
    for (idx_t column_idx = 0; column_idx < table_column_types.size();
         column_idx++) {
      if (!returning_needed[column_idx]) {
        continue;
      }
      needed_names.push_back(table_column_names[column_idx]);
      needed_types.push_back(table_column_types[column_idx]);
      needed_to_table.push_back(column_idx);
    }
    vector<LogicalType> needed_input_types;
    needed_input_types.reserve(needed_to_table.size());
    for (auto column_idx : needed_to_table) {
      needed_input_types.push_back(insert_values.data[column_idx].GetType());
    }
    DataChunk needed_values;
    needed_values.Initialize(context.client, needed_input_types);
    needed_values.SetChildCardinality(insert_values.size());
    for (idx_t i = 0; i < needed_to_table.size(); i++) {
      needed_values.data[i].Reference(insert_values.data[needed_to_table[i]]);
    }
    needed_values.Flatten();

    DataChunk returning_values;
    ConvertBlobInputChunkToBinary(context.client, needed_values, needed_names,
                                  needed_types, returning_values);

    DataChunk patched;
    patched.Initialize(context.client, table_column_types);
    patched.SetChildCardinality(insert_values.size());
    for (idx_t column_idx = 0; column_idx < table_column_types.size();
         column_idx++) {
      if (!returning_needed[column_idx]) {
        Value null_value(table_column_types[column_idx]);
        patched.data[column_idx].Reference(null_value, count_t(patched.size()));
      }
    }
    for (idx_t i = 0; i < needed_to_table.size(); i++) {
      patched.data[needed_to_table[i]].Reference(returning_values.data[i]);
    }
    AppendMergeReturningRows(context.client, gstate.return_collection, patched,
                             "INSERT");
  }

  gstate.changed_rows += insert_values.size();
}

void PhysicalLanceMergeInto::HandleMergeDeleteRows(
    ExecutionContext &context, DataChunk &input_chunk,
    LanceMergeGlobalState &gstate) const {
  vector<uint64_t> row_ids;
  string rowid_error;
  if (!TryExtractRowIdsFromChunk(input_chunk.data[row_id_index],
                                 input_chunk.size(), row_ids, rowid_error)) {
    throw ConstraintException(rowid_error);
  }
  RegisterMutatedTargetRows(row_ids, gstate.mutated_target_rowids);

  if (return_chunk) {
    vector<string> taken_names;
    vector<LogicalType> taken_types;
    vector<idx_t> taken_to_table;
    for (idx_t column_idx = 0; column_idx < table_column_types.size();
         column_idx++) {
      if (!returning_needed[column_idx]) {
        continue;
      }
      taken_names.push_back(table_column_names[column_idx]);
      taken_types.push_back(table_column_types[column_idx]);
      taken_to_table.push_back(column_idx);
    }

    auto patch_rows = [&](DataChunk &taken) {
      DataChunk patched;
      patched.Initialize(context.client, table_column_types);
      patched.SetChildCardinality(taken.size());
      for (idx_t column_idx = 0; column_idx < table_column_types.size();
           column_idx++) {
        if (!returning_needed[column_idx]) {
          Value null_value(table_column_types[column_idx]);
          patched.data[column_idx].Reference(null_value,
                                             count_t(patched.size()));
        }
      }
      for (idx_t column_idx = 0; column_idx < taken_to_table.size();
           column_idx++) {
        patched.data[taken_to_table[column_idx]].Reference(
            taken.data[column_idx]);
      }
      AppendMergeReturningRows(context.client, gstate.return_collection,
                               patched, "DELETE");
    };

    if (taken_names.empty()) {
      DataChunk empty_taken;
      empty_taken.Initialize(context.client, {LogicalType::BOOLEAN});
      empty_taken.SetChildCardinality(row_ids.size());
      patch_rows(empty_taken);
    } else {
      gstate.ForEachTakenChunkSubset(
          context.client, row_ids, taken_names, taken_types,
          [&](DataChunk &taken, idx_t) { patch_rows(taken); });
    }
  }

  gstate.EnsureMergeHandle(context.client);
  gstate.AddDeleteRowIds(row_ids);
  gstate.changed_rows += input_chunk.size();
}

void PhysicalLanceMergeInto::HandleMergeErrorRows(
    ExecutionContext &context, DataChunk &input_chunk,
    const LanceMergeAction &action, LanceMergeActionLocalState &action_state,
    MergeActionCondition condition) const {
  string merge_condition;
  merge_condition += MergeQueryNode::ActionConditionToString(condition);
  if (action.condition) {
    merge_condition += " AND " + action.condition->ToString();
  }

  if (!action.expressions.empty()) {
    if (!action_state.value_executor || !action_state.value_chunk) {
      throw InternalException("Lance MERGE ERROR action missing executor");
    }
    action_state.value_chunk->Reset();
    action_state.value_executor->Execute(input_chunk,
                                         *action_state.value_chunk);
    if (action_state.value_chunk->size() > 0 &&
        action_state.value_chunk->ColumnCount() > 0) {
      merge_condition +=
          ": " + action_state.value_chunk->data[0].GetValue(0).ToString();
    }
  }

  (void)context;
  throw ConstraintException("Merge error condition %s", merge_condition);
}

void PhysicalLanceMergeInto::HandleActionRows(
    ExecutionContext &context, DataChunk &input_chunk,
    const LanceMergeAction &action, LanceMergeActionLocalState &action_state,
    MergeActionCondition condition, LanceMergeGlobalState &gstate) const {
  switch (action.action_type) {
  case MergeActionType::MERGE_UPDATE: {
    if (!action_state.value_executor || !action_state.value_chunk) {
      throw InternalException("Lance MERGE UPDATE action missing executor");
    }
    action_state.value_chunk->Reset();
    action_state.value_executor->Execute(input_chunk,
                                         *action_state.value_chunk);
    HandleMergeUpdateRows(context, input_chunk, *action_state.value_chunk,
                          action, gstate);
    break;
  }
  case MergeActionType::MERGE_DELETE:
    HandleMergeDeleteRows(context, input_chunk, gstate);
    break;
  case MergeActionType::MERGE_INSERT: {
    if (!action_state.value_executor || !action_state.value_chunk) {
      throw InternalException("Lance MERGE INSERT action missing executor");
    }
    action_state.value_chunk->Reset();
    action_state.value_executor->Execute(input_chunk,
                                         *action_state.value_chunk);
    HandleMergeInsertRows(context, *action_state.value_chunk, gstate);
    break;
  }
  case MergeActionType::MERGE_DO_NOTHING:
    break;
  case MergeActionType::MERGE_ERROR:
    HandleMergeErrorRows(context, input_chunk, action, action_state, condition);
    break;
  default:
    throw InternalException("Unsupported Lance MERGE action type");
  }
}

void PhysicalLanceMergeInto::ProcessMatchedRows(
    ExecutionContext &context, DataChunk &chunk,
    const SelectionVector &initial_sel, idx_t initial_count,
    MergeActionCondition condition, idx_t range_index,
    LanceMergeGlobalState &gstate, LanceMergeLocalState &lstate) const {
  if (initial_count == 0) {
    return;
  }

  auto current_count = initial_count;
  lstate.current_sel.Initialize(initial_sel);

  auto &range = action_ranges[range_index];
  for (idx_t action_idx = range.start; action_idx < range.end; action_idx++) {
    if (current_count == 0) {
      break;
    }

    auto &action = *actions[action_idx];
    auto &action_state = lstate.action_states[action_idx];

    const SelectionVector *action_sel = &lstate.current_sel;
    idx_t action_count = current_count;

    if (action.condition) {
      if (!action_state.condition_executor) {
        throw InternalException(
            "Lance MERGE action missing condition executor");
      }

      action_count = action_state.condition_executor->SelectExpression(
          chunk, lstate.selected_sel, lstate.remaining_sel, lstate.current_sel,
          current_count);
      action_sel = &lstate.selected_sel;
    }

    if (action_count == 0) {
      continue;
    }

    auto input_chunk_opt = SliceActionInput(context.client, chunk, *action_sel,
                                            action_count, lstate);
    if (!input_chunk_opt) {
      continue;
    }

    auto &input_chunk = *input_chunk_opt;
    HandleActionRows(context, input_chunk, action, action_state, condition,
                     gstate);

    if (action.condition) {
      current_count -= action_count;
      lstate.current_sel.Initialize(lstate.remaining_sel);
    } else {
      current_count = 0;
      break;
    }
  }
}

SinkResultType PhysicalLanceMergeInto::Sink(ExecutionContext &context,
                                            DataChunk &chunk,
                                            OperatorSinkInput &input) const {
  if (chunk.size() == 0) {
    return SinkResultType::NEED_MORE_INPUT;
  }

  auto &gstate = input.global_state.Cast<LanceMergeGlobalState>();
  auto &lstate = input.local_state.Cast<LanceMergeLocalState>();

  ComputeMatches(lstate, chunk);

  auto matched_range = GetRangeIndex(MergeActionCondition::WHEN_MATCHED);
  ProcessMatchedRows(context, chunk, lstate.matched_sel, lstate.matched_count,
                     MergeActionCondition::WHEN_MATCHED, matched_range, gstate,
                     lstate);

  auto by_target_range =
      GetRangeIndex(MergeActionCondition::WHEN_NOT_MATCHED_BY_TARGET);
  ProcessMatchedRows(context, chunk, lstate.not_matched_by_target_sel,
                     lstate.not_matched_by_target_count,
                     MergeActionCondition::WHEN_NOT_MATCHED_BY_TARGET,
                     by_target_range, gstate, lstate);

  auto by_source_range =
      GetRangeIndex(MergeActionCondition::WHEN_NOT_MATCHED_BY_SOURCE);
  ProcessMatchedRows(context, chunk, lstate.not_matched_by_source_sel,
                     lstate.not_matched_by_source_count,
                     MergeActionCondition::WHEN_NOT_MATCHED_BY_SOURCE,
                     by_source_range, gstate, lstate);

  return SinkResultType::NEED_MORE_INPUT;
}

SinkFinalizeType
PhysicalLanceMergeInto::Finalize(Pipeline &, Event &, ClientContext &context,
                                 OperatorSinkFinalizeInput &input) const {
  auto &gstate = input.global_state.Cast<LanceMergeGlobalState>();
  if (!gstate.merge_handle) {
    return SinkFinalizeType::READY;
  }

  void *txn = nullptr;
  auto rc = lance_merge_finish_uncommitted(gstate.merge_handle, &txn);
  gstate.merge_handle = nullptr;

  if (rc != 0) {
    throw IOException("Failed to finish Lance MERGE transaction for '" +
                      gstate.open_path + "'" + LanceFormatErrorSuffix());
  }

  if (!txn) {
    return SinkFinalizeType::READY;
  }

  RegisterLancePendingAppend(
      context, table.catalog, std::move(gstate.open_path),
      std::move(gstate.option_keys), std::move(gstate.option_values),
      LanceBuildDatasetCacheKeyForTable(context, table), txn);
  return SinkFinalizeType::READY;
}

SourceResultType
PhysicalLanceMergeInto::GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                        OperatorSourceInput &input) const {
  auto &gstate = sink_state->Cast<LanceMergeGlobalState>();
  auto &source_state = input.global_state.Cast<LanceMergeSourceState>();

  if (!return_chunk) {
    if (source_state.emitted) {
      return SourceResultType::FINISHED;
    }
    source_state.emitted = true;
    chunk.data[0].Append(
        Value::BIGINT(NumericCast<int64_t>(gstate.changed_rows)));
    return SourceResultType::FINISHED;
  }

  gstate.return_collection.Scan(source_state.scan_state, chunk);
  if (chunk.size() == 0) {
    return SourceResultType::FINISHED;
  }
  return SourceResultType::HAVE_MORE_OUTPUT;
}

namespace {

static LogicalGet *FindGetByTableIndex(LogicalOperator &op,
                                       TableIndex table_index) {
  if (op.type == LogicalOperatorType::LOGICAL_GET) {
    auto &get = op.Cast<LogicalGet>();
    if (get.table_index == table_index) {
      return &get;
    }
  }
  for (auto &child : op.children) {
    if (child) {
      auto result = FindGetByTableIndex(*child, table_index);
      if (result) {
        return result;
      }
    }
  }
  return nullptr;
}

static void AddExpressionBindings(const Expression &expression,
                                  column_binding_set_t &required,
                                  vector<ColumnBinding> &worklist) {
  ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
      expression, [&](const BoundColumnRefExpression &ref) {
        if (required.insert(ref.Binding()).second) {
          worklist.push_back(ref.Binding());
        }
      });
}

static void CollectProjectionDefinitions(
    LogicalOperator &op,
    column_binding_map_t<const Expression *> &definitions) {
  if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
    auto &projection = op.Cast<LogicalProjection>();
    for (idx_t i = 0; i < projection.expressions.size(); i++) {
      auto &expression = projection.expressions[i];
      if (expression) {
        definitions.emplace(
            ColumnBinding(projection.table_index, ProjectionIndex(i)),
            expression.get());
      }
    }
  }
  for (auto &child : op.children) {
    if (child) {
      CollectProjectionDefinitions(*child, definitions);
    }
  }
}

static void CollectMergeSeedBindings(LogicalOperator &op,
                                     column_binding_set_t &required,
                                     vector<ColumnBinding> &worklist) {
  // Projections in the merge child build DuckDB's generic full-row RETURNING
  // image. Only action conditions/expressions and join/filter expressions
  // describe values the Lance sink actually consumes.
  if (op.type != LogicalOperatorType::LOGICAL_PROJECTION) {
    LogicalOperatorVisitor::EnumerateExpressions(
        op, [&](unique_ptr<Expression> *expression) {
          if (*expression) {
            AddExpressionBindings(**expression, required, worklist);
          }
        });
  }
  for (auto &child : op.children) {
    if (child) {
      CollectMergeSeedBindings(*child, required, worklist);
    }
  }
}

static column_binding_set_t RequiredMergeBindings(LogicalMergeInto &merge) {
  column_binding_map_t<const Expression *> projection_definitions;
  CollectProjectionDefinitions(merge, projection_definitions);

  column_binding_set_t required;
  vector<ColumnBinding> worklist;
  CollectMergeSeedBindings(merge, required, worklist);

  for (idx_t i = 0; i < worklist.size(); i++) {
    auto definition = projection_definitions.find(worklist[i]);
    if (definition != projection_definitions.end()) {
      AddExpressionBindings(*definition->second, required, worklist);
    }
  }
  return required;
}

static optional_ptr<LogicalGet>
FindLanceMergeTargetGet(LogicalMergeInto &merge) {
  if (merge.children.empty() || !merge.children[0] ||
      merge.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
    return nullptr;
  }
  auto &child = merge.children[0]->Cast<LogicalProjection>();
  if (merge.row_id_start >= child.expressions.size()) {
    return nullptr;
  }

  column_binding_map_t<const Expression *> projection_definitions;
  CollectProjectionDefinitions(*merge.children[0], projection_definitions);

  ColumnBinding binding(child.table_index, ProjectionIndex(merge.row_id_start));
  column_binding_set_t visited;
  while (visited.insert(binding).second) {
    auto definition = projection_definitions.find(binding);
    if (definition != projection_definitions.end()) {
      vector<ColumnBinding> references;
      ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
          *definition->second, [&](const BoundColumnRefExpression &ref) {
            references.push_back(ref.Binding());
          });
      if (references.size() != 1) {
        return nullptr;
      }
      binding = references[0];
      continue;
    }
    auto *get = FindGetByTableIndex(*merge.children[0], binding.table_index);
    if (!get || get->GetTable().get() != &merge.table) {
      return nullptr;
    }
    return get;
  }
  return nullptr;
}

// DuckDB marks every child column of a DML RETURNING operator as required, so
// a target scan would otherwise read unrelated heavy BLOB columns even when
// RETURNING and the merge actions never reference them. Mark those scan
// outputs as forced NULLs; Lance builds RETURNING values from its own take row
// image.
static void MaskUnusedLanceMergeTargetColumns(LogicalMergeInto &merge) {
  auto get_ptr = FindLanceMergeTargetGet(merge);
  if (!get_ptr) {
    return;
  }

  auto &get = *get_ptr;
  auto &bind_data = get.bind_data->Cast<LanceScanBindData>();
  auto old_bindings = get.GetColumnBindings();
  auto required = RequiredMergeBindings(merge);

  for (idx_t i = 0; i < old_bindings.size(); i++) {
    auto &column_index = get.GetColumnIndex(ProjectionIndex(i));
    auto is_physical_column =
        column_index.HasPrimaryIndex() &&
        column_index.GetPrimaryIndex() <
            merge.table.GetColumns().PhysicalColumnCount();
    if (is_physical_column && required.count(old_bindings[i]) == 0) {
      bind_data.forced_null_columns.insert(
          NumericCast<column_t>(column_index.GetPrimaryIndex()));
    }
  }
}

// A pass-through logical carrier placed between the SQL RETURNING projection
// and the MERGE operator. It keeps the full-width row image schema while
// telling the physical merge operator which columns the projection references.
class LanceMergeReturningOperator final : public LogicalExtensionOperator {
public:
  LanceMergeReturningOperator(unique_ptr<LogicalOperator> child,
                              vector<bool> returning_needed_p)
      : returning_needed(std::move(returning_needed_p)) {
    children.push_back(std::move(child));
  }

  vector<ColumnBinding> GetColumnBindings() override {
    return children[0]->GetColumnBindings();
  }

  void ResolveTypes() override { types = children[0]->types; }

  vector<TableIndex> GetTableIndex() const override {
    return children[0]->GetTableIndex();
  }

  unique_ptr<LogicalOperator> Copy(ClientContext &context) const override {
    return make_uniq<LanceMergeReturningOperator>(children[0]->Copy(context),
                                                  returning_needed);
  }

  PhysicalOperator &CreatePlan(ClientContext &context,
                               PhysicalPlanGenerator &planner) override {
    auto &plan = planner.CreatePlan(*children[0]);
    auto *physical = dynamic_cast<PhysicalLanceMergeInto *>(&plan);
    if (!physical) {
      throw InternalException(
          "Lance MERGE RETURNING wrapper did not plan a Lance MERGE operator");
    }
    physical->SetReturningNeeded(returning_needed);
    (void)context;
    return plan;
  }

  optional_ptr<const string>
  GetTypeBindingVerificationIdentifier() const noexcept override {
    static const string identifier = "lance_merge_returning";
    return identifier;
  }

  bool SupportSerialization() const override { return false; }

  string GetExtensionName() const override { return "lance_merge_returning"; }

private:
  vector<bool> returning_needed;
};

} // namespace

void LanceWrapMergeReturningOperator(LogicalOperator &plan) {
  if (plan.type == LogicalOperatorType::LOGICAL_PROJECTION &&
      !plan.children.empty() && plan.children[0] &&
      plan.children[0]->type == LogicalOperatorType::LOGICAL_MERGE_INTO) {
    auto &merge = plan.children[0]->Cast<LogicalMergeInto>();
    if (merge.return_chunk &&
        dynamic_cast<LanceTableEntry *>(&merge.table) != nullptr) {
      MaskUnusedLanceMergeTargetColumns(merge);
      std::set<idx_t> columns;
      auto &projection = plan.Cast<LogicalProjection>();
      for (auto &expression : projection.expressions) {
        if (!expression) {
          continue;
        }
        ExpressionIterator::VisitExpression<BoundColumnRefExpression>(
            *expression, [&](const BoundColumnRefExpression &ref) {
              if (ref.Binding().table_index == merge.table_index &&
                  ref.Binding().column_index <
                      merge.table.GetColumns().PhysicalColumnCount()) {
                columns.insert(ref.Binding().column_index);
              }
            });
      }
      vector<bool> returning_needed(
          merge.table.GetColumns().PhysicalColumnCount(), false);
      for (auto column_idx : columns) {
        returning_needed[column_idx] = true;
      }
      plan.children[0] = make_uniq<LanceMergeReturningOperator>(
          std::move(plan.children[0]), std::move(returning_needed));
    }
  }
  for (auto &child : plan.children) {
    if (child) {
      LanceWrapMergeReturningOperator(*child);
    }
  }
}

PhysicalOperator &PlanLanceMergeInto(ClientContext &context,
                                     PhysicalPlanGenerator &planner,
                                     LogicalMergeInto &op,
                                     PhysicalOperator &plan) {
  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceMergeInto called for non-Lance table");
  }
  if (lance_table->HasCoercedColumns()) {
    throw NotImplementedException(
        "MERGE INTO Lance table '" + lance_table->name +
        "' is not supported: column(s) [" +
        StringUtil::Join(lance_table->CoercedColumnNames(), ", ") +
        "] have Arrow types DuckDB cannot represent natively, so the "
        "catalog exposes a coerced type. Writing in the coerced type would "
        "silently change the on-disk storage.");
  }

  if (op.children.empty() || !op.children[0]) {
    throw InternalException("Lance MERGE expects a child plan");
  }

  map<MergeActionCondition, vector<unique_ptr<LanceMergeAction>>>
      planned_actions;

  for (auto &entry : op.actions) {
    vector<unique_ptr<LanceMergeAction>> actions_for_condition;
    for (auto &bound_action : entry.second) {
      if (!bound_action) {
        throw InternalException("Lance MERGE found null bound action");
      }

      auto action = make_uniq<LanceMergeAction>();
      action->action_type = bound_action->action_type;
      action->condition = std::move(bound_action->condition);

      switch (bound_action->action_type) {
      case MergeActionType::MERGE_UPDATE: {
        action->expressions = BuildMergeUpdateExpressions(op, *bound_action);
        action->update_columns = std::move(bound_action->columns);
        action->update_column_names.reserve(action->update_columns.size());
        for (auto &col_idx : action->update_columns) {
          action->update_column_names.push_back(op.table.GetColumns()
                                                    .GetColumn(col_idx)
                                                    .Name()
                                                    .GetIdentifierName());
        }
        break;
      }
      case MergeActionType::MERGE_INSERT:
        action->expressions = BuildMergeInsertExpressions(op, *bound_action);
        break;
      case MergeActionType::MERGE_ERROR:
        action->expressions = std::move(bound_action->expressions);
        break;
      case MergeActionType::MERGE_DELETE:
      case MergeActionType::MERGE_DO_NOTHING:
        break;
      default:
        throw InternalException("Unsupported Lance MERGE action type");
      }

      actions_for_condition.push_back(std::move(action));
    }
    planned_actions.emplace(entry.first, std::move(actions_for_condition));
  }

  vector<unique_ptr<LanceMergeAction>> flat_actions;
  vector<LanceMergeActionRange> action_ranges;

  map<MergeActionCondition, LanceMergeActionRange> ranges;
  for (auto &entry : planned_actions) {
    LanceMergeActionRange range;
    range.condition = entry.first;
    range.start = flat_actions.size();
    for (auto &action : entry.second) {
      flat_actions.push_back(std::move(action));
    }
    range.end = flat_actions.size();
    ranges.emplace(entry.first, range);
  }

  vector<MergeActionCondition> match_actions = {
      MergeActionCondition::WHEN_MATCHED,
      MergeActionCondition::WHEN_NOT_MATCHED_BY_TARGET,
      MergeActionCondition::WHEN_NOT_MATCHED_BY_SOURCE,
  };
  for (auto condition : match_actions) {
    auto it = ranges.find(condition);
    LanceMergeActionRange range;
    if (it != ranges.end()) {
      range = it->second;
    }
    range.condition = condition;
    action_ranges.push_back(range);
  }

  vector<string> table_column_names;
  vector<LogicalType> table_column_types;
  table_column_names.reserve(op.table.GetColumns().PhysicalColumnCount());
  table_column_types.reserve(op.table.GetColumns().PhysicalColumnCount());
  for (auto &col : op.table.GetColumns().Physical()) {
    table_column_names.push_back(col.Name().GetIdentifierName());
    table_column_types.push_back(col.Type());
  }

  // The RETURNING projection wrapper replaces the mask before the global sink
  // state is created; without one every column is available.
  vector<bool> returning_needed(table_column_names.size(), true);

  auto &merge = planner.Make<PhysicalLanceMergeInto>(
      op.types, *lance_table, std::move(table_column_names),
      std::move(table_column_types), std::move(flat_actions),
      std::move(action_ranges), op.row_id_start, op.source_marker,
      op.return_chunk, std::move(returning_needed), op.estimated_cardinality);
  merge.children.push_back(plan);
  (void)context;
  return merge;
}

} // namespace duckdb
