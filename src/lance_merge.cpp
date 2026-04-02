#include "duckdb.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/statement/merge_into_statement.hpp"
#include "duckdb/planner/expression/bound_default_expression.hpp"
#include "duckdb/planner/operator/logical_merge_into.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_merge.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include <cstdint>

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
  rowid_vec.ToUnifiedFormat(count, format);
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
  out.SetCardinality(rows.size());

  for (idx_t c = 0; c < rows.ColumnCount(); c++) {
    out.data[c].Reference(rows.data[c]);
  }
  out.data[rows.ColumnCount()].Reference(Value(merge_action));

  collection.Append(out);
}

static void AddInsertChunkToMergeHandle(void *merge_handle, DataChunk &chunk,
                                        ClientContext &context) {
  if (chunk.size() == 0) {
    return;
  }

  unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
      extension_type_cast;
  auto props = context.GetClientProperties();

  ArrowArray array;
  memset(&array, 0, sizeof(array));
  ArrowConverter::ToArrowArray(chunk, &array, props, extension_type_cast);

  auto rc = lance_merge_add_insert_batch(merge_handle,
                                         reinterpret_cast<void *>(&array));
  if (array.release) {
    array.release(&array);
  }

  if (rc != 0) {
    throw IOException("Failed to add MERGE insert batch to Lance transaction" +
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
  out.SetCardinality(NumericCast<idx_t>(scan_state.chunk->arrow_array.length));
  ArrowTableFunction::ArrowToDuckDB(scan_state, arrow_table.GetColumns(), out,
                                    true, DConstants::INVALID_INDEX);
}

struct MergeTakeBatch {
  DataChunk chunk;
  idx_t row_offset = 0;
};

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
  void ForEachTakenChunk(ClientContext &context,
                         const vector<uint64_t> &row_ids, FUNC &&fn);

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
                         bool return_chunk_p, idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), table_column_names(std::move(table_column_names_p)),
        table_column_types(std::move(table_column_types_p)),
        actions(std::move(actions_p)),
        action_ranges(std::move(action_ranges_p)), row_id_index(row_id_index_p),
        source_marker(source_marker_p), return_chunk(return_chunk_p) {}

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
      LANCE_DEFAULT_MAX_BYTES_PER_FILE, LanceGetSessionHandle(context),
      &merge_handle);
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
void LanceMergeGlobalState::ForEachTakenChunk(ClientContext &context,
                                              const vector<uint64_t> &row_ids,
                                              FUNC &&fn) {
  if (row_ids.empty()) {
    return;
  }

  EnsureTakeDataset(context);

  vector<const char *> column_ptrs;
  column_ptrs.reserve(op.table_column_names.size());
  for (auto &name : op.table_column_names) {
    column_ptrs.push_back(name.c_str());
  }

  auto *stream = lance_create_dataset_take_stream_unfiltered(
      take_dataset, row_ids.data(), row_ids.size(),
      column_ptrs.empty() ? nullptr : column_ptrs.data(), column_ptrs.size());
  if (!stream) {
    throw IOException("Failed to create Lance take stream for MERGE" +
                      LanceFormatErrorSuffix());
  }

  idx_t taken_rows = 0;
  while (true) {
    void *batch = nullptr;
    auto rc = lance_stream_next(stream, &batch);
    if (rc == 1) {
      break;
    }
    if (rc != 0) {
      lance_close_stream(stream);
      throw IOException("Failed to read Lance take stream for MERGE" +
                        LanceFormatErrorSuffix());
    }

    auto arrow_chunk = make_uniq<ArrowArrayWrapper>();
    memset(&arrow_chunk->arrow_array, 0, sizeof(arrow_chunk->arrow_array));
    ArrowSchema schema;
    memset(&schema, 0, sizeof(schema));

    if (lance_batch_to_arrow(batch, &arrow_chunk->arrow_array, &schema) != 0) {
      lance_free_batch(batch);
      lance_close_stream(stream);
      throw IOException("Failed to export Lance MERGE take batch to Arrow" +
                        LanceFormatErrorSuffix());
    }
    lance_free_batch(batch);

    DataChunk taken;
    ConvertArrowArrayToDuckDataChunk(context, *arrow_chunk, schema, taken,
                                     op.table_column_types);

    if (schema.release) {
      schema.release(&schema);
    }

    fn(taken, taken_rows);
    taken_rows += taken.size();
  }

  lance_close_stream(stream);

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
        value_types.push_back(expr->return_type);
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
  chunk.data[row_id_index].ToUnifiedFormat(chunk.size(), rowid_data);

  if (source_marker.IsValid()) {
    UnifiedVectorFormat source_data;
    chunk.data[source_marker.GetIndex()].ToUnifiedFormat(chunk.size(),
                                                         source_data);
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

  vector<uint64_t> row_ids;
  string rowid_error;
  if (!TryExtractRowIdsFromChunk(input_chunk.data[row_id_index],
                                 input_chunk.size(), row_ids, rowid_error)) {
    throw ConstraintException(rowid_error);
  }

  if (return_chunk) {
    gstate.ForEachTakenChunk(
        context.client, row_ids, [&](DataChunk &taken, idx_t row_offset) {
          DataChunk patched;
          patched.Initialize(context.client, table_column_types);
          patched.SetCardinality(taken.size());
          for (idx_t c = 0; c < table_column_types.size(); c++) {
            patched.data[c].Reference(taken.data[c]);
          }
          for (idx_t update_col_idx = 0;
               update_col_idx < action.update_columns.size();
               update_col_idx++) {
            auto physical_col_idx = action.update_columns[update_col_idx].index;
            if (physical_col_idx >= patched.ColumnCount()) {
              throw InternalException(
                  "Lance MERGE UPDATE target column index is out of range");
            }
            for (idx_t row_idx = 0; row_idx < taken.size(); row_idx++) {
              auto update_row_idx = row_offset + row_idx;
              patched.SetValue(
                  physical_col_idx, row_idx,
                  update_values.data[update_col_idx].GetValue(update_row_idx));
            }
          }
          AppendMergeReturningRows(context.client, gstate.return_collection,
                                   patched, "UPDATE");
        });
  }

  vector<uint64_t> unique_row_ids;
  vector<idx_t> unique_input_idxs;
  unique_row_ids.reserve(row_ids.size());
  unique_input_idxs.reserve(row_ids.size());
  for (idx_t i = 0; i < row_ids.size(); i++) {
    auto row_id = row_ids[i];
    if (gstate.mutated_target_rowids.insert(row_id).second) {
      unique_row_ids.push_back(row_id);
      unique_input_idxs.push_back(i);
    }
  }

  if (unique_row_ids.empty()) {
    gstate.changed_rows += input_chunk.size();
    return;
  }

  gstate.EnsureMergeHandle(context.client);
  gstate.AddDeleteRowIds(unique_row_ids);

  gstate.ForEachTakenChunk(
      context.client, unique_row_ids, [&](DataChunk &taken, idx_t row_offset) {
        DataChunk patched;
        patched.Initialize(context.client, table_column_types);
        patched.SetCardinality(taken.size());
        for (idx_t c = 0; c < table_column_types.size(); c++) {
          patched.data[c].Reference(taken.data[c]);
        }

        for (idx_t update_col_idx = 0;
             update_col_idx < action.update_columns.size(); update_col_idx++) {
          auto physical_col_idx = action.update_columns[update_col_idx].index;
          if (physical_col_idx >= patched.ColumnCount()) {
            throw InternalException(
                "Lance MERGE UPDATE target column index is out of range");
          }
          for (idx_t row_idx = 0; row_idx < taken.size(); row_idx++) {
            auto update_row_offset = row_offset + row_idx;
            if (update_row_offset >= unique_input_idxs.size()) {
              throw InternalException(
                  "Lance MERGE UPDATE row offset is out of range");
            }
            auto update_row_idx = unique_input_idxs[update_row_offset];
            patched.SetValue(
                physical_col_idx, row_idx,
                update_values.data[update_col_idx].GetValue(update_row_idx));
          }
        }

        AddInsertChunkToMergeHandle(gstate.merge_handle, patched,
                                    context.client);
      });

  gstate.changed_rows += input_chunk.size();
}

void PhysicalLanceMergeInto::HandleMergeInsertRows(
    ExecutionContext &context, DataChunk &insert_values,
    LanceMergeGlobalState &gstate) const {
  if (insert_values.ColumnCount() != table_column_types.size()) {
    throw InternalException(
        "Lance MERGE INSERT values column count does not match table schema");
  }

  gstate.EnsureMergeHandle(context.client);
  AddInsertChunkToMergeHandle(gstate.merge_handle, insert_values,
                              context.client);

  if (return_chunk) {
    AppendMergeReturningRows(context.client, gstate.return_collection,
                             insert_values, "INSERT");
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

  vector<uint64_t> unique_row_ids;
  unique_row_ids.reserve(row_ids.size());
  for (auto row_id : row_ids) {
    if (gstate.mutated_target_rowids.insert(row_id).second) {
      unique_row_ids.push_back(row_id);
    }
  }

  if (unique_row_ids.empty()) {
    gstate.changed_rows += input_chunk.size();
    return;
  }

  if (return_chunk) {
    gstate.ForEachTakenChunk(
        context.client, unique_row_ids, [&](DataChunk &taken, idx_t) {
          AppendMergeReturningRows(context.client, gstate.return_collection,
                                   taken, "DELETE");
        });
  }

  gstate.EnsureMergeHandle(context.client);
  gstate.AddDeleteRowIds(unique_row_ids);
  gstate.changed_rows += input_chunk.size();
}

void PhysicalLanceMergeInto::HandleMergeErrorRows(
    ExecutionContext &context, DataChunk &input_chunk,
    const LanceMergeAction &action, LanceMergeActionLocalState &action_state,
    MergeActionCondition condition) const {
  string merge_condition;
  merge_condition += MergeIntoStatement::ActionConditionToString(condition);
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
    chunk.SetCardinality(1);
    chunk.SetValue(0, 0,
                   Value::BIGINT(NumericCast<int64_t>(gstate.changed_rows)));
    return SourceResultType::FINISHED;
  }

  gstate.return_collection.Scan(source_state.scan_state, chunk);
  if (chunk.size() == 0) {
    return SourceResultType::FINISHED;
  }
  return SourceResultType::HAVE_MORE_OUTPUT;
}

PhysicalOperator &PlanLanceMergeInto(ClientContext &context,
                                     PhysicalPlanGenerator &planner,
                                     LogicalMergeInto &op,
                                     PhysicalOperator &plan) {
  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceMergeInto called for non-Lance table");
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
          action->update_column_names.push_back(
              op.table.GetColumns().GetColumn(col_idx).Name());
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
    table_column_names.push_back(col.Name());
    table_column_types.push_back(col.Type());
  }

  auto &merge = planner.Make<PhysicalLanceMergeInto>(
      op.types, *lance_table, std::move(table_column_names),
      std::move(table_column_types), std::move(flat_actions),
      std::move(action_ranges), op.row_id_start, op.source_marker,
      op.return_chunk, op.estimated_cardinality);
  merge.children.push_back(plan);
  (void)context;
  return merge;
}

} // namespace duckdb
