#include "duckdb.hpp"

#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_update.hpp"

#include "lance_blob.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "lance_update.hpp"

#include <cstdint>

namespace duckdb {

namespace {

idx_t ResolveUpdateSetChunkIndex(LogicalUpdate &op,
                                 const LogicalProjection *projection,
                                 idx_t expression_idx) {
  auto &expr = *op.expressions[expression_idx];
  switch (expr.GetExpressionClass()) {
  case ExpressionClass::BOUND_REF:
    return expr.Cast<BoundReferenceExpression>().Index();
  case ExpressionClass::BOUND_COLUMN_REF: {
    auto &ref = expr.Cast<BoundColumnRefExpression>();
    if (!projection || ref.Binding().table_index != projection->table_index) {
      throw NotImplementedException(
          "Lance UPDATE does not support UPDATE with joins or FROM");
    }
    return ref.Binding().column_index;
  }
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support SET expression of class %s",
        ExpressionClassToString(expr.GetExpressionClass()));
  }
}

class LanceUpdateValueGlobalState : public GlobalSinkState {
public:
  explicit LanceUpdateValueGlobalState(const LanceTableEntry &table_p)
      : table(&table_p) {}

  const LanceTableEntry *table = nullptr;
  string display_uri;
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  void *update_handle = nullptr;
  idx_t rows_updated = 0;
  mutex lock;

  ~LanceUpdateValueGlobalState() override {
    if (update_handle) {
      lance_update_abort(update_handle);
      update_handle = nullptr;
    }
  }
};

class LanceUpdateValueLocalState : public LocalSinkState {};

class PhysicalLanceUpdateValue final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceUpdateValue(PhysicalPlan &physical_plan,
                           vector<LogicalType> types_p,
                           LanceTableEntry &table_p,
                           vector<string> set_columns_p,
                           vector<LogicalType> set_types_p,
                           vector<idx_t> set_chunk_indices_p,
                           idx_t rowid_chunk_index_p, LogicalType rowid_type_p,
                           idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), set_columns(std::move(set_columns_p)),
        set_types(std::move(set_types_p)),
        set_chunk_indices(std::move(set_chunk_indices_p)),
        rowid_chunk_index(rowid_chunk_index_p) {
    if (set_columns.size() != set_types.size() ||
        set_columns.size() != set_chunk_indices.size()) {
      throw InternalException(
          "Lance UPDATE value plan has mismatched SET columns");
    }
    value_chunk_types = set_types;
    value_chunk_types.push_back(std::move(rowid_type_p));
  }

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return false; }
  bool SinkOrderDependent() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &) const override {
    return make_uniq<LanceUpdateValueGlobalState>(table);
  }

  unique_ptr<LocalSinkState>
  GetLocalSinkState(ExecutionContext &) const override {
    return make_uniq<LanceUpdateValueLocalState>();
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    if (chunk.size() == 0) {
      return SinkResultType::NEED_MORE_INPUT;
    }

    auto &gstate = input.global_state.Cast<LanceUpdateValueGlobalState>();
    lock_guard<mutex> guard(gstate.lock);

    if (!gstate.update_handle) {
      ResolveLanceStorageOptionsForTable(
          context.client, table, gstate.open_path, gstate.option_keys,
          gstate.option_values, gstate.display_uri);

      vector<const char *> key_ptrs;
      vector<const char *> value_ptrs;
      BuildStorageOptionPointerArrays(gstate.option_keys, gstate.option_values,
                                      key_ptrs, value_ptrs);

      auto rc = lance_update_begin_with_storage_options(
          gstate.open_path.c_str(),
          key_ptrs.empty() ? nullptr : key_ptrs.data(),
          value_ptrs.empty() ? nullptr : value_ptrs.data(),
          gstate.option_keys.size(),
          NumericCast<uint64_t>(LanceBlobIngestBatchBudget(context.client)),
          LanceGetSessionHandle(context.client), &gstate.update_handle);
      if (rc != 0 || !gstate.update_handle) {
        throw IOException("Failed to start Lance UPDATE transaction for '" +
                          gstate.open_path + "'" + LanceFormatErrorSuffix());
      }
    }

    chunk.Flatten();

    DataChunk value_chunk;
    value_chunk.Initialize(context.client, value_chunk_types);
    value_chunk.SetChildCardinality(chunk.size());
    for (idx_t i = 0; i < set_chunk_indices.size(); i++) {
      value_chunk.data[i].Reference(chunk.data[set_chunk_indices[i]]);
    }
    value_chunk.data[set_chunk_indices.size()].Reference(
        chunk.data[rowid_chunk_index]);
    // Materialize ingest rows exactly once; the same values feed the writer.
    NormalizeIngestRowsInChunk(context.client, value_chunk);

    auto props = context.client.GetClientProperties();
    unordered_map<idx_t, const shared_ptr<ArrowTypeExtensionData>>
        extension_type_cast;

    ArrowArray array;
    memset(&array, 0, sizeof(array));
    ArrowConverter::ToArrowArray(value_chunk, &array, props,
                                 extension_type_cast);

    vector<string> value_names = set_columns;
    value_names.push_back("_rowid");
    ArrowSchemaWrapper schema;
    schema.arrow_schema = {};
    ArrowConverter::ToArrowSchema(&schema.arrow_schema, value_chunk.GetTypes(),
                                  value_names, props);

    // Resolve ingest source credentials before exporting the Arrow array; a
    // secret/policy failure must not leak an unreleased array.
    LanceBatchSourceOptions source_options;
    LanceCollectIngestSourceOptions(context.client, value_chunk,
                                    source_options);
    LanceBlobSourceOptionArrays option_arrays;
    BuildLanceBlobSourceOptionArrays(source_options, option_arrays);

    auto rc = lance_update_add_value_batch(
        gstate.update_handle, &schema.arrow_schema, &array,
        option_arrays.Uris(), option_arrays.KeyOffsets(),
        option_arrays.ValueOffsets(), option_arrays.Counts(),
        option_arrays.counts.size());
    if (array.release) {
      array.release(&array);
    }
    if (rc != 0) {
      throw IOException("Failed to add Lance UPDATE value batch" +
                        LanceFormatErrorSuffix());
    }
    gstate.rows_updated += chunk.size();
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &gstate = input.global_state.Cast<LanceUpdateValueGlobalState>();
    lock_guard<mutex> guard(gstate.lock);
    if (!gstate.update_handle) {
      return SinkFinalizeType::READY;
    }

    void *txn = nullptr;
    auto rc = lance_update_finish_uncommitted(gstate.update_handle, &txn);
    // finish takes ownership of the handle and releases it either way.
    gstate.update_handle = nullptr;
    if (rc != 0) {
      throw IOException("Failed to finish Lance UPDATE transaction for '" +
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

  class LanceUpdateValueSourceState : public GlobalSourceState {
  public:
    bool emitted = false;
  };

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceUpdateValueSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceUpdateValueSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    auto &gstate = sink_state->Cast<LanceUpdateValueGlobalState>();
    chunk.data[0].Append(
        Value::BIGINT(NumericCast<int64_t>(gstate.rows_updated)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceUpdateValue"; }

  InsertionOrderPreservingMap<string> ParamsToString() const override {
    InsertionOrderPreservingMap<string> result;
    result["SET Columns"] = StringUtil::Join(set_columns, "\n");
    result["SET Chunk Indices"] =
        StringUtil::Join(set_chunk_indices, "\n",
                         [](const idx_t &index) { return to_string(index); });
    result["Row ID Chunk Index"] = to_string(rowid_chunk_index);
    return result;
  }

private:
  LanceTableEntry &table;
  vector<string> set_columns;
  vector<LogicalType> set_types;
  vector<idx_t> set_chunk_indices;
  idx_t rowid_chunk_index;
  vector<LogicalType> value_chunk_types;
};

} // namespace

PhysicalOperator &PlanLanceUpdateOverwrite(ClientContext &context,
                                           PhysicalPlanGenerator &planner,
                                           LogicalUpdate &op,
                                           PhysicalOperator &plan) {
  if (op.return_chunk) {
    throw NotImplementedException("Lance UPDATE does not support RETURNING");
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException(
        "PlanLanceUpdateOverwrite called for non-Lance table");
  }
  if (lance_table->HasCoercedColumns()) {
    throw NotImplementedException(
        "UPDATE on Lance table '" + lance_table->name +
        "' is not supported: column(s) [" +
        StringUtil::Join(lance_table->CoercedColumnNames(), ", ") +
        "] have Arrow types DuckDB cannot represent natively, so the "
        "catalog exposes a coerced type. Writing in the coerced type would "
        "silently change the on-disk storage.");
  }

  if (op.expressions.size() != op.columns.size()) {
    throw InternalException(
        "Lance UPDATE has mismatched SET columns/expressions");
  }
  if (op.children.empty() || !op.children[0]) {
    throw InternalException("Lance UPDATE expects a child plan");
  }
  auto *projection = dynamic_cast<LogicalProjection *>(op.children[0].get());
  if (!projection) {
    throw NotImplementedException(
        "Lance UPDATE only supports the standard UPDATE planning path");
  }
  if (plan.types.empty()) {
    throw InternalException("Lance UPDATE child plan has no output columns");
  }
  idx_t rowid_chunk_index = plan.types.size() - 1;

  vector<string> set_columns;
  vector<LogicalType> set_types;
  vector<idx_t> set_chunk_indices;
  set_columns.reserve(op.columns.size());
  set_types.reserve(op.columns.size());
  set_chunk_indices.reserve(op.columns.size());

  for (idx_t i = 0; i < op.columns.size(); i++) {
    auto &target_col = op.table.GetColumns().GetColumn(op.columns[i]);
    if (lance_table->IsJsonColumn(target_col.Name().GetIdentifierName())) {
      throw NotImplementedException(
          "UPDATE on Lance table '" + lance_table->name +
          "' is not supported because it writes JSON column(s) [" +
          target_col.Name().GetIdentifierName() + "]");
    }
    set_columns.push_back(target_col.Name().GetIdentifierName());
    if (!op.expressions[i]) {
      throw InternalException("Lance UPDATE has null SET expression");
    }
    auto chunk_index = ResolveUpdateSetChunkIndex(op, projection, i);
    if (chunk_index >= plan.types.size() - 1) {
      throw InternalException(
          "Lance UPDATE SET expression resolved past the row-id column");
    }
    set_types.push_back(op.expressions[i]->GetReturnType());
    set_chunk_indices.push_back(chunk_index);
  }

  auto &update = planner.Make<PhysicalLanceUpdateValue>(
      op.types, *lance_table, std::move(set_columns), std::move(set_types),
      std::move(set_chunk_indices), rowid_chunk_index,
      plan.types[rowid_chunk_index], op.estimated_cardinality);
  update.children.push_back(plan);
  (void)context;
  return update;
}

} // namespace duckdb
