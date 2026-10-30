#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_scan_bind_data.hpp"
#include "lance_table_entry.hpp"

#include <cstdint>

namespace duckdb {

namespace {

// A plain `DELETE FROM t` (no WHERE, no USING) plans as a bare scan of the
// target. Lance then drops whole fragments instead of DuckDB streaming every
// row id. Any filter left on the scan (table, dynamic, or pushed into the
// Lance bind data) disqualifies it.
bool IsUnfilteredLanceScan(LogicalOperator &node) {
  if (node.type == LogicalOperatorType::LOGICAL_PROJECTION) {
    return node.children.size() == 1 &&
           IsUnfilteredLanceScan(*node.children[0]);
  }
  if (node.type != LogicalOperatorType::LOGICAL_GET || !node.children.empty()) {
    return false;
  }
  auto &get = node.Cast<LogicalGet>();
  if (get.function.GetName().GetIdentifierName() != "__lance_table_scan" ||
      !get.bind_data || get.table_filters.HasFilters() ||
      get.table_filters.HasMultiColumnFilters() || get.dynamic_filters ||
      !get.scan_partition_indices.empty()) {
    return false;
  }
  auto &bind = get.bind_data->Cast<LanceScanBindData>();
  return bind.lance_pushed_filter_ir_parts.empty() &&
         bind.duckdb_pushed_filter_sql_parts.empty() &&
         bind.take_row_ids.empty() && !bind.sampling_pushed_down &&
         !bind.limit_offset_pushed_down;
}

// Hands a staged Lance transaction to the DuckDB transaction, which commits
// or frees it.
void RegisterLanceDeleteTransaction(ClientContext &context,
                                    LanceTableEntry &table,
                                    shared_ptr<LanceDatasetCacheEntry> dataset,
                                    void *txn) {
  // Keep the dataset's store, credentials and commit handler until commit or
  // rollback. Resolving the namespace again could target a replacement table.
  auto display_uri = dataset->DisplayUri();
  RegisterLancePendingAppend(context, table.catalog, std::move(display_uri), {},
                             {},
                             LanceBuildDatasetCacheKeyForTable(context, table),
                             txn, std::move(dataset));
}

class LanceDeleteGlobalState : public GlobalSinkState {
public:
  ~LanceDeleteGlobalState() override {
    if (delete_handle) {
      lance_delete_abort(delete_handle);
      delete_handle = nullptr;
    }
  }

  void *delete_handle = nullptr;
  idx_t rows_deleted = 0;
  mutex lock;
};

class LanceDeleteSourceState : public GlobalSourceState {
public:
  bool emitted = false;
};

// DuckDB's child plan evaluates the whole WHERE clause (subqueries and joins
// included); Lance receives the `_rowid` of every selected row and deletes
// them from the dataset version pinned for the statement.
class PhysicalLanceDelete final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceDelete(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                      LanceTableEntry &table_p,
                      shared_ptr<LanceDatasetCacheEntry> dataset_p,
                      idx_t row_id_index_p, idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), dataset(std::move(dataset_p)),
        row_id_index(row_id_index_p) {}

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return true; }
  bool SinkOrderDependent() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &) const override {
    auto state = make_uniq<LanceDeleteGlobalState>();
    if (lance_delete_begin(dataset->Handle(), &state->delete_handle) != 0 ||
        !state->delete_handle) {
      throw IOException("Failed to start Lance DELETE for '" +
                        dataset->DisplayUri() + "'" + LanceFormatErrorSuffix());
    }
    return std::move(state);
  }

  SinkResultType Sink(ExecutionContext &, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    if (chunk.size() == 0) {
      return SinkResultType::NEED_MORE_INPUT;
    }

    UnifiedVectorFormat format;
    chunk.data[row_id_index].ToUnifiedFormat(format);
    auto row_id_data = UnifiedVectorFormat::GetData<int64_t>(format);
    vector<uint64_t> row_ids;
    row_ids.reserve(chunk.size());
    for (idx_t i = 0; i < chunk.size(); i++) {
      auto idx = format.sel->get_index(i);
      if (!format.validity.RowIsValid(idx) || row_id_data[idx] < 0) {
        throw InternalException("Lance DELETE received an invalid row id");
      }
      row_ids.push_back(NumericCast<uint64_t>(row_id_data[idx]));
    }

    auto &gstate = input.global_state.Cast<LanceDeleteGlobalState>();
    lock_guard<mutex> guard(gstate.lock);
    if (lance_delete_add_rowids(gstate.delete_handle, row_ids.data(),
                                row_ids.size()) != 0) {
      throw IOException("Failed to add Lance DELETE row ids" +
                        LanceFormatErrorSuffix());
    }
    return SinkResultType::NEED_MORE_INPUT;
  }

  SinkCombineResultType Combine(ExecutionContext &,
                                OperatorSinkCombineInput &) const override {
    return SinkCombineResultType::FINISHED;
  }

  SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context,
                            OperatorSinkFinalizeInput &input) const override {
    auto &gstate = input.global_state.Cast<LanceDeleteGlobalState>();
    lock_guard<mutex> guard(gstate.lock);

    void *txn = nullptr;
    int64_t deleted_rows = 0;
    auto rc = lance_delete_finish_uncommitted(gstate.delete_handle, &txn,
                                              &deleted_rows);
    // finish takes ownership of the handle and releases it either way.
    gstate.delete_handle = nullptr;
    if (rc != 0) {
      throw IOException("Failed to create Lance DELETE transaction for '" +
                        dataset->DisplayUri() + "'" + LanceFormatErrorSuffix());
    }
    gstate.rows_deleted = NumericCast<idx_t>(deleted_rows);
    if (txn) {
      RegisterLanceDeleteTransaction(context, table, dataset, txn);
    }
    return SinkFinalizeType::READY;
  }

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceDeleteSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceDeleteSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    auto &gstate = sink_state->Cast<LanceDeleteGlobalState>();
    chunk.data[0].Append(
        Value::BIGINT(NumericCast<int64_t>(gstate.rows_deleted)));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceDelete"; }

private:
  LanceTableEntry &table;
  shared_ptr<LanceDatasetCacheEntry> dataset;
  idx_t row_id_index;
};

// A plain `DELETE FROM t`: drops every fragment of the dataset version pinned
// for the statement.
class PhysicalLanceDeleteAll final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceDeleteAll(PhysicalPlan &physical_plan,
                         vector<LogicalType> types_p, LanceTableEntry &table_p,
                         shared_ptr<LanceDatasetCacheEntry> dataset_p,
                         idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        table(table_p), dataset(std::move(dataset_p)) {}

  bool IsSource() const override { return true; }

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceDeleteSourceState>();
  }

  SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceDeleteSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    void *txn = nullptr;
    int64_t deleted_rows = 0;
    if (lance_delete_all_uncommitted(dataset->Handle(), &txn, &deleted_rows) !=
        0) {
      throw IOException("Failed to create Lance DELETE transaction for '" +
                        dataset->DisplayUri() + "'" + LanceFormatErrorSuffix());
    }
    if (txn) {
      RegisterLanceDeleteTransaction(context.client, table, dataset, txn);
    }
    chunk.data[0].Append(Value::BIGINT(deleted_rows));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceDelete"; }

private:
  LanceTableEntry &table;
  shared_ptr<LanceDatasetCacheEntry> dataset;
};

} // namespace

PhysicalOperator &PlanLanceDelete(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalDelete &op,
                                  optional_ptr<PhysicalOperator> plan) {
  if (op.return_chunk) {
    throw NotImplementedException(
        "Lance DELETE does not support RETURNING yet");
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceDelete called for non-Lance table");
  }
  if (op.expressions.empty() || (!plan && op.children.size() != 1)) {
    throw InternalException("Lance DELETE expects a child plan and a row id");
  }

  // The statement-scoped dataset cache returns the handle that native scans
  // of the target bound, so their row ids and the Lance transaction share one
  // version. All reads of the target in this statement, including CTEs and
  // correlated subqueries outside the DELETE subtree, use the same entry.
  string display_uri;
  auto dataset =
      LanceGetOrOpenDatasetEntryForTable(context, *lance_table, display_uri);
  if (!dataset || !dataset->Handle()) {
    throw IOException("Failed to open Lance dataset for DELETE: " +
                      display_uri + LanceFormatErrorSuffix());
  }
  LancePinDatasetForWrite(context, *lance_table, dataset);

  if (!plan && IsUnfilteredLanceScan(*op.children[0])) {
    return planner.Make<PhysicalLanceDeleteAll>(
        op.types, *lance_table, std::move(dataset), op.estimated_cardinality);
  }

  auto &child = plan ? *plan : planner.CreatePlan(*op.children[0]);
  auto &row_id = op.expressions[0]->Cast<BoundReferenceExpression>();
  auto &del = planner.Make<PhysicalLanceDelete>(
      op.types, *lance_table, std::move(dataset), row_id.Index(),
      op.estimated_cardinality);
  del.children.push_back(child);
  return del;
}

} // namespace duckdb
