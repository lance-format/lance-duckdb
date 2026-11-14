#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/execution/row_id_deduplicator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_insert.hpp"
#include "lance_returning.hpp"
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

class LanceDeleteGlobalState : public GlobalSinkState {
public:
  LanceDeleteGlobalState(ClientContext &context,
                         const vector<LogicalType> &types, bool return_chunk) {
    if (return_chunk) {
      returning = make_uniq<ColumnDataCollection>(context, types);
      seen = make_uniq<RowIdDeduplicator>(
          context, vector<LogicalType>{LogicalType::ROW_TYPE});
    }
  }

  ~LanceDeleteGlobalState() override {
    if (delete_handle) {
      lance_delete_abort(delete_handle);
      delete_handle = nullptr;
    }
  }

  void *delete_handle = nullptr;
  idx_t rows_deleted = 0;
  mutex lock;
  unique_ptr<ColumnDataCollection> returning;
  unique_ptr<RowIdDeduplicator> seen;
};

class LanceDeleteSourceState : public LanceReturningSourceState {
public:
  explicit LanceDeleteSourceState(
      optional_ptr<ColumnDataCollection> rows = nullptr)
      : LanceReturningSourceState(rows) {}
  bool emitted = false;
};

// DuckDB's child plan evaluates the whole WHERE clause (subqueries and joins
// included); Lance receives the `_rowid` of every selected row and deletes
// them from the dataset version pinned for the statement.
class PhysicalLanceDelete final : public PhysicalOperator,
                                  public LanceReturningOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceDelete(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                      LanceTableEntry &table_p,
                      shared_ptr<LanceDatasetCacheEntry> dataset_p,
                      idx_t row_id_index_p, vector<idx_t> return_columns_p,
                      bool return_chunk_p, idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        LanceReturningOperator(return_chunk_p ? types.size() : 0),
        table(table_p), dataset(std::move(dataset_p)),
        row_id_index(row_id_index_p),
        return_columns(std::move(return_columns_p)),
        return_chunk(return_chunk_p) {}

  bool IsSink() const override { return true; }
  bool IsSource() const override { return true; }
  bool ParallelSink() const override { return true; }
  bool SinkOrderDependent() const override { return false; }

  unique_ptr<GlobalSinkState>
  GetGlobalSinkState(ClientContext &context) const override {
    auto state =
        make_uniq<LanceDeleteGlobalState>(context, types, return_chunk);
    if (lance_delete_begin(dataset->Handle(), &state->delete_handle) != 0 ||
        !state->delete_handle) {
      throw IOException("Failed to start Lance DELETE for '" +
                        dataset->DisplayUri() + "'" + LanceFormatErrorSuffix());
    }
    return std::move(state);
  }

  SinkResultType Sink(ExecutionContext &context, DataChunk &chunk,
                      OperatorSinkInput &input) const override {
    if (chunk.size() == 0) {
      return SinkResultType::NEED_MORE_INPUT;
    }

    auto &gstate = input.global_state.Cast<LanceDeleteGlobalState>();
    unique_lock<mutex> guard(gstate.lock, std::defer_lock);
    if (return_chunk) {
      guard.lock();
      SelectionVector selection(chunk.size());
      auto count = gstate.seen->Register(chunk.data[row_id_index], chunk.size(),
                                         selection);
      chunk.Slice(selection, count);
      if (count == 0) {
        return SinkResultType::NEED_MORE_INPUT;
      }
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

    if (!return_chunk) {
      guard.lock();
    }
    if (lance_delete_add_rowids(gstate.delete_handle, row_ids.data(),
                                row_ids.size()) != 0) {
      throw IOException("Failed to add Lance DELETE row ids" +
                        LanceFormatErrorSuffix());
    }
    if (return_chunk) {
      DataChunk rows;
      rows.Initialize(context.client, types);
      rows.SetChildCardinality(chunk.size());
      for (idx_t i = 0; i < return_columns.size(); i++) {
        if (return_columns[i] == DConstants::INVALID_INDEX) {
          rows.data[i].Reference(Value(types[i]), count_t(chunk.size()));
        } else {
          rows.data[i].Reference(chunk.data[return_columns[i]]);
        }
      }
      // Lance exposes DuckDB's single virtual rowid after the physical fields.
      rows.data[return_columns.size()].Reference(chunk.data[row_id_index]);
      gstate.returning->Append(rows);
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
      RegisterLancePinnedTransaction(context, table, dataset, txn,
                                     LanceWriteKind::DELETE);
    }
    return SinkFinalizeType::READY;
  }

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceDeleteSourceState>(
        sink_state->Cast<LanceDeleteGlobalState>().returning.get());
  }

  SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk,
                                   OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceDeleteSourceState>();
    if (return_chunk) {
      return state.Scan(chunk);
    }
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
  vector<idx_t> return_columns;
  bool return_chunk;
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
      RegisterLancePinnedTransaction(context.client, table, dataset, txn,
                                     LanceWriteKind::REWRITE);
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
  // A plain `DELETE FROM t` drops whole fragments, which later deletes of the
  // same table conflict with.
  const bool delete_all =
      !op.return_chunk && !plan && IsUnfilteredLanceScan(*op.children[0]);
  LanceRejectWriteAfterPendingWrite(context, *lance_table,
                                    delete_all ? LanceWriteKind::REWRITE
                                               : LanceWriteKind::DELETE);
  LancePinDatasetForWrite(context, *lance_table, dataset);

  if (delete_all) {
    return planner.Make<PhysicalLanceDeleteAll>(
        op.types, *lance_table, std::move(dataset), op.estimated_cardinality);
  }

  auto &child = plan ? *plan : planner.CreatePlan(*op.children[0]);
  auto &row_id = op.expressions[0]->Cast<BoundReferenceExpression>();
  auto return_columns = op.return_columns;
  if (op.return_chunk) {
    if (op.types.size() != return_columns.size() + 1) {
      throw InternalException(
          "Lance DELETE RETURNING has unexpected virtual columns");
    }
    for (auto &index : return_columns) {
      if (index != DConstants::INVALID_INDEX) {
        index = op.expressions[index]->Cast<BoundReferenceExpression>().Index();
      }
    }
  }
  auto &del = planner.Make<PhysicalLanceDelete>(
      op.types, *lance_table, std::move(dataset), row_id.Index(),
      std::move(return_columns), op.return_chunk, op.estimated_cardinality);
  del.children.push_back(child);
  return del;
}

} // namespace duckdb
