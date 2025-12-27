#include "duckdb.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "lance_common.hpp"
#include "lance_delete.hpp"
#include "lance_ffi.hpp"
#include "lance_filter_ir.hpp"
#include "lance_table_entry.hpp"

namespace duckdb {

static bool TryCollectDeleteFilterPredicates(LogicalOperator &node,
                                             vector<const Expression *> &out,
                                             string &out_error) {
  switch (node.type) {
  case LogicalOperatorType::LOGICAL_FILTER: {
    auto &filter = node.Cast<LogicalFilter>();
    for (auto &expr : filter.expressions) {
      out.push_back(expr.get());
    }
    break;
  }
  case LogicalOperatorType::LOGICAL_PROJECTION:
    break;
  case LogicalOperatorType::LOGICAL_GET:
    return true;
  default:
    out_error = "unsupported DELETE plan shape: " + node.GetName();
    return false;
  }

  if (node.children.size() != 1) {
    out_error = "unsupported DELETE plan: expected 1 child, got " +
                to_string(node.children.size());
    return false;
  }
  return TryCollectDeleteFilterPredicates(*node.children[0], out, out_error);
}

static bool TryFindDeleteLogicalGet(LogicalOperator &node, LogicalGet *&out_get,
                                    string &out_error) {
  if (node.type == LogicalOperatorType::LOGICAL_GET) {
    auto &get = node.Cast<LogicalGet>();
    if (get.function.name != "__lance_table_scan") {
      out_error = "unsupported DELETE plan: expected __lance_table_scan, got " +
                  get.function.name;
      return false;
    }
    out_get = &get;
    return true;
  }
  if (node.children.size() != 1) {
    out_error = "unsupported DELETE plan: expected 1 child, got " +
                to_string(node.children.size());
    return false;
  }
  return TryFindDeleteLogicalGet(*node.children[0], out_get, out_error);
}

static bool TryBuildLanceDeleteFilterIR(LogicalDelete &op,
                                        string &out_filter_ir,
                                        string &out_error) {
  out_filter_ir.clear();
  out_error.clear();

  if (op.children.size() != 1) {
    out_error = "unsupported DELETE plan: expected 1 child, got " +
                to_string(op.children.size());
    return false;
  }

  LogicalGet *get = nullptr;
  if (!TryFindDeleteLogicalGet(*op.children[0], get, out_error)) {
    return false;
  }
  if (!get) {
    out_error = "unsupported DELETE plan: missing LogicalGet";
    return false;
  }

  vector<string> names;
  vector<LogicalType> types;
  names.reserve(op.table.GetColumns().PhysicalColumnCount());
  types.reserve(op.table.GetColumns().PhysicalColumnCount());
  for (auto &col : op.table.GetColumns().Physical()) {
    names.push_back(col.Name());
    types.push_back(col.Type());
  }

  vector<string> parts;

  vector<ColumnIndex> column_indexes = get->GetColumnIds();
  TableFunctionInitInput init_input(get->bind_data.get(),
                                    std::move(column_indexes),
                                    get->projection_ids, &get->table_filters);
  auto table_filters =
      BuildLanceTableFilterIRParts(names, types, init_input, false);
  parts = std::move(table_filters.parts);

  vector<const Expression *> predicates;
  if (!TryCollectDeleteFilterPredicates(*op.children[0], predicates,
                                        out_error)) {
    return false;
  }
  parts.reserve(parts.size() + predicates.size());
  for (auto *predicate : predicates) {
    if (!predicate) {
      out_error = "unsupported DELETE plan: null predicate expression";
      return false;
    }
    string part;
    if (!TryBuildLanceExprFilterIR(*get, names, types, false, *predicate,
                                   part)) {
      out_error =
          "unsupported DELETE predicate for Lance: " + predicate->ToString();
      return false;
    }
    parts.push_back(std::move(part));
  }

  if (!TryEncodeLanceFilterIRMessage(parts, out_filter_ir)) {
    out_error = "failed to encode Lance filter IR for DELETE";
    return false;
  }

  return true;
}

struct LanceDeleteSourceState : public GlobalSourceState {
  bool emitted = false;
};

class PhysicalLanceDelete final : public PhysicalOperator {
public:
  static constexpr const PhysicalOperatorType TYPE =
      PhysicalOperatorType::EXTENSION;

  PhysicalLanceDelete(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                      string dataset_uri_p, string filter_ir_p,
                      idx_t estimated_cardinality)
      : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION,
                         std::move(types_p), estimated_cardinality),
        dataset_uri(std::move(dataset_uri_p)),
        filter_ir(std::move(filter_ir_p)) {}

  bool IsSource() const override { return true; }

  unique_ptr<GlobalSourceState>
  GetGlobalSourceState(ClientContext &) const override {
    return make_uniq<LanceDeleteSourceState>();
  }

  SourceResultType GetData(ExecutionContext &context, DataChunk &chunk,
                           OperatorSourceInput &input) const override {
    auto &state = input.global_state.Cast<LanceDeleteSourceState>();
    if (state.emitted) {
      return SourceResultType::FINISHED;
    }
    state.emitted = true;

    if (!context.client.transaction.IsAutoCommit()) {
      throw NotImplementedException(
          "Lance DELETE does not support explicit transactions yet");
    }

    auto *dataset = LanceOpenDataset(context.client, dataset_uri);
    if (!dataset) {
      throw IOException("Failed to open Lance dataset: " + dataset_uri +
                        LanceFormatErrorSuffix());
    }

    int64_t deleted_rows = 0;
    auto *filter_ptr =
        filter_ir.empty() ? nullptr
                          : reinterpret_cast<const uint8_t *>(filter_ir.data());
    auto rc = lance_dataset_delete(dataset, filter_ptr, filter_ir.size(),
                                   &deleted_rows);
    lance_close_dataset(dataset);
    if (rc != 0) {
      throw IOException("Failed to delete from Lance dataset: " + dataset_uri +
                        LanceFormatErrorSuffix());
    }

    chunk.SetCardinality(1);
    chunk.SetValue(0, 0, Value::BIGINT(deleted_rows));
    return SourceResultType::FINISHED;
  }

  string GetName() const override { return "LanceDelete"; }

private:
  string dataset_uri;
  string filter_ir;
};

PhysicalOperator &PlanLanceDelete(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalDelete &op) {
  if (op.return_chunk) {
    throw NotImplementedException(
        "Lance DELETE does not support RETURNING yet");
  }

  auto *lance_table = dynamic_cast<LanceTableEntry *>(&op.table);
  if (!lance_table) {
    throw InternalException("PlanLanceDelete called for non-Lance table");
  }

  string filter_ir;
  string error;
  if (!TryBuildLanceDeleteFilterIR(op, filter_ir, error)) {
    if (error.empty()) {
      error = "unsupported DELETE plan";
    }
    throw NotImplementedException(error);
  }

  auto &del = planner.Make<PhysicalLanceDelete>(
      op.types, lance_table->DatasetUri(), std::move(filter_ir),
      op.estimated_cardinality);
  (void)context;
  return del;
}

} // namespace duckdb
