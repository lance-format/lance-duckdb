#include "lance_logical_exec.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

namespace duckdb {

LogicalLanceExec::LogicalLanceExec(idx_t group_index, idx_t aggregate_index,
                                   idx_t group_count,
                                   vector<LogicalType> output_types,
                                   unique_ptr<LogicalOperator> child_get)
    : LogicalExtensionOperator(), group_index_(group_index),
      aggregate_index_(aggregate_index), group_count_(group_count),
      output_types_(std::move(output_types)) {
  if (child_get) {
    children.push_back(std::move(child_get));
  }
}

PhysicalOperator &LogicalLanceExec::CreatePlan(ClientContext &,
                                               PhysicalPlanGenerator &planner) {
  D_ASSERT(children.size() == 1);
  return planner.CreatePlan(*children[0]);
}

vector<ColumnBinding> LogicalLanceExec::GetColumnBindings() {
  vector<ColumnBinding> bindings;
  bindings.reserve(output_types_.size());
  for (idx_t i = 0; i < output_types_.size(); i++) {
    if (i < group_count_) {
      bindings.emplace_back(group_index_, i);
    } else {
      bindings.emplace_back(aggregate_index_, i - group_count_);
    }
  }
  return bindings;
}

vector<idx_t> LogicalLanceExec::GetTableIndex() const {
  if (group_count_ == 0) {
    return vector<idx_t>{aggregate_index_};
  }
  return vector<idx_t>{group_index_, aggregate_index_};
}

string LogicalLanceExec::GetName() const { return "__LANCE_EXEC"; }

InsertionOrderPreservingMap<string> LogicalLanceExec::ParamsToString() const {
  InsertionOrderPreservingMap<string> result;
  SetParamsEstimatedCardinality(result);
  return result;
}

string LogicalLanceExec::GetExtensionName() const { return "lance"; }

void LogicalLanceExec::ResolveTypes() { types = output_types_; }

} // namespace duckdb
