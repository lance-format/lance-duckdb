#include "lance_logical_exec.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/planner/operator/logical_get.hpp"

#include <algorithm>

namespace duckdb {

LogicalLanceExec::LogicalLanceExec(
    TableIndex group_index, TableIndex aggregate_index, idx_t group_count,
    vector<LogicalType> output_types, unique_ptr<LogicalOperator> child_get,
    vector<unique_ptr<Expression>> output_expressions)
    : LogicalExtensionOperator(), group_index_(group_index),
      aggregate_index_(aggregate_index), group_count_(group_count),
      output_types_(std::move(output_types)),
      output_expressions_(std::move(output_expressions)) {
  if (child_get) {
    children.push_back(std::move(child_get));
  }
}

PhysicalOperator &LogicalLanceExec::CreatePlan(ClientContext &,
                                               PhysicalPlanGenerator &planner) {
  D_ASSERT(children.size() == 1);
  auto &child = planner.CreatePlan(*children[0]);
  if (output_expressions_.empty()) {
    return child;
  }
  vector<unique_ptr<Expression>> expressions;
  expressions.reserve(output_expressions_.size());
  for (auto &expr : output_expressions_) {
    expressions.push_back(expr->Copy());
  }
  auto &projection = planner.Make<PhysicalProjection>(
      output_types_, std::move(expressions), estimated_cardinality);
  projection.children.push_back(child);
  return projection;
}

vector<ColumnBinding> LogicalLanceExec::GetColumnBindings() {
  if (!column_bindings_.empty()) {
    return column_bindings_;
  }
  vector<ColumnBinding> bindings;
  bindings.reserve(output_types_.size());
  for (idx_t i = 0; i < output_types_.size(); i++) {
    if (i < group_count_) {
      bindings.emplace_back(group_index_, ProjectionIndex(i));
    } else {
      bindings.emplace_back(aggregate_index_,
                            ProjectionIndex(i - group_count_));
    }
  }
  return bindings;
}

vector<TableIndex> LogicalLanceExec::GetTableIndex() const {
  if (!column_bindings_.empty()) {
    vector<TableIndex> indexes;
    for (auto &binding : column_bindings_) {
      if (std::find(indexes.begin(), indexes.end(), binding.table_index) ==
          indexes.end()) {
        indexes.push_back(binding.table_index);
      }
    }
    return indexes;
  }
  if (group_count_ == 0) {
    return vector<TableIndex>{aggregate_index_};
  }
  return vector<TableIndex>{group_index_, aggregate_index_};
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
