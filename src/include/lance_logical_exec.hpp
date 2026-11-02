#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

// LogicalLanceExec is an internal-only logical operator used to preserve
// DuckDB's grouped aggregate column bindings while executing the subtree via
// the internal __lance_exec table function. When `output_expressions` is not
// empty, it computes the outputs from the columns of the table function (for
// example an AVG from the SUM and COUNT that Lance returned).
class LogicalLanceExec final : public LogicalExtensionOperator {
public:
  LogicalLanceExec(TableIndex group_index, TableIndex aggregate_index,
                   idx_t group_count, vector<LogicalType> output_types,
                   unique_ptr<LogicalOperator> child_get,
                   vector<unique_ptr<Expression>> output_expressions = {});

  PhysicalOperator &CreatePlan(ClientContext &context,
                               PhysicalPlanGenerator &planner) override;

  vector<ColumnBinding> GetColumnBindings() override;
  vector<TableIndex> GetTableIndex() const override;

  // The bindings of the operator this replaces when it is not the aggregate
  // itself (an ORDER BY over a projection of the aggregate).
  void SetColumnBindings(vector<ColumnBinding> bindings) {
    column_bindings_ = std::move(bindings);
  }

  string GetName() const override;
  InsertionOrderPreservingMap<string> ParamsToString() const override;

  string GetExtensionName() const override;

protected:
  void ResolveTypes() override;

private:
  TableIndex group_index_;
  TableIndex aggregate_index_;
  idx_t group_count_;
  vector<LogicalType> output_types_;
  // References into the table function's columns.
  vector<unique_ptr<Expression>> output_expressions_;
  // Overrides the group/aggregate bindings when set.
  vector<ColumnBinding> column_bindings_;
};

} // namespace duckdb
