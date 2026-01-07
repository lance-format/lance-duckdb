#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"

namespace duckdb {

// LogicalLanceExec is an internal-only logical operator used to preserve
// DuckDB's grouped aggregate column bindings while executing the subtree via
// the internal __lance_exec table function.
class LogicalLanceExec final : public LogicalExtensionOperator {
public:
  LogicalLanceExec(idx_t group_index, idx_t aggregate_index, idx_t group_count,
                   vector<LogicalType> output_types,
                   unique_ptr<LogicalOperator> child_get);

  PhysicalOperator &CreatePlan(ClientContext &context,
                               PhysicalPlanGenerator &planner) override;

  vector<ColumnBinding> GetColumnBindings() override;
  vector<idx_t> GetTableIndex() const override;

  string GetName() const override;
  InsertionOrderPreservingMap<string> ParamsToString() const override;

  string GetExtensionName() const override;

protected:
  void ResolveTypes() override;

private:
  idx_t group_index_;
  idx_t aggregate_index_;
  idx_t group_count_;
  vector<LogicalType> output_types_;
};

} // namespace duckdb
