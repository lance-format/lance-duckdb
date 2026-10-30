#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {

class LogicalDelete;
class PhysicalOperator;

// `plan` is DuckDB's already-built child plan; when null, the child plan is
// built here.
PhysicalOperator &PlanLanceDelete(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalDelete &op,
                                  optional_ptr<PhysicalOperator> plan);

} // namespace duckdb
