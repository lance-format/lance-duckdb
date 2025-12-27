#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"

namespace duckdb {

class LogicalDelete;
class PhysicalOperator;

PhysicalOperator &PlanLanceDelete(ClientContext &context,
                                  PhysicalPlanGenerator &planner,
                                  LogicalDelete &op);

} // namespace duckdb
