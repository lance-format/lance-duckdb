#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LogicalUpdate;
class PhysicalOperator;
class PhysicalPlanGenerator;

PhysicalOperator &PlanLanceUpdateOverwrite(ClientContext &context,
                                           PhysicalPlanGenerator &planner,
                                           LogicalUpdate &op,
                                           PhysicalOperator &plan);

} // namespace duckdb
