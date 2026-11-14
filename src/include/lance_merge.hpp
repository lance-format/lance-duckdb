#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LogicalMergeInto;
class PhysicalOperator;
class PhysicalPlanGenerator;

PhysicalOperator &PlanLanceMergeInto(ClientContext &context,
                                     PhysicalPlanGenerator &planner,
                                     LogicalMergeInto &op,
                                     PhysicalOperator &plan);

} // namespace duckdb
