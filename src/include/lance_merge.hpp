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

// Wrap a RETURNING projection above a MERGE with a pass-through extension
// operator that records which table columns the projection references.
void LanceWrapMergeReturningOperator(LogicalOperator &plan);

} // namespace duckdb
