#pragma once

#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/parser/parsed_data/copy_info.hpp"

namespace duckdb {
PhysicalOperator &PlanLanceCopy(ClientContext &context,
                                PhysicalPlanGenerator &planner, CopyInfo &info,
                                const vector<Identifier> &names,
                                const vector<LogicalType> &types,
                                PhysicalOperator &child);
} // namespace duckdb
