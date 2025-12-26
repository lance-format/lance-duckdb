#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LogicalInsert;
class PhysicalPlanGenerator;

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan);

void RegisterLancePendingAppend(ClientContext &context, Catalog &catalog,
                                string dataset_uri, vector<string> option_keys,
                                vector<string> option_values,
                                void *lance_transaction);

} // namespace duckdb
