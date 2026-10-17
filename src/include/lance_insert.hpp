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
                                vector<string> option_values, string cache_key,
                                void *lance_transaction);

// Whether a transaction in this database instance holds Lance writes
// registered above and not yet handed to its commit or rollback. Their files
// exist, but no manifest references them.
bool LanceHasUncommittedWrites(ClientContext &context);

} // namespace duckdb
