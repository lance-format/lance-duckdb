#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LanceTableEntry;
class LogicalInsert;
class PhysicalPlanGenerator;
class LanceDatasetCacheEntry;

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan);

void RegisterLancePendingAppend(
    ClientContext &context, Catalog &catalog, string dataset_uri,
    vector<string> option_keys, vector<string> option_values, string cache_key,
    void *lance_transaction,
    shared_ptr<LanceDatasetCacheEntry> pinned_dataset = nullptr);

// Hands a staged DELETE, UPDATE or MERGE transaction to the DuckDB transaction,
// which commits or frees it. The transaction was built on `dataset`, the
// version the statement's scans read; the commit goes to that handle, so Lance
// checks it against every commit made since.
void RegisterLancePinnedTransaction(ClientContext &context,
                                    LanceTableEntry &table,
                                    shared_ptr<LanceDatasetCacheEntry> dataset,
                                    void *lance_transaction);

// Whether a transaction in this database instance holds Lance writes
// registered above and not yet handed to its commit or rollback. Their files
// exist, but no manifest references them.
bool LanceHasUncommittedWrites(ClientContext &context);

} // namespace duckdb
