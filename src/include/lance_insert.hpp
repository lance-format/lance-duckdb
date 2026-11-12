#pragma once

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LanceTableEntry;
class LogicalInsert;
class PhysicalPlanGenerator;
class LanceDatasetCacheEntry;

// What a staged Lance transaction does to its table.
enum class LanceWriteKind : uint8_t {
  APPEND,  // INSERT
  DELETE,  // DELETE: marks rows deleted, rebases over deletes of other rows
  REWRITE, // UPDATE or MERGE: rewrites fragments
};

PhysicalOperator &PlanLanceInsertAppend(ClientContext &context,
                                        PhysicalPlanGenerator &planner,
                                        LogicalInsert &op,
                                        optional_ptr<PhysicalOperator> plan);

void RegisterLancePendingAppend(
    ClientContext &context, Catalog &catalog, string dataset_uri,
    vector<string> option_keys, vector<string> option_values, string cache_key,
    void *lance_transaction,
    shared_ptr<LanceDatasetCacheEntry> pinned_dataset = nullptr,
    LanceWriteKind kind = LanceWriteKind::APPEND);

// Hands a staged DELETE, UPDATE or MERGE transaction to the DuckDB transaction,
// which commits or frees it. The transaction was built on `dataset`, the
// version the statement's scans read; the commit goes to that handle, so Lance
// checks it against every commit made since.
void RegisterLancePinnedTransaction(ClientContext &context,
                                    LanceTableEntry &table,
                                    shared_ptr<LanceDatasetCacheEntry> dataset,
                                    void *lance_transaction,
                                    LanceWriteKind kind);

// Throws if the current transaction already holds an uncommitted write to this
// table that a DELETE (`kind` DELETE) or an UPDATE or MERGE (`kind` REWRITE)
// cannot be combined with. Lance commits each statement's transaction
// separately against the version the statement read, so the statement can
// neither see the earlier write nor be committed with it atomically. Only a
// DELETE after DELETEs is accepted: Lance rebases it over deletes of other
// rows. INSERT is not checked; appends do not conflict.
void LanceRejectWriteAfterPendingWrite(ClientContext &context,
                                       LanceTableEntry &table,
                                       LanceWriteKind kind);

// Whether a transaction in this database instance holds Lance writes
// registered above and not yet handed to its commit or rollback. Their files
// exist, but no manifest references them.
bool LanceHasUncommittedWrites(ClientContext &context);

} // namespace duckdb
