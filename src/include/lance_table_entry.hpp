#pragma once

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/storage/table_storage_info.hpp"

struct ArrowSchema;

namespace duckdb {

struct AlterInfo;
struct CatalogTransaction;
class CatalogEntry;
class ClientContext;

enum class LanceNamespaceKind { Directory, Rest };

struct LanceNamespaceTableConfig {
  LanceNamespaceKind kind = LanceNamespaceKind::Rest;

  string root;
  vector<string> option_keys;
  vector<string> option_values;

  string endpoint;
  string table_id;
  string delimiter;
  string bearer_token_override;
  string api_key_override;
  string headers_tsv;
  string display_uri;

  bool IsDirectory() const { return kind == LanceNamespaceKind::Directory; }
  bool IsRest() const { return kind == LanceNamespaceKind::Rest; }
};

// C++-only ALTER marker (never produced by SQL) that asks a LanceTableEntry
// to rebuild itself from the current dataset state; used to replace a stale
// entry through DuckDB's transactional catalog version chain (see
// LanceTableEntry::AlterEntry and LanceSchemaEntry::ReplaceStaleTableEntry).
// It must derive from a stock AlterTableInfo subclass because
// CatalogSet::AlterEntry serializes the info into the undo buffer and the
// commit path deserializes it through the stock registry; the payload must
// also report an empty GetColumnName(), since the commit path casts the old
// entry to DuckTableEntry only for column-affecting alters (see
// CommitState::CommitEntryDrop). SET_PARTITIONED_BY with no keys satisfies
// both: the inherited Serialize() emits a valid stock encoding and the base
// GetColumnName() stays empty; the deserialized stock object is discarded at
// commit. Detected via dynamic_cast in LanceTableEntry::AlterEntry.
// (SET_COMMENT is unsuitable: CatalogSet::AlterEntry special-cases it with
// entry->Copy instead of calling the entry's AlterEntry.)
struct LanceRefreshTableAlterInfo final : SetPartitionedByInfo {
  explicit LanceRefreshTableAlterInfo(AlterEntryData data)
      : SetPartitionedByInfo(std::move(data),
                             vector<unique_ptr<ParsedExpression>>()) {}

  // Preserve the marker type across Copy() so a copied info still triggers
  // the refresh path instead of a stock (rejected) SET_PARTITIONED_BY.
  unique_ptr<AlterInfo> Copy() const override {
    auto result = make_uniq<LanceRefreshTableAlterInfo>(GetAlterEntryData());
    result->allow_internal = allow_internal;
    return std::move(result);
  }
};

// LanceTableEntry represents a Lance dataset as a DuckDB base table entry.
// It supports scanning via a Lance-backed table scan function and appending via
// DuckDB's INSERT planning path (implemented at the catalog level).
class LanceTableEntry final : public TableCatalogEntry {
public:
  LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                  CreateTableInfo &info, string dataset_uri);
  LanceTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                  CreateTableInfo &info, LanceNamespaceTableConfig config);

  unique_ptr<CatalogEntry> AlterEntry(ClientContext &context,
                                      AlterInfo &info) override;
  unique_ptr<CatalogEntry> AlterEntry(CatalogTransaction transaction,
                                      AlterInfo &info) override;

  unique_ptr<CatalogEntry> Copy(ClientContext &context) const override;

  TableFunction GetScanFunction(ClientContext &context,
                                unique_ptr<FunctionData> &bind_data) override;

  // Validate that this entry's declared schema state (columns, coerced
  // columns, NOT NULL constraints) still matches the dataset on storage.
  // Statements that bind a scan get this implicitly via GetScanFunction;
  // write-only statements (e.g. plain INSERT) must call it before planning.
  // On mismatch the stale entry is replaced through the catalog version
  // chain (this object stays alive until undo-buffer cleanup, so existing
  // references remain valid) and a "changed externally" error is thrown; the
  // next access serves the entry rebuilt from the current schema.
  void VerifySchemaFreshness(ClientContext &context);

  // Non-throwing-on-mismatch staleness probe for catalog resolution
  // (DESCRIBE / information_schema expose entry metadata without binding a
  // scan): reports whether the declared schema state diverged from the
  // dataset on storage without replacing anything. Infrastructure errors
  // (e.g. dataset unreachable) still propagate as exceptions.
  bool IsSchemaStale(ClientContext &context);

  unique_ptr<BaseStatistics> GetStatistics(ClientContext &, column_t) override {
    return nullptr;
  }

  TableStorageInfo GetStorageInfo(ClientContext &) override { return {}; }

  const string &DatasetUri() const { return dataset_uri; }
  bool IsNamespaceBacked() const { return namespace_config != nullptr; }
  const LanceNamespaceTableConfig &NamespaceConfig() const {
    if (!namespace_config) {
      throw InternalException("LanceTableEntry is not namespace-backed");
    }
    return *namespace_config;
  }

  // Top-level catalog columns whose declared type was coerced to a
  // DuckDB-compatible shape by the Arrow-compat reader-boundary layer
  // (e.g. FloatingPoint(HALF) → FloatingPoint(SINGLE)). Writers must refuse
  // to operate on such columns — DuckDB would hand back values in the
  // coerced type and silently widen / otherwise corrupt the on-disk storage.
  bool HasCoercedColumns() const { return !coerced_column_names.empty(); }
  const vector<string> &CoercedColumnNames() const {
    return coerced_column_names;
  }
  void SetCoercedColumnNames(vector<string> names) {
    coerced_column_names = std::move(names);
  }

private:
  // Pure comparison of this entry's declared schema state against a live
  // dataset schema produced by the shared population pipeline.
  bool
  MatchesLiveSchemaState(const vector<string> &live_names,
                         const vector<LogicalType> &live_types,
                         const std::vector<std::string> &live_coerced_columns,
                         const ArrowSchema &live_schema_root) const;

  // Fetch the (revalidated) dataset handle and compare; shared by
  // IsSchemaStale and VerifySchemaFreshness.
  bool FetchLiveSchemaMatches(ClientContext &context, string &out_display_uri);

  // Shared freshness comparator for the scan bind path and
  // VerifySchemaFreshness; replaces this entry through the catalog version
  // chain and throws on mismatch.
  void ValidateLiveSchemaOrReplace(
      ClientContext &context, const vector<string> &live_names,
      const vector<LogicalType> &live_types,
      const std::vector<std::string> &live_coerced_columns,
      const ArrowSchema &live_schema_root, const string &display_uri);

private:
  string dataset_uri;
  unique_ptr<LanceNamespaceTableConfig> namespace_config;
  vector<string> coerced_column_names;
};

// Replace a catalog table entry whose declared schema state no longer
// matches the dataset on storage (external schema evolution) with an entry
// rebuilt from the current dataset state. The replacement goes through
// DuckDB's transactional catalog version chain (CatalogSet::AlterEntry with
// the caller's real transaction), so the old generation - `table` itself -
// stays alive, and raw references held by active scans, DML operators, or
// prepared plans stay valid, until normal undo-buffer cleanup reclaims it
// once no active transaction can reference it. Returns false (serving the
// existing entry unchanged) when the entry is not part of a Lance-attached
// schema, was already replaced or dropped, the caller has no real
// transaction to own the undo entry, or the replace lost a write-write
// conflict; the replacement is best-effort and always fail-open.
bool LanceTryReplaceStaleTableEntry(ClientContext &context,
                                    LanceTableEntry &table);

} // namespace duckdb
