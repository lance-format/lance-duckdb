#pragma once

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
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
  // On mismatch the stale entry is evicted (destroying this object) and a
  // "changed externally" error is thrown; the next access re-discovers the
  // table with its current schema.
  void VerifySchemaFreshness(ClientContext &context);

  // Non-throwing-on-mismatch staleness probe for catalog resolution
  // (DESCRIBE / information_schema expose entry metadata without binding a
  // scan): reports whether the declared schema state diverged from the
  // dataset on storage without evicting anything. Infrastructure errors
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
  // VerifySchemaFreshness; evicts this entry and throws on mismatch.
  void ValidateLiveSchemaOrEvict(
      ClientContext &context, const vector<string> &live_names,
      const vector<LogicalType> &live_types,
      const std::vector<std::string> &live_coerced_columns,
      const ArrowSchema &live_schema_root, const string &display_uri);

private:
  string dataset_uri;
  unique_ptr<LanceNamespaceTableConfig> namespace_config;
  vector<string> coerced_column_names;
};

// Evict a catalog table entry whose declared columns no longer match the
// dataset schema on storage (external schema evolution). Uses the same
// system-transaction drop + tombstone cleanup mechanism as DROP TABLE so that
// the next access lazily re-discovers the table with its current schema.
//
// WARNING: on success this destroys `table` (the catalog set owns it); the
// caller must not touch the entry afterwards. Returns false when the entry is
// not part of a Lance-attached schema or was already replaced.
bool LanceTryEvictStaleTableEntry(ClientContext &context,
                                  LanceTableEntry &table);

} // namespace duckdb
