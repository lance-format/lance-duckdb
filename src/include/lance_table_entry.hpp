#pragma once

#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/parsed_data/alter_table_info.hpp"
#include "duckdb/storage/table_storage_info.hpp"

struct ArrowSchema;

namespace duckdb {

struct AlterInfo;
struct Binder;
struct CatalogTransaction;
class LogicalGet;
class LogicalProjection;
class LogicalUpdate;
class CatalogEntry;
class ClientContext;
struct ArrowTableSchema;

// A transient interpretation shared by catalog discovery, rebuild, and
// freshness checks. Comments are loaded separately and never affect binding.
struct LanceTableSchema {
  // Read original Arrow formats, coerce the schema in place, and populate
  // the same Arrow mapping that scan binding uses.
  static LanceTableSchema FromArrow(ClientContext &context, ArrowSchema &schema,
                                    ArrowTableSchema &arrow_table);
  static LanceTableSchema FromDataset(ClientContext &context, void *dataset);

  ColumnList columns;
  vector<unique_ptr<Constraint>> constraints;
  vector<string> coerced_columns;
  // Top-level columns whose source Arrow field contains an actual Lance blob.
  vector<bool> blob_columns;
  // Top-level columns whose Lance field is genuinely Blob v2. Only these
  // accept the LANCE_BLOB_INPUT write type; plain binary/legacy Blob columns
  // keep the ordinary BLOB cast path and reject URI values at the write
  // boundary. Kept distinct from the read-side per-column encoding markers.
  vector<string> writable_blob_v2_columns;
};

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
  // Rebuild metadata without changing storage. Executing queries retain their
  // previous generation until QueryEnd; prepared statements rebind on EXECUTE.
  unique_ptr<CatalogEntry> Rebuild(ClientContext &context) const;

  unique_ptr<CatalogEntry> Copy(ClientContext &context) const override;

  TableFunction GetScanFunction(ClientContext &context,
                                unique_ptr<FunctionData> &bind_data) override;

  // BLOB columns read as BLOB bytes but accept the shared Blob write input so
  // SQL can express bytes, external references, and managed ingest through the
  // same INSERT/UPDATE/MERGE value contract. Whether a URI value is legal
  // depends on the destination field's encoding and is enforced at the write
  // boundary.
  LogicalType
  GetExpectedTypeForInsert(const ColumnDefinition &column) const override;

  // Preserve an already-input-typed value on DEFAULT/omitted-column binding
  // instead of casting it back to the declared BLOB read type.
  unique_ptr<Expression> GetDefaultExpressionForColumn(
      ClientContext &context, const LogicalType &input_type,
      const LogicalType &result_type, ColumnBinding binding,
      const Expression &constant_value) const override;

  // Validate that this entry's declared schema state (columns, coerced
  // columns, NOT NULL constraints) still matches the dataset on storage.
  // Statements that bind a scan get this implicitly via GetScanFunction;
  // write-only statements (e.g. plain INSERT) must call it before planning.
  // On mismatch the cache entry is replaced (executing queries retain the old
  // generation) and a "changed externally" error is thrown; the
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

  const ColumnList &GetColumns() const override { return columns; }

  TableStorageInfo GetStorageInfo(ClientContext &) override { return {}; }

  // Lance tables have no local CHECK constraints or secondary-index
  // maintenance for UPDATE. DuckDB's default also binds every physical column
  // whenever RETURNING is present; the Lance MERGE sink builds the returned row
  // image itself from the columns the RETURNING projection references.
  void BindUpdateConstraints(Binder &binder, LogicalGet &get,
                             LogicalProjection &proj, LogicalUpdate &update,
                             ClientContext &context) override {}

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
  void SetWritableBlobV2Columns(vector<string> names) {
    writable_blob_v2_columns = std::move(names);
  }

  bool IsWritableBlobV2Column(const ColumnDefinition &column) const;

  const vector<bool> &GetBlobColumns() const { return blob_column_flags; }
  bool HasBlobColumns() const {
    for (auto is_blob : blob_column_flags) {
      if (is_blob) {
        return true;
      }
    }
    return false;
  }
  void SetBlobColumns(vector<bool> columns) {
    blob_column_flags = std::move(columns);
  }

private:
  bool MatchesLiveSchemaState(const LanceTableSchema &live) const;

  // Uses the revalidated dataset handle; infrastructure errors propagate.
  LanceTableSchema FetchLiveSchema(ClientContext &context,
                                   string &out_display_uri);

  // Best-effort cache replacement followed by a fail-closed error on mismatch.
  // The current query keeps this entry's generation pinned.
  void ValidateLiveSchemaOrReplace(ClientContext &context,
                                   const LanceTableSchema &live,
                                   const string &display_uri);

private:
  ColumnList columns;
  string dataset_uri;
  unique_ptr<LanceNamespaceTableConfig> namespace_config;
  vector<string> coerced_column_names;
  vector<bool> blob_column_flags;
  vector<string> writable_blob_v2_columns;
};

// Rebuild a stale cache entry without changing storage. Executing queries pin
// their bound generations until QueryEnd; prepared statements rebind before
// execution. Return false if the entry was superseded, is not Lance-owned,
// or cannot be rebuilt because storage is unavailable.
bool LanceTryReplaceStaleTableEntry(ClientContext &context,
                                    LanceTableEntry &table);

} // namespace duckdb
