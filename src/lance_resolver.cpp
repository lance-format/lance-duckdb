#include "lance_resolver.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/tableref/bound_at_clause.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_storage.hpp"
#include "lance_table_entry.hpp"

namespace duckdb {

static bool HasUnquotedPathSeparator(const string &input) {
  bool quoted = false;
  for (auto c : input) {
    if (c == '"') {
      quoted = !quoted;
    } else if (!quoted && (c == '/' || c == '\\')) {
      return true;
    }
  }
  return false;
}

static bool CanResolveDirectoryTable(ClientContext &context,
                                     const QualifiedName &name) {
  auto is_directory = [&](const Identifier &catalog_name) {
    auto catalog = Catalog::GetCatalogEntry(context, catalog_name);
    return catalog && IsLanceDirectoryCatalog(*catalog);
  };
  if (!name.Catalog().empty()) {
    return is_directory(name.Catalog());
  }
  if (name.Path().size() == 1) {
    // DuckDB also resolves a bare catalog name to its ATTACH DEFAULT_TABLE.
    auto catalog = Catalog::GetCatalogEntry(context, name.Name());
    if (catalog && catalog->HasDefaultTable() &&
        IsLanceDirectoryCatalog(*catalog)) {
      return true;
    }
  }
  auto &search_path = *ClientData::Get(context).catalog_search_path;
  for (const auto &entry : search_path.Get()) {
    if ((name.Schema().empty() || entry.GetSchema() == name.Schema()) &&
        is_directory(entry.GetCatalog())) {
      return true;
    }
  }
  return is_directory(search_path.ResolveCatalog(name.Schema()));
}

static optional_ptr<CatalogEntry> LookupDatasetEntry(ClientContext &context,
                                                     const QualifiedName &name,
                                                     optional_idx version) {
  // Only a directory lookup needs AT. Other catalogs can reject AT outright,
  // even for a missing name that would otherwise resolve to a relative path.
  if (version.IsValid() && CanResolveDirectoryTable(context, name)) {
    BoundAtClause at(Identifier("VERSION"), Value::UBIGINT(version.GetIndex()));
    EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, name, &at,
                           QueryErrorContext());
    return Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
  }
  EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, name);
  return Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
}

static optional_ptr<CatalogEntry> LookupDatasetTable(ClientContext &context,
                                                     QualifiedName &name,
                                                     optional_idx version) {
  // A two-part string can name an attached table or a relative dataset path.
  // Apply DuckDB's ambiguity check only when the attachment contains a match.
  if (version.IsValid() && name.Path().size() == 2) {
    auto catalog = Catalog::GetCatalogEntry(context, name.Schema());
    if (catalog && IsLanceCatalog(*catalog)) {
      QualifiedName qualified(name.Schema(), Identifier(), name.Name());
      auto entry = LookupDatasetEntry(context, qualified, version);
      if (entry) {
        Binder::BindSchemaOrCatalog(context, name);
        return entry;
      }
    }
  }
  return LookupDatasetEntry(context, name, version);
}

LanceResolvedDataset ResolveLanceDataset(ClientContext &context,
                                         const Value &input,
                                         LanceResolvePolicy policy,
                                         const string &function_name,
                                         optional_idx search_version) {
  if (input.IsNull()) {
    throw InvalidInputException(
        function_name + " requires a non-null dataset path or table name");
  }
  auto input_str = input.GetValue<string>();
  if (input_str.empty()) {
    throw InvalidInputException(
        function_name + " requires a non-empty dataset path or table name");
  }

  // QualifiedName accepts filesystem separators as identifier characters.
  // Do not send paths to catalog discovery; quoted identifiers may contain
  // the same separators and must still be resolved as table names.
  if (policy == LanceResolvePolicy::FALLBACK_TO_PATH &&
      HasUnquotedPathSeparator(input_str)) {
    return {std::move(input_str), nullptr};
  }

  QualifiedName name;
  string error;
  if (QualifiedName::TryParse(input_str, name, error)) {
    // Only a missing/non-Lance entry falls back to a path. Namespace and
    // storage failures must propagate instead of being retried as path opens.
    auto entry = LookupDatasetTable(context, name, search_version);
    if (entry) {
      auto *table = dynamic_cast<LanceTableEntry *>(entry.get());
      if (table) {
        if (search_version.IsValid() &&
            (!table->IsNamespaceBacked() ||
             !table->NamespaceConfig().IsDirectory() ||
             !table->IsHistorical())) {
          throw BinderException("Search by version requires a dataset path, "
                                "URI or directory-attached Lance table");
        }
        return {string(), table};
      }
      error = "Table '" + input_str + "' exists but is not a Lance table";
    } else {
      error = "Table '" + input_str + "' not found in catalog";
    }
  } else {
    error = "Failed to resolve table '" + input_str + "': " + error;
  }
  if (policy == LanceResolvePolicy::STRICT) {
    throw BinderException(function_name + ": " + error);
  }
  return {std::move(input_str), nullptr};
}

LanceDatasetReference LanceResolvedDataset::Reference() const {
  if (table) {
    return {table->schema.GetQualifiedName(table->name).ToString(), true};
  }
  return {path, false};
}

LanceResolvedDataset
LanceDatasetReference::Resolve(ClientContext &context) const {
  if (!is_table) {
    return {identifier, nullptr};
  }
  // A bound table cannot silently become a path if it disappears or the
  // connection's search path changes before execution.
  return ResolveLanceDataset(context, Value(identifier),
                             LanceResolvePolicy::STRICT, "Lance dataset");
}

void *LanceResolvedDataset::Open(ClientContext &context,
                                 string &display_uri) const {
  if (table) {
    return LanceOpenDatasetForTable(context, *table, display_uri);
  }
  display_uri = path;
  return LanceOpenDataset(context, path);
}

shared_ptr<LanceDatasetCacheEntry> LanceResolvedDataset::GetOrOpenCached(
    ClientContext &context, string &display_uri, bool *cache_hit) const {
  if (table) {
    return LanceGetOrOpenDatasetEntryForTable(context, *table, display_uri,
                                              cache_hit);
  }
  display_uri = path;
  auto entry = LanceGetOrOpenDatasetEntry(context, path, cache_hit);
  if (entry) {
    display_uri = entry->DisplayUri();
  }
  return entry;
}

} // namespace duckdb
