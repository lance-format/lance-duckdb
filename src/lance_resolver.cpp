#include "lance_resolver.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/parser/qualified_name.hpp"
#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
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

LanceResolvedDataset ResolveLanceDataset(ClientContext &context,
                                         const Value &input,
                                         LanceResolvePolicy policy,
                                         const string &function_name) {
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
    EntryLookupInfo lookup(CatalogType::TABLE_ENTRY, name);
    auto entry =
        Catalog::GetEntry(context, lookup, OnEntryNotFound::RETURN_NULL);
    if (entry) {
      auto *table = dynamic_cast<LanceTableEntry *>(entry.get());
      if (table) {
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
