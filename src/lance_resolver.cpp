#include "lance_resolver.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_table_entry.hpp"

#include <algorithm>

namespace duckdb {

//===--------------------------------------------------------------------===//
// DefaultCatalogResolver Implementation
//===--------------------------------------------------------------------===//
LanceResolveResult DefaultCatalogResolver::TryResolve(ClientContext &context,
                                                      const string &input) {
  try {
    auto qname = QualifiedName::Parse(input);
    auto &entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY,
                                    qname.catalog, qname.schema, qname.name);
    auto &table_entry = entry.Cast<TableCatalogEntry>();
    auto *lance_entry = dynamic_cast<LanceTableEntry *>(&table_entry);
    if (lance_entry) {
      return LanceResolveResult::Success(lance_entry->DatasetUri());
    }
    // Found a table but not a Lance table
    return LanceResolveResult::Failure("Table '" + input +
                                       "' exists but is not a Lance table");
  } catch (CatalogException &e) {
    // Table not found in catalog
    return LanceResolveResult::Failure("Table '" + input +
                                       "' not found in catalog: " + e.what());
  } catch (Exception &e) {
    // Other errors (e.g., invalid qualified name)
    return LanceResolveResult::Failure("Failed to resolve table '" + input +
                                       "': " + e.what());
  }
}

//===--------------------------------------------------------------------===//
// LanceDatasetResolverRegistry Implementation
//===--------------------------------------------------------------------===//
LanceDatasetResolverRegistry &LanceDatasetResolverRegistry::Get() {
  static LanceDatasetResolverRegistry instance;
  return instance;
}

LanceDatasetResolverRegistry::LanceDatasetResolverRegistry() {
  // Register default resolver
  resolvers_.push_back(make_uniq<DefaultCatalogResolver>());
}

void LanceDatasetResolverRegistry::RegisterResolver(
    unique_ptr<ILanceDatasetResolver> resolver) {
  if (!resolver) {
    return;
  }

  lock_guard<mutex> guard(lock_);

  // Check for duplicate names and replace if exists
  auto name = resolver->Name();
  for (auto &existing : resolvers_) {
    if (existing->Name() == name) {
      existing = std::move(resolver);
      SortResolvers();
      return;
    }
  }

  // Add new resolver
  resolvers_.push_back(std::move(resolver));
  SortResolvers();
}

bool LanceDatasetResolverRegistry::UnregisterResolver(const string &name) {
  lock_guard<mutex> guard(lock_);

  auto it = std::remove_if(resolvers_.begin(), resolvers_.end(),
                           [&name](const unique_ptr<ILanceDatasetResolver> &r) {
                             return r->Name() == name;
                           });

  if (it != resolvers_.end()) {
    resolvers_.erase(it, resolvers_.end());
    return true;
  }
  return false;
}

vector<string> LanceDatasetResolverRegistry::GetResolverNames() const {
  lock_guard<mutex> guard(lock_);
  vector<string> names;
  names.reserve(resolvers_.size());
  for (const auto &resolver : resolvers_) {
    names.push_back(resolver->Name());
  }
  return names;
}

void LanceDatasetResolverRegistry::SortResolvers() {
  // Sort by priority (descending) - higher priority runs first
  std::sort(resolvers_.begin(), resolvers_.end(),
            [](const unique_ptr<ILanceDatasetResolver> &a,
               const unique_ptr<ILanceDatasetResolver> &b) {
              return a->Priority() > b->Priority();
            });
}

string LanceDatasetResolverRegistry::Resolve(ClientContext &context,
                                             const Value &input,
                                             LanceResolvePolicy policy,
                                             const string &function_name) {
  // Validate input
  if (input.IsNull()) {
    throw InvalidInputException(
        function_name + " requires a non-null dataset path or table name");
  }
  auto input_str = input.GetValue<string>();
  if (input_str.empty()) {
    throw InvalidInputException(
        function_name + " requires a non-empty dataset path or table name");
  }

  // Try each resolver in priority order
  string last_error;
  {
    lock_guard<mutex> guard(lock_);
    for (const auto &resolver : resolvers_) {
      auto result = resolver->TryResolve(context, input_str);
      if (result.success) {
        return result.dataset_uri;
      }
      // Keep track of the last error for STRICT mode
      if (!result.error_message.empty()) {
        last_error = result.error_message;
      }
    }
  }

  // All resolvers failed - handle based on policy
  switch (policy) {
  case LanceResolvePolicy::STRICT:
    // In STRICT mode, throw an exception with the last error
    if (last_error.empty()) {
      last_error = "Could not resolve '" + input_str + "' to a Lance dataset";
    }
    throw BinderException(function_name + ": " + last_error);

  case LanceResolvePolicy::FALLBACK_TO_PATH:
    // In FALLBACK_TO_PATH mode, treat input as direct file path
    return input_str;
  }

  // Should not reach here
  throw InternalException("Unknown LanceResolvePolicy");
}

//===--------------------------------------------------------------------===//
// Convenience function
//===--------------------------------------------------------------------===//
string ResolveLanceDatasetUri(ClientContext &context, const Value &input,
                              LanceResolvePolicy policy,
                              const string &function_name) {
  return LanceDatasetResolverRegistry::Get().Resolve(context, input, policy,
                                                     function_name);
}

} // namespace duckdb
