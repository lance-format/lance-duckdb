#pragma once

#include "duckdb.hpp"
#include "duckdb/common/mutex.hpp"

#include <memory>
#include <string>
#include <vector>

namespace duckdb {

// Forward declaration
class LanceTableEntry;

//===--------------------------------------------------------------------===//
// Policy for dataset path resolution behavior
//===--------------------------------------------------------------------===//
enum class LanceResolvePolicy : uint8_t {
  // Must be a Lance table, throw exception if not found or not Lance table
  STRICT = 0,
  // First try catalog lookup, fallback to file path if failed
  FALLBACK_TO_PATH = 1,
};

//===--------------------------------------------------------------------===//
// Result of dataset resolution
//===--------------------------------------------------------------------===//
struct LanceResolveResult {
  bool success;
  string dataset_uri;
  string error_message;

  // Explicit constructor for C++11 compatibility
  LanceResolveResult() : success(false), dataset_uri(), error_message() {}

  LanceResolveResult(bool success_, string uri, string error)
      : success(success_), dataset_uri(std::move(uri)),
        error_message(std::move(error)) {}

  static LanceResolveResult Success(string uri) {
    return LanceResolveResult(true, std::move(uri), "");
  }

  static LanceResolveResult Failure(string error) {
    return LanceResolveResult(false, "", std::move(error));
  }
};

//===--------------------------------------------------------------------===//
// Interface for Lance dataset resolver
// Third-party can implement this to provide custom resolution logic
//===--------------------------------------------------------------------===//
class ILanceDatasetResolver {
public:
  virtual ~ILanceDatasetResolver() = default;

  //! Try to resolve the input string to a Lance dataset URI
  //! Returns LanceResolveResult indicating success/failure
  virtual LanceResolveResult TryResolve(ClientContext &context,
                                        const string &input) = 0;

  //! Priority of this resolver (higher priority runs first)
  //! Default implementation has priority 0
  //! Third-party can use higher priority to override default behavior
  //! or use negative priority to serve as fallback
  virtual int Priority() const { return 0; }

  //! Name of this resolver for debugging and unregistration
  virtual string Name() const = 0;
};

//===--------------------------------------------------------------------===//
// Registry for dataset resolvers (singleton pattern)
//===--------------------------------------------------------------------===//
class LanceDatasetResolverRegistry {
public:
  //! Get the singleton instance
  static LanceDatasetResolverRegistry &Get();

  //! Register a custom resolver
  //! Resolvers are sorted by priority (descending) after registration
  void RegisterResolver(unique_ptr<ILanceDatasetResolver> resolver);

  //! Unregister a resolver by name
  //! Returns true if a resolver was removed
  bool UnregisterResolver(const string &name);

  //! Get all registered resolver names (for debugging)
  vector<string> GetResolverNames() const;

  //! Main resolution function
  //! This is the public API that should be called by all Lance functions
  //! @param context The client context
  //! @param input The input value (table name or file path)
  //! @param policy The resolution policy (STRICT or FALLBACK_TO_PATH)
  //! @param function_name The name of the calling function (for error messages)
  //! @return The resolved dataset URI
  string Resolve(ClientContext &context, const Value &input,
                 LanceResolvePolicy policy, const string &function_name);

private:
  LanceDatasetResolverRegistry();
  ~LanceDatasetResolverRegistry() = default;

  // Non-copyable
  LanceDatasetResolverRegistry(const LanceDatasetResolverRegistry &) = delete;
  LanceDatasetResolverRegistry &
  operator=(const LanceDatasetResolverRegistry &) = delete;

  //! Sort resolvers by priority (descending)
  void SortResolvers();

  mutable mutex lock_;
  vector<unique_ptr<ILanceDatasetResolver>> resolvers_;
};

//===--------------------------------------------------------------------===//
// Default catalog resolver implementation
// Resolves table names through DuckDB's catalog system
//===--------------------------------------------------------------------===//
class DefaultCatalogResolver : public ILanceDatasetResolver {
public:
  LanceResolveResult TryResolve(ClientContext &context,
                                const string &input) override;

  int Priority() const override { return 0; }

  string Name() const override { return "default_catalog"; }
};

//===--------------------------------------------------------------------===//
// Convenience function (wraps registry call)
// This is the recommended API for resolving Lance dataset URIs
//===--------------------------------------------------------------------===//
string ResolveLanceDatasetUri(ClientContext &context, const Value &input,
                              LanceResolvePolicy policy,
                              const string &function_name);

} // namespace duckdb
