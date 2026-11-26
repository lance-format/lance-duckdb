#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LanceTableEntry;
class LanceDatasetCacheEntry;
struct LanceResolvedDataset;

enum class LanceResolvePolicy : uint8_t {
  STRICT,
  FALLBACK_TO_PATH,
};

// Bind data retains identity, not catalog entry pointers or credentials.
struct LanceDatasetReference {
  string identifier;
  bool is_table;

  LanceResolvedDataset Resolve(ClientContext &context) const;
  bool operator==(const LanceDatasetReference &other) const {
    return is_table == other.is_table && identifier == other.identifier;
  }
};

// A catalog pointer is valid only in the current query. Use Reference() when
// carrying a target from binding into execution or a prepared statement.
struct LanceResolvedDataset {
  string path;
  optional_ptr<LanceTableEntry> table;

  LanceDatasetReference Reference() const;
  void *Open(ClientContext &context, string &display_uri) const;
  shared_ptr<LanceDatasetCacheEntry> GetOrOpenCached(ClientContext &context,
                                                     string &display_uri,
                                                     bool *cache_hit) const;
};

LanceResolvedDataset ResolveLanceDataset(ClientContext &context,
                                         const Value &input,
                                         LanceResolvePolicy policy,
                                         const string &function_name);

} // namespace duckdb
