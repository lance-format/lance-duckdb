#pragma once

#include "duckdb.hpp"

#include <mutex>
#include <unordered_map>

namespace duckdb {

class LanceTableEntry;

// Outcome of a vector-index metric lookup. Cached on LanceDatasetCacheEntry
// so the per-plan KNN optimizer does not pay the index_statistics open cost
// on repeated queries against the same (cached) dataset.
struct LanceVectorIndexMetricLookup {
  // 0 = ok, 1 = no index, -1 = error.
  int status = 0;
  string metric;
  string error_message;
};

class LanceDatasetCacheEntry {
public:
  LanceDatasetCacheEntry(void *dataset_p, string display_uri_p);
  ~LanceDatasetCacheEntry();

  void *Handle() const { return dataset; }
  const string &DisplayUri() const { return display_uri; }

  // Returns the metric lookup for `column`, calling into Lance only on the
  // first request per column. Subsequent calls return the cached value.
  // Cache invalidation happens automatically: writes invalidate the dataset
  // cache entry, which discards this cache along with the handle.
  LanceVectorIndexMetricLookup
  GetOrLookupVectorIndexMetric(const string &column);

private:
  void *dataset = nullptr;
  string display_uri;

  std::mutex vector_metric_mutex;
  std::unordered_map<string, LanceVectorIndexMetricLookup> vector_metric_cache;
};

shared_ptr<LanceDatasetCacheEntry>
LanceGetOrOpenDatasetEntry(ClientContext &context, const string &path,
                           bool *out_cache_hit = nullptr);

string
LanceBuildResolvedPathDatasetCacheKey(const string &open_path,
                                      const vector<string> &option_keys,
                                      const vector<string> &option_values);

string LanceBuildPathDatasetCacheKey(ClientContext &context,
                                     const string &path);

string LanceBuildNamespaceDatasetCacheKey(
    const string &endpoint, const string &table_id, const string &bearer_token,
    const string &api_key, const string &delimiter, const string &headers_tsv);

string LanceBuildDatasetCacheKeyForTable(ClientContext &context,
                                         const LanceTableEntry &table);

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryInNamespace(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_display_uri,
    bool *out_cache_hit = nullptr);

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryForTable(
    ClientContext &context, const LanceTableEntry &table,
    string &out_display_uri, bool *out_cache_hit = nullptr);

void LanceInvalidateDatasetCache(ClientContext &context,
                                 const string &cache_key);
void LanceInvalidateDatasetCacheForPath(ClientContext &context,
                                        const string &path);
void LanceInvalidateDatasetCacheForTable(ClientContext &context,
                                         const LanceTableEntry &table);

} // namespace duckdb
