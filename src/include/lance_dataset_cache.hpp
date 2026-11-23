#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LanceTableEntry;
struct LanceNamespaceTableConfig;

class LanceDatasetCacheEntry {
public:
  LanceDatasetCacheEntry(void *dataset_p, string display_uri_p);
  ~LanceDatasetCacheEntry();

  void *Handle() const { return dataset; }
  const string &DisplayUri() const { return display_uri; }

private:
  void *dataset = nullptr;
  string display_uri;
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
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv);

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

shared_ptr<LanceDatasetCacheEntry>
LanceGetNamedSnapshotForTable(ClientContext &context,
                              const LanceTableEntry &table,
                              const string &reference);

shared_ptr<LanceDatasetCacheEntry> LanceGetDirectorySnapshot(
    ClientContext &context, const LanceNamespaceTableConfig &table,
    const string &binding_key, idx_t version, bool pinned_only);

string LanceRestVersionKey(ClientContext &context, const string &namespace_key,
                           idx_t version);
shared_ptr<LanceDatasetCacheEntry>
LanceGetPinnedSnapshot(ClientContext &context, const string &key);
void LancePinSnapshot(ClientContext &context, const string &key,
                      shared_ptr<LanceDatasetCacheEntry> snapshot);

// A write's target reads must use its pinned generation across the complete
// statement, even when DuckDB plans CTE producers outside the write subtree.
void LancePinDatasetForWrite(ClientContext &context,
                             const LanceTableEntry &table,
                             shared_ptr<LanceDatasetCacheEntry> dataset);
shared_ptr<LanceDatasetCacheEntry>
LanceGetDatasetPinnedForWrite(ClientContext &context,
                              const LanceTableEntry &table);
// Whether the current statement pinned any write target.
bool LanceHasDatasetPinnedForWrite(ClientContext &context);

void LanceInvalidateDatasetCache(ClientContext &context,
                                 const string &cache_key);
void LanceInvalidateDatasetCacheForPath(ClientContext &context,
                                        const string &path);
void LanceInvalidateDatasetCacheForTable(ClientContext &context,
                                         const LanceTableEntry &table);

} // namespace duckdb
