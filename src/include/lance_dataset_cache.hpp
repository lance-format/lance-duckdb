#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LanceTableEntry;

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

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryInNamespace(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_display_uri,
    bool *out_cache_hit = nullptr);

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryForTable(
    ClientContext &context, const LanceTableEntry &table,
    string &out_display_uri, bool *out_cache_hit = nullptr);

void LanceInvalidateDatasetCache(ClientContext &context);

} // namespace duckdb
