#include "lance_dataset_cache.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"

#include "duckdb/main/client_context_state.hpp"

#include <functional>

namespace duckdb {

static constexpr const char *LANCE_DATASET_CACHE_STATE_KEY =
    "lance_dataset_cache_state";

class LanceDatasetCacheState final : public ClientContextState {
public:
  shared_ptr<LanceDatasetCacheEntry> Get(const string &key) {
    lock_guard<mutex> guard(lock);
    auto entry = entries.find(key);
    if (entry == entries.end()) {
      query_misses++;
      return nullptr;
    }
    query_hits++;
    return entry->second;
  }

  shared_ptr<LanceDatasetCacheEntry>
  PutOrGetExisting(const string &key,
                   shared_ptr<LanceDatasetCacheEntry> entry) {
    lock_guard<mutex> guard(lock);
    auto existing = entries.find(key);
    if (existing != entries.end()) {
      return existing->second;
    }
    entries[key] = entry;
    return entry;
  }

  void Clear() {
    lock_guard<mutex> guard(lock);
    entries.clear();
  }

  void QueryBegin(ClientContext &) override {
    lock_guard<mutex> guard(lock);
    query_hits = 0;
    query_misses = 0;
  }

  void WriteProfilingInformation(std::ostream &ss) override {
    lock_guard<mutex> guard(lock);
    ss << "Lance Dataset Cache: entries=" << entries.size()
       << " hits=" << query_hits << " misses=" << query_misses << "\n";
  }

private:
  mutex lock;
  unordered_map<string, shared_ptr<LanceDatasetCacheEntry>> entries;
  idx_t query_hits = 0;
  idx_t query_misses = 0;
};

LanceDatasetCacheEntry::LanceDatasetCacheEntry(void *dataset_p,
                                               string display_uri_p)
    : dataset(dataset_p), display_uri(std::move(display_uri_p)) {}

LanceDatasetCacheEntry::~LanceDatasetCacheEntry() {
  if (dataset) {
    lance_close_dataset(dataset);
    dataset = nullptr;
  }
}

static shared_ptr<LanceDatasetCacheState>
GetOrCreateLanceDatasetCacheState(ClientContext &context) {
  return context.registered_state->GetOrCreate<LanceDatasetCacheState>(
      LANCE_DATASET_CACHE_STATE_KEY);
}

static void AppendCacheKeyPart(string &key, const string &value) {
  key += to_string(value.size());
  key += ':';
  key += value;
  key += ';';
}

static void AppendCacheKeyPart(string &key, idx_t value) {
  AppendCacheKeyPart(key, to_string(value));
}

static string BuildPathCacheKey(const string &open_path,
                                const vector<string> &option_keys,
                                const vector<string> &option_values) {
  if (option_keys.size() != option_values.size()) {
    throw InternalException(
        "Storage option keys/values size mismatch for Lance dataset cache");
  }

  string key = "path|";
  AppendCacheKeyPart(key, open_path);
  AppendCacheKeyPart(key, option_keys.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    AppendCacheKeyPart(key, option_keys[i]);
    AppendCacheKeyPart(key, option_values[i]);
  }
  return key;
}

static string
BuildNamespaceCacheKey(const string &endpoint, const string &table_id,
                       const string &bearer_token, const string &api_key,
                       const string &delimiter, const string &headers_tsv) {
  string key = "namespace|";
  AppendCacheKeyPart(key, endpoint);
  AppendCacheKeyPart(key, table_id);
  AppendCacheKeyPart(key, bearer_token);
  AppendCacheKeyPart(key, api_key);
  AppendCacheKeyPart(key, delimiter);
  AppendCacheKeyPart(key, headers_tsv);
  return key;
}

static void *OpenResolvedPathDataset(ClientContext &context,
                                     const string &open_path,
                                     const vector<string> &option_keys,
                                     const vector<string> &option_values) {
  auto *session = LanceGetSessionHandle(context);
  if (option_keys.empty()) {
    return lance_open_dataset_with_session(open_path.c_str(), session);
  }

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);
  return lance_open_dataset_with_storage_options_and_session(
      open_path.c_str(), key_ptrs.data(), value_ptrs.data(), option_keys.size(),
      session);
}

static void *
OpenNamespaceDataset(ClientContext &context, const string &endpoint,
                     const string &table_id, const string &bearer_token,
                     const string &api_key, const string &delimiter,
                     const string &headers_tsv, string &out_table_uri) {
  out_table_uri.clear();
  auto *session = LanceGetSessionHandle(context);
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *uri_ptr = nullptr;
  auto *dataset = lance_open_dataset_in_namespace_with_session(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, session, &uri_ptr);
  if (uri_ptr) {
    out_table_uri = uri_ptr;
    lance_free_string(uri_ptr);
  }
  return dataset;
}

static shared_ptr<LanceDatasetCacheEntry> GetOrOpenDatasetCacheEntry(
    ClientContext &context, const string &cache_key,
    const std::function<shared_ptr<LanceDatasetCacheEntry>()> &open_dataset,
    bool *out_cache_hit) {
  auto state = GetOrCreateLanceDatasetCacheState(context);
  auto entry = state->Get(cache_key);
  if (entry) {
    if (out_cache_hit) {
      *out_cache_hit = true;
    }
    return entry;
  }

  auto opened = open_dataset();
  if (!opened) {
    return nullptr;
  }
  if (out_cache_hit) {
    *out_cache_hit = false;
  }
  return state->PutOrGetExisting(cache_key, opened);
}

shared_ptr<LanceDatasetCacheEntry>
LanceGetOrOpenDatasetEntry(ClientContext &context, const string &path,
                           bool *out_cache_hit) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
  auto cache_key = BuildPathCacheKey(open_path, option_keys, option_values);

  return GetOrOpenDatasetCacheEntry(
      context, cache_key,
      [&]() {
        auto *dataset = OpenResolvedPathDataset(context, open_path, option_keys,
                                                option_values);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }
        return make_shared_ptr<LanceDatasetCacheEntry>(dataset, path);
      },
      out_cache_hit);
}

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryInNamespace(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_display_uri, bool *out_cache_hit) {
  auto cache_key = BuildNamespaceCacheKey(endpoint, table_id, bearer_token,
                                          api_key, delimiter, headers_tsv);
  auto entry = GetOrOpenDatasetCacheEntry(
      context, cache_key,
      [&]() {
        string table_uri;
        auto *dataset =
            OpenNamespaceDataset(context, endpoint, table_id, bearer_token,
                                 api_key, delimiter, headers_tsv, table_uri);
        if (!dataset) {
          return shared_ptr<LanceDatasetCacheEntry>();
        }

        string display_uri = table_uri.empty() ? endpoint + "/" + table_id
                                               : std::move(table_uri);
        return make_shared_ptr<LanceDatasetCacheEntry>(dataset,
                                                       std::move(display_uri));
      },
      out_cache_hit);
  if (entry) {
    out_display_uri = entry->DisplayUri();
  } else {
    out_display_uri.clear();
  }
  return entry;
}

shared_ptr<LanceDatasetCacheEntry> LanceGetOrOpenDatasetEntryForTable(
    ClientContext &context, const LanceTableEntry &table,
    string &out_display_uri, bool *out_cache_hit) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    auto entry =
        LanceGetOrOpenDatasetEntry(context, table.DatasetUri(), out_cache_hit);
    if (entry) {
      out_display_uri = entry->DisplayUri();
    }
    return entry;
  }

  auto &cfg = table.NamespaceConfig();
  unordered_map<string, Value> overrides;
  if (!cfg.bearer_token_override.empty()) {
    overrides["bearer_token"] = Value(cfg.bearer_token_override);
  }
  if (!cfg.api_key_override.empty()) {
    overrides["api_key"] = Value(cfg.api_key_override);
  }

  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);
  return LanceGetOrOpenDatasetEntryInNamespace(
      context, cfg.endpoint, cfg.table_id, bearer_token, api_key, cfg.delimiter,
      cfg.headers_tsv, out_display_uri, out_cache_hit);
}

void LanceInvalidateDatasetCache(ClientContext &context) {
  auto state = context.registered_state->Get<LanceDatasetCacheState>(
      LANCE_DATASET_CACHE_STATE_KEY);
  if (state) {
    state->Clear();
  }
}

} // namespace duckdb
