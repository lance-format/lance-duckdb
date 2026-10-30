#pragma once

#include "duckdb.hpp"

struct LanceNamespaceQueryConfig;

namespace duckdb {

string LanceConsumeLastError();
string LanceFormatErrorSuffix();

bool IsComputedSearchColumn(const string &name);

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel);

void *LanceOpenDataset(ClientContext &context, const string &path);
void *LanceOpenDatasetAtVersion(ClientContext &context, const string &path,
                                const optional_idx &version);

string LanceNormalizeS3Scheme(const string &path);
void LanceFillStorageOptionsFromSecrets(ClientContext &context,
                                        const string &path,
                                        vector<string> &out_keys,
                                        vector<string> &out_values);
void ResolveLanceStorageOptions(ClientContext &context, const string &path,
                                string &out_open_path,
                                vector<string> &out_option_keys,
                                vector<string> &out_option_values);
void BuildStorageOptionPointerArrays(const vector<string> &option_keys,
                                     const vector<string> &option_values,
                                     vector<const char *> &out_key_ptrs,
                                     vector<const char *> &out_value_ptrs);

static constexpr uint64_t LANCE_DEFAULT_MAX_ROWS_PER_FILE = 1024ULL * 1024ULL;
static constexpr uint64_t LANCE_DEFAULT_MAX_ROWS_PER_GROUP = 1024ULL;
static constexpr uint64_t LANCE_DEFAULT_MAX_BYTES_PER_FILE =
    90ULL * 1024ULL * 1024ULL * 1024ULL;
static constexpr const char *LANCE_DEFAULT_DATA_STORAGE_VERSION = "2.2";

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const unordered_map<string, Value> &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_argument_map_t &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuthOverrides(
    const unordered_map<string, Value> &options, string &out_bearer_token,
    string &out_api_key);

bool TryLanceNamespaceListTables(ClientContext &context, const string &endpoint,
                                 const string &namespace_id,
                                 const string &bearer_token,
                                 const string &api_key, const string &delimiter,
                                 const string &headers_tsv,
                                 vector<string> &out_tables, string &out_error);

bool TryLanceDirNamespaceListTables(ClientContext &context, const string &root,
                                    vector<string> &out_tables,
                                    string &out_error);

void *
LanceOpenDatasetInNamespace(ClientContext &context, const string &endpoint,
                            const string &table_id, const string &bearer_token,
                            const string &api_key, const string &delimiter,
                            const string &headers_tsv, string &out_table_uri);

bool TryLanceNamespaceDescribeTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error);

bool TryLanceNamespaceCreateEmptyTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error);

bool TryLanceNamespaceDropTable(ClientContext &context, const string &endpoint,
                                const string &table_id,
                                const string &bearer_token,
                                const string &api_key, const string &delimiter,
                                const string &headers_tsv, string &out_error);

struct LanceNamespaceTableConfig;
class LanceTableEntry;

void FillLanceNamespaceQueryConfig(
    ClientContext &context, const LanceNamespaceTableConfig &cfg, uint64_t k,
    bool prefilter, const string &filter, const vector<string> &columns,
    vector<const char *> &option_key_ptrs,
    vector<const char *> &option_value_ptrs, vector<const char *> &column_ptrs,
    string &bearer_token, string &api_key,
    ::LanceNamespaceQueryConfig &out_config);

string LanceDirectoryNamespaceDatasetUri(const LanceNamespaceTableConfig &cfg);

void *LanceOpenDatasetForTable(ClientContext &context,
                               const LanceTableEntry &table,
                               string &out_display_uri);

void ResolveLanceStorageOptionsForTable(ClientContext &context,
                                        const LanceTableEntry &table,
                                        string &out_open_path,
                                        vector<string> &out_option_keys,
                                        vector<string> &out_option_values,
                                        string &out_display_uri);

} // namespace duckdb
