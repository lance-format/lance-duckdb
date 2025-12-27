#pragma once

#include "duckdb.hpp"

namespace duckdb {

string LanceConsumeLastError();
string LanceFormatErrorSuffix();

bool IsComputedSearchColumn(const string &name);

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel);

void *LanceOpenDataset(ClientContext &context, const string &path);

string LanceNormalizeS3Scheme(const string &path);
void LanceFillS3StorageOptionsFromSecrets(ClientContext &context,
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

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const unordered_map<string, Value> &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_parameter_map_t &options,
                               string &out_bearer_token, string &out_api_key);
void ResolveLanceNamespaceAuthOverrides(
    const unordered_map<string, Value> &options, string &out_bearer_token,
    string &out_api_key);

bool TryLanceNamespaceListTables(ClientContext &context, const string &endpoint,
                                 const string &namespace_id,
                                 const string &bearer_token,
                                 const string &api_key, const string &delimiter,
                                 vector<string> &out_tables, string &out_error);

bool TryLanceDirNamespaceListTables(ClientContext &context, const string &root,
                                    vector<string> &out_tables,
                                    string &out_error);

void *
LanceOpenDatasetInNamespace(ClientContext &context, const string &endpoint,
                            const string &table_id, const string &bearer_token,
                            const string &api_key, const string &delimiter,
                            string &out_table_uri);

int64_t LanceTruncateDataset(ClientContext &context, const string &dataset_uri);

} // namespace duckdb
