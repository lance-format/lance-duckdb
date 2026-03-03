#pragma once

#include "duckdb.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

string LanceConsumeLastError();
string LanceFormatErrorSuffix();

bool IsComputedSearchColumn(const string &name);

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel);

void *LanceOpenDataset(ClientContext &context, const string &path);

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

class LanceTableEntry;

void *LanceOpenDatasetForTable(ClientContext &context,
                               const LanceTableEntry &table,
                               string &out_display_uri);

void ResolveLanceStorageOptionsForTable(ClientContext &context,
                                        const LanceTableEntry &table,
                                        string &out_open_path,
                                        vector<string> &out_option_keys,
                                        vector<string> &out_option_values,
                                        string &out_display_uri);

int64_t LanceTruncateDatasetWithStorageOptions(
    const string &open_path, const vector<string> &option_keys,
    const vector<string> &option_values, const string &display_uri);

int64_t LanceTruncateDataset(ClientContext &context, const string &dataset_uri);

template <typename ContextType>
inline auto PopulateArrowTableSchemaCompatImpl(ContextType &context,
                                               ArrowTableSchema &arrow_table,
                                               const ArrowSchema &arrow_schema,
                                               int)
    -> decltype(ArrowTableFunction::PopulateArrowTableSchema(context,
                                                             arrow_table,
                                                             arrow_schema),
                void()) {
  ArrowTableFunction::PopulateArrowTableSchema(context, arrow_table,
                                               arrow_schema);
}

template <typename ContextType>
inline auto PopulateArrowTableSchemaCompatImpl(ContextType &context,
                                               ArrowTableSchema &arrow_table,
                                               const ArrowSchema &arrow_schema,
                                               long)
    -> decltype(ArrowTableFunction::PopulateArrowTableSchema(
                    DBConfig::GetConfig(context), arrow_table, arrow_schema),
                void()) {
  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(config, arrow_table,
                                               arrow_schema);
}

template <typename ContextType>
inline void PopulateArrowTableSchemaCompat(ContextType &context,
                                           ArrowTableSchema &arrow_table,
                                           const ArrowSchema &arrow_schema) {
  PopulateArrowTableSchemaCompatImpl(context, arrow_table, arrow_schema, 0);
}

} // namespace duckdb
