#include "lance_common.hpp"

#include "lance_ffi.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

#include <cstring>

namespace duckdb {

string LanceConsumeLastError() {
  auto code = lance_last_error_code();
  string message;
  if (auto *ptr = lance_last_error_message()) {
    message = ptr;
    lance_free_string(ptr);
  }

  if (code == 0 && message.empty()) {
    return "";
  }
  if (message.empty()) {
    return "code=" + to_string(code);
  }
  if (code == 0) {
    return message;
  }
  return message + " (code=" + to_string(code) + ")";
}

string LanceFormatErrorSuffix() {
  auto err = LanceConsumeLastError();
  if (err.empty()) {
    return "";
  }
  return " (Lance error: " + err + ")";
}

bool IsComputedSearchColumn(const string &name) {
  return name == "_distance" || name == "_score" || name == "_hybrid_score";
}

string LanceNormalizeS3Scheme(const string &path) {
  if (StringUtil::StartsWith(path, "s3a://")) {
    return "s3://" + path.substr(6);
  }
  if (StringUtil::StartsWith(path, "s3n://")) {
    return "s3://" + path.substr(6);
  }
  return path;
}

static string SecretValueToString(const Value &value) {
  if (value.IsNull()) {
    return "";
  }
  return value.ToString();
}

static void AddIfNotEmpty(vector<string> &keys, vector<string> &values,
                          const string &key, const string &value) {
  if (value.empty()) {
    return;
  }
  keys.push_back(key);
  values.push_back(value);
}

void LanceFillS3StorageOptionsFromSecrets(ClientContext &context,
                                          const string &path,
                                          vector<string> &out_keys,
                                          vector<string> &out_values) {
  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match = secret_manager.LookupSecret(transaction, path, "s3");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  auto key_id = SecretValueToString(kv_secret->TryGetValue("key_id"));
  auto secret_access_key =
      SecretValueToString(kv_secret->TryGetValue("secret"));
  auto session_token =
      SecretValueToString(kv_secret->TryGetValue("session_token"));
  auto region = SecretValueToString(kv_secret->TryGetValue("region"));
  auto endpoint = SecretValueToString(kv_secret->TryGetValue("endpoint"));
  auto url_style = SecretValueToString(kv_secret->TryGetValue("url_style"));
  auto use_ssl = SecretValueToString(kv_secret->TryGetValue("use_ssl"));

  if (key_id.empty() && secret_access_key.empty()) {
    AddIfNotEmpty(out_keys, out_values, "skip_signature", "true");
  } else {
    AddIfNotEmpty(out_keys, out_values, "access_key_id", key_id);
    AddIfNotEmpty(out_keys, out_values, "secret_access_key", secret_access_key);
    AddIfNotEmpty(out_keys, out_values, "session_token", session_token);
  }

  AddIfNotEmpty(out_keys, out_values, "region", region);
  AddIfNotEmpty(out_keys, out_values, "endpoint", endpoint);

  if (StringUtil::CIEquals(url_style, "vhost") ||
      StringUtil::CIEquals(url_style, "virtual_hosted")) {
    AddIfNotEmpty(out_keys, out_values, "virtual_hosted_style_request", "true");
  } else if (StringUtil::CIEquals(url_style, "path")) {
    AddIfNotEmpty(out_keys, out_values, "virtual_hosted_style_request",
                  "false");
  }

  if (!use_ssl.empty()) {
    if (StringUtil::CIEquals(use_ssl, "false") ||
        StringUtil::CIEquals(use_ssl, "0")) {
      AddIfNotEmpty(out_keys, out_values, "allow_http", "true");
    }
  }
}

static void FillLanceNamespaceAuthFromSecrets(ClientContext &context,
                                              const string &endpoint,
                                              string &out_bearer_token,
                                              string &out_api_key) {
  out_bearer_token.clear();
  out_api_key.clear();

  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match =
      secret_manager.LookupSecret(transaction, endpoint, "lance_namespace");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  out_bearer_token = SecretValueToString(kv_secret->TryGetValue("token"));
  if (out_bearer_token.empty()) {
    out_bearer_token =
        SecretValueToString(kv_secret->TryGetValue("bearer_token"));
  }
  out_api_key = SecretValueToString(kv_secret->TryGetValue("api_key"));
}

static void ApplyAuthOverrideValue(const Value &value, string &out) {
  if (value.IsNull()) {
    return;
  }
  auto s = value.DefaultCastAs(LogicalType::VARCHAR).GetValue<string>();
  if (!s.empty()) {
    out = std::move(s);
  }
}

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const unordered_map<string, Value> &options,
                               string &out_bearer_token, string &out_api_key) {
  FillLanceNamespaceAuthFromSecrets(context, endpoint, out_bearer_token,
                                    out_api_key);
  (void)options;
}

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_parameter_map_t &options,
                               string &out_bearer_token, string &out_api_key) {
  FillLanceNamespaceAuthFromSecrets(context, endpoint, out_bearer_token,
                                    out_api_key);
  auto token_it = options.find("token");
  if (token_it != options.end()) {
    ApplyAuthOverrideValue(token_it->second, out_bearer_token);
  }
  auto bearer_it = options.find("bearer_token");
  if (bearer_it != options.end()) {
    ApplyAuthOverrideValue(bearer_it->second, out_bearer_token);
  }
  auto key_it = options.find("api_key");
  if (key_it != options.end()) {
    ApplyAuthOverrideValue(key_it->second, out_api_key);
  }
}

static void ResolveStorageOptions(ClientContext &context, const string &path,
                                  string &out_open_path,
                                  vector<string> &out_keys,
                                  vector<string> &out_values) {
  out_open_path = path;
  out_keys.clear();
  out_values.clear();

  if (StringUtil::StartsWith(out_open_path, "s3://") ||
      StringUtil::StartsWith(out_open_path, "s3a://") ||
      StringUtil::StartsWith(out_open_path, "s3n://")) {
    out_open_path = LanceNormalizeS3Scheme(out_open_path);
    LanceFillS3StorageOptionsFromSecrets(context, out_open_path, out_keys,
                                         out_values);
  }
}

bool TryLanceNamespaceListTables(ClientContext &context, const string &endpoint,
                                 const string &namespace_id,
                                 const string &bearer_token,
                                 const string &api_key, const string &delimiter,
                                 vector<string> &out_tables,
                                 string &out_error) {
  out_tables.clear();
  out_error.clear();

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();

  auto *ptr =
      lance_namespace_list_tables(endpoint.c_str(), namespace_id.c_str(),
                                  bearer_ptr, api_key_ptr, delimiter_ptr);
  if (!ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  string joined = ptr;
  lance_free_string(ptr);

  vector<string> parts = StringUtil::Split(joined, '\n');
  for (auto &p : parts) {
    if (!p.empty()) {
      out_tables.push_back(std::move(p));
    }
  }
  return true;
}

bool TryLanceDirNamespaceListTables(ClientContext &context, const string &root,
                                    vector<string> &out_tables,
                                    string &out_error) {
  out_tables.clear();
  out_error.clear();

  string open_root;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveStorageOptions(context, root, open_root, option_keys, option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  key_ptrs.reserve(option_keys.size());
  value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    key_ptrs.push_back(option_keys[i].c_str());
    value_ptrs.push_back(option_values[i].c_str());
  }

  auto *ptr = lance_dir_namespace_list_tables(
      open_root.c_str(), key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size());
  if (!ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }

  string joined = ptr;
  lance_free_string(ptr);

  vector<string> parts = StringUtil::Split(joined, '\n');
  for (auto &p : parts) {
    if (!p.empty()) {
      out_tables.push_back(std::move(p));
    }
  }
  return true;
}

void *
LanceOpenDatasetInNamespace(ClientContext &context, const string &endpoint,
                            const string &table_id, const string &bearer_token,
                            const string &api_key, const string &delimiter,
                            string &out_table_uri) {
  (void)context;
  out_table_uri.clear();
  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();

  const char *uri_ptr = nullptr;
  auto *dataset = lance_open_dataset_in_namespace(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, &uri_ptr);
  if (uri_ptr) {
    out_table_uri = uri_ptr;
    lance_free_string(uri_ptr);
  }
  return dataset;
}

void *LanceOpenDataset(ClientContext &context, const string &path) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveStorageOptions(context, path, open_path, option_keys, option_values);

  if (option_keys.empty()) {
    return lance_open_dataset(open_path.c_str());
  }

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  key_ptrs.reserve(option_keys.size());
  value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    key_ptrs.push_back(option_keys[i].c_str());
    value_ptrs.push_back(option_values[i].c_str());
  }
  return lance_open_dataset_with_storage_options(
      open_path.c_str(), key_ptrs.data(), value_ptrs.data(),
      option_keys.size());
}

static constexpr uint64_t DEFAULT_MAX_ROWS_PER_FILE = 1024ULL * 1024ULL;
static constexpr uint64_t DEFAULT_MAX_ROWS_PER_GROUP = 1024ULL;
static constexpr uint64_t DEFAULT_MAX_BYTES_PER_FILE =
    90ULL * 1024ULL * 1024ULL * 1024ULL;

int64_t LanceTruncateDataset(ClientContext &context,
                             const string &dataset_uri) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveStorageOptions(context, dataset_uri, open_path, option_keys,
                        option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  key_ptrs.reserve(option_keys.size());
  value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    key_ptrs.push_back(option_keys[i].c_str());
    value_ptrs.push_back(option_values[i].c_str());
  }

  void *dataset = nullptr;
  if (option_keys.empty()) {
    dataset = lance_open_dataset(open_path.c_str());
  } else {
    dataset = lance_open_dataset_with_storage_options(
        open_path.c_str(), key_ptrs.data(), value_ptrs.data(),
        option_keys.size());
  }
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + dataset_uri +
                      LanceFormatErrorSuffix());
  }

  auto row_count = lance_dataset_count_rows(dataset);
  if (row_count < 0) {
    lance_close_dataset(dataset);
    throw IOException("Failed to count rows from Lance dataset: " +
                      dataset_uri + LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_schema(dataset);
  if (!schema_handle) {
    lance_close_dataset(dataset);
    throw IOException("Failed to get schema from Lance dataset: " +
                      dataset_uri + LanceFormatErrorSuffix());
  }

  ArrowSchemaWrapper schema_root;
  memset(&schema_root.arrow_schema, 0, sizeof(schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &schema_root.arrow_schema) != 0) {
    lance_free_schema(schema_handle);
    lance_close_dataset(dataset);
    throw IOException(
        "Failed to export Lance schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }

  lance_free_schema(schema_handle);
  lance_close_dataset(dataset);

  auto *writer = lance_open_writer_with_storage_options(
      open_path.c_str(), "overwrite",
      key_ptrs.empty() ? nullptr : key_ptrs.data(),
      value_ptrs.empty() ? nullptr : value_ptrs.data(), option_keys.size(),
      DEFAULT_MAX_ROWS_PER_FILE, DEFAULT_MAX_ROWS_PER_GROUP,
      DEFAULT_MAX_BYTES_PER_FILE, &schema_root.arrow_schema);
  if (!writer) {
    throw IOException("Failed to open Lance writer: " + open_path +
                      LanceFormatErrorSuffix());
  }
  auto rc = lance_writer_finish(writer);
  lance_close_writer(writer);
  if (rc != 0) {
    throw IOException("Failed to finalize Lance dataset write" +
                      LanceFormatErrorSuffix());
  }

  return row_count;
}

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel) {
  if (chunk.size() == 0) {
    return;
  }
  unique_ptr<Expression> combined;
  for (auto &it : filters.filters) {
    auto scan_col_idx = it.first;
    if (scan_col_idx >= chunk.ColumnCount()) {
      continue;
    }
    BoundReferenceExpression col_expr(chunk.data[scan_col_idx].GetType(),
                                      NumericCast<storage_t>(scan_col_idx));
    auto expr = it.second->ToExpression(col_expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conj = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conj->children.push_back(std::move(combined));
      conj->children.push_back(std::move(expr));
      combined = std::move(conj);
    }
  }
  if (!combined) {
    return;
  }
  ExpressionExecutor executor(context, *combined);
  auto selected = executor.SelectExpression(chunk, sel);
  if (selected != chunk.size()) {
    chunk.Slice(sel, selected);
  }
}

} // namespace duckdb
