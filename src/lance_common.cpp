#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "lance_common.hpp"

#include "lance_ffi.hpp"
#include "lance_session_state.hpp"
#include "lance_table_entry.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
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

static void BuildStringPointerArray(const vector<string> &values,
                                    vector<const char *> &out_ptrs) {
  out_ptrs.clear();
  out_ptrs.reserve(values.size());
  for (auto &value : values) {
    out_ptrs.push_back(value.c_str());
  }
}

void FillLanceNamespaceQueryConfig(
    ClientContext &context, const LanceNamespaceTableConfig &cfg, uint64_t k,
    bool prefilter, const string &filter, const vector<string> &columns,
    vector<const char *> &option_key_ptrs,
    vector<const char *> &option_value_ptrs, vector<const char *> &column_ptrs,
    string &bearer_token, string &api_key,
    LanceNamespaceQueryConfig &out_config) {
  static constexpr uint8_t NAMESPACE_KIND_DIRECTORY = 0;
  static constexpr uint8_t NAMESPACE_KIND_REST = 1;

  out_config = {};
  out_config.table_id = cfg.table_id.c_str();
  out_config.k = k;
  out_config.prefilter = prefilter ? 1 : 0;
  out_config.filter = filter.empty() ? nullptr : filter.c_str();

  BuildStringPointerArray(columns, column_ptrs);
  out_config.columns = column_ptrs.empty() ? nullptr : column_ptrs.data();
  out_config.columns_len = column_ptrs.size();

  if (cfg.IsDirectory()) {
    out_config.namespace_kind = NAMESPACE_KIND_DIRECTORY;
    out_config.root = cfg.root.c_str();
    BuildStorageOptionPointerArrays(cfg.option_keys, cfg.option_values,
                                    option_key_ptrs, option_value_ptrs);
    out_config.option_keys =
        option_key_ptrs.empty() ? nullptr : option_key_ptrs.data();
    out_config.option_values =
        option_value_ptrs.empty() ? nullptr : option_value_ptrs.data();
    out_config.options_len = option_key_ptrs.size();
    return;
  }

  out_config.namespace_kind = NAMESPACE_KIND_REST;
  out_config.endpoint = cfg.endpoint.c_str();
  out_config.delimiter =
      cfg.delimiter.empty() ? nullptr : cfg.delimiter.c_str();
  out_config.headers_tsv =
      cfg.headers_tsv.empty() ? nullptr : cfg.headers_tsv.c_str();

  unordered_map<string, Value> overrides;
  if (!cfg.bearer_token_override.empty()) {
    overrides["bearer_token"] = Value(cfg.bearer_token_override);
  }
  if (!cfg.api_key_override.empty()) {
    overrides["api_key"] = Value(cfg.api_key_override);
  }
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);
  out_config.bearer_token =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  out_config.api_key = api_key.empty() ? nullptr : api_key.c_str();
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

string LanceDirectoryNamespaceDatasetUri(const LanceNamespaceTableConfig &cfg) {
  if (!cfg.display_uri.empty()) {
    return LanceNormalizeS3Scheme(cfg.display_uri);
  }

  auto child = cfg.table_id;
  if (!StringUtil::EndsWith(child, ".lance")) {
    child += ".lance";
  }
  string uri;
  if (cfg.root.empty()) {
    uri = std::move(child);
  } else if (cfg.root.back() == '/' || cfg.root.back() == '\\') {
    uri = cfg.root + child;
  } else {
    uri = cfg.root + "/" + child;
  }
  return LanceNormalizeS3Scheme(uri);
}

static string SecretValueToString(const Value &value) {
  if (value.IsNull()) {
    return "";
  }
  return value.ToString();
}

void LanceFillStorageOptionsFromSecrets(ClientContext &context,
                                        const string &path,
                                        vector<string> &out_keys,
                                        vector<string> &out_values) {
  auto &secret_manager = SecretManager::Get(context);
  auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
  auto secret_match = secret_manager.LookupSecret(transaction, path, "lance");
  if (!secret_match.HasMatch() || !secret_match.secret_entry ||
      !secret_match.secret_entry->secret) {
    return;
  }

  auto *kv_secret = dynamic_cast<const KeyValueSecret *>(
      secret_match.secret_entry->secret.get());
  if (!kv_secret) {
    return;
  }

  for (auto &kv : kv_secret->secret_map) {
    if (kv.second.IsNull()) {
      continue;
    }
    out_keys.push_back(kv.first.GetIdentifierName());
    out_values.push_back(kv.second.ToString());
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
  ResolveLanceNamespaceAuthOverrides(options, out_bearer_token, out_api_key);
}

void ResolveLanceNamespaceAuth(ClientContext &context, const string &endpoint,
                               const named_argument_map_t &options,
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

void ResolveLanceNamespaceAuthOverrides(
    const unordered_map<string, Value> &options, string &out_bearer_token,
    string &out_api_key) {
  for (auto &kv : options) {
    if (StringUtil::CIEquals(kv.first, "token")) {
      ApplyAuthOverrideValue(kv.second, out_bearer_token);
      continue;
    }
    if (StringUtil::CIEquals(kv.first, "bearer_token")) {
      ApplyAuthOverrideValue(kv.second, out_bearer_token);
      continue;
    }
    if (StringUtil::CIEquals(kv.first, "api_key")) {
      ApplyAuthOverrideValue(kv.second, out_api_key);
      continue;
    }
  }
}

void ResolveLanceStorageOptions(ClientContext &context, const string &path,
                                string &out_open_path, vector<string> &out_keys,
                                vector<string> &out_values) {
  out_open_path = path;
  out_keys.clear();
  out_values.clear();

  out_open_path = LanceNormalizeS3Scheme(out_open_path);
  LanceFillStorageOptionsFromSecrets(context, out_open_path, out_keys,
                                     out_values);
}

void BuildStorageOptionPointerArrays(const vector<string> &option_keys,
                                     const vector<string> &option_values,
                                     vector<const char *> &out_key_ptrs,
                                     vector<const char *> &out_value_ptrs) {
  if (option_keys.size() != option_values.size()) {
    throw InternalException(
        "Storage option keys/values size mismatch for Lance");
  }
  out_key_ptrs.clear();
  out_value_ptrs.clear();
  out_key_ptrs.reserve(option_keys.size());
  out_value_ptrs.reserve(option_values.size());
  for (idx_t i = 0; i < option_keys.size(); i++) {
    out_key_ptrs.push_back(option_keys[i].c_str());
    out_value_ptrs.push_back(option_values[i].c_str());
  }
}

bool TryLanceNamespaceListTables(
    ClientContext &context, const string &endpoint, const string &namespace_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, vector<string> &out_tables, string &out_error) {
  out_tables.clear();
  out_error.clear();

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto *ptr = lance_namespace_list_tables(
      endpoint.c_str(), namespace_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr);
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

static void ParseStorageOptionsTsv(const char *ptr, vector<string> &out_keys,
                                   vector<string> &out_values) {
  out_keys.clear();
  out_values.clear();
  if (!ptr) {
    return;
  }
  string joined = ptr;
  lance_free_string(ptr);

  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto parts = StringUtil::Split(line, '\t');
    if (parts.size() != 2) {
      continue;
    }
    out_keys.push_back(std::move(parts[0]));
    out_values.push_back(std::move(parts[1]));
  }
}

bool TryLanceNamespaceDescribeTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error) {
  (void)context;
  out_location.clear();
  out_option_keys.clear();
  out_option_values.clear();
  out_error.clear();

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *location_ptr = nullptr;
  const char *options_ptr = nullptr;
  auto rc = lance_namespace_describe_table(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, &location_ptr, &options_ptr);
  if (rc != 0) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  if (location_ptr) {
    out_location = location_ptr;
    lance_free_string(location_ptr);
  }
  ParseStorageOptionsTsv(options_ptr, out_option_keys, out_option_values);
  return true;
}

bool TryLanceNamespaceCreateEmptyTable(
    ClientContext &context, const string &endpoint, const string &table_id,
    const string &bearer_token, const string &api_key, const string &delimiter,
    const string &headers_tsv, string &out_location,
    vector<string> &out_option_keys, vector<string> &out_option_values,
    string &out_error) {
  (void)context;
  out_location.clear();
  out_option_keys.clear();
  out_option_values.clear();
  out_error.clear();

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  const char *location_ptr = nullptr;
  const char *options_ptr = nullptr;
  auto rc = lance_namespace_create_empty_table(
      endpoint.c_str(), table_id.c_str(), bearer_ptr, api_key_ptr,
      delimiter_ptr, headers_ptr, &location_ptr, &options_ptr);
  if (rc != 0) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }
  if (location_ptr) {
    out_location = location_ptr;
    lance_free_string(location_ptr);
  }
  ParseStorageOptionsTsv(options_ptr, out_option_keys, out_option_values);
  return true;
}

bool TryLanceNamespaceDropTable(ClientContext &context, const string &endpoint,
                                const string &table_id,
                                const string &bearer_token,
                                const string &api_key, const string &delimiter,
                                const string &headers_tsv, string &out_error) {
  (void)context;
  out_error.clear();

  const char *bearer_ptr =
      bearer_token.empty() ? nullptr : bearer_token.c_str();
  const char *api_key_ptr = api_key.empty() ? nullptr : api_key.c_str();
  const char *delimiter_ptr = delimiter.empty() ? nullptr : delimiter.c_str();
  const char *headers_ptr = headers_tsv.empty() ? nullptr : headers_tsv.c_str();

  auto rc =
      lance_namespace_drop_table(endpoint.c_str(), table_id.c_str(), bearer_ptr,
                                 api_key_ptr, delimiter_ptr, headers_ptr);
  if (rc != 0) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
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
  ResolveLanceStorageOptions(context, root, open_root, option_keys,
                             option_values);

  vector<const char *> key_ptrs;
  vector<const char *> value_ptrs;
  BuildStorageOptionPointerArrays(option_keys, option_values, key_ptrs,
                                  value_ptrs);

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

void *LanceOpenDataset(ClientContext &context, const string &path) {
  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  ResolveLanceStorageOptions(context, path, open_path, option_keys,
                             option_values);
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

static unordered_map<string, Value>
BuildNamespaceAuthOverrideOptions(const string &bearer_token_override,
                                  const string &api_key_override) {
  unordered_map<string, Value> options;
  if (!bearer_token_override.empty()) {
    options["bearer_token"] = Value(bearer_token_override);
  }
  if (!api_key_override.empty()) {
    options["api_key"] = Value(api_key_override);
  }
  return options;
}

void *LanceOpenDatasetForTable(ClientContext &context,
                               const LanceTableEntry &table,
                               string &out_display_uri) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    return LanceOpenDataset(context, table.DatasetUri());
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    vector<const char *> key_ptrs;
    vector<const char *> value_ptrs;
    BuildStorageOptionPointerArrays(cfg.option_keys, cfg.option_values,
                                    key_ptrs, value_ptrs);

    const char *uri_ptr = nullptr;
    auto *dataset = lance_open_dataset_in_dir_namespace_with_session(
        cfg.root.c_str(), cfg.table_id.c_str(),
        key_ptrs.empty() ? nullptr : key_ptrs.data(),
        value_ptrs.empty() ? nullptr : value_ptrs.data(),
        cfg.option_keys.size(), LanceGetSessionHandle(context), &uri_ptr);
    if (uri_ptr) {
      out_display_uri = uri_ptr;
      lance_free_string(uri_ptr);
    } else if (!cfg.display_uri.empty()) {
      out_display_uri = cfg.display_uri;
    } else {
      out_display_uri = LanceDirectoryNamespaceDatasetUri(cfg);
    }
    return dataset;
  }

  unordered_map<string, Value> overrides = BuildNamespaceAuthOverrideOptions(
      cfg.bearer_token_override, cfg.api_key_override);
  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);

  string table_uri;
  auto *dataset = LanceOpenDatasetInNamespace(
      context, cfg.endpoint, cfg.table_id, bearer_token, api_key, cfg.delimiter,
      cfg.headers_tsv, table_uri);
  if (!table_uri.empty()) {
    out_display_uri = table_uri;
  } else {
    out_display_uri = cfg.endpoint + "/" + cfg.table_id;
  }
  return dataset;
}

void ResolveLanceStorageOptionsForTable(ClientContext &context,
                                        const LanceTableEntry &table,
                                        string &out_open_path,
                                        vector<string> &out_option_keys,
                                        vector<string> &out_option_values,
                                        string &out_display_uri) {
  out_display_uri = table.DatasetUri();
  if (!table.IsNamespaceBacked()) {
    ResolveLanceStorageOptions(context, table.DatasetUri(), out_open_path,
                               out_option_keys, out_option_values);
    return;
  }

  auto &cfg = table.NamespaceConfig();
  if (cfg.IsDirectory()) {
    out_display_uri = LanceDirectoryNamespaceDatasetUri(cfg);
    out_open_path = out_display_uri;
    out_option_keys = cfg.option_keys;
    out_option_values = cfg.option_values;
    return;
  }

  unordered_map<string, Value> overrides = BuildNamespaceAuthOverrideOptions(
      cfg.bearer_token_override, cfg.api_key_override);
  string bearer_token;
  string api_key;
  ResolveLanceNamespaceAuth(context, cfg.endpoint, overrides, bearer_token,
                            api_key);

  string location;
  string error;
  vector<string> option_keys;
  vector<string> option_values;
  if (!TryLanceNamespaceDescribeTable(context, cfg.endpoint, cfg.table_id,
                                      bearer_token, api_key, cfg.delimiter,
                                      cfg.headers_tsv, location, option_keys,
                                      option_values, error)) {
    throw IOException("Failed to describe Lance table via namespace: " +
                      (error.empty() ? "unknown error" : error));
  }

  out_display_uri =
      location.empty() ? (cfg.endpoint + "/" + cfg.table_id) : location;

  if (!option_keys.empty()) {
    out_open_path = LanceNormalizeS3Scheme(location);
    out_option_keys = std::move(option_keys);
    out_option_values = std::move(option_values);
    return;
  }

  ResolveLanceStorageOptions(context, location, out_open_path, out_option_keys,
                             out_option_values);
}

void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
                        DataChunk &chunk, SelectionVector &sel) {
  if (chunk.size() == 0) {
    return;
  }
  unique_ptr<Expression> combined;
  for (auto &it : filters) {
    auto scan_col_idx = it.GetIndex();
    if (scan_col_idx >= chunk.ColumnCount()) {
      throw InternalException("Lance filter refers to a missing scan column");
    }
    BoundReferenceExpression col_expr(
        chunk.data[scan_col_idx].GetType(),
        NumericCast<storage_t>(scan_col_idx.GetIndex()));
    auto expr = it.Filter().ToExpression(col_expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conj = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conj->GetChildrenMutable().push_back(std::move(combined));
      conj->GetChildrenMutable().push_back(std::move(expr));
      combined = std::move(conj);
    }
  }
  for (const auto &filter : filters.GetMultiColumnFilters()) {
    const auto &expression_filter =
        ExpressionFilter::GetExpressionFilter(*filter, "Lance residual filter");
    auto expr = expression_filter.expr->Copy();
    std::function<void(Expression &)> remap = [&](Expression &node) {
      if (node.GetExpressionClass() == ExpressionClass::BOUND_REF) {
        auto &ref = node.Cast<BoundReferenceExpression>();
        if (ref.Index() >= expression_filter.column_indexes.size()) {
          throw InternalException(
              "Lance multi-column filter reference is out of bounds");
        }
        auto projection = expression_filter.column_indexes[ref.Index()];
        if (projection >= chunk.ColumnCount()) {
          throw InternalException(
              "Lance multi-column filter refers to a missing scan column");
        }
        ref.IndexMutable() = projection.GetIndex();
      }
      ExpressionIterator::EnumerateChildren(
          node, [&](Expression &child) { remap(child); });
    };
    remap(*expr);
    if (!combined) {
      combined = std::move(expr);
    } else {
      auto conjunction = make_uniq<BoundConjunctionExpression>(
          ExpressionType::CONJUNCTION_AND);
      conjunction->GetChildrenMutable().push_back(std::move(combined));
      conjunction->GetChildrenMutable().push_back(std::move(expr));
      combined = std::move(conjunction);
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
