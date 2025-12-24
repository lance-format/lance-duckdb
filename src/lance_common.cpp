#include "lance_common.hpp"

#include "lance_ffi.hpp"
#include "duckdb/catalog/catalog_transaction.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"

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

static string NormalizeS3Scheme(const string &path) {
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

static void FillS3StorageOptionsFromSecrets(ClientContext &context,
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

void *LanceOpenDataset(ClientContext &context, const string &path) {
  auto open_path = path;
  vector<string> option_keys;
  vector<string> option_values;

  if (StringUtil::StartsWith(open_path, "s3://") ||
      StringUtil::StartsWith(open_path, "s3a://") ||
      StringUtil::StartsWith(open_path, "s3n://")) {
    open_path = NormalizeS3Scheme(open_path);
    FillS3StorageOptionsFromSecrets(context, open_path, option_keys,
                                    option_values);
  }

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
  return lance_open_dataset_with_storage_options(open_path.c_str(),
                                                 key_ptrs.data(),
                                                 value_ptrs.data(),
                                                 option_keys.size());
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

