#include "duckdb.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/database_manager.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_resolver.hpp"
#include "lance_table_entry.hpp"

#include <cctype>
#include <cstring>
#include <unordered_map>

namespace duckdb {

// When the user passes a qualified table name (e.g.
// "catalog.schema.table") to the maintenance table functions, the
// registered dataset resolvers may produce a "virtual" URI that
// lance-io cannot open directly -- namespace-backed Lance tables
// expose themselves through an "http://<endpoint>/<table-id>" handle
// whose real location (cos:// / file://) is only reachable through
// the namespace REST API.
//
// To stay correct in that case we keep the original first-argument
// literal in the bind data, and at execution time re-resolve it via
// TryResolveLanceTableEntry() so we can take the namespace-aware
// LanceOpenDatasetForTable() path.  If the input does not name a
// Lance table (plain file path, or table not yet in the catalog for
// any reason), we transparently fall back to LanceOpenDataset() on
// the resolved URI -- preserving the original "open by path" contract.
static void *TryOpenDatasetForMaintenanceInput(ClientContext &context,
                                               const string &input_str,
                                               string &out_display_uri) {
  auto *lance_entry = TryResolveLanceTableEntry(context, input_str);
  if (!lance_entry) {
    return nullptr;
  }
  return LanceOpenDatasetForTable(context, *lance_entry, out_display_uri);
}

static constexpr const char *LANCE_AUTO_CLEANUP_INTERVAL_KEY =
    "lance.auto_cleanup.interval";
static constexpr const char *LANCE_AUTO_CLEANUP_OLDER_THAN_KEY =
    "lance.auto_cleanup.older_than";
static constexpr const char *LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY =
    "lance.auto_cleanup.retain_versions";
static string EscapeJsonString(const string &s);

static vector<pair<string, string>> ParseTsvRows(const char *ptr) {
  if (!ptr) {
    throw IOException("Failed to read Lance dataset config" +
                      LanceFormatErrorSuffix());
  }

  string joined = ptr;
  lance_free_string(ptr);

  vector<pair<string, string>> out;
  for (auto &line : StringUtil::Split(joined, '\n')) {
    if (line.empty()) {
      continue;
    }
    auto parts = StringUtil::Split(line, '\t');
    if (parts.size() != 2) {
      continue;
    }
    out.emplace_back(std::move(parts[0]), std::move(parts[1]));
  }
  return out;
}

enum class LanceMaintenanceOp : uint8_t {
  COMPACT = 1,
  CLEANUP = 2,
  OPTIMIZE_INDEX = 3,
};

struct LanceMaintenanceBindData final : public FunctionData {
  LanceMaintenanceBindData(string dataset_uri_p, string input_str_p,
                           LanceMaintenanceOp op_p, string options_json_p,
                           string index_name_p)
      : dataset_uri(std::move(dataset_uri_p)),
        input_str(std::move(input_str_p)), op(op_p),
        options_json(std::move(options_json_p)),
        index_name(std::move(index_name_p)) {}

  string dataset_uri;
  // Original first-argument literal preserved so the executor can re-resolve
  // it as a Lance catalog table entry when the registered resolver produced
  // a virtual URI (e.g. namespace-backed tables).  See
  // TryOpenDatasetForMaintenanceInput().
  string input_str;
  LanceMaintenanceOp op;
  string options_json;
  string index_name;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceMaintenanceBindData>(dataset_uri, input_str, op,
                                               options_json, index_name);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceMaintenanceBindData>();
    return dataset_uri == other.dataset_uri && input_str == other.input_str &&
           op == other.op && options_json == other.options_json &&
           index_name == other.index_name;
  }
};

struct LanceMaintenanceGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<GlobalTableFunctionState>
LanceMaintenanceInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceMaintenanceGlobalState>();
}

static unique_ptr<FunctionData>
LanceMaintenanceBind(ClientContext &context, TableFunctionBindInput &input,
                     vector<LogicalType> &return_types, vector<string> &names,
                     LanceMaintenanceOp op, idx_t expected_inputs) {
  if (input.inputs.size() != expected_inputs) {
    throw BinderException("invalid argument count");
  }

  auto dataset_uri = ResolveLanceDatasetUri(
      context, input.inputs[0], LanceResolvePolicy::FALLBACK_TO_PATH,
      "__lance_maintenance");
  string input_str =
      input.inputs[0].IsNull() ? string() : input.inputs[0].GetValue<string>();

  string options_json;
  string index_name;
  switch (op) {
  case LanceMaintenanceOp::COMPACT:
  case LanceMaintenanceOp::CLEANUP:
    if (!input.inputs[1].IsNull()) {
      options_json = input.inputs[1].GetValue<string>();
    }
    break;
  case LanceMaintenanceOp::OPTIMIZE_INDEX:
    if (input.inputs[1].IsNull() ||
        input.inputs[1].GetValue<string>().empty()) {
      throw BinderException("index_name must be non-empty");
    }
    index_name = input.inputs[1].GetValue<string>();
    if (!input.inputs[2].IsNull()) {
      options_json = input.inputs[2].GetValue<string>();
    }
    break;
  default:
    throw InternalException("unknown maintenance op");
  }

  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
                  LogicalType::VARCHAR};
  names = {"Operation", "Target", "MetricsJSON"};
  return make_uniq<LanceMaintenanceBindData>(
      std::move(dataset_uri), std::move(input_str), op, std::move(options_json),
      std::move(index_name));
}

static unique_ptr<FunctionData>
LanceCompactBind(ClientContext &context, TableFunctionBindInput &input,
                 vector<LogicalType> &return_types, vector<string> &names) {
  return LanceMaintenanceBind(context, input, return_types, names,
                              LanceMaintenanceOp::COMPACT, 2);
}

static unique_ptr<FunctionData>
LanceCleanupBind(ClientContext &context, TableFunctionBindInput &input,
                 vector<LogicalType> &return_types, vector<string> &names) {
  return LanceMaintenanceBind(context, input, return_types, names,
                              LanceMaintenanceOp::CLEANUP, 2);
}

static unique_ptr<FunctionData>
LanceOptimizeIndexBind(ClientContext &context, TableFunctionBindInput &input,
                       vector<LogicalType> &return_types,
                       vector<string> &names) {
  return LanceMaintenanceBind(context, input, return_types, names,
                              LanceMaintenanceOp::OPTIMIZE_INDEX, 3);
}

static void LanceMaintenanceFunc(ClientContext &context,
                                 TableFunctionInput &data, DataChunk &output);

static TableFunction LanceInternalCompactTableFunction() {
  return TableFunction(
      "__lance_compact_files", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceMaintenanceFunc, LanceCompactBind, LanceMaintenanceInitGlobal);
}

static TableFunction LanceInternalCleanupTableFunction() {
  return TableFunction("__lance_cleanup_old_versions",
                       {LogicalType::VARCHAR, LogicalType::VARCHAR},
                       LanceMaintenanceFunc, LanceCleanupBind,
                       LanceMaintenanceInitGlobal);
}

static TableFunction LanceInternalOptimizeIndexTableFunction() {
  return TableFunction(
      "__lance_optimize_index",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceMaintenanceFunc, LanceOptimizeIndexBind, LanceMaintenanceInitGlobal);
}

static void LanceMaintenanceFunc(ClientContext &context,
                                 TableFunctionInput &data, DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceMaintenanceGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceMaintenanceBindData>();
  const char *options_ptr =
      bind_data.options_json.empty() ? nullptr : bind_data.options_json.c_str();

  string display_uri = bind_data.dataset_uri;
  void *dataset = TryOpenDatasetForMaintenanceInput(
      context, bind_data.input_str, display_uri);
  if (!dataset) {
    display_uri = bind_data.dataset_uri;
    dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  }
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  int32_t rc = -1;
  const char *metrics_ptr = nullptr;
  string op_name;
  switch (bind_data.op) {
  case LanceMaintenanceOp::COMPACT:
    rc = lance_dataset_compact_files_with_options(dataset, options_ptr,
                                                  &metrics_ptr);
    op_name = "compact";
    break;
  case LanceMaintenanceOp::CLEANUP:
    rc = lance_dataset_cleanup_old_versions_with_options(dataset, options_ptr,
                                                         &metrics_ptr);
    op_name = "cleanup";
    break;
  case LanceMaintenanceOp::OPTIMIZE_INDEX:
    rc = lance_dataset_optimize_index_with_options(
        dataset, bind_data.index_name.c_str(), options_ptr, &metrics_ptr);
    op_name = "optimize_index";
    break;
  default:
    rc = -1;
    break;
  }
  lance_close_dataset(dataset);

  if (rc != 0) {
    throw IOException("Failed to execute Lance maintenance operation '" +
                      op_name + "' on dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  string metrics_json = "{}";
  if (metrics_ptr) {
    metrics_json = metrics_ptr;
    lance_free_string(metrics_ptr);
  }

  output.SetCardinality(1);
  output.SetValue(0, 0, Value(op_name));
  output.SetValue(1, 0, Value(display_uri));
  output.SetValue(2, 0, Value(metrics_json));
}

struct LanceAutoCleanupSetBindData final : public FunctionData {
  LanceAutoCleanupSetBindData(string dataset_uri_p, string input_str_p,
                              bool unset_p, int64_t interval_p,
                              string older_than_p, bool has_retain_versions_p,
                              int64_t retain_versions_p)
      : dataset_uri(std::move(dataset_uri_p)),
        input_str(std::move(input_str_p)), unset(unset_p), interval(interval_p),
        older_than(std::move(older_than_p)),
        has_retain_versions(has_retain_versions_p),
        retain_versions(retain_versions_p) {}

  string dataset_uri;
  string input_str;
  bool unset = false;
  int64_t interval = 0;
  string older_than;
  bool has_retain_versions = false;
  int64_t retain_versions = 0;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceAutoCleanupSetBindData>(
        dataset_uri, input_str, unset, interval, older_than,
        has_retain_versions, retain_versions);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceAutoCleanupSetBindData>();
    return dataset_uri == other.dataset_uri && input_str == other.input_str &&
           unset == other.unset && interval == other.interval &&
           older_than == other.older_than &&
           has_retain_versions == other.has_retain_versions &&
           retain_versions == other.retain_versions;
  }
};

struct LanceAutoCleanupShowBindData final : public FunctionData {
  LanceAutoCleanupShowBindData(string dataset_uri_p, string input_str_p)
      : dataset_uri(std::move(dataset_uri_p)),
        input_str(std::move(input_str_p)) {}

  string dataset_uri;
  string input_str;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceAutoCleanupShowBindData>(dataset_uri, input_str);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceAutoCleanupShowBindData>();
    return dataset_uri == other.dataset_uri && input_str == other.input_str;
  }
};

struct LanceAutoCleanupShowGlobalState final : public GlobalTableFunctionState {
  explicit LanceAutoCleanupShowGlobalState(vector<pair<string, string>> rows_p)
      : rows(std::move(rows_p)) {}

  vector<pair<string, string>> rows;
  idx_t offset = 0;
};

static unique_ptr<FunctionData>
LanceSetAutoCleanupBind(ClientContext &context, TableFunctionBindInput &input,
                        vector<LogicalType> &return_types,
                        vector<string> &names) {
  if (input.inputs.size() != 5) {
    throw BinderException(
        "__lance_set_auto_cleanup requires (dataset, interval, older_than, "
        "retain_versions, unset)");
  }

  auto dataset_uri = ResolveLanceDatasetUri(
      context, input.inputs[0], LanceResolvePolicy::FALLBACK_TO_PATH,
      "__lance_set_auto_cleanup");
  string input_str =
      input.inputs[0].IsNull() ? string() : input.inputs[0].GetValue<string>();

  bool unset = false;
  if (!input.inputs[4].IsNull()) {
    unset =
        input.inputs[4].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  int64_t interval = 0;
  string older_than;
  bool has_retain_versions = false;
  int64_t retain_versions = 0;
  if (!unset) {
    if (input.inputs[1].IsNull()) {
      throw BinderException("auto cleanup interval must be provided");
    }
    interval =
        input.inputs[1].DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
    if (interval <= 0) {
      throw BinderException("auto cleanup interval must be > 0");
    }

    if (input.inputs[2].IsNull()) {
      throw BinderException("auto cleanup older_than must be provided");
    }
    older_than = input.inputs[2].GetValue<string>();
    if (older_than.empty()) {
      throw BinderException("auto cleanup older_than cannot be empty");
    }

    if (!input.inputs[3].IsNull()) {
      retain_versions = input.inputs[3]
                            .DefaultCastAs(LogicalType::BIGINT)
                            .GetValue<int64_t>();
      if (retain_versions <= 0) {
        throw BinderException("auto cleanup retain_versions must be > 0");
      }
      has_retain_versions = true;
    }
  }

  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR,
                  LogicalType::VARCHAR};
  names = {"Operation", "Target", "MetricsJSON"};
  return make_uniq<LanceAutoCleanupSetBindData>(
      std::move(dataset_uri), std::move(input_str), unset, interval,
      std::move(older_than), has_retain_versions, retain_versions);
}

static unique_ptr<FunctionData>
LanceShowAutoCleanupBind(ClientContext &context, TableFunctionBindInput &input,
                         vector<LogicalType> &return_types,
                         vector<string> &names) {
  if (input.inputs.size() != 1) {
    throw BinderException("__lance_show_auto_cleanup requires 1 argument");
  }
  auto dataset_uri = ResolveLanceDatasetUri(
      context, input.inputs[0], LanceResolvePolicy::FALLBACK_TO_PATH,
      "__lance_show_auto_cleanup");
  string input_str =
      input.inputs[0].IsNull() ? string() : input.inputs[0].GetValue<string>();
  return_types = {LogicalType::VARCHAR, LogicalType::VARCHAR};
  names = {"Key", "Value"};
  return make_uniq<LanceAutoCleanupShowBindData>(std::move(dataset_uri),
                                                 std::move(input_str));
}

static void UpdateDatasetConfigOrThrow(void *dataset, const string &dataset_uri,
                                       const char *key, const char *value) {
  auto rc = lance_dataset_update_config(dataset, key, value);
  if (rc != 0) {
    throw IOException("Failed to update Lance maintenance config on dataset: " +
                      dataset_uri + LanceFormatErrorSuffix());
  }
}

static void LanceSetAutoCleanupFunc(ClientContext &context,
                                    TableFunctionInput &data,
                                    DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceMaintenanceGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceAutoCleanupSetBindData>();
  string display_uri = bind_data.dataset_uri;
  void *dataset = TryOpenDatasetForMaintenanceInput(
      context, bind_data.input_str, display_uri);
  if (!dataset) {
    display_uri = bind_data.dataset_uri;
    dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  }
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  string op_name;
  string metrics_json;
  try {
    if (bind_data.unset) {
      UpdateDatasetConfigOrThrow(dataset, display_uri,
                                 LANCE_AUTO_CLEANUP_INTERVAL_KEY, nullptr);
      UpdateDatasetConfigOrThrow(dataset, display_uri,
                                 LANCE_AUTO_CLEANUP_OLDER_THAN_KEY, nullptr);
      UpdateDatasetConfigOrThrow(dataset, display_uri,
                                 LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY,
                                 nullptr);
      op_name = "unset_auto_cleanup";
      metrics_json = "{\"enabled\":false}";
    } else {
      auto interval = to_string(bind_data.interval);
      UpdateDatasetConfigOrThrow(dataset, display_uri,
                                 LANCE_AUTO_CLEANUP_INTERVAL_KEY,
                                 interval.c_str());
      UpdateDatasetConfigOrThrow(dataset, display_uri,
                                 LANCE_AUTO_CLEANUP_OLDER_THAN_KEY,
                                 bind_data.older_than.c_str());

      if (bind_data.has_retain_versions) {
        auto retain_versions = to_string(bind_data.retain_versions);
        UpdateDatasetConfigOrThrow(dataset, display_uri,
                                   LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY,
                                   retain_versions.c_str());
      } else {
        UpdateDatasetConfigOrThrow(dataset, display_uri,
                                   LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY,
                                   nullptr);
      }

      op_name = "set_auto_cleanup";
      metrics_json = "{\"enabled\":true,\"interval\":" + interval +
                     ",\"older_than\":\"" +
                     EscapeJsonString(bind_data.older_than) + "\"";
      if (bind_data.has_retain_versions) {
        metrics_json +=
            ",\"retain_versions\":" + to_string(bind_data.retain_versions);
      }
      metrics_json += "}";
    }
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }
  lance_close_dataset(dataset);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value(op_name));
  output.SetValue(1, 0, Value(display_uri));
  output.SetValue(2, 0, Value(metrics_json));
}

static unique_ptr<GlobalTableFunctionState>
LanceShowAutoCleanupInitGlobal(ClientContext &context,
                               TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceAutoCleanupShowBindData>();
  string display_uri = bind_data.dataset_uri;
  void *dataset = TryOpenDatasetForMaintenanceInput(
      context, bind_data.input_str, display_uri);
  if (!dataset) {
    display_uri = bind_data.dataset_uri;
    dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  }
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + display_uri +
                      LanceFormatErrorSuffix());
  }

  vector<pair<string, string>> rows;
  try {
    auto all_configs = ParseTsvRows(lance_dataset_list_config(dataset));
    unordered_map<string, string> filtered;
    for (auto &kv : all_configs) {
      if (kv.first == LANCE_AUTO_CLEANUP_INTERVAL_KEY ||
          kv.first == LANCE_AUTO_CLEANUP_OLDER_THAN_KEY ||
          kv.first == LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY) {
        filtered[kv.first] = kv.second;
      }
    }

    auto enabled =
        filtered.find(LANCE_AUTO_CLEANUP_INTERVAL_KEY) != filtered.end() &&
        filtered.find(LANCE_AUTO_CLEANUP_OLDER_THAN_KEY) != filtered.end();
    rows.emplace_back("enabled", enabled ? "true" : "false");
    if (filtered.find(LANCE_AUTO_CLEANUP_INTERVAL_KEY) != filtered.end()) {
      rows.emplace_back("interval", filtered[LANCE_AUTO_CLEANUP_INTERVAL_KEY]);
    }
    if (filtered.find(LANCE_AUTO_CLEANUP_OLDER_THAN_KEY) != filtered.end()) {
      rows.emplace_back("older_than",
                        filtered[LANCE_AUTO_CLEANUP_OLDER_THAN_KEY]);
    }
    if (filtered.find(LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY) !=
        filtered.end()) {
      rows.emplace_back("retain_versions",
                        filtered[LANCE_AUTO_CLEANUP_RETAIN_VERSIONS_KEY]);
    }
  } catch (...) {
    lance_close_dataset(dataset);
    throw;
  }
  lance_close_dataset(dataset);
  return make_uniq<LanceAutoCleanupShowGlobalState>(std::move(rows));
}

static void LanceShowAutoCleanupFunc(ClientContext &, TableFunctionInput &data,
                                     DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceAutoCleanupShowGlobalState>();
  if (gstate.offset >= gstate.rows.size()) {
    output.SetCardinality(0);
    return;
  }

  auto remaining = gstate.rows.size() - gstate.offset;
  auto count = MinValue<idx_t>(remaining, STANDARD_VECTOR_SIZE);
  output.SetCardinality(count);

  for (idx_t i = 0; i < count; i++) {
    auto &kv = gstate.rows[gstate.offset + i];
    output.SetValue(0, i, Value(kv.first));
    output.SetValue(1, i, Value(kv.second));
  }
  gstate.offset += count;
}

static TableFunction LanceInternalSetAutoCleanupTableFunction() {
  return TableFunction("__lance_set_auto_cleanup",
                       {LogicalType::VARCHAR, LogicalType::BIGINT,
                        LogicalType::VARCHAR, LogicalType::BIGINT,
                        LogicalType::BOOLEAN},
                       LanceSetAutoCleanupFunc, LanceSetAutoCleanupBind,
                       LanceMaintenanceInitGlobal);
}

static TableFunction LanceInternalShowAutoCleanupTableFunction() {
  return TableFunction("__lance_show_auto_cleanup", {LogicalType::VARCHAR},
                       LanceShowAutoCleanupFunc, LanceShowAutoCleanupBind,
                       LanceShowAutoCleanupInitGlobal);
}

enum class LanceMaintenanceStmtKind : uint8_t {
  COMPACT = 1,
  CLEANUP = 2,
  OPTIMIZE_INDEX = 3,
  SET_AUTO_CLEANUP = 4,
  UNSET_AUTO_CLEANUP = 5,
  SHOW_AUTO_CLEANUP = 6,
};

static bool IsSpace(char c) {
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

static string TrimCopy(string s) {
  StringUtil::Trim(s);
  return s;
}

static string TrimTrailingSemicolons(string s) {
  StringUtil::Trim(s);
  while (!s.empty() && s.back() == ';') {
    s.pop_back();
    StringUtil::Trim(s);
  }
  return s;
}

static bool HasKeywordPrefix(const string &lower, const char *keyword) {
  auto kw_len = strlen(keyword);
  if (lower.size() < kw_len) {
    return false;
  }
  if (lower.compare(0, kw_len, keyword) != 0) {
    return false;
  }
  if (lower.size() == kw_len) {
    return true;
  }
  return IsSpace(lower[kw_len]);
}

static bool ConsumeKeyword(string &sql, const string &keyword) {
  auto trimmed = TrimCopy(sql);
  auto lower = StringUtil::Lower(trimmed);
  if (!HasKeywordPrefix(lower, keyword.c_str())) {
    return false;
  }
  sql = TrimCopy(trimmed.substr(keyword.size()));
  return true;
}

static bool IsIdentChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' ||
         c == '.';
}

static bool StartsWithQuotedString(const string &sql) {
  return !sql.empty() && sql[0] == '\'';
}

static bool TryParseSqlStringLiteral(const string &sql, string &out_value,
                                     idx_t &out_consumed) {
  out_value.clear();
  out_consumed = 0;
  if (sql.empty() || sql[0] != '\'') {
    return false;
  }
  idx_t i = 1;
  while (i < sql.size()) {
    auto c = sql[i];
    if (c == '\'') {
      if (i + 1 < sql.size() && sql[i + 1] == '\'') {
        out_value.push_back('\'');
        i += 2;
        continue;
      }
      out_consumed = i + 1;
      return true;
    }
    out_value.push_back(c);
    i++;
  }
  return false;
}

static bool TryParseParenList(const string &sql, vector<string> &out_items,
                              idx_t &out_consumed) {
  out_items.clear();
  out_consumed = 0;
  auto s = TrimCopy(sql);
  if (s.empty() || s[0] != '(') {
    return false;
  }
  idx_t i = 1;
  idx_t start = i;
  bool in_str = false;
  for (; i < s.size(); i++) {
    auto c = s[i];
    if (c == '\'') {
      if (in_str) {
        if (i + 1 < s.size() && s[i + 1] == '\'') {
          i++;
          continue;
        }
        in_str = false;
      } else {
        in_str = true;
      }
      continue;
    }
    if (in_str) {
      continue;
    }
    if (c == ',') {
      auto part = TrimCopy(s.substr(start, i - start));
      if (!part.empty()) {
        out_items.push_back(part);
      }
      start = i + 1;
      continue;
    }
    if (c == ')') {
      auto part = TrimCopy(s.substr(start, i - start));
      if (!part.empty()) {
        out_items.push_back(part);
      }
      out_consumed = i + 1;
      return true;
    }
  }
  return false;
}

static bool IsNumberLiteral(const string &s) {
  if (s.empty()) {
    return false;
  }
  idx_t i = 0;
  if (s[i] == '-' || s[i] == '+') {
    i++;
  }
  bool saw_digit = false;
  bool saw_dot = false;
  for (; i < s.size(); i++) {
    auto c = s[i];
    if (c >= '0' && c <= '9') {
      saw_digit = true;
      continue;
    }
    if (c == '.' && !saw_dot) {
      saw_dot = true;
      continue;
    }
    return false;
  }
  return saw_digit;
}

static string EscapeJsonString(const string &s) {
  string out;
  out.reserve(s.size() + 8);
  for (auto c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      out.push_back(c);
      break;
    }
  }
  return out;
}

static bool TryBuildOptionsJsonFromWithClause(const string &with_clause_sql,
                                              string &out_options_json,
                                              string &out_error) {
  out_error.clear();
  out_options_json.clear();

  auto rest = TrimCopy(with_clause_sql);
  if (rest.empty()) {
    return true;
  }

  auto lower = StringUtil::Lower(rest);
  if (HasKeywordPrefix(lower, "with")) {
    rest = TrimCopy(rest.substr(strlen("with")));
  }

  vector<string> kvs;
  idx_t consumed = 0;
  if (!TryParseParenList(rest, kvs, consumed)) {
    out_error = "WITH clause must be a parenthesized key=value list";
    return false;
  }
  auto trailing = TrimCopy(rest.substr(consumed));
  if (!trailing.empty()) {
    out_error = "unexpected trailing tokens after WITH (...)";
    return false;
  }

  string json = "{";
  bool first = true;
  for (auto &kv : kvs) {
    auto eq_pos = kv.find('=');
    if (eq_pos == string::npos) {
      out_error = "WITH (...) entries must be key=value";
      return false;
    }
    auto key = TrimCopy(kv.substr(0, eq_pos));
    auto val = TrimCopy(kv.substr(eq_pos + 1));
    if (key.empty()) {
      out_error = "WITH (...) entry key cannot be empty";
      return false;
    }

    if (!first) {
      json += ",";
    }
    first = false;
    json += "\"";
    json += EscapeJsonString(key);
    json += "\":";

    if (StartsWithQuotedString(val)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(val, lit, lit_consumed) ||
          TrimCopy(val.substr(lit_consumed)) != "") {
        out_error = "WITH string values must be single-quoted string literals";
        return false;
      }
      json += "\"";
      json += EscapeJsonString(lit);
      json += "\"";
      continue;
    }

    auto v_lower = StringUtil::Lower(val);
    if (v_lower == "true" || v_lower == "false") {
      json += v_lower;
    } else if (IsNumberLiteral(val)) {
      json += val;
    } else {
      json += "\"";
      json += EscapeJsonString(val);
      json += "\"";
    }
  }
  json += "}";
  if (json != "{}") {
    out_options_json = json;
  }
  return true;
}

static bool TryParseIntegerLiteral(const string &value, int64_t &out) {
  out = 0;
  if (value.empty()) {
    return false;
  }
  idx_t i = 0;
  if (value[0] == '+' || value[0] == '-') {
    i = 1;
  }
  if (i >= value.size()) {
    return false;
  }
  for (; i < value.size(); i++) {
    auto c = value[i];
    if (c < '0' || c > '9') {
      return false;
    }
  }
  try {
    auto pos = size_t(0);
    auto parsed = std::stoll(value, &pos);
    if (pos != value.size()) {
      return false;
    }
    out = parsed;
    return true;
  } catch (...) {
    return false;
  }
}

static bool TryParseAutoCleanupSetOptions(const string &with_clause_sql,
                                          int64_t &out_interval,
                                          string &out_older_than,
                                          bool &out_has_retain_versions,
                                          int64_t &out_retain_versions,
                                          string &out_error) {
  out_interval = 0;
  out_older_than.clear();
  out_has_retain_versions = false;
  out_retain_versions = 0;
  out_error.clear();

  auto rest = TrimCopy(with_clause_sql);
  if (rest.empty()) {
    out_error = "SET AUTO_CLEANUP requires WITH (...)";
    return false;
  }

  auto lower = StringUtil::Lower(rest);
  if (HasKeywordPrefix(lower, "with")) {
    rest = TrimCopy(rest.substr(strlen("with")));
  }

  vector<string> kvs;
  idx_t consumed = 0;
  if (!TryParseParenList(rest, kvs, consumed)) {
    out_error = "WITH clause must be a parenthesized key=value list";
    return false;
  }
  auto trailing = TrimCopy(rest.substr(consumed));
  if (!trailing.empty()) {
    out_error = "unexpected trailing tokens after WITH (...)";
    return false;
  }

  bool has_interval = false;
  bool has_older_than = false;
  for (auto &kv : kvs) {
    auto eq_pos = kv.find('=');
    if (eq_pos == string::npos) {
      out_error = "WITH (...) entries must be key=value";
      return false;
    }

    auto key = StringUtil::Lower(TrimCopy(kv.substr(0, eq_pos)));
    auto value = TrimCopy(kv.substr(eq_pos + 1));
    if (key.empty()) {
      out_error = "WITH (...) entry key cannot be empty";
      return false;
    }

    if (key == "interval") {
      int64_t interval = 0;
      if (!TryParseIntegerLiteral(value, interval)) {
        out_error = "AUTO_CLEANUP interval must be an integer";
        return false;
      }
      if (interval <= 0) {
        out_error = "AUTO_CLEANUP interval must be > 0";
        return false;
      }
      out_interval = interval;
      has_interval = true;
      continue;
    }

    if (key == "older_than") {
      string older_than;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(value, older_than, lit_consumed) ||
          !TrimCopy(value.substr(lit_consumed)).empty()) {
        out_error =
            "AUTO_CLEANUP older_than must be a single-quoted string literal";
        return false;
      }
      if (older_than.empty()) {
        out_error = "AUTO_CLEANUP older_than cannot be empty";
        return false;
      }
      out_older_than = older_than;
      has_older_than = true;
      continue;
    }

    if (key == "retain_versions") {
      int64_t retain_versions = 0;
      if (!TryParseIntegerLiteral(value, retain_versions)) {
        out_error = "AUTO_CLEANUP retain_versions must be an integer";
        return false;
      }
      if (retain_versions <= 0) {
        out_error = "AUTO_CLEANUP retain_versions must be > 0";
        return false;
      }
      out_retain_versions = retain_versions;
      out_has_retain_versions = true;
      continue;
    }

    out_error = "unsupported AUTO_CLEANUP option: " + key;
    return false;
  }

  if (!has_interval || !has_older_than) {
    out_error = "AUTO_CLEANUP requires interval and older_than";
    return false;
  }
  return true;
}

static bool TryParseIdentifier(const string &sql, string &out_ident,
                               idx_t &out_consumed) {
  out_ident.clear();
  out_consumed = 0;
  if (sql.empty()) {
    return false;
  }
  idx_t i = 0;
  while (i < sql.size() && IsIdentChar(sql[i])) {
    i++;
  }
  if (i == 0) {
    return false;
  }
  out_ident = sql.substr(0, i);
  out_consumed = i;
  return true;
}

struct LanceMaintenanceParseData final : public ParserExtensionParseData {
  explicit LanceMaintenanceParseData(LanceMaintenanceStmtKind kind_p)
      : kind(kind_p) {}

  LanceMaintenanceStmtKind kind;
  bool target_is_path = false;
  string dataset_uri;
  string target_sql;
  string options_json;
  string index_name;
  int64_t auto_cleanup_interval = 0;
  string auto_cleanup_older_than;
  bool auto_cleanup_has_retain_versions = false;
  int64_t auto_cleanup_retain_versions = 0;

  unique_ptr<ParserExtensionParseData> Copy() const override {
    auto out = make_uniq<LanceMaintenanceParseData>(kind);
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->target_sql = target_sql;
    out->options_json = options_json;
    out->index_name = index_name;
    out->auto_cleanup_interval = auto_cleanup_interval;
    out->auto_cleanup_older_than = auto_cleanup_older_than;
    out->auto_cleanup_has_retain_versions = auto_cleanup_has_retain_versions;
    out->auto_cleanup_retain_versions = auto_cleanup_retain_versions;
    return std::move(out);
  }

  string ToString() const override {
    switch (kind) {
    case LanceMaintenanceStmtKind::COMPACT:
      return "OPTIMIZE " + target_sql;
    case LanceMaintenanceStmtKind::CLEANUP:
      return "VACUUM LANCE " + target_sql;
    case LanceMaintenanceStmtKind::OPTIMIZE_INDEX:
      return "ALTER INDEX " + index_name + " ON " + target_sql + " OPTIMIZE";
    case LanceMaintenanceStmtKind::SET_AUTO_CLEANUP:
      return "ALTER TABLE " + target_sql + " SET AUTO_CLEANUP";
    case LanceMaintenanceStmtKind::UNSET_AUTO_CLEANUP:
      return "ALTER TABLE " + target_sql + " UNSET AUTO_CLEANUP";
    case LanceMaintenanceStmtKind::SHOW_AUTO_CLEANUP:
      return "SHOW MAINTENANCE ON " + target_sql;
    default:
      return "LANCE MAINTENANCE";
    }
  }
};

static bool TryParseTarget(string &rest, bool &target_is_path,
                           string &dataset_uri, string &target_sql,
                           string &out_error) {
  out_error.clear();
  target_is_path = false;
  dataset_uri.clear();
  target_sql.clear();

  idx_t consumed = 0;
  if (StartsWithQuotedString(rest)) {
    string lit;
    idx_t lit_consumed = 0;
    if (!TryParseSqlStringLiteral(rest, lit, lit_consumed)) {
      out_error = "invalid dataset path string literal";
      return false;
    }
    dataset_uri = lit;
    target_sql = rest.substr(0, lit_consumed);
    rest = TrimCopy(rest.substr(lit_consumed));
    target_is_path = true;
    return true;
  }

  string ident;
  if (!TryParseIdentifier(rest, ident, consumed)) {
    out_error = "expected dataset path or table identifier";
    return false;
  }
  target_sql = ident;
  rest = TrimCopy(rest.substr(consumed));
  return true;
}

static ParserExtensionParseResult LanceMaintenanceParse(ParserExtensionInfo *,
                                                        const string &query) {
  auto trimmed = TrimTrailingSemicolons(query);
  if (trimmed.empty()) {
    return ParserExtensionParseResult();
  }
  auto lower = StringUtil::Lower(trimmed);

  if (HasKeywordPrefix(lower, "optimize")) {
    auto rest = TrimCopy(trimmed.substr(strlen("optimize")));
    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    string parse_error;
    if (!TryParseTarget(rest, target_is_path, dataset_uri, target_sql,
                        parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }

    string options_json;
    if (!TryBuildOptionsJsonFromWithClause(rest, options_json, parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }

    auto out =
        make_uniq<LanceMaintenanceParseData>(LanceMaintenanceStmtKind::COMPACT);
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->target_sql = target_sql;
    out->options_json = options_json;
    return ParserExtensionParseResult(std::move(out));
  }

  if (HasKeywordPrefix(lower, "vacuum")) {
    auto rest = TrimCopy(trimmed.substr(strlen("vacuum")));
    if (!ConsumeKeyword(rest, "lance")) {
      return ParserExtensionParseResult();
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    string parse_error;
    if (!TryParseTarget(rest, target_is_path, dataset_uri, target_sql,
                        parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }

    string options_json;
    if (!TryBuildOptionsJsonFromWithClause(rest, options_json, parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }

    auto out =
        make_uniq<LanceMaintenanceParseData>(LanceMaintenanceStmtKind::CLEANUP);
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->target_sql = target_sql;
    out->options_json = options_json;
    return ParserExtensionParseResult(std::move(out));
  }

  if (HasKeywordPrefix(lower, "show")) {
    auto rest = TrimCopy(trimmed.substr(strlen("show")));
    if (!ConsumeKeyword(rest, "maintenance")) {
      return ParserExtensionParseResult();
    }
    if (!ConsumeKeyword(rest, "on") && !ConsumeKeyword(rest, "from")) {
      return ParserExtensionParseResult(
          "SHOW MAINTENANCE requires ON/FROM <target>");
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    string parse_error;
    if (!TryParseTarget(rest, target_is_path, dataset_uri, target_sql,
                        parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }
    if (!TrimCopy(rest).empty()) {
      return ParserExtensionParseResult("unexpected trailing tokens");
    }

    auto out = make_uniq<LanceMaintenanceParseData>(
        LanceMaintenanceStmtKind::SHOW_AUTO_CLEANUP);
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->target_sql = target_sql;
    return ParserExtensionParseResult(std::move(out));
  }

  if (HasKeywordPrefix(lower, "alter")) {
    auto rest = TrimCopy(trimmed.substr(strlen("alter")));
    if (ConsumeKeyword(rest, "index")) {
      string index_name;
      idx_t consumed = 0;
      if (!TryParseIdentifier(rest, index_name, consumed)) {
        return ParserExtensionParseResult("ALTER INDEX requires an index name");
      }
      rest = TrimCopy(rest.substr(consumed));
      if (!ConsumeKeyword(rest, "on")) {
        return ParserExtensionParseResult("ALTER INDEX requires ON <target>");
      }

      bool target_is_path = false;
      string dataset_uri;
      string target_sql;
      string parse_error;
      if (!TryParseTarget(rest, target_is_path, dataset_uri, target_sql,
                          parse_error)) {
        return ParserExtensionParseResult(parse_error);
      }
      if (!ConsumeKeyword(rest, "optimize")) {
        return ParserExtensionParseResult(
            "ALTER INDEX requires trailing OPTIMIZE");
      }

      string options_json;
      if (!TryBuildOptionsJsonFromWithClause(rest, options_json, parse_error)) {
        return ParserExtensionParseResult(parse_error);
      }

      auto out = make_uniq<LanceMaintenanceParseData>(
          LanceMaintenanceStmtKind::OPTIMIZE_INDEX);
      out->target_is_path = target_is_path;
      out->dataset_uri = dataset_uri;
      out->target_sql = target_sql;
      out->options_json = options_json;
      out->index_name = index_name;
      return ParserExtensionParseResult(std::move(out));
    }

    if (!ConsumeKeyword(rest, "table")) {
      return ParserExtensionParseResult();
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    string parse_error;
    if (!TryParseTarget(rest, target_is_path, dataset_uri, target_sql,
                        parse_error)) {
      return ParserExtensionParseResult(parse_error);
    }

    if (ConsumeKeyword(rest, "set")) {
      if (!ConsumeKeyword(rest, "auto_cleanup")) {
        return ParserExtensionParseResult(
            "ALTER TABLE ... SET AUTO_CLEANUP is required");
      }

      int64_t interval = 0;
      string older_than;
      bool has_retain_versions = false;
      int64_t retain_versions = 0;
      if (!TryParseAutoCleanupSetOptions(rest, interval, older_than,
                                         has_retain_versions, retain_versions,
                                         parse_error)) {
        return ParserExtensionParseResult(parse_error);
      }
      auto out = make_uniq<LanceMaintenanceParseData>(
          LanceMaintenanceStmtKind::SET_AUTO_CLEANUP);
      out->target_is_path = target_is_path;
      out->dataset_uri = dataset_uri;
      out->target_sql = target_sql;
      out->auto_cleanup_interval = interval;
      out->auto_cleanup_older_than = older_than;
      out->auto_cleanup_has_retain_versions = has_retain_versions;
      out->auto_cleanup_retain_versions = retain_versions;
      return ParserExtensionParseResult(std::move(out));
    }

    if (ConsumeKeyword(rest, "unset")) {
      if (!ConsumeKeyword(rest, "auto_cleanup")) {
        return ParserExtensionParseResult(
            "ALTER TABLE ... UNSET AUTO_CLEANUP is required");
      }
      if (!TrimCopy(rest).empty()) {
        return ParserExtensionParseResult("unexpected trailing tokens");
      }
      auto out = make_uniq<LanceMaintenanceParseData>(
          LanceMaintenanceStmtKind::UNSET_AUTO_CLEANUP);
      out->target_is_path = target_is_path;
      out->dataset_uri = dataset_uri;
      out->target_sql = target_sql;
      return ParserExtensionParseResult(std::move(out));
    }

    return ParserExtensionParseResult(
        "ALTER TABLE maintenance syntax must be SET/UNSET AUTO_CLEANUP");
  }

  return ParserExtensionParseResult();
}

static ParserExtensionPlanResult
LanceMaintenancePlan(ParserExtensionInfo *, ClientContext &context,
                     unique_ptr<ParserExtensionParseData> parse_data_p) {
  auto *parse_data =
      dynamic_cast<LanceMaintenanceParseData *>(parse_data_p.get());
  if (!parse_data) {
    throw InternalException(
        "LanceMaintenancePlan received unexpected parse data");
  }

  string target;
  if (parse_data->target_is_path) {
    target = parse_data->dataset_uri;
  } else {
    auto qname = QualifiedName::Parse(parse_data->target_sql);
    if (qname.catalog.empty()) {
      qname.catalog = DatabaseManager::GetDefaultDatabase(context);
    }
    if (qname.schema.empty()) {
      qname.schema = DEFAULT_SCHEMA;
    }
    target = qname.catalog + "." + qname.schema + "." + qname.name;
  }

  ParserExtensionPlanResult result;
  switch (parse_data->kind) {
  case LanceMaintenanceStmtKind::COMPACT:
    result.function = LanceInternalCompactTableFunction();
    result.parameters = {Value(target), Value(parse_data->options_json)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  case LanceMaintenanceStmtKind::CLEANUP:
    result.function = LanceInternalCleanupTableFunction();
    result.parameters = {Value(target), Value(parse_data->options_json)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  case LanceMaintenanceStmtKind::OPTIMIZE_INDEX:
    result.function = LanceInternalOptimizeIndexTableFunction();
    result.parameters = {Value(target), Value(parse_data->index_name),
                         Value(parse_data->options_json)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  case LanceMaintenanceStmtKind::SET_AUTO_CLEANUP:
    result.function = LanceInternalSetAutoCleanupTableFunction();
    result.parameters = {
        Value(target), Value::BIGINT(parse_data->auto_cleanup_interval),
        Value(parse_data->auto_cleanup_older_than),
        parse_data->auto_cleanup_has_retain_versions
            ? Value::BIGINT(parse_data->auto_cleanup_retain_versions)
            : Value(),
        Value::BOOLEAN(false)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  case LanceMaintenanceStmtKind::UNSET_AUTO_CLEANUP:
    result.function = LanceInternalSetAutoCleanupTableFunction();
    result.parameters = {Value(target), Value(), Value(), Value(),
                         Value::BOOLEAN(true)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  case LanceMaintenanceStmtKind::SHOW_AUTO_CLEANUP:
    result.function = LanceInternalShowAutoCleanupTableFunction();
    result.parameters = {Value(target)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  default:
    throw InternalException("unknown maintenance statement kind");
  }
  return result;
}

void RegisterLanceMaintenance(ExtensionLoader &loader) {
  loader.RegisterFunction(LanceInternalCompactTableFunction());
  loader.RegisterFunction(LanceInternalCleanupTableFunction());
  loader.RegisterFunction(LanceInternalOptimizeIndexTableFunction());
  loader.RegisterFunction(LanceInternalSetAutoCleanupTableFunction());
  loader.RegisterFunction(LanceInternalShowAutoCleanupTableFunction());
}

void RegisterLanceMaintenanceParser(DBConfig &config) {
  ParserExtension extension;
  extension.parse_function = LanceMaintenanceParse;
  extension.plan_function = LanceMaintenancePlan;
  extension.parser_info = make_shared_ptr<ParserExtensionInfo>();
  ParserExtension::Register(config, std::move(extension));
}

} // namespace duckdb
