#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_common.hpp"
#include "lance_ffi.hpp"
#include "lance_table_entry.hpp"

#include <atomic>
#include <cctype>
#include <cstring>
#include <mutex>

namespace duckdb {

// --- Utilities ---

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

static string NormalizeIndexType(string t) {
  StringUtil::Trim(t);
  t = StringUtil::Upper(t);
  t = StringUtil::Replace(t, "-", "_");
  t = StringUtil::Replace(t, " ", "_");
  return t;
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

static bool TryBuildParamsJsonFromWithClause(const string &with_clause_sql,
                                             bool &out_replace, bool &out_train,
                                             bool &out_retrain,
                                             string &out_params_json,
                                             string &out_error) {
  out_error.clear();
  out_params_json.clear();
  out_replace = false;
  out_train = true;
  out_retrain = false;

  auto rest = TrimCopy(with_clause_sql);
  if (rest.empty()) {
    out_params_json.clear();
    return true;
  }

  // Accept either "(...)" or "WITH (...)"
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

  bool has_raw_params_json = false;
  string raw_params_json;
  vector<pair<string, string>> passthrough;
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
    auto key_lower = StringUtil::Lower(key);
    if (key_lower == "replace") {
      out_replace = StringUtil::Lower(val) == "true" || val == "1";
      continue;
    }
    if (key_lower == "train") {
      out_train = !(StringUtil::Lower(val) == "false" || val == "0");
      continue;
    }
    if (key_lower == "retrain") {
      out_retrain = StringUtil::Lower(val) == "true" || val == "1";
      continue;
    }
    if (key_lower == "params") {
      if (StartsWithQuotedString(val)) {
        string lit;
        idx_t lit_consumed = 0;
        if (!TryParseSqlStringLiteral(val, lit, lit_consumed) ||
            TrimCopy(val.substr(lit_consumed)) != "") {
          out_error = "WITH params must be a single-quoted string literal";
          return false;
        }
        raw_params_json = std::move(lit);
        has_raw_params_json = true;
      } else {
        out_error = "WITH params must be a single-quoted string literal";
        return false;
      }
      continue;
    }
    passthrough.emplace_back(std::move(key), std::move(val));
  }

  if (has_raw_params_json) {
    out_params_json = raw_params_json;
    return true;
  }

  // Build a minimal JSON object from the remaining options.
  string json = "{";
  bool first = true;
  for (auto &kv : passthrough) {
    if (!first) {
      json += ",";
    }
    first = false;
    json += "\"";
    json += EscapeJsonString(kv.first);
    json += "\":";

    auto v = kv.second;
    if (StartsWithQuotedString(v)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(v, lit, lit_consumed) ||
          TrimCopy(v.substr(lit_consumed)) != "") {
        out_error = "WITH string values must be single-quoted string literals";
        return false;
      }
      json += "\"";
      json += EscapeJsonString(lit);
      json += "\"";
    } else {
      auto v_lower = StringUtil::Lower(v);
      if (v_lower == "true" || v_lower == "false") {
        json += v_lower;
      } else if (IsNumberLiteral(v)) {
        json += v;
      } else {
        json += "\"";
        json += EscapeJsonString(v);
        json += "\"";
      }
    }
  }
  json += "}";
  if (json == "{}") {
    out_params_json.clear();
  } else {
    out_params_json = json;
  }
  return true;
}

// --- Index table function: lance_indexes(path) ---

struct LanceIndexListBindData final : public TableFunctionData {
  string file_path;

  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  ~LanceIndexListBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceIndexListGlobalState final : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};

  idx_t MaxThreads() const override { return 1; }
};

struct LanceIndexListLocalState final : public ArrowScanLocalState {
  explicit LanceIndexListLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                                    ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context) {}

  void *stream = nullptr;
  LanceIndexListGlobalState *global_state = nullptr;

  ~LanceIndexListLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static unique_ptr<FunctionData>
LanceIndexListBind(ClientContext &context, TableFunctionBindInput &input,
                   vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 1) {
    throw BinderException("lance_indexes requires exactly one input");
  }
  if (input.inputs[0].IsNull()) {
    throw BinderException("lance_indexes dataset uri cannot be NULL");
  }

  auto result = make_uniq<LanceIndexListBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  if (result->file_path.empty()) {
    throw BinderException("lance_indexes dataset uri cannot be empty");
  }

  result->dataset = LanceOpenDataset(context, result->file_path);
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_index_list_schema(result->dataset);
  if (!schema_handle) {
    throw IOException("Failed to get Lance index list schema: " +
                      result->file_path + LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance index list schema to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_schema(schema_handle);

  auto &config = DBConfig::GetConfig(context);
  ArrowTableFunction::PopulateArrowTableSchema(
      config, result->arrow_table, result->schema_root.arrow_schema);
  result->names = result->arrow_table.GetNames();
  result->types = result->arrow_table.GetTypes();
  names = result->names;
  return_types = result->types;
  return std::move(result);
}

static unique_ptr<GlobalTableFunctionState>
LanceIndexListInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceIndexListBindData>();
  auto state =
      make_uniq_base<GlobalTableFunctionState, LanceIndexListGlobalState>();
  (void)bind_data;
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceIndexListInitLocal(ExecutionContext &context,
                        TableFunctionInitInput &input,
                        GlobalTableFunctionState *global_state) {
  auto &global = global_state->Cast<LanceIndexListGlobalState>();
  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result =
      make_uniq<LanceIndexListLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->global_state = &global;
  return std::move(result);
}

static bool LanceIndexListLoadNextBatch(LanceIndexListLocalState &local_state,
                                        const LanceIndexListBindData &bind_data,
                                        LanceIndexListGlobalState &global) {
  if (!local_state.stream) {
    local_state.stream = lance_create_index_list_stream(bind_data.dataset);
    if (!local_state.stream) {
      throw IOException("Failed to open Lance index list stream" +
                        LanceFormatErrorSuffix());
    }
  }

  void *batch = nullptr;
  auto rc = lance_stream_next(local_state.stream, &batch);
  if (rc == 1) {
    lance_close_stream(local_state.stream);
    local_state.stream = nullptr;
    return false;
  }
  if (rc != 0) {
    throw IOException("Failed to read next Lance RecordBatch" +
                      LanceFormatErrorSuffix());
  }

  auto new_chunk = make_shared_ptr<ArrowArrayWrapper>();
  memset(&new_chunk->arrow_array, 0, sizeof(new_chunk->arrow_array));
  ArrowSchema tmp_schema;
  memset(&tmp_schema, 0, sizeof(tmp_schema));

  if (lance_batch_to_arrow(batch, &new_chunk->arrow_array, &tmp_schema) != 0) {
    lance_free_batch(batch);
    throw IOException(
        "Failed to export Lance RecordBatch to Arrow C Data Interface" +
        LanceFormatErrorSuffix());
  }
  lance_free_batch(batch);

  global.record_batches.fetch_add(1);
  auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
  global.record_batch_rows.fetch_add(rows);

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.chunk_offset = 0;
  local_state.Reset();
  return true;
}

static void LanceIndexListFunc(ClientContext &context, TableFunctionInput &data,
                               DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceIndexListBindData>();
  auto &global_state = data.global_state->Cast<LanceIndexListGlobalState>();
  auto &local_state = data.local_state->Cast<LanceIndexListLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceIndexListLoadNextBatch(local_state, bind_data, global_state)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    auto start = global_state.lines_read.fetch_add(output_size);

    output.SetCardinality(output_size);
    // The Lance index list stream always returns all columns (no projection
    // pushdown), so we must map DuckDB projection column ids to Arrow children.
    ArrowTableFunction::ArrowToDuckDB(
        local_state, bind_data.arrow_table.GetColumns(), output, start,
        /*arrow_scan_is_projected=*/false);
    local_state.chunk_offset += output_size;

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static void RegisterLanceIndexList(ExtensionLoader &loader) {
  TableFunction f("lance_indexes", {LogicalType::VARCHAR}, LanceIndexListFunc,
                  LanceIndexListBind, LanceIndexListInitGlobal,
                  LanceIndexListInitLocal);
  f.projection_pushdown = true;
  loader.RegisterFunction(f);
}

// --- Internal DDL table functions ---

struct LanceIndexDdlBindData final : public FunctionData {
  explicit LanceIndexDdlBindData(string dataset_uri_p, string index_name_p,
                                 vector<string> columns_p, string index_type_p,
                                 string params_json_p, bool replace_p,
                                 bool train_p, bool retrain_p)
      : dataset_uri(std::move(dataset_uri_p)),
        index_name(std::move(index_name_p)), columns(std::move(columns_p)),
        index_type(std::move(index_type_p)),
        params_json(std::move(params_json_p)), replace(replace_p),
        train(train_p), retrain(retrain_p) {}

  string dataset_uri;
  string index_name;
  vector<string> columns;
  string index_type;
  string params_json;
  bool replace = false;
  bool train = true;
  bool retrain = false;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceIndexDdlBindData>(dataset_uri, index_name, columns,
                                            index_type, params_json, replace,
                                            train, retrain);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceIndexDdlBindData>();
    return dataset_uri == other.dataset_uri && index_name == other.index_name &&
           columns == other.columns && index_type == other.index_type &&
           params_json == other.params_json && replace == other.replace &&
           train == other.train && retrain == other.retrain;
  }
};

struct LanceIndexDdlGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<FunctionData>
LanceCreateIndexBind(ClientContext &, TableFunctionBindInput &input,
                     vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 7) {
    throw BinderException("__lance_create_index requires 7 inputs");
  }
  for (idx_t i = 0; i < input.inputs.size(); i++) {
    if (input.inputs[i].IsNull()) {
      throw BinderException("__lance_create_index inputs cannot be NULL");
    }
  }

  auto dataset_uri = input.inputs[0].GetValue<string>();
  auto index_name = input.inputs[1].GetValue<string>();
  auto column = input.inputs[2].GetValue<string>();
  auto index_type = input.inputs[3].GetValue<string>();
  auto params_json = input.inputs[4].GetValue<string>();
  auto replace =
      input.inputs[5].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  auto train =
      input.inputs[6].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();

  if (dataset_uri.empty()) {
    throw BinderException("__lance_create_index dataset uri cannot be empty");
  }
  if (column.empty()) {
    throw BinderException("__lance_create_index column cannot be empty");
  }
  if (index_type.empty()) {
    throw BinderException("__lance_create_index index type cannot be empty");
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  vector<string> columns;
  columns.push_back(std::move(column));
  return make_uniq<LanceIndexDdlBindData>(
      std::move(dataset_uri), std::move(index_name), std::move(columns),
      NormalizeIndexType(std::move(index_type)), std::move(params_json),
      replace, train, false);
}

static unique_ptr<FunctionData>
LanceDropIndexBind(ClientContext &, TableFunctionBindInput &input,
                   vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 2) {
    throw BinderException("__lance_drop_index requires 2 inputs");
  }
  for (idx_t i = 0; i < input.inputs.size(); i++) {
    if (input.inputs[i].IsNull()) {
      throw BinderException("__lance_drop_index inputs cannot be NULL");
    }
  }
  auto dataset_uri = input.inputs[0].GetValue<string>();
  auto index_name = input.inputs[1].GetValue<string>();
  if (dataset_uri.empty()) {
    throw BinderException("__lance_drop_index dataset uri cannot be empty");
  }
  if (index_name.empty()) {
    throw BinderException("__lance_drop_index index name cannot be empty");
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceIndexDdlBindData>(
      std::move(dataset_uri), std::move(index_name), vector<string>{}, "", "{}",
      false, true, false);
}

static unique_ptr<FunctionData>
LanceReindexBind(ClientContext &, TableFunctionBindInput &input,
                 vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 3) {
    throw BinderException("__lance_reindex requires 3 inputs");
  }
  for (idx_t i = 0; i < input.inputs.size(); i++) {
    if (input.inputs[i].IsNull()) {
      throw BinderException("__lance_reindex inputs cannot be NULL");
    }
  }
  auto dataset_uri = input.inputs[0].GetValue<string>();
  auto index_name = input.inputs[1].GetValue<string>();
  auto retrain =
      input.inputs[2].DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  if (dataset_uri.empty()) {
    throw BinderException("__lance_reindex dataset uri cannot be empty");
  }
  if (index_name.empty()) {
    throw BinderException("__lance_reindex index name cannot be empty");
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceIndexDdlBindData>(
      std::move(dataset_uri), std::move(index_name), vector<string>{}, "", "{}",
      false, true, retrain);
}

static unique_ptr<GlobalTableFunctionState>
LanceIndexDdlInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceIndexDdlGlobalState>();
}

static void LanceCreateIndexFunc(ClientContext &context,
                                 TableFunctionInput &data, DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceIndexDdlGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceIndexDdlBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }

  vector<const char *> col_ptrs;
  col_ptrs.reserve(bind_data.columns.size());
  for (auto &c : bind_data.columns) {
    col_ptrs.push_back(c.c_str());
  }
  const char *name_ptr =
      bind_data.index_name.empty() ? nullptr : bind_data.index_name.c_str();
  const char *params_ptr =
      bind_data.params_json.empty() ? nullptr : bind_data.params_json.c_str();

  auto rc = lance_dataset_create_index(
      dataset, name_ptr, col_ptrs.data(), col_ptrs.size(),
      bind_data.index_type.c_str(), params_ptr, bind_data.replace ? 1 : 0,
      bind_data.train ? 1 : 0);
  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to create Lance index" +
                      LanceFormatErrorSuffix());
  }

  output.SetCardinality(0);
}

static void LanceDropIndexFunc(ClientContext &context, TableFunctionInput &data,
                               DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceIndexDdlGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceIndexDdlBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }

  auto rc = lance_dataset_drop_index(dataset, bind_data.index_name.c_str());
  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to drop Lance index" + LanceFormatErrorSuffix());
  }

  output.SetCardinality(0);
}

static void LanceReindexFunc(ClientContext &context, TableFunctionInput &data,
                             DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceIndexDdlGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceIndexDdlBindData>();
  void *dataset = LanceOpenDataset(context, bind_data.dataset_uri);
  if (!dataset) {
    throw IOException("Failed to open Lance dataset: " + bind_data.dataset_uri +
                      LanceFormatErrorSuffix());
  }

  auto rc = lance_dataset_optimize_index(dataset, bind_data.index_name.c_str(),
                                         bind_data.retrain ? 1 : 0);
  lance_close_dataset(dataset);
  if (rc != 0) {
    throw IOException("Failed to optimize Lance index" +
                      LanceFormatErrorSuffix());
  }

  output.SetCardinality(0);
}

static TableFunction LanceCreateIndexTableFunction() {
  TableFunction function(
      "__lance_create_index",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN,
       LogicalType::BOOLEAN},
      LanceCreateIndexFunc, LanceCreateIndexBind, LanceIndexDdlInitGlobal);
  return function;
}

static TableFunction LanceDropIndexTableFunction() {
  TableFunction function(
      "__lance_drop_index", {LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceDropIndexFunc, LanceDropIndexBind, LanceIndexDdlInitGlobal);
  return function;
}

static TableFunction LanceReindexTableFunction() {
  TableFunction function(
      "__lance_reindex",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::BOOLEAN},
      LanceReindexFunc, LanceReindexBind, LanceIndexDdlInitGlobal);
  return function;
}

// --- Parser extension ---

enum class LanceIndexStmtKind : uint8_t {
  Create = 0,
  Drop = 1,
  Show = 2,
  Reindex = 3
};

struct LanceIndexParseData final : public ParserExtensionParseData {
  explicit LanceIndexParseData(LanceIndexStmtKind kind_p) : kind(kind_p) {}

  LanceIndexStmtKind kind;

  string index_name;
  string target_sql;
  bool target_is_path = false;
  string dataset_uri;

  vector<string> columns;
  string index_type;
  string params_json;
  bool replace = false;
  bool train = true;
  bool retrain = false;

  unique_ptr<ParserExtensionParseData> Copy() const override {
    auto out = make_uniq<LanceIndexParseData>(kind);
    out->index_name = index_name;
    out->target_sql = target_sql;
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->columns = columns;
    out->index_type = index_type;
    out->params_json = params_json;
    out->replace = replace;
    out->train = train;
    out->retrain = retrain;
    return std::move(out);
  }

  string ToString() const override {
    switch (kind) {
    case LanceIndexStmtKind::Create:
      return "CREATE INDEX " + index_name + " ON " + target_sql + " (" +
             StringUtil::Join(columns, ", ") + ") USING " + index_type;
    case LanceIndexStmtKind::Drop:
      return "DROP INDEX " + index_name + " ON " + target_sql;
    case LanceIndexStmtKind::Show:
      return "SHOW INDEXES ON " + target_sql;
    case LanceIndexStmtKind::Reindex:
      return "REINDEX " + index_name + " ON " + target_sql;
    default:
      return "LANCE INDEX STMT";
    }
  }
};

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

static ParserExtensionParseResult LanceIndexParse(ParserExtensionInfo *,
                                                  const string &query) {
  auto trimmed = TrimTrailingSemicolons(query);
  if (trimmed.empty()) {
    return ParserExtensionParseResult();
  }
  auto lower = StringUtil::Lower(trimmed);

  // CREATE INDEX ... USING <type> ...
  if (HasKeywordPrefix(lower, "create")) {
    auto rest = TrimCopy(trimmed.substr(strlen("create")));
    auto rest_lower = StringUtil::Lower(rest);
    if (!HasKeywordPrefix(rest_lower, "index")) {
      return ParserExtensionParseResult();
    }
    rest = TrimCopy(rest.substr(strlen("index")));

    string index_name;
    idx_t consumed = 0;
    if (!TryParseIdentifier(rest, index_name, consumed)) {
      return ParserExtensionParseResult("CREATE INDEX requires an index name");
    }
    rest = TrimCopy(rest.substr(consumed));
    if (!ConsumeKeyword(rest, "on")) {
      return ParserExtensionParseResult("CREATE INDEX requires ON <dataset>");
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    if (StartsWithQuotedString(rest)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(rest, lit, lit_consumed)) {
        return ParserExtensionParseResult(
            "invalid dataset path string literal");
      }
      dataset_uri = lit;
      target_sql = rest.substr(0, lit_consumed);
      rest = TrimCopy(rest.substr(lit_consumed));
      target_is_path = true;
    } else {
      string ident;
      if (!TryParseIdentifier(rest, ident, consumed)) {
        return ParserExtensionParseResult("CREATE INDEX requires ON <dataset>");
      }
      target_sql = ident;
      rest = TrimCopy(rest.substr(consumed));
    }

    // Avoid intercepting DuckDB's CREATE INDEX on regular tables (including
    // cases that use a USING clause). Lance index DDL is only supported when
    // the target is a dataset path string literal.
    if (!target_is_path) {
      return ParserExtensionParseResult();
    }

    vector<string> columns;
    idx_t paren_consumed = 0;
    if (!TryParseParenList(rest, columns, paren_consumed) || columns.empty()) {
      return ParserExtensionParseResult(
          "CREATE INDEX requires a non-empty column list");
    }
    rest = TrimCopy(rest.substr(paren_consumed));

    // If there is no USING clause then this is likely a DuckDB table index.
    if (!ConsumeKeyword(rest, "using")) {
      return ParserExtensionParseResult();
    }
    string index_type;
    if (!TryParseIdentifier(rest, index_type, consumed)) {
      return ParserExtensionParseResult("CREATE INDEX requires USING <type>");
    }
    rest = TrimCopy(rest.substr(consumed));
    index_type = NormalizeIndexType(index_type);

    bool replace = false;
    bool train = true;
    bool retrain = false;
    string params_json;
    if (!rest.empty()) {
      string err;
      if (!TryBuildParamsJsonFromWithClause(rest, replace, train, retrain,
                                            params_json, err)) {
        return ParserExtensionParseResult(err);
      }
      if (retrain) {
        return ParserExtensionParseResult(
            "CREATE INDEX does not accept retrain; use REINDEX instead");
      }
    }

    auto out = make_uniq<LanceIndexParseData>(LanceIndexStmtKind::Create);
    out->index_name = index_name;
    out->target_sql = target_sql;
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->columns = columns;
    out->index_type = index_type;
    out->params_json = params_json;
    out->replace = replace;
    out->train = train;
    return ParserExtensionParseResult(std::move(out));
  }

  // DROP INDEX <name> ON <dataset>
  if (HasKeywordPrefix(lower, "drop")) {
    auto rest = TrimCopy(trimmed.substr(strlen("drop")));
    auto rest_lower = StringUtil::Lower(rest);
    if (!HasKeywordPrefix(rest_lower, "index")) {
      return ParserExtensionParseResult();
    }
    rest = TrimCopy(rest.substr(strlen("index")));
    string index_name;
    idx_t consumed = 0;
    if (!TryParseIdentifier(rest, index_name, consumed)) {
      return ParserExtensionParseResult("DROP INDEX requires an index name");
    }
    rest = TrimCopy(rest.substr(consumed));
    if (!ConsumeKeyword(rest, "on")) {
      return ParserExtensionParseResult();
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    if (StartsWithQuotedString(rest)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(rest, lit, lit_consumed)) {
        return ParserExtensionParseResult(
            "invalid dataset path string literal");
      }
      dataset_uri = lit;
      target_sql = rest.substr(0, lit_consumed);
      rest = TrimCopy(rest.substr(lit_consumed));
      target_is_path = true;
    } else {
      string ident;
      if (!TryParseIdentifier(rest, ident, consumed)) {
        return ParserExtensionParseResult();
      }
      target_sql = ident;
      rest = TrimCopy(rest.substr(consumed));
    }

    if (!rest.empty()) {
      return ParserExtensionParseResult();
    }

    auto out = make_uniq<LanceIndexParseData>(LanceIndexStmtKind::Drop);
    out->index_name = index_name;
    out->target_sql = target_sql;
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    return ParserExtensionParseResult(std::move(out));
  }

  // SHOW INDEXES ON <dataset>
  if (HasKeywordPrefix(lower, "show")) {
    auto rest = TrimCopy(trimmed.substr(strlen("show")));
    auto rest_lower = StringUtil::Lower(rest);
    if (!HasKeywordPrefix(rest_lower, "indexes")) {
      return ParserExtensionParseResult();
    }
    rest = TrimCopy(rest.substr(strlen("indexes")));
    auto rest_lower2 = StringUtil::Lower(rest);
    if (HasKeywordPrefix(rest_lower2, "on")) {
      rest = TrimCopy(rest.substr(strlen("on")));
    } else if (HasKeywordPrefix(rest_lower2, "from")) {
      rest = TrimCopy(rest.substr(strlen("from")));
    } else {
      return ParserExtensionParseResult();
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    idx_t consumed = 0;
    if (StartsWithQuotedString(rest)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(rest, lit, lit_consumed)) {
        return ParserExtensionParseResult(
            "invalid dataset path string literal");
      }
      dataset_uri = lit;
      target_sql = rest.substr(0, lit_consumed);
      rest = TrimCopy(rest.substr(lit_consumed));
      target_is_path = true;
    } else {
      string ident;
      if (!TryParseIdentifier(rest, ident, consumed)) {
        return ParserExtensionParseResult();
      }
      target_sql = ident;
      rest = TrimCopy(rest.substr(consumed));
    }

    if (!rest.empty()) {
      return ParserExtensionParseResult();
    }

    auto out = make_uniq<LanceIndexParseData>(LanceIndexStmtKind::Show);
    out->target_sql = target_sql;
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    return ParserExtensionParseResult(std::move(out));
  }

  // REINDEX <name> ON <dataset> [RETRAIN]
  if (HasKeywordPrefix(lower, "reindex")) {
    auto rest = TrimCopy(trimmed.substr(strlen("reindex")));
    string index_name;
    idx_t consumed = 0;
    if (!TryParseIdentifier(rest, index_name, consumed)) {
      return ParserExtensionParseResult();
    }
    rest = TrimCopy(rest.substr(consumed));
    if (!ConsumeKeyword(rest, "on")) {
      return ParserExtensionParseResult();
    }

    bool target_is_path = false;
    string dataset_uri;
    string target_sql;
    if (StartsWithQuotedString(rest)) {
      string lit;
      idx_t lit_consumed = 0;
      if (!TryParseSqlStringLiteral(rest, lit, lit_consumed)) {
        return ParserExtensionParseResult(
            "invalid dataset path string literal");
      }
      dataset_uri = lit;
      target_sql = rest.substr(0, lit_consumed);
      rest = TrimCopy(rest.substr(lit_consumed));
      target_is_path = true;
    } else {
      string ident;
      if (!TryParseIdentifier(rest, ident, consumed)) {
        return ParserExtensionParseResult();
      }
      target_sql = ident;
      rest = TrimCopy(rest.substr(consumed));
    }

    bool replace = false;
    bool train = true;
    bool retrain = false;
    string params_json = "{}";
    string err;
    if (!rest.empty()) {
      auto rest_lower = StringUtil::Lower(rest);
      if (HasKeywordPrefix(rest_lower, "retrain")) {
        retrain = true;
        rest = TrimCopy(rest.substr(strlen("retrain")));
      } else if (HasKeywordPrefix(rest_lower, "with")) {
        if (!TryBuildParamsJsonFromWithClause(rest, replace, train, retrain,
                                              params_json, err)) {
          return ParserExtensionParseResult(err);
        }
        rest.clear();
      }
    }
    if (!rest.empty()) {
      return ParserExtensionParseResult();
    }

    auto out = make_uniq<LanceIndexParseData>(LanceIndexStmtKind::Reindex);
    out->index_name = index_name;
    out->target_sql = target_sql;
    out->target_is_path = target_is_path;
    out->dataset_uri = dataset_uri;
    out->retrain = retrain;
    return ParserExtensionParseResult(std::move(out));
  }

  return ParserExtensionParseResult();
}

static string
ResolveDatasetUriFromTarget(ClientContext &context,
                            const LanceIndexParseData &parse_data) {
  if (parse_data.target_is_path) {
    if (parse_data.dataset_uri.empty()) {
      throw BinderException("dataset uri cannot be empty");
    }
    return parse_data.dataset_uri;
  }

  auto qname = QualifiedName::Parse(parse_data.target_sql);
  auto &entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY,
                                  qname.catalog, qname.schema, qname.name);
  auto &table_entry = entry.Cast<TableCatalogEntry>();
  auto *lance_entry = dynamic_cast<LanceTableEntry *>(&table_entry);
  if (!lance_entry) {
    throw NotImplementedException(
        "Lance CREATE/DROP/SHOW/REINDEX INDEX only supports dataset paths or "
        "tables in ATTACH TYPE LANCE directory namespaces");
  }
  return lance_entry->DatasetUri();
}

static ParserExtensionPlanResult
LanceIndexPlan(ParserExtensionInfo *, ClientContext &context,
               unique_ptr<ParserExtensionParseData> parse_data_p) {
  auto *parse_data = dynamic_cast<LanceIndexParseData *>(parse_data_p.get());
  if (!parse_data) {
    throw InternalException("LanceIndexPlan received unexpected parse data");
  }

  auto dataset_uri = ResolveDatasetUriFromTarget(context, *parse_data);

  ParserExtensionPlanResult result;
  switch (parse_data->kind) {
  case LanceIndexStmtKind::Create: {
    if (parse_data->columns.size() != 1) {
      throw NotImplementedException(
          "Lance CREATE INDEX currently supports a single column");
    }
    result.function = LanceCreateIndexTableFunction();
    result.parameters = {
        Value(dataset_uri),
        Value(parse_data->index_name),
        Value(parse_data->columns[0]),
        Value(parse_data->index_type),
        Value(parse_data->params_json),
        Value::BOOLEAN(parse_data->replace),
        Value::BOOLEAN(parse_data->train),
    };
    result.return_type = StatementReturnType::NOTHING;
    break;
  }
  case LanceIndexStmtKind::Drop: {
    result.function = LanceDropIndexTableFunction();
    result.parameters = {Value(dataset_uri), Value(parse_data->index_name)};
    result.return_type = StatementReturnType::NOTHING;
    break;
  }
  case LanceIndexStmtKind::Show: {
    TableFunction show_fun("lance_indexes", {LogicalType::VARCHAR},
                           LanceIndexListFunc, LanceIndexListBind,
                           LanceIndexListInitGlobal, LanceIndexListInitLocal);
    show_fun.projection_pushdown = true;
    result.function = show_fun;
    result.parameters = {Value(dataset_uri)};
    result.return_type = StatementReturnType::QUERY_RESULT;
    break;
  }
  case LanceIndexStmtKind::Reindex: {
    result.function = LanceReindexTableFunction();
    result.parameters = {Value(dataset_uri), Value(parse_data->index_name),
                         Value::BOOLEAN(parse_data->retrain)};
    result.return_type = StatementReturnType::NOTHING;
    break;
  }
  default:
    throw InternalException("unknown Lance index statement kind");
  }

  return result;
}

void RegisterLanceIndex(DBConfig &config, ExtensionLoader &loader) {
  RegisterLanceIndexList(loader);

  ParserExtension extension;
  extension.parse_function = LanceIndexParse;
  extension.plan_function = LanceIndexPlan;
  extension.parser_info = make_shared_ptr<ParserExtensionInfo>();
  config.parser_extensions.push_back(std::move(extension));
}

} // namespace duckdb
