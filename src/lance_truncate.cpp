#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_common.hpp"
#include "lance_dataset_cache.hpp"
#include "lance_table_entry.hpp"

#include <cctype>
#include <cstring>

namespace duckdb {

struct LanceTruncateParseData final : public ParserExtensionParseData {
  explicit LanceTruncateParseData(string table_name_sql_p)
      : table_name_sql(std::move(table_name_sql_p)) {}

  string table_name_sql;

  unique_ptr<ParserExtensionParseData> Copy() const override {
    return make_uniq<LanceTruncateParseData>(table_name_sql);
  }

  string ToString() const override {
    return "TRUNCATE TABLE " + table_name_sql;
  }
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

static ParserExtensionParseResult LanceTruncateParse(ParserExtensionInfo *,
                                                     const string &query) {
  auto trimmed = TrimTrailingSemicolons(query);
  if (trimmed.empty()) {
    return ParserExtensionParseResult();
  }

  auto lower = StringUtil::Lower(trimmed);
  if (!HasKeywordPrefix(lower, "truncate")) {
    return ParserExtensionParseResult();
  }

  auto rest = TrimCopy(trimmed.substr(strlen("truncate")));
  if (rest.empty()) {
    return ParserExtensionParseResult("TRUNCATE TABLE requires a table name");
  }
  auto rest_lower = StringUtil::Lower(rest);
  if (HasKeywordPrefix(rest_lower, "table")) {
    rest = TrimCopy(rest.substr(strlen("table")));
  }
  if (rest.empty()) {
    return ParserExtensionParseResult("TRUNCATE TABLE requires a table name");
  }
  return ParserExtensionParseResult(make_uniq<LanceTruncateParseData>(rest));
}

struct LanceTruncateBindData final : public FunctionData {
  LanceTruncateBindData(string catalog_p, string schema_p, string table_p)
      : catalog(std::move(catalog_p)), schema(std::move(schema_p)),
        table(std::move(table_p)) {}

  string catalog;
  string schema;
  string table;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceTruncateBindData>(catalog, schema, table);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceTruncateBindData>();
    return catalog == other.catalog && schema == other.schema &&
           table == other.table;
  }
};

struct LanceTruncateGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<FunctionData>
LanceTruncateBind(ClientContext &, TableFunctionBindInput &input,
                  vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 3) {
    throw BinderException(
        "__lance_truncate_table requires (catalog, schema, table)");
  }
  if (input.inputs[0].IsNull() || input.inputs[1].IsNull() ||
      input.inputs[2].IsNull()) {
    throw BinderException(
        "__lance_truncate_table catalog/schema/table cannot be NULL");
  }
  auto catalog = input.inputs[0].GetValue<string>();
  auto schema = input.inputs[1].GetValue<string>();
  auto table = input.inputs[2].GetValue<string>();
  if (catalog.empty() || schema.empty() || table.empty()) {
    throw BinderException(
        "__lance_truncate_table catalog/schema/table cannot be empty");
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceTruncateBindData>(std::move(catalog), std::move(schema),
                                          std::move(table));
}

static unique_ptr<GlobalTableFunctionState>
LanceTruncateInitGlobal(ClientContext &, TableFunctionInitInput &) {
  return make_uniq<LanceTruncateGlobalState>();
}

static void LanceTruncateFunc(ClientContext &context, TableFunctionInput &data,
                              DataChunk &output) {
  auto &gstate = data.global_state->Cast<LanceTruncateGlobalState>();
  if (gstate.finished) {
    output.SetCardinality(0);
    return;
  }
  gstate.finished = true;

  auto &bind_data = data.bind_data->Cast<LanceTruncateBindData>();
  auto &entry =
      Catalog::GetEntry(context, CatalogType::TABLE_ENTRY, bind_data.catalog,
                        bind_data.schema, bind_data.table);
  auto &table_entry = entry.Cast<TableCatalogEntry>();

  auto *lance_entry = dynamic_cast<LanceTableEntry *>(&table_entry);
  if (!lance_entry) {
    throw NotImplementedException(
        "TRUNCATE TABLE is only supported for tables backed by Lance");
  }

  string open_path;
  vector<string> option_keys;
  vector<string> option_values;
  string display_uri;
  ResolveLanceStorageOptionsForTable(context, *lance_entry, open_path,
                                     option_keys, option_values, display_uri);
  auto row_count = LanceTruncateDatasetWithStorageOptions(
      context, open_path, option_keys, option_values, display_uri);
  LanceInvalidateDatasetCacheForTable(context, *lance_entry);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(row_count));
}

static TableFunction LanceTruncateTableFunction() {
  TableFunction function(
      "__lance_truncate_table",
      {LogicalType::VARCHAR, LogicalType::VARCHAR, LogicalType::VARCHAR},
      LanceTruncateFunc, LanceTruncateBind, LanceTruncateInitGlobal);
  return function;
}

static ParserExtensionPlanResult
LanceTruncatePlan(ParserExtensionInfo *, ClientContext &context,
                  unique_ptr<ParserExtensionParseData> parse_data_p) {
  auto *parse_data = dynamic_cast<LanceTruncateParseData *>(parse_data_p.get());
  if (!parse_data) {
    throw InternalException("LanceTruncatePlan received unexpected parse data");
  }
  auto qname = QualifiedName::Parse(parse_data->table_name_sql);

  auto &entry = Catalog::GetEntry(context, CatalogType::TABLE_ENTRY,
                                  qname.catalog, qname.schema, qname.name);
  auto &table_entry = entry.Cast<TableCatalogEntry>();

  auto *lance_entry = dynamic_cast<LanceTableEntry *>(&table_entry);
  if (!lance_entry) {
    throw NotImplementedException(
        "TRUNCATE TABLE is only supported for tables backed by Lance");
  }

  ParserExtensionPlanResult result;
  result.function = LanceTruncateTableFunction();
  result.parameters = {Value(qname.catalog), Value(qname.schema),
                       Value(qname.name)};

  auto &catalog = table_entry.ParentCatalog();
  result.modified_databases[catalog.GetName()] =
      StatementProperties::ModificationInfo{
          StatementProperties::CatalogIdentity{
              catalog.GetOid(), catalog.GetCatalogVersion(context)},
          DatabaseModificationType::DELETE_DATA};
  result.return_type = StatementReturnType::CHANGED_ROWS;
  return result;
}

void RegisterLanceTruncate(DBConfig &config, ExtensionLoader &loader) {
  CreateTableFunctionInfo func_info(LanceTruncateTableFunction());
  func_info.internal = true;
  loader.RegisterFunction(std::move(func_info));

  ParserExtension extension;
  extension.parse_function = LanceTruncateParse;
  extension.plan_function = LanceTruncatePlan;
  extension.parser_info = make_shared_ptr<ParserExtensionInfo>();
  ParserExtension::Register(config, std::move(extension));
}

} // namespace duckdb
