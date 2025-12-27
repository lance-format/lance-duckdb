#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/parser_extension.hpp"
#include "duckdb/parser/qualified_name.hpp"

#include "lance_common.hpp"
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
  explicit LanceTruncateBindData(string dataset_uri_p)
      : dataset_uri(std::move(dataset_uri_p)) {}

  string dataset_uri;

  unique_ptr<FunctionData> Copy() const override {
    return make_uniq<LanceTruncateBindData>(dataset_uri);
  }

  bool Equals(const FunctionData &other_p) const override {
    auto &other = other_p.Cast<LanceTruncateBindData>();
    return dataset_uri == other.dataset_uri;
  }
};

struct LanceTruncateGlobalState final : public GlobalTableFunctionState {
  bool finished = false;
};

static unique_ptr<FunctionData>
LanceTruncateBind(ClientContext &, TableFunctionBindInput &input,
                  vector<LogicalType> &return_types, vector<string> &names) {
  if (input.inputs.size() != 1) {
    throw BinderException("__lance_truncate_table requires exactly one input");
  }
  if (input.inputs[0].IsNull()) {
    throw BinderException("__lance_truncate_table dataset uri cannot be NULL");
  }
  auto dataset_uri = input.inputs[0].GetValue<string>();
  if (dataset_uri.empty()) {
    throw BinderException("__lance_truncate_table dataset uri cannot be empty");
  }

  return_types = {LogicalType::BIGINT};
  names = {"Count"};
  return make_uniq<LanceTruncateBindData>(std::move(dataset_uri));
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
  auto row_count = LanceTruncateDataset(context, bind_data.dataset_uri);

  output.SetCardinality(1);
  output.SetValue(0, 0, Value::BIGINT(row_count));
}

static TableFunction LanceTruncateTableFunction() {
  TableFunction function("__lance_truncate_table", {LogicalType::VARCHAR},
                         LanceTruncateFunc, LanceTruncateBind,
                         LanceTruncateInitGlobal);
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
        "TRUNCATE TABLE is only supported for tables in ATTACH TYPE LANCE "
        "directory namespaces");
  }

  ParserExtensionPlanResult result;
  result.function = LanceTruncateTableFunction();
  result.parameters = {Value(lance_entry->DatasetUri())};

  auto &catalog = table_entry.ParentCatalog();
  result.modified_databases[catalog.GetName()] =
      StatementProperties::CatalogIdentity{catalog.GetOid(),
                                           catalog.GetCatalogVersion(context)};
  result.return_type = StatementReturnType::CHANGED_ROWS;
  return result;
}

void RegisterLanceTruncate(DBConfig &config) {
  ParserExtension extension;
  extension.parse_function = LanceTruncateParse;
  extension.plan_function = LanceTruncatePlan;
  extension.parser_info = make_shared_ptr<ParserExtensionInfo>();
  config.parser_extensions.push_back(std::move(extension));
}

} // namespace duckdb
