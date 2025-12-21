#include "duckdb.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/function/replacement_scan.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/function_expression.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"

namespace duckdb {

static unique_ptr<TableRef>
LanceReplacementScan(ClientContext &context, ReplacementScanInput &input,
                     optional_ptr<ReplacementScanData> data) {
  const auto &table_name = input.table_name;
  if (!StringUtil::EndsWith(table_name, ".lance")) {
    return nullptr;
  }

  auto table_function = make_uniq<TableFunctionRef>();
  auto function_expr = make_uniq<FunctionExpression>(
      "lance_scan", vector<unique_ptr<ParsedExpression>>());
  function_expr->children.push_back(
      make_uniq<ConstantExpression>(Value(table_name)));
  table_function->function = std::move(function_expr);
  return std::move(table_function);
}

void RegisterLanceReplacement(DBConfig &config) {
  auto replacement_scan = ReplacementScan(LanceReplacementScan);
  config.replacement_scans.push_back(std::move(replacement_scan));
}

} // namespace duckdb
