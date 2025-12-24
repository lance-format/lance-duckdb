#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

extern "C" {
void *lance_open_dataset(const char *path);
void lance_close_dataset(void *dataset);

void *lance_get_knn_schema(void *dataset, const char *vector_column,
                           const float *query_values, size_t query_len,
                           uint64_t k, uint8_t prefilter, uint8_t use_index);
void lance_free_schema(void *schema);
int32_t lance_schema_to_arrow(void *schema, ArrowSchema *out_schema);

void *lance_create_knn_stream(void *dataset, const char *vector_column,
                              const float *query_values, size_t query_len,
                              uint64_t k, const char *filter_sql,
                              uint8_t prefilter, uint8_t use_index);
int32_t lance_stream_next(void *stream, void **out_batch);
void lance_close_stream(void *stream);

int32_t lance_last_error_code();
const char *lance_last_error_message();
void lance_free_string(const char *s);

const char *lance_explain_knn_scan(void *dataset, const char *vector_column,
                                   const float *query_values, size_t query_len,
                                   uint64_t k, const char *filter_sql,
                                   uint8_t prefilter, uint8_t use_index,
                                   uint8_t verbose);

void lance_free_batch(void *batch);
int32_t lance_batch_to_arrow(void *batch, ArrowArray *out_array,
                             ArrowSchema *out_schema);
}

namespace duckdb {

static string LanceConsumeLastError() {
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

static string LanceFormatErrorSuffix() {
  auto err = LanceConsumeLastError();
  if (err.empty()) {
    return "";
  }
  return " (Lance error: " + err + ")";
}

static bool TryLanceExplainKnn(void *dataset, const string &vector_column,
                               const vector<float> &query, uint64_t k,
                               const string &filter_sql, bool prefilter,
                               bool use_index, bool verbose, string &out_plan,
                               string &out_error) {
  out_plan.clear();
  out_error.clear();

  if (!dataset) {
    out_error = "dataset is null";
    return false;
  }
  if (query.empty()) {
    out_error = "query is empty";
    return false;
  }

  auto filter_ptr = filter_sql.empty() ? nullptr : filter_sql.c_str();
  auto *plan_ptr = lance_explain_knn_scan(
      dataset, vector_column.c_str(), query.data(), query.size(), k, filter_ptr,
      prefilter ? 1 : 0, use_index ? 1 : 0, verbose ? 1 : 0);
  if (!plan_ptr) {
    out_error = LanceConsumeLastError();
    if (out_error.empty()) {
      out_error = "unknown error";
    }
    return false;
  }

  out_plan = plan_ptr;
  lance_free_string(plan_ptr);
  return true;
}

static vector<float> ParseQueryVector(const Value &value) {
  if (value.IsNull()) {
    throw InvalidInputException("lance_knn requires a non-null query vector");
  }
  if (value.type().id() != LogicalTypeId::LIST) {
    throw InvalidInputException("lance_knn requires query vector to be a LIST");
  }
  auto children = ListValue::GetChildren(value);
  if (children.empty()) {
    throw InvalidInputException("lance_knn requires a non-empty query vector");
  }

  auto cast_f32 = [](double v) {
    if (!std::isfinite(v)) {
      throw InvalidInputException(
          "lance_knn query vector contains non-finite value");
    }
    auto max_v = static_cast<double>(std::numeric_limits<float>::max());
    if (v > max_v || v < -max_v) {
      throw InvalidInputException(
          "lance_knn query vector value is out of float32 range");
    }
    return static_cast<float>(v);
  };

  vector<float> out;
  out.reserve(children.size());
  for (auto &child : children) {
    if (child.IsNull()) {
      throw InvalidInputException("lance_knn query vector contains NULL");
    }
    switch (child.type().id()) {
    case LogicalTypeId::FLOAT:
      out.push_back(cast_f32(child.GetValue<float>()));
      break;
    case LogicalTypeId::DOUBLE:
      out.push_back(cast_f32(child.GetValue<double>()));
      break;
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<int64_t>())));
      break;
    case LogicalTypeId::UTINYINT:
    case LogicalTypeId::USMALLINT:
    case LogicalTypeId::UINTEGER:
    case LogicalTypeId::UBIGINT:
      out.push_back(cast_f32(static_cast<double>(child.GetValue<uint64_t>())));
      break;
    default:
      try {
        auto dbl = child.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
        out.push_back(cast_f32(dbl));
      } catch (Exception &) {
        throw InvalidInputException(
            "lance_knn query vector elements must be numeric");
      }
    }
  }
  return out;
}

static string EscapeLanceColumnName(const string &name) {
  string result;
  idx_t start = 0;
  for (idx_t i = 0; i <= name.size(); i++) {
    if (i == name.size() || name[i] == '.') {
      auto segment = name.substr(start, i - start);
      if (!result.empty()) {
        result += ".";
      }
      result += "`";
      result += StringUtil::Replace(segment, "`", "``");
      result += "`";
      start = i + 1;
    }
  }
  return result;
}

static bool LanceFilterPushdownSupported(const TableFilter &filter) {
  switch (filter.filter_type) {
  case TableFilterType::CONSTANT_COMPARISON:
  case TableFilterType::IS_NULL:
  case TableFilterType::IS_NOT_NULL:
  case TableFilterType::IN_FILTER:
    return true;
  case TableFilterType::CONJUNCTION_AND: {
    auto &f = filter.Cast<ConjunctionAndFilter>();
    for (auto &child : f.child_filters) {
      if (!LanceFilterPushdownSupported(*child)) {
        return false;
      }
    }
    return true;
  }
  case TableFilterType::CONJUNCTION_OR: {
    auto &f = filter.Cast<ConjunctionOrFilter>();
    for (auto &child : f.child_filters) {
      if (!LanceFilterPushdownSupported(*child)) {
        return false;
      }
    }
    return true;
  }
  default:
    return false;
  }
}

static bool LanceSupportsPushdownLogicalType(const LogicalType &type) {
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
  case LogicalTypeId::FLOAT:
  case LogicalTypeId::DOUBLE:
  case LogicalTypeId::VARCHAR:
    return true;
  default:
    return false;
  }
}

static bool TryBuildLanceFilterSQL(const vector<string> &names,
                                   const vector<LogicalType> &types,
                                   const TableFunctionInitInput &input,
                                   string &out_sql) {
  out_sql.clear();
  if (!input.filters) {
    return true;
  }
  if (input.filters->filters.empty()) {
    return true;
  }

  vector<string> predicates;
  predicates.reserve(input.filters->filters.size());

  for (auto &it : input.filters->filters) {
    auto scan_col_idx = it.first;
    auto &filter = *it.second;
    if (scan_col_idx >= input.column_ids.size()) {
      return false;
    }
    if (!LanceFilterPushdownSupported(filter)) {
      return false;
    }
    auto col_id = input.column_ids[scan_col_idx];
    if (col_id >= names.size() || col_id >= types.size()) {
      return false;
    }
    if (!LanceSupportsPushdownLogicalType(types[col_id])) {
      return false;
    }
    auto col_name = EscapeLanceColumnName(names[col_id]);
    predicates.push_back(filter.ToString(col_name));
  }

  if (predicates.empty()) {
    return true;
  }
  out_sql = StringUtil::Join(predicates, " AND ");
  return true;
}

static bool TrySerializeLanceLiteral(const Value &value, string &out_sql) {
  if (value.IsNull()) {
    out_sql = "NULL";
    return true;
  }
  switch (value.type().id()) {
  case LogicalTypeId::BOOLEAN:
    out_sql = value.GetValue<bool>() ? "TRUE" : "FALSE";
    return true;
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
    out_sql = to_string(value.GetValue<int64_t>());
    return true;
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
    out_sql = to_string(value.GetValue<uint64_t>());
    return true;
  case LogicalTypeId::FLOAT: {
    auto v = value.GetValue<float>();
    if (!std::isfinite(v)) {
      return false;
    }
    out_sql = to_string(v);
    return true;
  }
  case LogicalTypeId::DOUBLE: {
    auto v = value.GetValue<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    out_sql = to_string(v);
    return true;
  }
  case LogicalTypeId::VARCHAR: {
    auto v = value.GetValue<string>();
    out_sql = "'";
    out_sql += StringUtil::Replace(v, "'", "''");
    out_sql += "'";
    return true;
  }
  default:
    return false;
  }
}

static bool TrySerializeLanceColumnRef(const LogicalGet &get,
                                       const vector<string> &names,
                                       const vector<LogicalType> &types,
                                       const Expression &expr,
                                       string &out_sql) {
  if (expr.expression_class != ExpressionClass::BOUND_COLUMN_REF) {
    return false;
  }
  auto &colref = expr.Cast<BoundColumnRefExpression>();
  if (colref.depth != 0) {
    return false;
  }
  if (colref.binding.table_index != get.table_index) {
    return false;
  }
  auto &column_ids = get.GetColumnIds();
  if (colref.binding.column_index >= column_ids.size()) {
    return false;
  }
  auto &col_index = column_ids[colref.binding.column_index];
  if (col_index.IsVirtualColumn()) {
    return false;
  }
  if (!col_index.GetChildIndexes().empty()) {
    return false;
  }
  auto col_id = col_index.GetPrimaryIndex();
  if (col_id >= names.size() || col_id >= types.size()) {
    return false;
  }
  if (!LanceSupportsPushdownLogicalType(types[col_id])) {
    return false;
  }
  out_sql = EscapeLanceColumnName(names[col_id]);
  return true;
}

static bool TrySerializeLanceExpr(const LogicalGet &get,
                                  const vector<string> &names,
                                  const vector<LogicalType> &types,
                                  const Expression &expr, string &out_sql) {
  switch (expr.expression_class) {
  case ExpressionClass::BOUND_COLUMN_REF:
    return TrySerializeLanceColumnRef(get, names, types, expr, out_sql);
  case ExpressionClass::BOUND_CONSTANT: {
    auto &c = expr.Cast<BoundConstantExpression>();
    return TrySerializeLanceLiteral(c.value, out_sql);
  }
  case ExpressionClass::BOUND_CAST: {
    auto &cast = expr.Cast<BoundCastExpression>();
    if (cast.try_cast) {
      return false;
    }
    if (cast.child->expression_class != ExpressionClass::BOUND_CONSTANT) {
      return false;
    }
    auto &c = cast.child->Cast<BoundConstantExpression>();
    return TrySerializeLanceLiteral(c.value, out_sql);
  }
  case ExpressionClass::BOUND_COMPARISON: {
    auto &cmp = expr.Cast<BoundComparisonExpression>();
    if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
        cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      return false;
    }
    string lhs, rhs;
    if (!TrySerializeLanceExpr(get, names, types, *cmp.left, lhs) ||
        !TrySerializeLanceExpr(get, names, types, *cmp.right, rhs)) {
      return false;
    }
    out_sql =
        "(" + lhs + " " + ExpressionTypeToOperator(cmp.type) + " " + rhs + ")";
    return true;
  }
  case ExpressionClass::BOUND_CONJUNCTION: {
    auto &conj = expr.Cast<BoundConjunctionExpression>();
    const char *op = nullptr;
    if (conj.type == ExpressionType::CONJUNCTION_AND) {
      op = " AND ";
    } else if (conj.type == ExpressionType::CONJUNCTION_OR) {
      op = " OR ";
    } else {
      return false;
    }
    vector<string> parts;
    parts.reserve(conj.children.size());
    for (auto &child : conj.children) {
      string child_sql;
      if (!TrySerializeLanceExpr(get, names, types, *child, child_sql)) {
        return false;
      }
      parts.push_back(std::move(child_sql));
    }
    out_sql = "(" + StringUtil::Join(parts, op) + ")";
    return true;
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.type == ExpressionType::OPERATOR_NOT) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_sql;
      if (!TrySerializeLanceExpr(get, names, types, *op.children[0],
                                 child_sql)) {
        return false;
      }
      out_sql = "(NOT " + child_sql + ")";
      return true;
    }
    if (op.type == ExpressionType::OPERATOR_IS_NULL ||
        op.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_sql;
      if (!TrySerializeLanceExpr(get, names, types, *op.children[0],
                                 child_sql)) {
        return false;
      }
      out_sql = "(" + child_sql +
                (op.type == ExpressionType::OPERATOR_IS_NULL ? " IS NULL)"
                                                             : " IS NOT NULL)");
      return true;
    }
    if (op.type == ExpressionType::COMPARE_IN ||
        op.type == ExpressionType::COMPARE_NOT_IN) {
      if (op.children.size() < 2) {
        return false;
      }
      string lhs_sql;
      if (!TrySerializeLanceExpr(get, names, types, *op.children[0], lhs_sql)) {
        return false;
      }
      vector<string> values;
      values.reserve(op.children.size() - 1);
      for (idx_t i = 1; i < op.children.size(); i++) {
        if (op.children[i]->expression_class !=
            ExpressionClass::BOUND_CONSTANT) {
          return false;
        }
        auto &c = op.children[i]->Cast<BoundConstantExpression>();
        string lit;
        if (!TrySerializeLanceLiteral(c.value, lit)) {
          return false;
        }
        values.push_back(std::move(lit));
      }
      out_sql =
          "(" + lhs_sql +
          (op.type == ExpressionType::COMPARE_IN ? " IN (" : " NOT IN (") +
          StringUtil::Join(values, ", ") + "))";
      return true;
    }
    return false;
  }
  case ExpressionClass::BOUND_BETWEEN: {
    auto &between = expr.Cast<BoundBetweenExpression>();
    string input_sql, lower_sql, upper_sql;
    if (!TrySerializeLanceExpr(get, names, types, *between.input, input_sql) ||
        !TrySerializeLanceExpr(get, names, types, *between.lower, lower_sql) ||
        !TrySerializeLanceExpr(get, names, types, *between.upper, upper_sql)) {
      return false;
    }
    auto lower_op = ExpressionTypeToOperator(between.LowerComparisonType());
    auto upper_op = ExpressionTypeToOperator(between.UpperComparisonType());
    out_sql = "((" + input_sql + " " + lower_op + " " + lower_sql + ") AND (" +
              input_sql + " " + upper_op + " " + upper_sql + "))";
    return true;
  }
  default:
    return false;
  }
}

struct LanceKnnBindData : public TableFunctionData {
  string file_path;
  string vector_column;
  vector<float> query;
  uint64_t k = 0;
  bool prefilter = true;
  bool use_index = true;
  bool explain_verbose = false;

  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;

  string lance_complex_filter_sql;

  ~LanceKnnBindData() override {
    if (dataset) {
      lance_close_dataset(dataset);
    }
  }
};

struct LanceKnnGlobalState : public GlobalTableFunctionState {
  std::atomic<idx_t> lines_read{0};
  std::atomic<idx_t> record_batches{0};
  std::atomic<idx_t> record_batch_rows{0};
  string lance_filter_sql;
  bool filter_pushed_down = false;

  vector<idx_t> projection_ids;
  vector<LogicalType> scanned_types;

  std::atomic<bool> explain_computed{false};
  string explain_plan;
  string explain_error;
  std::mutex explain_mutex;

  idx_t MaxThreads() const override { return 1; }
  bool CanRemoveFilterColumns() const { return !projection_ids.empty(); }
};

struct LanceKnnLocalState : public ArrowScanLocalState {
  explicit LanceKnnLocalState(unique_ptr<ArrowArrayWrapper> current_chunk,
                              ClientContext &context)
      : ArrowScanLocalState(std::move(current_chunk), context),
        filter_sel(STANDARD_VECTOR_SIZE) {}

  void *stream = nullptr;
  LanceKnnGlobalState *global_state = nullptr;
  bool filter_pushed_down = false;
  SelectionVector filter_sel;

  ~LanceKnnLocalState() override {
    if (stream) {
      lance_close_stream(stream);
    }
  }
};

static void
LancePushdownComplexFilter(ClientContext &, LogicalGet &get,
                           FunctionData *bind_data,
                           vector<unique_ptr<Expression>> &filters) {
  if (!bind_data || filters.empty()) {
    return;
  }
  auto &scan_bind = bind_data->Cast<LanceKnnBindData>();

  vector<string> predicates;
  predicates.reserve(filters.size());
  for (auto &expr : filters) {
    if (!expr || expr->HasParameter() || expr->IsVolatile() ||
        expr->CanThrow()) {
      continue;
    }
    string sql;
    if (!TrySerializeLanceExpr(get, scan_bind.names, scan_bind.types, *expr,
                               sql)) {
      continue;
    }
    predicates.push_back(std::move(sql));
  }

  if (predicates.empty()) {
    return;
  }
  auto pushed_sql = StringUtil::Join(predicates, " AND ");
  if (scan_bind.lance_complex_filter_sql.empty()) {
    scan_bind.lance_complex_filter_sql = std::move(pushed_sql);
  } else {
    scan_bind.lance_complex_filter_sql =
        "(" + scan_bind.lance_complex_filter_sql + ") AND (" + pushed_sql + ")";
  }
}

static unique_ptr<FunctionData> LanceKnnBind(ClientContext &context,
                                             TableFunctionBindInput &input,
                                             vector<LogicalType> &return_types,
                                             vector<string> &names) {
  if (input.inputs.size() < 4) {
    throw InvalidInputException(
        "lance_knn requires (path, vector_column, query, k)");
  }
  if (input.inputs[0].IsNull()) {
    throw InvalidInputException("lance_knn requires a dataset root path");
  }
  if (input.inputs[1].IsNull()) {
    throw InvalidInputException("lance_knn requires a non-null vector column");
  }
  if (input.inputs[3].IsNull()) {
    throw InvalidInputException("lance_knn requires a non-null k");
  }

  auto result = make_uniq<LanceKnnBindData>();
  result->file_path = input.inputs[0].GetValue<string>();
  result->vector_column = input.inputs[1].GetValue<string>();
  result->query = ParseQueryVector(input.inputs[2]);
  auto verbose_it = input.named_parameters.find("explain_verbose");
  if (verbose_it != input.named_parameters.end() &&
      !verbose_it->second.IsNull()) {
    result->explain_verbose =
        verbose_it->second.DefaultCastAs(LogicalType::BOOLEAN).GetValue<bool>();
  }

  auto k_val = input.inputs[3].GetValue<int64_t>();
  if (k_val <= 0) {
    throw InvalidInputException("lance_knn requires k > 0");
  }
  result->k = NumericCast<uint64_t>(k_val);

  if (input.inputs.size() >= 5 && !input.inputs[4].IsNull()) {
    result->prefilter = input.inputs[4].GetValue<bool>();
  }
  if (input.inputs.size() >= 6 && !input.inputs[5].IsNull()) {
    result->use_index = input.inputs[5].GetValue<bool>();
  }

  result->dataset = lance_open_dataset(result->file_path.c_str());
  if (!result->dataset) {
    throw IOException("Failed to open Lance dataset: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  auto *schema_handle = lance_get_knn_schema(
      result->dataset, result->vector_column.c_str(), result->query.data(),
      result->query.size(), result->k, result->prefilter ? 1 : 0,
      result->use_index ? 1 : 0);
  if (!schema_handle) {
    throw IOException("Failed to get Lance KNN schema: " + result->file_path +
                      LanceFormatErrorSuffix());
  }

  memset(&result->schema_root.arrow_schema, 0,
         sizeof(result->schema_root.arrow_schema));
  if (lance_schema_to_arrow(schema_handle, &result->schema_root.arrow_schema) !=
      0) {
    lance_free_schema(schema_handle);
    throw IOException(
        "Failed to export Lance KNN schema to Arrow C Data Interface" +
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
LanceKnnInitGlobal(ClientContext &, TableFunctionInitInput &input) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto state = make_uniq_base<GlobalTableFunctionState, LanceKnnGlobalState>();
  auto &global = state->Cast<LanceKnnGlobalState>();

  global.projection_ids = input.projection_ids;
  if (!input.projection_ids.empty()) {
    global.scanned_types.reserve(input.column_ids.size());
    for (auto col_id : input.column_ids) {
      if (col_id >= bind_data.types.size()) {
        throw IOException("Invalid column id in projection");
      }
      global.scanned_types.push_back(bind_data.types[col_id]);
    }
  }

  string table_filter_sql;
  bool supported = TryBuildLanceFilterSQL(bind_data.names, bind_data.types,
                                          input, table_filter_sql);

  if (!supported) {
    if (bind_data.prefilter) {
      throw InvalidInputException(
          "lance_knn requires filter pushdown when prefilter=true");
    }
    global.lance_filter_sql.clear();
    global.filter_pushed_down = false;
    return state;
  }

  if (bind_data.lance_complex_filter_sql.empty()) {
    global.lance_filter_sql = std::move(table_filter_sql);
  } else if (table_filter_sql.empty()) {
    global.lance_filter_sql = bind_data.lance_complex_filter_sql;
  } else {
    global.lance_filter_sql = "(" + bind_data.lance_complex_filter_sql +
                              ") AND (" + table_filter_sql + ")";
  }
  global.filter_pushed_down = !global.lance_filter_sql.empty();

  if (bind_data.prefilter && input.filters && !input.filters->filters.empty() &&
      !global.filter_pushed_down) {
    throw InvalidInputException(
        "lance_knn requires filter pushdown when prefilter=true");
  }
  return state;
}

static unique_ptr<LocalTableFunctionState>
LanceKnnLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                  GlobalTableFunctionState *global_state) {
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global = global_state->Cast<LanceKnnGlobalState>();

  auto chunk = make_uniq<ArrowArrayWrapper>();
  auto result = make_uniq<LanceKnnLocalState>(std::move(chunk), context.client);
  result->column_ids = input.column_ids;
  result->filters = input.filters.get();
  result->global_state = &global;
  result->filter_pushed_down = global.filter_pushed_down;
  if (global.CanRemoveFilterColumns()) {
    result->all_columns.Initialize(context.client, global.scanned_types);
  }

  result->stream = lance_create_knn_stream(
      bind_data.dataset, bind_data.vector_column.c_str(),
      bind_data.query.data(), bind_data.query.size(), bind_data.k,
      global.lance_filter_sql.c_str(), bind_data.prefilter ? 1 : 0,
      bind_data.use_index ? 1 : 0);
  if (!result->stream) {
    throw IOException("Failed to create Lance KNN stream" +
                      LanceFormatErrorSuffix());
  }

  return std::move(result);
}

static bool LanceKnnLoadNextBatch(LanceKnnLocalState &local_state) {
  if (!local_state.stream) {
    return false;
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

  if (local_state.global_state) {
    local_state.global_state->record_batches.fetch_add(1);
    auto rows = NumericCast<idx_t>(new_chunk->arrow_array.length);
    local_state.global_state->record_batch_rows.fetch_add(rows);
  }

  if (tmp_schema.release) {
    tmp_schema.release(&tmp_schema);
  }

  local_state.chunk = std::move(new_chunk);
  local_state.Reset();
  return true;
}

static void ApplyDuckDBFilters(ClientContext &context, TableFilterSet &filters,
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

static void LanceKnnFunc(ClientContext &context, TableFunctionInput &data,
                         DataChunk &output) {
  if (!data.local_state) {
    return;
  }

  auto &bind_data = data.bind_data->Cast<LanceKnnBindData>();
  auto &global_state = data.global_state->Cast<LanceKnnGlobalState>();
  auto &local_state = data.local_state->Cast<LanceKnnLocalState>();

  while (true) {
    if (local_state.chunk_offset >=
        NumericCast<idx_t>(local_state.chunk->arrow_array.length)) {
      if (!LanceKnnLoadNextBatch(local_state)) {
        return;
      }
    }

    auto remaining = NumericCast<idx_t>(local_state.chunk->arrow_array.length) -
                     local_state.chunk_offset;
    auto output_size = MinValue<idx_t>(STANDARD_VECTOR_SIZE, remaining);
    auto start = global_state.lines_read.fetch_add(output_size);

    if (global_state.CanRemoveFilterColumns()) {
      local_state.all_columns.Reset();
      local_state.all_columns.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        local_state.all_columns, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters,
                           local_state.all_columns, local_state.filter_sel);
      }
      output.ReferenceColumns(local_state.all_columns,
                              global_state.projection_ids);
      output.SetCardinality(local_state.all_columns);
    } else {
      output.SetCardinality(output_size);
      ArrowTableFunction::ArrowToDuckDB(local_state,
                                        bind_data.arrow_table.GetColumns(),
                                        output, start, false);
      local_state.chunk_offset += output_size;
      if (local_state.filters && !local_state.filter_pushed_down) {
        ApplyDuckDBFilters(context, *local_state.filters, output,
                           local_state.filter_sel);
      }
    }

    if (output.size() == 0) {
      continue;
    }
    output.Verify();
    return;
  }
}

static InsertionOrderPreservingMap<string>
LanceKnnToString(TableFunctionToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";

  if (!bind_data.lance_complex_filter_sql.empty()) {
    result["Lance Filter SQL (Bind)"] = bind_data.lance_complex_filter_sql;
  }

  string plan;
  string error;
  if (TryLanceExplainKnn(
          bind_data.dataset, bind_data.vector_column, bind_data.query,
          bind_data.k, bind_data.lance_complex_filter_sql, bind_data.prefilter,
          bind_data.use_index, bind_data.explain_verbose, plan, error)) {
    result["Lance Plan (Bind)"] = plan;
  } else if (!error.empty()) {
    result["Lance Plan Error (Bind)"] = error;
  }

  return result;
}

static InsertionOrderPreservingMap<string>
LanceKnnDynamicToString(TableFunctionDynamicToStringInput &input) {
  InsertionOrderPreservingMap<string> result;
  auto &bind_data = input.bind_data->Cast<LanceKnnBindData>();
  auto &global_state = input.global_state->Cast<LanceKnnGlobalState>();

  result["Lance Path"] = bind_data.file_path;
  result["Lance Vector Column"] = bind_data.vector_column;
  result["Lance K"] = to_string(bind_data.k);
  result["Lance Query Dim"] = to_string(bind_data.query.size());
  result["Lance Prefilter"] = bind_data.prefilter ? "true" : "false";
  result["Lance Use Index"] = bind_data.use_index ? "true" : "false";
  result["Lance Explain Verbose"] =
      bind_data.explain_verbose ? "true" : "false";

  result["Lance Filter Pushed Down"] =
      global_state.filter_pushed_down ? "true" : "false";
  if (!global_state.lance_filter_sql.empty()) {
    result["Lance Filter SQL"] = global_state.lance_filter_sql;
  }

  result["Lance Record Batches"] =
      to_string(global_state.record_batches.load());
  result["Lance Record Batch Rows"] =
      to_string(global_state.record_batch_rows.load());
  result["Lance Rows Out"] = to_string(global_state.lines_read.load());

  if (!global_state.explain_computed.load()) {
    std::lock_guard<std::mutex> guard(global_state.explain_mutex);
    if (!global_state.explain_computed.load()) {
      string plan;
      string error;
      auto ok = TryLanceExplainKnn(
          bind_data.dataset, bind_data.vector_column, bind_data.query,
          bind_data.k, global_state.lance_filter_sql, bind_data.prefilter,
          bind_data.use_index, bind_data.explain_verbose, plan, error);
      if (ok) {
        global_state.explain_plan = std::move(plan);
      } else {
        global_state.explain_error = std::move(error);
      }
      global_state.explain_computed.store(true);
    }
  }

  if (!global_state.explain_plan.empty()) {
    result["Lance Plan"] = global_state.explain_plan;
  } else if (!global_state.explain_error.empty()) {
    result["Lance Plan Error"] = global_state.explain_error;
  }

  return result;
}

void RegisterLanceKnn(ExtensionLoader &loader) {
  TableFunction knn4(
      "lance_knn",
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::LIST(LogicalType::FLOAT), LogicalType::BIGINT},
      LanceKnnFunc, LanceKnnBind, LanceKnnInitGlobal, LanceKnnLocalInit);
  knn4.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  knn4.projection_pushdown = true;
  knn4.filter_pushdown = true;
  knn4.filter_prune = true;
  knn4.pushdown_complex_filter = LancePushdownComplexFilter;
  knn4.to_string = LanceKnnToString;
  knn4.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(knn4);

  TableFunction knn6(
      "lance_knn",
      {LogicalType::VARCHAR, LogicalType::VARCHAR,
       LogicalType::LIST(LogicalType::FLOAT), LogicalType::BIGINT,
       LogicalType::BOOLEAN, LogicalType::BOOLEAN},
      LanceKnnFunc, LanceKnnBind, LanceKnnInitGlobal, LanceKnnLocalInit);
  knn6.named_parameters["explain_verbose"] = LogicalType::BOOLEAN;
  knn6.projection_pushdown = true;
  knn6.filter_pushdown = true;
  knn6.filter_prune = true;
  knn6.pushdown_complex_filter = LancePushdownComplexFilter;
  knn6.to_string = LanceKnnToString;
  knn6.dynamic_to_string = LanceKnnDynamicToString;
  loader.RegisterFunction(knn6);
}

} // namespace duckdb
