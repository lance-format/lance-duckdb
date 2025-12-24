#include "lance_filter_ir.hpp"

#include "lance_common.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/in_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/table_filter.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace duckdb {

bool LanceFilterIRSupportsLogicalType(const LogicalType &type) {
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

static constexpr char LANCE_FILTER_IR_MAGIC[] = {'L', 'F', 'T', '1'};
static constexpr uint8_t LANCE_FILTER_IR_VERSION = 1;

enum class LanceFilterIRTag : uint8_t {
  COLUMN_REF = 1,
  LITERAL = 2,
  AND = 3,
  OR = 4,
  NOT = 5,
  COMPARISON = 6,
  IS_NULL = 7,
  IS_NOT_NULL = 8,
  IN_LIST = 9,
};

enum class LanceFilterIRLiteralTag : uint8_t {
  NULL_VALUE = 0,
  BOOL = 1,
  I64 = 2,
  U64 = 3,
  F32 = 4,
  F64 = 5,
  STRING = 6,
};

enum class LanceFilterIRComparisonOp : uint8_t {
  EQ = 0,
  NOT_EQ = 1,
  LT = 2,
  LT_EQ = 3,
  GT = 4,
  GT_EQ = 5,
};

static void LanceFilterIRAppendU8(string &out, uint8_t v) {
  out.push_back(static_cast<char>(v));
}

static void LanceFilterIRAppendU32(string &out, uint32_t v) {
  uint8_t buf[4];
  buf[0] = static_cast<uint8_t>(v & 0xFF);
  buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void LanceFilterIRAppendU64(string &out, uint64_t v) {
  uint8_t buf[8];
  for (idx_t i = 0; i < 8; i++) {
    buf[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
  }
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void LanceFilterIRAppendI64(string &out, int64_t v) {
  LanceFilterIRAppendU64(out, static_cast<uint64_t>(v));
}

static void LanceFilterIRAppendF32(string &out, float v) {
  uint32_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  LanceFilterIRAppendU32(out, bits);
}

static void LanceFilterIRAppendF64(string &out, double v) {
  uint64_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  LanceFilterIRAppendU64(out, bits);
}

static bool LanceFilterIRAppendLenPrefixed(string &out, const string &bytes) {
  if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out, static_cast<uint32_t>(bytes.size()));
  out.append(bytes);
  return true;
}

static bool LanceFilterIRAppendLenPrefixedString(string &out,
                                                 const string &value) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out, static_cast<uint32_t>(value.size()));
  out.append(value);
  return true;
}

static string LanceFilterIREncodeMessage(const string &root_node) {
  string out;
  out.append(LANCE_FILTER_IR_MAGIC, sizeof(LANCE_FILTER_IR_MAGIC));
  LanceFilterIRAppendU8(out, LANCE_FILTER_IR_VERSION);
  out.append(root_node);
  return out;
}

static bool SplitLanceColumnPath(const string &name, vector<string> &segments) {
  segments.clear();
  idx_t start = 0;
  for (idx_t i = 0; i <= name.size(); i++) {
    if (i == name.size() || name[i] == '.') {
      if (i == start) {
        return false;
      }
      segments.push_back(name.substr(start, i - start));
      start = i + 1;
    }
  }
  return !segments.empty();
}

static bool TryEncodeLanceFilterIRColumnRef(const string &name,
                                            string &out_ir) {
  vector<string> segments;
  if (!SplitLanceColumnPath(name, segments)) {
    return false;
  }

  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::COLUMN_REF));
  if (segments.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(segments.size()));
  for (auto &segment : segments) {
    if (!LanceFilterIRAppendLenPrefixedString(out_ir, segment)) {
      return false;
    }
  }
  return true;
}

static bool TryEncodeLanceFilterIRLiteral(const Value &value, string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::LITERAL));

  if (value.IsNull()) {
    LanceFilterIRAppendU8(
        out_ir, static_cast<uint8_t>(LanceFilterIRLiteralTag::NULL_VALUE));
    return true;
  }

  switch (value.type().id()) {
  case LogicalTypeId::BOOLEAN:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::BOOL));
    LanceFilterIRAppendU8(out_ir, value.GetValue<bool>() ? 1 : 0);
    return true;
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::I64));
    LanceFilterIRAppendI64(out_ir, value.GetValue<int64_t>());
    return true;
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::U64));
    LanceFilterIRAppendU64(out_ir, value.GetValue<uint64_t>());
    return true;
  case LogicalTypeId::FLOAT: {
    auto v = value.GetValue<float>();
    if (!std::isfinite(v)) {
      return false;
    }
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::F32));
    LanceFilterIRAppendF32(out_ir, v);
    return true;
  }
  case LogicalTypeId::DOUBLE: {
    auto v = value.GetValue<double>();
    if (!std::isfinite(v)) {
      return false;
    }
    LanceFilterIRAppendU8(out_ir,
                          static_cast<uint8_t>(LanceFilterIRLiteralTag::F64));
    LanceFilterIRAppendF64(out_ir, v);
    return true;
  }
  case LogicalTypeId::VARCHAR: {
    auto v = value.GetValue<string>();
    LanceFilterIRAppendU8(
        out_ir, static_cast<uint8_t>(LanceFilterIRLiteralTag::STRING));
    return LanceFilterIRAppendLenPrefixedString(out_ir, v);
  }
  default:
    return false;
  }
}

static bool TryEncodeLanceFilterIRComparisonOp(ExpressionType type,
                                               uint8_t &out_op) {
  switch (type) {
  case ExpressionType::COMPARE_EQUAL:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::EQ);
    return true;
  case ExpressionType::COMPARE_NOTEQUAL:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::NOT_EQ);
    return true;
  case ExpressionType::COMPARE_LESSTHAN:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::LT);
    return true;
  case ExpressionType::COMPARE_LESSTHANOREQUALTO:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::LT_EQ);
    return true;
  case ExpressionType::COMPARE_GREATERTHAN:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::GT);
    return true;
  case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    out_op = static_cast<uint8_t>(LanceFilterIRComparisonOp::GT_EQ);
    return true;
  default:
    return false;
  }
}

static bool TryEncodeLanceFilterIRConjunction(LanceFilterIRTag tag,
                                              const vector<string> &children,
                                              string &out_ir) {
  if (children.empty()) {
    return false;
  }
  if (children.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir, static_cast<uint8_t>(tag));
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(children.size()));
  for (auto &child : children) {
    if (!LanceFilterIRAppendLenPrefixed(out_ir, child)) {
      return false;
    }
  }
  return true;
}

static bool TryEncodeLanceFilterIRUnary(LanceFilterIRTag tag,
                                        const string &child, string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir, static_cast<uint8_t>(tag));
  return LanceFilterIRAppendLenPrefixed(out_ir, child);
}

static bool TryEncodeLanceFilterIRComparison(uint8_t op, const string &left,
                                             const string &right,
                                             string &out_ir) {
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::COMPARISON));
  LanceFilterIRAppendU8(out_ir, op);
  return LanceFilterIRAppendLenPrefixed(out_ir, left) &&
         LanceFilterIRAppendLenPrefixed(out_ir, right);
}

static bool TryEncodeLanceFilterIRInList(bool negated, const string &expr,
                                         const vector<string> &list,
                                         string &out_ir) {
  if (list.size() > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  out_ir.clear();
  LanceFilterIRAppendU8(out_ir,
                        static_cast<uint8_t>(LanceFilterIRTag::IN_LIST));
  LanceFilterIRAppendU8(out_ir, negated ? 1 : 0);
  if (!LanceFilterIRAppendLenPrefixed(out_ir, expr)) {
    return false;
  }
  LanceFilterIRAppendU32(out_ir, static_cast<uint32_t>(list.size()));
  for (auto &item : list) {
    if (!LanceFilterIRAppendLenPrefixed(out_ir, item)) {
      return false;
    }
  }
  return true;
}

static bool TryBuildLanceTableFilterIRExpr(const string &col_ref_ir,
                                           const TableFilter &filter,
                                           string &out_ir) {
  switch (filter.filter_type) {
  case TableFilterType::CONSTANT_COMPARISON: {
    auto &f = filter.Cast<ConstantFilter>();
    uint8_t op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(f.comparison_type, op)) {
      return false;
    }
    string lit_ir;
    if (!TryEncodeLanceFilterIRLiteral(f.constant, lit_ir)) {
      return false;
    }
    return TryEncodeLanceFilterIRComparison(op, col_ref_ir, lit_ir, out_ir);
  }
  case TableFilterType::IS_NULL:
    return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::IS_NULL, col_ref_ir,
                                       out_ir);
  case TableFilterType::IS_NOT_NULL:
    return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::IS_NOT_NULL,
                                       col_ref_ir, out_ir);
  case TableFilterType::IN_FILTER: {
    auto &f = filter.Cast<InFilter>();
    vector<string> items;
    items.reserve(f.values.size());
    for (auto &value : f.values) {
      string lit_ir;
      if (!TryEncodeLanceFilterIRLiteral(value, lit_ir)) {
        return false;
      }
      items.push_back(std::move(lit_ir));
    }
    return TryEncodeLanceFilterIRInList(false, col_ref_ir, items, out_ir);
  }
  case TableFilterType::CONJUNCTION_AND: {
    auto &f = filter.Cast<ConjunctionAndFilter>();
    vector<string> children;
    children.reserve(f.child_filters.size());
    for (auto &child : f.child_filters) {
      string child_ir;
      if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, children,
                                             out_ir);
  }
  case TableFilterType::CONJUNCTION_OR: {
    auto &f = filter.Cast<ConjunctionOrFilter>();
    vector<string> children;
    children.reserve(f.child_filters.size());
    for (auto &child : f.child_filters) {
      string child_ir;
      if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::OR, children,
                                             out_ir);
  }
  default:
    return false;
  }
}

LanceFilterIRBuildResult BuildLanceTableFilterIRParts(
    const vector<string> &names, const vector<LogicalType> &types,
    const TableFunctionInitInput &input, bool exclude_computed_columns) {
  LanceFilterIRBuildResult result;
  if (!input.filters || input.filters->filters.empty()) {
    return result;
  }

  result.parts.reserve(input.filters->filters.size());

  for (auto &it : input.filters->filters) {
    auto scan_col_idx = it.first;
    auto &filter = *it.second;

    if (scan_col_idx >= input.column_ids.size()) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    auto col_id = input.column_ids[scan_col_idx];
    if (col_id >= names.size() || col_id >= types.size()) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    if (exclude_computed_columns && IsComputedSearchColumn(names[col_id])) {
      result.all_filters_pushed = false;
      continue;
    }
    if (!LanceFilterIRSupportsLogicalType(types[col_id])) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }

    string col_ref_ir;
    if (!TryEncodeLanceFilterIRColumnRef(names[col_id], col_ref_ir)) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }
    string filter_ir;
    if (!TryBuildLanceTableFilterIRExpr(col_ref_ir, filter, filter_ir)) {
      result.all_filters_pushed = false;
      result.all_prefilterable_filters_pushed = false;
      continue;
    }

    result.parts.push_back(std::move(filter_ir));
  }

  return result;
}

static bool TryBuildLanceExprColumnRefIR(const LogicalGet &get,
                                         const vector<string> &names,
                                         const vector<LogicalType> &types,
                                         bool exclude_computed_columns,
                                         const Expression &expr,
                                         string &out_ir) {
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
  if (exclude_computed_columns && IsComputedSearchColumn(names[col_id])) {
    return false;
  }
  if (!LanceFilterIRSupportsLogicalType(types[col_id])) {
    return false;
  }
  return TryEncodeLanceFilterIRColumnRef(names[col_id], out_ir);
}

bool TryBuildLanceExprFilterIR(const LogicalGet &get,
                               const vector<string> &names,
                               const vector<LogicalType> &types,
                               bool exclude_computed_columns,
                               const Expression &expr, string &out_ir) {
  switch (expr.expression_class) {
  case ExpressionClass::BOUND_COLUMN_REF:
    return TryBuildLanceExprColumnRefIR(get, names, types,
                                        exclude_computed_columns, expr, out_ir);
  case ExpressionClass::BOUND_CONSTANT: {
    auto &c = expr.Cast<BoundConstantExpression>();
    return TryEncodeLanceFilterIRLiteral(c.value, out_ir);
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
    return TryEncodeLanceFilterIRLiteral(c.value, out_ir);
  }
  case ExpressionClass::BOUND_COMPARISON: {
    auto &cmp = expr.Cast<BoundComparisonExpression>();
    if (cmp.type == ExpressionType::COMPARE_DISTINCT_FROM ||
        cmp.type == ExpressionType::COMPARE_NOT_DISTINCT_FROM) {
      return false;
    }
    uint8_t op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(cmp.type, op)) {
      return false;
    }
    string lhs_ir, rhs_ir;
    if (!TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *cmp.left, lhs_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *cmp.right, rhs_ir)) {
      return false;
    }
    return TryEncodeLanceFilterIRComparison(op, lhs_ir, rhs_ir, out_ir);
  }
  case ExpressionClass::BOUND_CONJUNCTION: {
    auto &conj = expr.Cast<BoundConjunctionExpression>();
    LanceFilterIRTag tag;
    if (conj.type == ExpressionType::CONJUNCTION_AND) {
      tag = LanceFilterIRTag::AND;
    } else if (conj.type == ExpressionType::CONJUNCTION_OR) {
      tag = LanceFilterIRTag::OR;
    } else {
      return false;
    }
    vector<string> children;
    children.reserve(conj.children.size());
    for (auto &child : conj.children) {
      string child_ir;
      if (!TryBuildLanceExprFilterIR(
              get, names, types, exclude_computed_columns, *child, child_ir)) {
        return false;
      }
      children.push_back(std::move(child_ir));
    }
    return TryEncodeLanceFilterIRConjunction(tag, children, out_ir);
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.type == ExpressionType::OPERATOR_NOT) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns, *op.children[0],
                                     child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(LanceFilterIRTag::NOT, child_ir,
                                         out_ir);
    }
    if (op.type == ExpressionType::OPERATOR_IS_NULL ||
        op.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
      if (op.children.size() != 1) {
        return false;
      }
      string child_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns, *op.children[0],
                                     child_ir)) {
        return false;
      }
      return TryEncodeLanceFilterIRUnary(
          op.type == ExpressionType::OPERATOR_IS_NULL
              ? LanceFilterIRTag::IS_NULL
              : LanceFilterIRTag::IS_NOT_NULL,
          child_ir, out_ir);
    }
    if (op.type == ExpressionType::COMPARE_IN ||
        op.type == ExpressionType::COMPARE_NOT_IN) {
      if (op.children.size() < 2) {
        return false;
      }
      string lhs_ir;
      if (!TryBuildLanceExprFilterIR(get, names, types,
                                     exclude_computed_columns, *op.children[0],
                                     lhs_ir)) {
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
        string lit_ir;
        if (!TryEncodeLanceFilterIRLiteral(c.value, lit_ir)) {
          return false;
        }
        values.push_back(std::move(lit_ir));
      }
      return TryEncodeLanceFilterIRInList(
          op.type == ExpressionType::COMPARE_NOT_IN, lhs_ir, values, out_ir);
    }
    return false;
  }
  case ExpressionClass::BOUND_BETWEEN: {
    auto &between = expr.Cast<BoundBetweenExpression>();
    uint8_t lower_op = 0;
    uint8_t upper_op = 0;
    if (!TryEncodeLanceFilterIRComparisonOp(between.LowerComparisonType(),
                                            lower_op) ||
        !TryEncodeLanceFilterIRComparisonOp(between.UpperComparisonType(),
                                            upper_op)) {
      return false;
    }
    string input_ir, lower_ir, upper_ir;
    if (!TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.input, input_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.lower, lower_ir) ||
        !TryBuildLanceExprFilterIR(get, names, types, exclude_computed_columns,
                                   *between.upper, upper_ir)) {
      return false;
    }
    string lower_cmp_ir;
    if (!TryEncodeLanceFilterIRComparison(lower_op, input_ir, lower_ir,
                                          lower_cmp_ir)) {
      return false;
    }
    string upper_cmp_ir;
    if (!TryEncodeLanceFilterIRComparison(upper_op, input_ir, upper_ir,
                                          upper_cmp_ir)) {
      return false;
    }
    vector<string> children;
    children.push_back(std::move(lower_cmp_ir));
    children.push_back(std::move(upper_cmp_ir));
    return TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, children,
                                             out_ir);
  }
  default:
    return false;
  }
}

bool TryEncodeLanceFilterIRMessage(const vector<string> &parts,
                                   string &out_msg) {
  out_msg.clear();
  if (parts.empty()) {
    return true;
  }
  string root_node;
  if (parts.size() == 1) {
    root_node = parts[0];
  } else if (!TryEncodeLanceFilterIRConjunction(LanceFilterIRTag::AND, parts,
                                                root_node)) {
    return false;
  }
  out_msg = LanceFilterIREncodeMessage(root_node);
  return true;
}

} // namespace duckdb
