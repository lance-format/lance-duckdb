#include "lance_exec_ir.hpp"

#include "lance_filter_ir.hpp"
#include "lance_scan_bind_data.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/enums/order_type.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>
#include <unordered_set>

namespace duckdb {

static bool ExecIrFailfast() {
  return std::getenv("LANCE_EXEC_PUSHDOWN_FAILFAST") != nullptr;
}

static void WriteU8(string &out, uint8_t v) {
  out.push_back(static_cast<char>(v));
}

static void WriteU32(string &out, uint32_t v) {
  for (int i = 0; i < 4; i++) {
    out.push_back(static_cast<char>((v >> (i * 8)) & 0xFF));
  }
}

static void WriteI64(string &out, int64_t v) {
  auto u = static_cast<uint64_t>(v);
  for (int i = 0; i < 8; i++) {
    out.push_back(static_cast<char>((u >> (i * 8)) & 0xFF));
  }
}

static void WriteF64(string &out, double v) {
  static_assert(sizeof(double) == sizeof(uint64_t),
                "double must be 64-bit IEEE754");
  uint64_t bits;
  memcpy(&bits, &v, sizeof(bits));
  for (int i = 0; i < 8; i++) {
    out.push_back(static_cast<char>((bits >> (i * 8)) & 0xFF));
  }
}

static void WriteBytes(string &out, const_data_ptr_t ptr, size_t len) {
  out.append(reinterpret_cast<const char *>(ptr), len);
}

static void WriteHugeint(string &out, hugeint_t v) {
  uint64_t lower = v.lower;
  int64_t upper = v.upper;
  for (int i = 0; i < 8; i++) {
    out.push_back(static_cast<char>((lower >> (i * 8)) & 0xFF));
  }
  auto upper_u = static_cast<uint64_t>(upper);
  for (int i = 0; i < 8; i++) {
    out.push_back(static_cast<char>((upper_u >> (i * 8)) & 0xFF));
  }
}

static void WriteString(string &out, const string &s) {
  if (s.size() > NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    throw InvalidInputException("ExecIR string too large");
  }
  WriteU32(out, NumericCast<uint32_t>(s.size()));
  WriteBytes(out, reinterpret_cast<const_data_ptr_t>(s.data()), s.size());
}

enum class ExecIrAggFunc : uint8_t {
  SUM = 1,
  COUNT = 2,
  COUNT_STAR = 3,
  MIN = 4,
  MAX = 5,
  AVG = 6,
};

enum class ExecIrExprTag : uint8_t {
  COLUMN_REF = 1,
  LITERAL = 2,
  BINARY = 3,
  CAST = 4,
};

enum class ExecIrLiteralTag : uint8_t {
  NULL_ = 0,
  INT64 = 1,
  DOUBLE = 2,
  BOOL = 3,
  STRING = 4,
  DECIMAL128 = 5,
};

enum class ExecIrBinaryOp : uint8_t {
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
};

enum class ExecIrOutputTypeTag : uint8_t {
  BOOL = 1,
  INT64 = 2,
  DOUBLE = 3,
  DATE = 4,
  TIMESTAMP_US = 5,
  VARCHAR = 6,
  DECIMAL = 7,
};

static bool TryEncodeOutputTypeHint(const LogicalType &type, string &out) {
  out.clear();
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::BOOL));
    return true;
  case LogicalTypeId::BIGINT:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::INT64));
    return true;
  case LogicalTypeId::DOUBLE:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::DOUBLE));
    return true;
  case LogicalTypeId::DATE:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::DATE));
    return true;
  case LogicalTypeId::TIMESTAMP:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::TIMESTAMP_US));
    return true;
  case LogicalTypeId::VARCHAR:
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::VARCHAR));
    return true;
  case LogicalTypeId::DECIMAL: {
    auto width = DecimalType::GetWidth(type);
    auto scale = DecimalType::GetScale(type);
    if (width < 0 || width > NumericLimits<uint8_t>::Maximum() || scale < 0 ||
        scale > NumericLimits<uint8_t>::Maximum()) {
      return false;
    }
    WriteU8(out, static_cast<uint8_t>(ExecIrOutputTypeTag::DECIMAL));
    WriteU8(out, NumericCast<uint8_t>(width));
    WriteU8(out, NumericCast<uint8_t>(scale));
    return true;
  }
  default:
    return false;
  }
}

static bool TryMapAggFunc(const BoundAggregateExpression &agg,
                          ExecIrAggFunc &out) {
  auto name = StringUtil::Lower(agg.function.name);
  if (name == "sum") {
    out = ExecIrAggFunc::SUM;
    return true;
  }
  if (name == "count_star") {
    out = ExecIrAggFunc::COUNT_STAR;
    return true;
  }
  if (name == "count") {
    if (agg.children.empty()) {
      out = ExecIrAggFunc::COUNT_STAR;
      return true;
    }
    if (agg.children.size() == 1) {
      out = ExecIrAggFunc::COUNT;
      return true;
    }
    return false;
  }
  if (name == "min") {
    out = ExecIrAggFunc::MIN;
    return true;
  }
  if (name == "max") {
    out = ExecIrAggFunc::MAX;
    return true;
  }
  if (name == "avg" || name == "average") {
    out = ExecIrAggFunc::AVG;
    return true;
  }
  return false;
}

static bool TryMapBinaryOp(const BoundFunctionExpression &fn,
                           ExecIrBinaryOp &out) {
  if (fn.children.size() != 2) {
    return false;
  }
  auto name = StringUtil::Lower(fn.function.name);
  if (name == "+" || name == "add") {
    out = ExecIrBinaryOp::ADD;
    return true;
  }
  if (name == "-" || name == "subtract") {
    out = ExecIrBinaryOp::SUB;
    return true;
  }
  if (name == "*" || name == "multiply") {
    out = ExecIrBinaryOp::MUL;
    return true;
  }
  if (name == "/" || name == "divide") {
    out = ExecIrBinaryOp::DIV;
    return true;
  }
  return false;
}

static bool TryEncodeExecExpr(
    const LogicalGet &scan_get, const LanceScanBindData &scan_bind,
    const vector<const vector<unique_ptr<Expression>> *> &projection_stack,
    idx_t projection_stack_offset, const Expression &expr, string &out,
    unordered_set<idx_t> &out_col_idxs);

static bool TryEncodeLiteral(const Value &v, string &out) {
  WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::LITERAL));
  if (v.IsNull()) {
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::NULL_));
    return true;
  }
  switch (v.type().id()) {
  case LogicalTypeId::BIGINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::UBIGINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UTINYINT: {
    auto i64 = v.DefaultCastAs(LogicalType::BIGINT).GetValue<int64_t>();
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::INT64));
    WriteI64(out, i64);
    return true;
  }
  case LogicalTypeId::DOUBLE:
  case LogicalTypeId::FLOAT: {
    auto f64 = v.DefaultCastAs(LogicalType::DOUBLE).GetValue<double>();
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::DOUBLE));
    WriteF64(out, f64);
    return true;
  }
  case LogicalTypeId::BOOLEAN: {
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::BOOL));
    WriteU8(out, v.GetValue<bool>() ? 1 : 0);
    return true;
  }
  case LogicalTypeId::VARCHAR: {
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::STRING));
    WriteString(out, v.GetValue<string>());
    return true;
  }
  case LogicalTypeId::DECIMAL: {
    auto width = DecimalType::GetWidth(v.type());
    auto scale = DecimalType::GetScale(v.type());
    if (width < 0 || width > NumericLimits<uint8_t>::Maximum() || scale < 0 ||
        scale > NumericLimits<uint8_t>::Maximum()) {
      return false;
    }
    WriteU8(out, static_cast<uint8_t>(ExecIrLiteralTag::DECIMAL128));
    WriteU8(out, NumericCast<uint8_t>(width));
    WriteU8(out, NumericCast<uint8_t>(scale));
    hugeint_t dec;
    switch (v.type().InternalType()) {
    case PhysicalType::INT16:
      dec = Hugeint::Convert(v.GetValueUnsafe<int16_t>());
      break;
    case PhysicalType::INT32:
      dec = Hugeint::Convert(v.GetValueUnsafe<int32_t>());
      break;
    case PhysicalType::INT64:
      dec = Hugeint::Convert(v.GetValueUnsafe<int64_t>());
      break;
    case PhysicalType::INT128:
      dec = v.GetValueUnsafe<hugeint_t>();
      break;
    default:
      return false;
    }
    WriteHugeint(out, dec);
    return true;
  }
  default:
    return false;
  }
}

static bool IsExecIrSafeCast(const LogicalType &from, const LogicalType &to) {
  if (from == to) {
    return true;
  }
  // Timestamp <-> Date casts are truncation/extension and should not fail.
  if (from.id() == LogicalTypeId::TIMESTAMP && to.id() == LogicalTypeId::DATE) {
    return true;
  }
  if (from.id() == LogicalTypeId::DATE && to.id() == LogicalTypeId::TIMESTAMP) {
    return true;
  }

  // Safe widening decimal cast: allow only if target width is at least the
  // source width. Narrowing may overflow.
  if (from.id() == LogicalTypeId::DECIMAL &&
      to.id() == LogicalTypeId::DECIMAL) {
    auto from_width = DecimalType::GetWidth(from);
    auto to_width = DecimalType::GetWidth(to);
    return to_width >= from_width;
  }

  // Safe integer widening (signed only): allow upcast to BIGINT.
  if (to.id() == LogicalTypeId::BIGINT) {
    switch (from.id()) {
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
      return true;
    default:
      break;
    }
  }

  // Numeric to DOUBLE is safe.
  if (to.id() == LogicalTypeId::DOUBLE) {
    switch (from.id()) {
    case LogicalTypeId::TINYINT:
    case LogicalTypeId::SMALLINT:
    case LogicalTypeId::INTEGER:
    case LogicalTypeId::BIGINT:
    case LogicalTypeId::FLOAT:
    case LogicalTypeId::DOUBLE:
      return true;
    default:
      break;
    }
  }

  return false;
}

static bool IsExecIrPushdownableExpr(const Expression &expr) {
  if (expr.HasParameter() || expr.IsVolatile()) {
    return false;
  }
  if (!expr.CanThrow()) {
    return true;
  }
  if (expr.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
    return false;
  }
  auto &cast = expr.Cast<BoundCastExpression>();
  if (!cast.child) {
    return false;
  }
  if (!IsExecIrSafeCast(cast.child->return_type, cast.return_type)) {
    return false;
  }
  return IsExecIrPushdownableExpr(*cast.child);
}

static bool TryEncodeExecExpr(
    const LogicalGet &scan_get, const LanceScanBindData &scan_bind,
    const vector<const vector<unique_ptr<Expression>> *> &projection_stack,
    idx_t projection_stack_offset, const Expression &expr, string &out,
    unordered_set<idx_t> &out_col_idxs) {
  auto projection_exprs = projection_stack_offset < projection_stack.size()
                              ? projection_stack[projection_stack_offset]
                              : nullptr;
  switch (expr.GetExpressionClass()) {
  case ExpressionClass::BOUND_COLUMN_REF: {
    auto &colref = expr.Cast<BoundColumnRefExpression>();
    if (colref.binding.table_index != scan_get.table_index) {
      if (!projection_exprs ||
          colref.binding.column_index >= projection_exprs->size() ||
          !(*projection_exprs)[colref.binding.column_index]) {
        return false;
      }
      return TryEncodeExecExpr(
          scan_get, scan_bind, projection_stack, projection_stack_offset + 1,
          *(*projection_exprs)[colref.binding.column_index], out, out_col_idxs);
    }
    auto &column_ids = scan_get.GetColumnIds();
    if (colref.binding.column_index >= column_ids.size()) {
      return false;
    }
    auto &col_index = column_ids[colref.binding.column_index];
    if (col_index.IsVirtualColumn()) {
      return false;
    }
    auto col_id = col_index.GetPrimaryIndex();
    if (col_id >= scan_bind.names.size()) {
      return false;
    }
    auto &name = scan_bind.names[col_id];
    if (name.empty()) {
      return false;
    }
    out_col_idxs.emplace(col_id);
    WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::COLUMN_REF));
    WriteString(out, name);
    return true;
  }
  case ExpressionClass::BOUND_CONSTANT: {
    auto &c = expr.Cast<BoundConstantExpression>();
    return TryEncodeLiteral(c.value, out);
  }
  case ExpressionClass::BOUND_CAST: {
    auto &cast = expr.Cast<BoundCastExpression>();
    if (!cast.child) {
      return false;
    }
    if (cast.child->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
      auto &c = cast.child->Cast<BoundConstantExpression>();
      Value v;
      try {
        v = c.value.DefaultCastAs(cast.return_type);
      } catch (...) {
        return false;
      }
      return TryEncodeLiteral(v, out);
    }
    string type_hint;
    if (!TryEncodeOutputTypeHint(cast.return_type, type_hint)) {
      return false;
    }
    WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::CAST));
    WriteBytes(out, reinterpret_cast<const_data_ptr_t>(type_hint.data()),
               type_hint.size());
    return TryEncodeExecExpr(scan_get, scan_bind, projection_stack,
                             projection_stack_offset, *cast.child, out,
                             out_col_idxs);
  }
  case ExpressionClass::BOUND_REF: {
    auto &ref = expr.Cast<BoundReferenceExpression>();
    if (projection_exprs && ref.index < projection_exprs->size() &&
        (*projection_exprs)[ref.index] &&
        (*projection_exprs)[ref.index].get() != &expr) {
      return TryEncodeExecExpr(
          scan_get, scan_bind, projection_stack, projection_stack_offset + 1,
          *(*projection_exprs)[ref.index], out, out_col_idxs);
    }

    auto &column_ids = scan_get.GetColumnIds();
    if (ref.index >= column_ids.size()) {
      return false;
    }
    auto &col_index = column_ids[ref.index];
    if (col_index.IsVirtualColumn()) {
      return false;
    }
    auto col_id = col_index.GetPrimaryIndex();
    if (col_id >= scan_bind.names.size()) {
      return false;
    }
    auto &name = scan_bind.names[col_id];
    if (name.empty()) {
      return false;
    }
    out_col_idxs.emplace(col_id);
    WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::COLUMN_REF));
    WriteString(out, name);
    return true;
  }
  case ExpressionClass::BOUND_FUNCTION: {
    auto &fn = expr.Cast<BoundFunctionExpression>();
    ExecIrBinaryOp bop;
    if (!TryMapBinaryOp(fn, bop)) {
      return false;
    }
    WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::BINARY));
    WriteU8(out, static_cast<uint8_t>(bop));

    const LogicalType *decimal_type = nullptr;
    if (fn.children.size() == 2 && fn.children[0] &&
        fn.children[0]->return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &fn.children[0]->return_type;
    } else if (fn.children.size() == 2 && fn.children[1] &&
               fn.children[1]->return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &fn.children[1]->return_type;
    } else if (expr.return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &expr.return_type;
    }

    auto encode_child = [&](const unique_ptr<Expression> &child_expr) -> bool {
      if (!child_expr) {
        return false;
      }
      if (decimal_type &&
          child_expr->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
        auto &c = child_expr->Cast<BoundConstantExpression>();
        Value v;
        try {
          v = c.value.DefaultCastAs(*decimal_type);
        } catch (...) {
          return false;
        }
        return TryEncodeLiteral(v, out);
      }
      return TryEncodeExecExpr(scan_get, scan_bind, projection_stack,
                               projection_stack_offset, *child_expr, out,
                               out_col_idxs);
    };

    if (!encode_child(fn.children[0])) {
      return false;
    }
    if (!encode_child(fn.children[1])) {
      return false;
    }
    return true;
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.children.size() != 2 || !op.children[0] || !op.children[1]) {
      return false;
    }
    auto op_str = ExpressionTypeToOperator(expr.GetExpressionType());
    ExecIrBinaryOp bop;
    if (op_str == "+") {
      bop = ExecIrBinaryOp::ADD;
    } else if (op_str == "-") {
      bop = ExecIrBinaryOp::SUB;
    } else if (op_str == "*") {
      bop = ExecIrBinaryOp::MUL;
    } else if (op_str == "/") {
      bop = ExecIrBinaryOp::DIV;
    } else {
      return false;
    }
    WriteU8(out, static_cast<uint8_t>(ExecIrExprTag::BINARY));
    WriteU8(out, static_cast<uint8_t>(bop));

    const LogicalType *decimal_type = nullptr;
    if (op.children[0] &&
        op.children[0]->return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &op.children[0]->return_type;
    } else if (op.children[1] &&
               op.children[1]->return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &op.children[1]->return_type;
    } else if (expr.return_type.id() == LogicalTypeId::DECIMAL) {
      decimal_type = &expr.return_type;
    }

    auto encode_child = [&](const unique_ptr<Expression> &child_expr) -> bool {
      if (!child_expr) {
        return false;
      }
      if (decimal_type &&
          child_expr->GetExpressionClass() == ExpressionClass::BOUND_CONSTANT) {
        auto &c = child_expr->Cast<BoundConstantExpression>();
        Value v;
        try {
          v = c.value.DefaultCastAs(*decimal_type);
        } catch (...) {
          return false;
        }
        return TryEncodeLiteral(v, out);
      }
      return TryEncodeExecExpr(scan_get, scan_bind, projection_stack,
                               projection_stack_offset, *child_expr, out,
                               out_col_idxs);
    };

    if (!encode_child(op.children[0])) {
      return false;
    }
    if (!encode_child(op.children[1])) {
      return false;
    }
    return true;
  }
  default:
    return false;
  }
}

bool TryEncodeLanceExecIRv1(
    const LogicalGet &scan_get, const LanceScanBindData &scan_bind,
    const string &filter_ir_msg, const vector<idx_t> &extra_scan_col_ids,
    const vector<const vector<unique_ptr<Expression>> *> &projection_stack,
    const LogicalAggregate &aggregate, const LogicalOrder *order,
    const LogicalProjection *post_projection, string &out_exec_ir) {
  if (!aggregate.grouping_functions.empty()) {
    if (ExecIrFailfast()) {
      throw IOException("ExecIR grouping functions not supported");
    }
    return false;
  }
  if (!aggregate.grouping_sets.empty()) {
    if (aggregate.grouping_sets.size() != 1) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR multiple grouping sets not supported");
      }
      return false;
    }
    auto &set = aggregate.grouping_sets[0];
    if (set.size() != aggregate.groups.size()) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR partial grouping set not supported");
      }
      return false;
    }
    for (idx_t i = 0; i < aggregate.groups.size(); i++) {
      if (set.count(i) == 0) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR reordered grouping set not supported");
        }
        return false;
      }
    }
  }
  if (aggregate.children.size() != 1 || !aggregate.children[0]) {
    return false;
  }
  if (scan_bind.names.size() != scan_bind.types.size()) {
    return false;
  }

  unordered_set<idx_t> col_idxs;
  col_idxs.reserve(16);
  for (auto idx : extra_scan_col_ids) {
    col_idxs.emplace(idx);
  }

  struct EncodedGroup {
    string output_name;
    string encoded_expr;
    string output_type_hint;
  };

  struct EncodedAgg {
    ExecIrAggFunc func;
    string output_name;
    string encoded_args;
    string output_type_hint;
    uint32_t arg_count;
  };

  std::function<bool(const Expression &)> is_supported_group_expr;
  is_supported_group_expr = [&](const Expression &expr) -> bool {
    if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF ||
        expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
      return true;
    }
    if (expr.GetExpressionClass() != ExpressionClass::BOUND_CAST) {
      return false;
    }
    auto &cast = expr.Cast<BoundCastExpression>();
    if (!cast.child) {
      return false;
    }
    // Allow nested CASTs but require the leaf to be a direct column reference.
    return is_supported_group_expr(*cast.child);
  };

  vector<EncodedGroup> groups;
  groups.reserve(aggregate.groups.size());
  for (auto &expr : aggregate.groups) {
    if (!expr || !IsExecIrPushdownableExpr(*expr)) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR group expression is not pushdownable");
      }
      return false;
    }
    if (!is_supported_group_expr(*expr)) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR group expression not supported: " +
                          expr->ToString());
      }
      return false;
    }
    string expr_buf;
    unordered_set<idx_t> tmp_idxs;
    tmp_idxs.reserve(4);
    if (!TryEncodeExecExpr(scan_get, scan_bind, projection_stack, 0, *expr,
                           expr_buf, tmp_idxs)) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR group expr encode failed: " +
                          expr->ToString());
      }
      return false;
    }

    EncodedGroup g;
    g.output_name = expr->GetName();
    g.encoded_expr = std::move(expr_buf);
    if (!TryEncodeOutputTypeHint(expr->return_type, g.output_type_hint)) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR group output type unsupported: " +
                          expr->return_type.ToString());
      }
      return false;
    }
    for (auto idx : tmp_idxs) {
      col_idxs.emplace(idx);
    }
    groups.push_back(std::move(g));
  }

  vector<EncodedAgg> aggs;
  aggs.reserve(aggregate.expressions.size());

  for (auto &expr : aggregate.expressions) {
    if (!expr) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR aggregate expression is null");
      }
      return false;
    }
    ExecIrAggFunc func;
    const Expression *arg_expr = nullptr;
    bool count_star = false;

    if (expr->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE) {
      auto &agg = expr->Cast<BoundAggregateExpression>();
      if (agg.IsDistinct() || agg.filter || agg.order_bys) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR aggregate has DISTINCT/FILTER/ORDER BY");
        }
        return false;
      }
      if (!TryMapAggFunc(agg, func)) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR unsupported aggregate: " +
                            agg.function.name);
        }
        return false;
      }
      if (func == ExecIrAggFunc::COUNT_STAR) {
        if (!agg.children.empty()) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR count_star has args");
          }
          return false;
        }
        count_star = true;
      } else {
        if (agg.children.size() != 1) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR aggregate arg count != 1");
          }
          return false;
        }
        arg_expr = agg.children[0].get();
      }
    } else if (expr->GetExpressionClass() == ExpressionClass::BOUND_FUNCTION) {
      auto &fn = expr->Cast<BoundFunctionExpression>();
      auto name = StringUtil::Lower(fn.function.name);
      if (name == "sum") {
        func = ExecIrAggFunc::SUM;
      } else if (name == "count") {
        func = fn.children.empty() ? ExecIrAggFunc::COUNT_STAR
                                   : ExecIrAggFunc::COUNT;
      } else if (name == "min") {
        func = ExecIrAggFunc::MIN;
      } else if (name == "max") {
        func = ExecIrAggFunc::MAX;
      } else if (name == "avg" || name == "average") {
        func = ExecIrAggFunc::AVG;
      } else {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR unsupported aggregate function: " +
                            fn.function.name);
        }
        return false;
      }
      if (func == ExecIrAggFunc::COUNT_STAR) {
        if (!fn.children.empty()) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR count_star has args");
          }
          return false;
        }
        count_star = true;
      } else {
        if (fn.children.size() != 1) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR aggregate arg count != 1");
          }
          return false;
        }
        arg_expr = fn.children[0].get();
      }
    } else {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR unsupported aggregate expression class: " +
                          ExpressionClassToString(expr->GetExpressionClass()));
      }
      return false;
    }

    EncodedAgg enc;
    enc.func = func;
    enc.output_name = expr->GetName();
    if (!TryEncodeOutputTypeHint(expr->return_type, enc.output_type_hint)) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR aggregate output type unsupported: " +
                          expr->return_type.ToString());
      }
      return false;
    }

    string args_buf;
    uint32_t arg_count = 0;
    unordered_set<idx_t> tmp_idxs;
    tmp_idxs.reserve(8);
    if (count_star) {
      arg_count = 0;
    } else {
      if (!arg_expr) {
        return false;
      }
      if (!TryEncodeExecExpr(scan_get, scan_bind, projection_stack, 0,
                             *arg_expr, args_buf, tmp_idxs)) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR aggregate arg encode failed: " +
                            arg_expr->ToString());
        }
        return false;
      }
      for (auto idx : tmp_idxs) {
        col_idxs.emplace(idx);
      }
      arg_count = 1;
    }

    enc.encoded_args = std::move(args_buf);
    enc.arg_count = arg_count;
    aggs.push_back(std::move(enc));
  }

  vector<string> scan_projection;
  scan_projection.reserve(col_idxs.size());
  vector<idx_t> sorted_idxs;
  sorted_idxs.reserve(col_idxs.size());
  for (auto idx : col_idxs) {
    sorted_idxs.push_back(idx);
  }
  std::sort(sorted_idxs.begin(), sorted_idxs.end());
  for (auto idx : sorted_idxs) {
    if (idx >= scan_bind.names.size()) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR scan projection index out of bounds: " +
                          to_string(idx));
      }
      return false;
    }
    auto &name = scan_bind.names[idx];
    if (name.empty()) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR scan projection name is empty");
      }
      return false;
    }
    scan_projection.push_back(name);
  }

  out_exec_ir.clear();
  out_exec_ir.reserve(64 + filter_ir_msg.size() + scan_projection.size() * 16 +
                      groups.size() * 16 + aggregate.expressions.size() * 32);

  WriteBytes(out_exec_ir, reinterpret_cast<const_data_ptr_t>("LEX1"), 4);
  WriteU32(out_exec_ir, 4); // version
  WriteU32(out_exec_ir, 0); // reserved flags

  if (filter_ir_msg.size() >
      NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(filter_ir_msg.size()));
  WriteBytes(out_exec_ir,
             reinterpret_cast<const_data_ptr_t>(filter_ir_msg.data()),
             filter_ir_msg.size());

  if (scan_projection.size() >
      NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(scan_projection.size()));
  for (auto &name : scan_projection) {
    WriteString(out_exec_ir, name);
  }

  if (groups.size() > NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(groups.size()));
  for (auto &g : groups) {
    WriteString(out_exec_ir, g.output_name);
    if (!g.encoded_expr.empty()) {
      WriteBytes(out_exec_ir,
                 reinterpret_cast<const_data_ptr_t>(g.encoded_expr.data()),
                 g.encoded_expr.size());
    }
    if (!g.output_type_hint.empty()) {
      WriteBytes(out_exec_ir,
                 reinterpret_cast<const_data_ptr_t>(g.output_type_hint.data()),
                 g.output_type_hint.size());
    }
  }

  if (aggs.size() > NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(aggs.size()));
  for (auto &agg : aggs) {
    WriteU8(out_exec_ir, static_cast<uint8_t>(agg.func));
    WriteString(out_exec_ir, agg.output_name);
    WriteU32(out_exec_ir, agg.arg_count);
    if (!agg.encoded_args.empty()) {
      WriteBytes(out_exec_ir,
                 reinterpret_cast<const_data_ptr_t>(agg.encoded_args.data()),
                 agg.encoded_args.size());
    }
    if (!agg.output_type_hint.empty()) {
      WriteBytes(
          out_exec_ir,
          reinterpret_cast<const_data_ptr_t>(agg.output_type_hint.data()),
          agg.output_type_hint.size());
    }
  }

  vector<string> output_names;
  output_names.reserve(groups.size() + aggs.size());
  for (auto &g : groups) {
    output_names.push_back(g.output_name);
  }
  for (auto &a : aggs) {
    output_names.push_back(a.output_name);
  }

  // Optional post-aggregate projection: allow simple rename (and identity
  // selection) so we can replace ORDER BY / PROJECTION roots without changing
  // user-visible column names.
  if (post_projection) {
    if (post_projection->children.size() != 1 ||
        !post_projection->children[0]) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR post projection child shape not supported");
      }
      return false;
    }
    if (post_projection->expressions.size() != output_names.size()) {
      if (ExecIrFailfast()) {
        throw IOException("ExecIR post projection column count mismatch");
      }
      return false;
    }
    for (idx_t i = 0; i < post_projection->expressions.size(); i++) {
      auto &expr_ptr = post_projection->expressions[i];
      if (!expr_ptr || expr_ptr->HasParameter() || expr_ptr->IsVolatile() ||
          expr_ptr->CanThrow()) {
        if (ExecIrFailfast()) {
          throw IOException(
              "ExecIR post projection expression not pushdownable");
        }
        return false;
      }

      const Expression *base = expr_ptr.get();
      while (base->GetExpressionClass() == ExpressionClass::BOUND_CAST) {
        auto &cast = base->Cast<BoundCastExpression>();
        if (!cast.child) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR post projection cast child is null");
          }
          return false;
        }
        base = cast.child.get();
      }

      idx_t out_idx = DConstants::INVALID_INDEX;
      if (base->GetExpressionClass() == ExpressionClass::BOUND_REF) {
        auto &ref = base->Cast<BoundReferenceExpression>();
        out_idx = NumericCast<idx_t>(ref.index);
      } else if (base->GetExpressionClass() ==
                 ExpressionClass::BOUND_COLUMN_REF) {
        auto &colref = base->Cast<BoundColumnRefExpression>();
        if (colref.binding.table_index == aggregate.group_index) {
          out_idx = NumericCast<idx_t>(colref.binding.column_index);
        } else if (colref.binding.table_index == aggregate.aggregate_index) {
          out_idx =
              groups.size() + NumericCast<idx_t>(colref.binding.column_index);
        }
      }
      if (out_idx == DConstants::INVALID_INDEX || out_idx != i ||
          out_idx >= output_names.size()) {
        if (ExecIrFailfast()) {
          throw IOException(
              "ExecIR post projection is not an identity mapping");
        }
        return false;
      }

      auto name = expr_ptr->GetName();
      if (name.empty()) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR post projection output name is empty");
        }
        return false;
      }
      output_names[out_idx] = name;
      if (out_idx < groups.size()) {
        groups[out_idx].output_name = name;
      } else {
        aggs[out_idx - groups.size()].output_name = name;
      }
    }
  }

  vector<std::tuple<string, bool, bool>> order_by;
  if (order) {
    order_by.reserve(order->orders.size());
    for (auto &node : order->orders) {
      if (!node.expression || node.expression->HasParameter() ||
          node.expression->IsVolatile() || node.expression->CanThrow()) {
        if (ExecIrFailfast()) {
          throw IOException("ExecIR order by expression not pushdownable");
        }
        return false;
      }

      string name;
      if (node.expression->GetExpressionClass() == ExpressionClass::BOUND_REF) {
        auto &ref = node.expression->Cast<BoundReferenceExpression>();
        auto idx = NumericCast<idx_t>(ref.index);
        if (idx >= output_names.size()) {
          if (ExecIrFailfast()) {
            throw IOException("ExecIR order by bound ref out of bounds");
          }
          return false;
        }
        name = output_names[idx];
      } else if (node.expression->GetExpressionClass() ==
                 ExpressionClass::BOUND_COLUMN_REF) {
        auto &colref = node.expression->Cast<BoundColumnRefExpression>();
        if (colref.binding.table_index == aggregate.group_index) {
          if (colref.binding.column_index >= groups.size()) {
            if (ExecIrFailfast()) {
              throw IOException("ExecIR order by group column out of bounds");
            }
            return false;
          }
          name = groups[colref.binding.column_index].output_name;
        } else if (colref.binding.table_index == aggregate.aggregate_index) {
          if (colref.binding.column_index >= aggs.size()) {
            if (ExecIrFailfast()) {
              throw IOException(
                  "ExecIR order by aggregate column out of bounds");
            }
            return false;
          }
          name = aggs[colref.binding.column_index].output_name;
        } else if (post_projection &&
                   colref.binding.table_index == post_projection->table_index) {
          auto idx = NumericCast<idx_t>(colref.binding.column_index);
          if (idx >= output_names.size()) {
            if (ExecIrFailfast()) {
              throw IOException(
                  "ExecIR order by post projection column out of bounds");
            }
            return false;
          }
          name = output_names[idx];
        } else {
          if (ExecIrFailfast()) {
            throw IOException(
                "ExecIR order by column ref uses unexpected binding");
          }
          return false;
        }
      } else {
        if (ExecIrFailfast()) {
          throw IOException(
              "ExecIR order by expression class not supported: " +
              ExpressionClassToString(node.expression->GetExpressionClass()));
        }
        return false;
      }

      bool asc = node.type == OrderType::ASCENDING;
      bool nulls_first = node.null_order == OrderByNullType::NULLS_FIRST;
      order_by.emplace_back(std::move(name), asc, nulls_first);
    }
  }

  if (order_by.size() >
      NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(order_by.size()));
  for (auto &entry : order_by) {
    WriteString(out_exec_ir, std::get<0>(entry));
    WriteU8(out_exec_ir, std::get<1>(entry) ? 1 : 0);
    WriteU8(out_exec_ir, std::get<2>(entry) ? 1 : 0);
  }

  return true;
}

} // namespace duckdb
