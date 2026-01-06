#include "lance_exec_ir.hpp"

#include "lance_filter_ir.hpp"
#include "lance_scan_bind_data.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <algorithm>
#include <cstring>
#include <unordered_set>

namespace duckdb {

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
};

enum class ExecIrLiteralTag : uint8_t {
  NULL_ = 0,
  INT64 = 1,
  DOUBLE = 2,
  BOOL = 3,
  STRING = 4,
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

static bool TryEncodeExecExpr(const LogicalGet &scan_get,
                              const LanceScanBindData &scan_bind,
                              const vector<unique_ptr<Expression>> *projection_exprs,
                              const Expression &expr, string &out,
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
  default:
    return false;
  }
}

static bool TryEncodeExecExpr(const LogicalGet &scan_get,
                              const LanceScanBindData &scan_bind,
                              const vector<unique_ptr<Expression>> *projection_exprs,
                              const Expression &expr, string &out,
                              unordered_set<idx_t> &out_col_idxs) {
  switch (expr.GetExpressionClass()) {
  case ExpressionClass::BOUND_COLUMN_REF: {
    auto &colref = expr.Cast<BoundColumnRefExpression>();
    if (colref.binding.table_index != scan_get.table_index) {
      if (!projection_exprs || colref.binding.column_index >= projection_exprs->size() ||
          !(*projection_exprs)[colref.binding.column_index]) {
        return false;
      }
      return TryEncodeExecExpr(scan_get, scan_bind, nullptr,
                               *(*projection_exprs)[colref.binding.column_index], out,
                               out_col_idxs);
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
  case ExpressionClass::BOUND_REF: {
    auto &ref = expr.Cast<BoundReferenceExpression>();
    if (projection_exprs && ref.index < projection_exprs->size() &&
        (*projection_exprs)[ref.index] &&
        (*projection_exprs)[ref.index].get() != &expr) {
      return TryEncodeExecExpr(scan_get, scan_bind, nullptr,
                               *(*projection_exprs)[ref.index], out,
                               out_col_idxs);
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
    if (!TryEncodeExecExpr(scan_get, scan_bind, projection_exprs,
                           *fn.children[0], out,
                           out_col_idxs)) {
      return false;
    }
    if (!TryEncodeExecExpr(scan_get, scan_bind, projection_exprs,
                           *fn.children[1], out,
                           out_col_idxs)) {
      return false;
    }
    return true;
  }
  default:
    return false;
  }
}

bool TryEncodeLanceExecIRv1(const LogicalGet &scan_get,
                            const LanceScanBindData &scan_bind,
                            const string &filter_ir_msg,
                            const vector<idx_t> &extra_scan_col_ids,
                            const vector<unique_ptr<Expression>> *projection_exprs,
                            const LogicalAggregate &aggregate,
                            string &out_exec_ir) {
  if (!aggregate.groups.empty()) {
    return false;
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

  struct EncodedAgg {
    ExecIrAggFunc func;
    string output_name;
    string encoded_args;
    string output_type_hint;
    uint32_t arg_count;
  };

  vector<EncodedAgg> aggs;
  aggs.reserve(aggregate.expressions.size());

  for (auto &expr : aggregate.expressions) {
    if (!expr) {
      return false;
    }
    ExecIrAggFunc func;
    const Expression *arg_expr = nullptr;
    bool count_star = false;

    if (expr->GetExpressionClass() == ExpressionClass::BOUND_AGGREGATE) {
      auto &agg = expr->Cast<BoundAggregateExpression>();
      if (agg.IsDistinct() || agg.filter || agg.order_bys) {
        return false;
      }
      if (!TryMapAggFunc(agg, func)) {
        return false;
      }
      if (func == ExecIrAggFunc::COUNT_STAR) {
        if (!agg.children.empty()) {
          return false;
        }
        count_star = true;
      } else {
        if (agg.children.size() != 1) {
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
        func = fn.children.empty() ? ExecIrAggFunc::COUNT_STAR : ExecIrAggFunc::COUNT;
      } else if (name == "min") {
        func = ExecIrAggFunc::MIN;
      } else if (name == "max") {
        func = ExecIrAggFunc::MAX;
      } else if (name == "avg" || name == "average") {
        func = ExecIrAggFunc::AVG;
      } else {
        return false;
      }
      if (func == ExecIrAggFunc::COUNT_STAR) {
        if (!fn.children.empty()) {
          return false;
        }
        count_star = true;
      } else {
        if (fn.children.size() != 1) {
          return false;
        }
        arg_expr = fn.children[0].get();
      }
    } else {
      return false;
    }

    EncodedAgg enc;
    enc.func = func;
    enc.output_name = expr->GetName();
    if (!TryEncodeOutputTypeHint(expr->return_type, enc.output_type_hint)) {
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
      if (!TryEncodeExecExpr(scan_get, scan_bind, projection_exprs,
                             *arg_expr, args_buf, tmp_idxs)) {
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
      return false;
    }
    auto &name = scan_bind.names[idx];
    if (name.empty()) {
      return false;
    }
    scan_projection.push_back(name);
  }

  out_exec_ir.clear();
  out_exec_ir.reserve(64 + filter_ir_msg.size() +
                      scan_projection.size() * 16 +
                      aggregate.expressions.size() * 32);

  WriteBytes(out_exec_ir, reinterpret_cast<const_data_ptr_t>("LEX1"), 4);
  WriteU32(out_exec_ir, 2); // version
  WriteU32(out_exec_ir, 0); // reserved flags

  if (filter_ir_msg.size() >
      NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(filter_ir_msg.size()));
  WriteBytes(out_exec_ir, reinterpret_cast<const_data_ptr_t>(filter_ir_msg.data()),
             filter_ir_msg.size());

  if (scan_projection.size() >
      NumericCast<size_t>(NumericLimits<uint32_t>::Maximum())) {
    return false;
  }
  WriteU32(out_exec_ir, NumericCast<uint32_t>(scan_projection.size()));
  for (auto &name : scan_projection) {
    WriteString(out_exec_ir, name);
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
      WriteBytes(out_exec_ir,
                 reinterpret_cast<const_data_ptr_t>(agg.output_type_hint.data()),
                 agg.output_type_hint.size());
    }
  }

  return true;
}

} // namespace duckdb
