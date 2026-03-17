#include "lance_expr_ir.hpp"

#include "lance_scan_bind_data.hpp"

#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/common/types/date.hpp"
#include "duckdb/common/types/decimal.hpp"
#include "duckdb/common/types/hugeint.hpp"
#include "duckdb/common/types/timestamp.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

namespace duckdb {

bool LanceExprIRSupportsLogicalType(const LogicalType &type) {
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
  case LogicalTypeId::DATE:
  case LogicalTypeId::TIMESTAMP:
  case LogicalTypeId::TIMESTAMP_SEC:
  case LogicalTypeId::TIMESTAMP_MS:
  case LogicalTypeId::TIMESTAMP_NS:
  case LogicalTypeId::TIMESTAMP_TZ:
  case LogicalTypeId::DECIMAL:
  case LogicalTypeId::STRUCT:
    return true;
  default:
    return false;
  }
}

static constexpr char LANCE_EXPR_IR_MAGIC[] = {'L', 'U', 'E', '1'};

enum class LanceUpdateExprIRTag : uint8_t {
  COLUMN_REF = 1,
  LITERAL = 2,
  CAST = 3,
  BINARY = 4,
  COMPARISON = 5,
  CONJUNCTION = 6,
  NOT = 7,
  SCALAR_FUNCTION = 8,
  CASE = 9,
  IS_NULL = 10,
  IS_NOT_NULL = 11,
  IN_LIST = 12,
  LIKE = 13,
  REGEXP = 14,
};

enum class LanceUpdateExprIRLiteralTag : uint8_t {
  NULL_VALUE = 0,
  BOOL = 1,
  I64 = 2,
  U64 = 3,
  F32 = 4,
  F64 = 5,
  STRING = 6,
  DATE32 = 7,
  TIMESTAMP = 8,
  DECIMAL128 = 9,
};

enum class LanceUpdateExprIRTimestampUnit : uint8_t {
  SECOND = 0,
  MILLISECOND = 1,
  MICROSECOND = 2,
  NANOSECOND = 3,
};

enum class LanceUpdateExprIRTypeTag : uint8_t {
  BOOL = 1,
  INT8 = 2,
  INT16 = 3,
  INT32 = 4,
  INT64 = 5,
  UINT8 = 6,
  UINT16 = 7,
  UINT32 = 8,
  UINT64 = 9,
  FLOAT32 = 10,
  FLOAT64 = 11,
  VARCHAR = 12,
  DATE32 = 13,
  TIMESTAMP = 14,
  DECIMAL128 = 15,
};

enum class LanceUpdateExprIRBinaryOp : uint8_t {
  ADD = 1,
  SUB = 2,
  MUL = 3,
  DIV = 4,
};

enum class LanceUpdateExprIRComparisonOp : uint8_t {
  EQ = 0,
  NOT_EQ = 1,
  LT = 2,
  LT_EQ = 3,
  GT = 4,
  GT_EQ = 5,
  DISTINCT_FROM = 6,
  NOT_DISTINCT_FROM = 7,
};

enum class LanceUpdateExprIRConjunctionOp : uint8_t {
  AND = 1,
  OR = 2,
};

static constexpr uint8_t LANCE_EXPR_IR_LIKE_FLAG_CASE_INSENSITIVE = 1;
static constexpr uint8_t LANCE_EXPR_IR_LIKE_FLAG_HAS_ESCAPE = 2;

static void AppendU8(string &out, uint8_t v) {
  out.push_back(static_cast<char>(v));
}

static void AppendU32(string &out, uint32_t v) {
  uint8_t buf[4];
  buf[0] = static_cast<uint8_t>(v & 0xFF);
  buf[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
  buf[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
  buf[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void AppendI32(string &out, int32_t v) {
  AppendU32(out, static_cast<uint32_t>(v));
}

static void AppendU64(string &out, uint64_t v) {
  uint8_t buf[8];
  for (idx_t i = 0; i < 8; i++) {
    buf[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
  }
  out.append(reinterpret_cast<const char *>(buf), sizeof(buf));
}

static void AppendI64(string &out, int64_t v) {
  AppendU64(out, static_cast<uint64_t>(v));
}

static void AppendI128(string &out, hugeint_t v) {
  AppendU64(out, v.lower);
  AppendU64(out, static_cast<uint64_t>(v.upper));
}

static void AppendF32(string &out, float v) {
  uint32_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  AppendU32(out, bits);
}

static void AppendF64(string &out, double v) {
  uint64_t bits = 0;
  memcpy(&bits, &v, sizeof(bits));
  AppendU64(out, bits);
}

static void AppendBytes(string &out, const string &bytes) { out.append(bytes); }

static void AppendLenPrefixed(string &out, const string &bytes) {
  if (bytes.size() > std::numeric_limits<uint32_t>::max()) {
    throw InvalidInputException("UpdateExprIR payload too large");
  }
  AppendU32(out, static_cast<uint32_t>(bytes.size()));
  AppendBytes(out, bytes);
}

static void AppendLenPrefixedString(string &out, const string &value) {
  AppendLenPrefixed(out, value);
}

static string EncodeMessage(const string &root) {
  string out;
  out.append(LANCE_EXPR_IR_MAGIC, sizeof(LANCE_EXPR_IR_MAGIC));
  out.append(root);
  return out;
}

static void EncodeColumnRef(const vector<string> &segments, string &out) {
  if (segments.empty()) {
    throw NotImplementedException(
        "Lance expression IR does not support empty column paths");
  }
  for (auto &segment : segments) {
    if (segment.empty()) {
      throw NotImplementedException(
          "Lance expression IR does not support empty column path segments");
    }
  }
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::COLUMN_REF));
  if (segments.size() > std::numeric_limits<uint32_t>::max()) {
    throw InvalidInputException("ExprIR column path too deep");
  }
  AppendU32(out, static_cast<uint32_t>(segments.size()));
  for (auto &segment : segments) {
    AppendLenPrefixedString(out, segment);
  }
}

static void EncodeColumnRef(const string &name, string &out) {
  EncodeColumnRef(vector<string>{name}, out);
}

static void EncodeLiteral(const Value &value, string &out) {
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::LITERAL));

  if (value.IsNull()) {
    AppendU8(out,
             static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::NULL_VALUE));
    return;
  }

  switch (value.type().id()) {
  case LogicalTypeId::BOOLEAN:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::BOOL));
    AppendU8(out, value.GetValue<bool>() ? 1 : 0);
    return;
  case LogicalTypeId::TINYINT:
  case LogicalTypeId::SMALLINT:
  case LogicalTypeId::INTEGER:
  case LogicalTypeId::BIGINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::I64));
    AppendI64(out, value.GetValue<int64_t>());
    return;
  case LogicalTypeId::UTINYINT:
  case LogicalTypeId::USMALLINT:
  case LogicalTypeId::UINTEGER:
  case LogicalTypeId::UBIGINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::U64));
    AppendU64(out, value.GetValue<uint64_t>());
    return;
  case LogicalTypeId::FLOAT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::F32));
    AppendF32(out, value.GetValue<float>());
    return;
  case LogicalTypeId::DOUBLE:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::F64));
    AppendF64(out, value.GetValue<double>());
    return;
  case LogicalTypeId::VARCHAR:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::STRING));
    AppendLenPrefixedString(out, value.GetValue<string>());
    return;
  case LogicalTypeId::DATE: {
    auto d = value.GetValue<date_t>();
    if (!Date::IsFinite(d)) {
      throw NotImplementedException(
          "Lance UPDATE does not support infinite DATE literals");
    }
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::DATE32));
    AppendI32(out, d.days);
    return;
  }
  case LogicalTypeId::TIMESTAMP:
  case LogicalTypeId::TIMESTAMP_SEC:
  case LogicalTypeId::TIMESTAMP_MS:
  case LogicalTypeId::TIMESTAMP_NS:
  case LogicalTypeId::TIMESTAMP_TZ: {
    auto v = value.GetValue<int64_t>();
    if (!Timestamp::IsFinite(timestamp_t(v))) {
      throw NotImplementedException(
          "Lance UPDATE does not support infinite TIMESTAMP literals");
    }
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::TIMESTAMP));
    LanceUpdateExprIRTimestampUnit unit;
    switch (value.type().id()) {
    case LogicalTypeId::TIMESTAMP_SEC:
      unit = LanceUpdateExprIRTimestampUnit::SECOND;
      break;
    case LogicalTypeId::TIMESTAMP_MS:
      unit = LanceUpdateExprIRTimestampUnit::MILLISECOND;
      break;
    case LogicalTypeId::TIMESTAMP_NS:
      unit = LanceUpdateExprIRTimestampUnit::NANOSECOND;
      break;
    case LogicalTypeId::TIMESTAMP:
    case LogicalTypeId::TIMESTAMP_TZ:
    default:
      unit = LanceUpdateExprIRTimestampUnit::MICROSECOND;
      break;
    }
    AppendU8(out, static_cast<uint8_t>(unit));
    AppendI64(out, v);
    return;
  }
  case LogicalTypeId::DECIMAL: {
    auto precision = DecimalType::GetWidth(value.type());
    auto scale_u8 = DecimalType::GetScale(value.type());
    if (scale_u8 > static_cast<uint8_t>(std::numeric_limits<int8_t>::max())) {
      throw NotImplementedException(
          "Lance UPDATE does not support DECIMAL scale > 127");
    }
    auto scale = static_cast<int8_t>(scale_u8);
    auto str = value.ToString();
    hugeint_t dec(0);
    idx_t pos = 0;
    bool neg = false;
    if (pos < str.size() && (str[pos] == '-' || str[pos] == '+')) {
      neg = (str[pos] == '-');
      pos++;
    }
    hugeint_t acc(0);
    bool seen_dot = false;
    uint8_t frac_digits = 0;
    for (; pos < str.size(); pos++) {
      auto c = str[pos];
      if (c == '.') {
        if (seen_dot) {
          throw NotImplementedException(
              "Lance UPDATE does not support malformed DECIMAL literals");
        }
        seen_dot = true;
        continue;
      }
      if (c < '0' || c > '9') {
        throw NotImplementedException(
            "Lance UPDATE does not support malformed DECIMAL literals");
      }
      acc = acc * hugeint_t(10) + hugeint_t(static_cast<int64_t>(c - '0'));
      if (seen_dot) {
        if (frac_digits >= scale_u8) {
          throw NotImplementedException(
              "Lance UPDATE does not support DECIMAL literals with excess "
              "fractional digits");
        }
        frac_digits++;
      }
    }
    while (frac_digits < scale_u8) {
      acc = acc * hugeint_t(10);
      frac_digits++;
    }
    dec = neg ? -acc : acc;
    AppendU8(out,
             static_cast<uint8_t>(LanceUpdateExprIRLiteralTag::DECIMAL128));
    AppendU8(out, precision);
    AppendU8(out, static_cast<uint8_t>(scale));
    AppendI128(out, dec);
    return;
  }
  default:
    throw NotImplementedException("Lance UPDATE does not support literal type "
                                  "%s in SET expressions",
                                  value.type().ToString());
  }
}

static void EncodeTypeHint(const LogicalType &type, string &out) {
  out.clear();
  switch (type.id()) {
  case LogicalTypeId::BOOLEAN:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::BOOL));
    return;
  case LogicalTypeId::TINYINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::INT8));
    return;
  case LogicalTypeId::SMALLINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::INT16));
    return;
  case LogicalTypeId::INTEGER:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::INT32));
    return;
  case LogicalTypeId::BIGINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::INT64));
    return;
  case LogicalTypeId::UTINYINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::UINT8));
    return;
  case LogicalTypeId::USMALLINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::UINT16));
    return;
  case LogicalTypeId::UINTEGER:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::UINT32));
    return;
  case LogicalTypeId::UBIGINT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::UINT64));
    return;
  case LogicalTypeId::FLOAT:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::FLOAT32));
    return;
  case LogicalTypeId::DOUBLE:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::FLOAT64));
    return;
  case LogicalTypeId::VARCHAR:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::VARCHAR));
    return;
  case LogicalTypeId::DATE:
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::DATE32));
    return;
  case LogicalTypeId::TIMESTAMP:
  case LogicalTypeId::TIMESTAMP_SEC:
  case LogicalTypeId::TIMESTAMP_MS:
  case LogicalTypeId::TIMESTAMP_NS:
  case LogicalTypeId::TIMESTAMP_TZ: {
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::TIMESTAMP));
    LanceUpdateExprIRTimestampUnit unit;
    switch (type.id()) {
    case LogicalTypeId::TIMESTAMP_SEC:
      unit = LanceUpdateExprIRTimestampUnit::SECOND;
      break;
    case LogicalTypeId::TIMESTAMP_MS:
      unit = LanceUpdateExprIRTimestampUnit::MILLISECOND;
      break;
    case LogicalTypeId::TIMESTAMP_NS:
      unit = LanceUpdateExprIRTimestampUnit::NANOSECOND;
      break;
    case LogicalTypeId::TIMESTAMP:
    case LogicalTypeId::TIMESTAMP_TZ:
    default:
      unit = LanceUpdateExprIRTimestampUnit::MICROSECOND;
      break;
    }
    AppendU8(out, static_cast<uint8_t>(unit));
    return;
  }
  case LogicalTypeId::DECIMAL: {
    auto precision = DecimalType::GetWidth(type);
    auto scale_u8 = DecimalType::GetScale(type);
    if (scale_u8 > static_cast<uint8_t>(std::numeric_limits<int8_t>::max())) {
      throw NotImplementedException(
          "Lance UPDATE does not support DECIMAL scale > 127");
    }
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTypeTag::DECIMAL128));
    AppendU8(out, precision);
    AppendU8(out, static_cast<uint8_t>(scale_u8));
    return;
  }
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support cast target type %s in SET expressions",
        type.ToString());
  }
}

static void EncodeComparisonOp(ExpressionType type, uint8_t &out) {
  switch (type) {
  case ExpressionType::COMPARE_EQUAL:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::EQ);
    return;
  case ExpressionType::COMPARE_NOTEQUAL:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::NOT_EQ);
    return;
  case ExpressionType::COMPARE_LESSTHAN:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::LT);
    return;
  case ExpressionType::COMPARE_LESSTHANOREQUALTO:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::LT_EQ);
    return;
  case ExpressionType::COMPARE_GREATERTHAN:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::GT);
    return;
  case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::GT_EQ);
    return;
  case ExpressionType::COMPARE_DISTINCT_FROM:
    out = static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::DISTINCT_FROM);
    return;
  case ExpressionType::COMPARE_NOT_DISTINCT_FROM:
    out =
        static_cast<uint8_t>(LanceUpdateExprIRComparisonOp::NOT_DISTINCT_FROM);
    return;
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support comparison operator %s in SET "
        "expressions",
        ExpressionTypeToString(type));
  }
}

static LanceUpdateExprIRBinaryOp MapBinaryOpName(const string &name) {
  auto lowered = StringUtil::Lower(name);
  if (lowered == "+" || lowered == "add") {
    return LanceUpdateExprIRBinaryOp::ADD;
  }
  if (lowered == "-" || lowered == "subtract") {
    return LanceUpdateExprIRBinaryOp::SUB;
  }
  if (lowered == "*" || lowered == "multiply") {
    return LanceUpdateExprIRBinaryOp::MUL;
  }
  if (lowered == "/" || lowered == "divide") {
    return LanceUpdateExprIRBinaryOp::DIV;
  }
  throw NotImplementedException(
      "Lance UPDATE does not support scalar function %s in SET expressions",
      name);
}

static void EncodeUpdateExprNode(const LogicalGet &scan_get,
                                 const LanceScanBindData &scan_bind,
                                 const LogicalOperator &expression_scope,
                                 const Expression &expr, string &out);

static void EncodeUnaryNode(LanceUpdateExprIRTag tag, const string &child,
                            string &out) {
  out.clear();
  AppendU8(out, static_cast<uint8_t>(tag));
  AppendLenPrefixed(out, child);
}

static void EncodeBinaryNode(LanceUpdateExprIRBinaryOp op, const string &lhs,
                             const string &rhs, string &out) {
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::BINARY));
  AppendU8(out, static_cast<uint8_t>(op));
  AppendLenPrefixed(out, lhs);
  AppendLenPrefixed(out, rhs);
}

static void EncodeComparisonNode(uint8_t op, const string &lhs,
                                 const string &rhs, string &out) {
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::COMPARISON));
  AppendU8(out, op);
  AppendLenPrefixed(out, lhs);
  AppendLenPrefixed(out, rhs);
}

static void EncodeConjunctionNode(LanceUpdateExprIRConjunctionOp op,
                                  const vector<string> &children, string &out) {
  if (children.empty()) {
    throw InternalException("UpdateExprIR conjunction requires children");
  }
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::CONJUNCTION));
  AppendU8(out, static_cast<uint8_t>(op));
  if (children.size() > std::numeric_limits<uint32_t>::max()) {
    throw InvalidInputException("UpdateExprIR conjunction too large");
  }
  AppendU32(out, static_cast<uint32_t>(children.size()));
  for (auto &child : children) {
    AppendLenPrefixed(out, child);
  }
}

static void EncodeFunctionNode(const string &name, const vector<string> &args,
                               string &out) {
  if (args.size() > std::numeric_limits<uint32_t>::max()) {
    throw InvalidInputException("UpdateExprIR function arity too large");
  }
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::SCALAR_FUNCTION));
  AppendLenPrefixedString(out, StringUtil::Lower(name));
  AppendU32(out, static_cast<uint32_t>(args.size()));
  for (auto &arg : args) {
    AppendLenPrefixed(out, arg);
  }
}

static void EncodeLikeNode(bool case_insensitive, bool has_escape,
                           uint8_t escape_char, const string &expr,
                           const string &pattern, string &out) {
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::LIKE));
  uint8_t flags = 0;
  if (case_insensitive) {
    flags |= LANCE_EXPR_IR_LIKE_FLAG_CASE_INSENSITIVE;
  }
  if (has_escape) {
    flags |= LANCE_EXPR_IR_LIKE_FLAG_HAS_ESCAPE;
  }
  AppendU8(out, flags);
  AppendLenPrefixed(out, expr);
  AppendLenPrefixed(out, pattern);
  if (has_escape) {
    AppendU8(out, escape_char);
  }
}

static void EncodeRegexpNode(uint8_t mode, bool has_flags, const string &expr,
                             const string &pattern, const string &flags,
                             string &out) {
  if (mode != LANCE_EXPR_IR_REGEXP_MODE_PARTIAL_MATCH &&
      mode != LANCE_EXPR_IR_REGEXP_MODE_FULL_MATCH) {
    throw InvalidInputException("ExprIR regexp mode is invalid");
  }
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::REGEXP));
  AppendU8(out, mode);
  AppendU8(out, has_flags ? 1 : 0);
  AppendLenPrefixed(out, expr);
  AppendLenPrefixed(out, pattern);
  if (has_flags) {
    AppendLenPrefixed(out, flags);
  }
}

static void EncodeCaseNode(const vector<std::pair<string, string>> &checks,
                           const string *else_ir, string &out) {
  if (checks.empty()) {
    throw NotImplementedException(
        "Lance UPDATE does not support CASE without WHEN clauses");
  }
  if (checks.size() > std::numeric_limits<uint32_t>::max()) {
    throw InvalidInputException("UpdateExprIR CASE too large");
  }
  out.clear();
  AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::CASE));
  AppendU32(out, static_cast<uint32_t>(checks.size()));
  for (auto &check : checks) {
    AppendLenPrefixed(out, check.first);
    AppendLenPrefixed(out, check.second);
  }
  AppendU8(out, else_ir ? 1 : 0);
  if (else_ir) {
    AppendLenPrefixed(out, *else_ir);
  }
}

bool TryEncodeLanceExprIRColumnRef(const string &name, string &out_ir) {
  try {
    EncodeColumnRef(name, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRColumnRef(const vector<string> &segments,
                                   string &out_ir) {
  try {
    EncodeColumnRef(segments, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRLiteral(const Value &value, string &out_ir) {
  try {
    EncodeLiteral(value, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRComparisonOp(ExpressionType type, uint8_t &out_op) {
  try {
    EncodeComparisonOp(type, out_op);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRAnd(const vector<string> &children, string &out_ir) {
  try {
    EncodeConjunctionNode(LanceUpdateExprIRConjunctionOp::AND, children,
                          out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIROr(const vector<string> &children, string &out_ir) {
  try {
    EncodeConjunctionNode(LanceUpdateExprIRConjunctionOp::OR, children, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRNot(const string &child, string &out_ir) {
  try {
    EncodeUnaryNode(LanceUpdateExprIRTag::NOT, child, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRIsNull(const string &child, string &out_ir) {
  try {
    EncodeUnaryNode(LanceUpdateExprIRTag::IS_NULL, child, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRIsNotNull(const string &child, string &out_ir) {
  try {
    EncodeUnaryNode(LanceUpdateExprIRTag::IS_NOT_NULL, child, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRComparison(uint8_t op, const string &left,
                                    const string &right, string &out_ir) {
  try {
    EncodeComparisonNode(op, left, right, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRInList(bool negated, const string &expr,
                                const vector<string> &list, string &out_ir) {
  try {
    out_ir.clear();
    AppendU8(out_ir, static_cast<uint8_t>(LanceUpdateExprIRTag::IN_LIST));
    AppendU8(out_ir, negated ? 1 : 0);
    AppendLenPrefixed(out_ir, expr);
    if (list.size() > std::numeric_limits<uint32_t>::max()) {
      return false;
    }
    AppendU32(out_ir, static_cast<uint32_t>(list.size()));
    for (auto &item : list) {
      AppendLenPrefixed(out_ir, item);
    }
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRScalarFunction(const string &name,
                                        const vector<string> &args,
                                        string &out_ir) {
  try {
    EncodeFunctionNode(name, args, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRLike(bool case_insensitive, bool has_escape,
                              uint8_t escape_char, const string &expr,
                              const string &pattern, string &out_ir) {
  try {
    EncodeLikeNode(case_insensitive, has_escape, escape_char, expr, pattern,
                   out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryEncodeLanceExprIRRegexp(uint8_t mode, bool has_flags,
                                const string &expr, const string &pattern,
                                const string &flags, string &out_ir) {
  try {
    EncodeRegexpNode(mode, has_flags, expr, pattern, flags, out_ir);
    return true;
  } catch (...) {
    return false;
  }
}

bool TryWrapLanceExprIRMessage(const string &root_node, string &out_msg) {
  try {
    out_msg = EncodeMessage(root_node);
    return true;
  } catch (...) {
    return false;
  }
}

static string ResolveBaseColumnName(const LogicalGet &scan_get,
                                    const LanceScanBindData &scan_bind,
                                    idx_t column_index) {
  auto &column_ids = scan_get.GetColumnIds();
  if (column_index >= column_ids.size()) {
    throw InternalException(
        "Lance UPDATE expression references invalid column");
  }
  auto &col_index = column_ids[column_index];
  if (col_index.IsVirtualColumn()) {
    throw NotImplementedException("Lance UPDATE does not support expressions "
                                  "referencing virtual columns");
  }
  auto col_id = col_index.GetPrimaryIndex();
  if (col_id >= scan_bind.names.size()) {
    throw InternalException(
        "Lance UPDATE expression column id is out of range");
  }
  auto &name = scan_bind.names[col_id];
  if (name.empty()) {
    throw NotImplementedException(
        "Lance UPDATE does not support unnamed columns in SET expressions");
  }
  return name;
}

static const LogicalOperator &GetSingleChildScope(const LogicalOperator &op) {
  if (op.children.empty() || !op.children[0]) {
    throw InternalException("Lance UPDATE scope operator has no child");
  }
  return *op.children[0];
}

static void EncodeScopeColumnRef(const LogicalGet &scan_get,
                                 const LanceScanBindData &scan_bind,
                                 const LogicalOperator &expression_scope,
                                 idx_t column_index, string &out) {
  switch (expression_scope.type) {
  case LogicalOperatorType::LOGICAL_FILTER:
    return EncodeScopeColumnRef(scan_get, scan_bind,
                                GetSingleChildScope(expression_scope),
                                column_index, out);
  case LogicalOperatorType::LOGICAL_PROJECTION: {
    auto &proj = expression_scope.Cast<LogicalProjection>();
    if (column_index >= proj.expressions.size() ||
        !proj.expressions[column_index]) {
      throw InternalException(
          "Lance UPDATE projection reference is out of range");
    }
    return EncodeUpdateExprNode(scan_get, scan_bind, GetSingleChildScope(proj),
                                *proj.expressions[column_index], out);
  }
  case LogicalOperatorType::LOGICAL_GET:
    return EncodeColumnRef(
        ResolveBaseColumnName(scan_get, scan_bind, column_index), out);
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support this expression scope operator");
  }
}

static void EncodeScopeColumnBinding(const LogicalGet &scan_get,
                                     const LanceScanBindData &scan_bind,
                                     const LogicalOperator &expression_scope,
                                     const ColumnBinding &binding,
                                     string &out) {
  switch (expression_scope.type) {
  case LogicalOperatorType::LOGICAL_FILTER:
    return EncodeScopeColumnBinding(scan_get, scan_bind,
                                    GetSingleChildScope(expression_scope),
                                    binding, out);
  case LogicalOperatorType::LOGICAL_PROJECTION: {
    auto &proj = expression_scope.Cast<LogicalProjection>();
    if (binding.table_index != proj.table_index) {
      throw NotImplementedException(
          "Lance UPDATE does not support cross-scope binding in SET "
          "expressions");
    }
    if (binding.column_index >= proj.expressions.size() ||
        !proj.expressions[binding.column_index]) {
      throw InternalException(
          "Lance UPDATE projection binding is out of range");
    }
    return EncodeUpdateExprNode(scan_get, scan_bind, GetSingleChildScope(proj),
                                *proj.expressions[binding.column_index], out);
  }
  case LogicalOperatorType::LOGICAL_GET:
    if (binding.table_index != scan_get.table_index) {
      throw NotImplementedException(
          "Lance UPDATE does not support non-scan bindings in SET "
          "expressions");
    }
    return EncodeColumnRef(
        ResolveBaseColumnName(scan_get, scan_bind, binding.column_index), out);
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support this expression scope operator");
  }
}

static void EncodeUpdateExprNode(const LogicalGet &scan_get,
                                 const LanceScanBindData &scan_bind,
                                 const LogicalOperator &expression_scope,
                                 const Expression &expr, string &out) {

  switch (expr.GetExpressionClass()) {
  case ExpressionClass::BOUND_COLUMN_REF: {
    auto &ref = expr.Cast<BoundColumnRefExpression>();
    return EncodeScopeColumnBinding(scan_get, scan_bind, expression_scope,
                                    ref.binding, out);
  }
  case ExpressionClass::BOUND_REF: {
    auto &ref = expr.Cast<BoundReferenceExpression>();
    if (!ref.alias.empty()) {
      return EncodeColumnRef(ref.alias, out);
    }
    return EncodeScopeColumnRef(scan_get, scan_bind, expression_scope,
                                ref.index, out);
  }
  case ExpressionClass::BOUND_CONSTANT: {
    auto &constant = expr.Cast<BoundConstantExpression>();
    return EncodeLiteral(constant.value, out);
  }
  case ExpressionClass::BOUND_CAST: {
    auto &cast = expr.Cast<BoundCastExpression>();
    if (!cast.child) {
      throw InternalException("Lance UPDATE cast expression is missing child");
    }
    if (cast.try_cast) {
      throw NotImplementedException(
          "Lance UPDATE does not support TRY_CAST in SET expressions");
    }
    string child_ir;
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *cast.child,
                         child_ir);
    string type_ir;
    EncodeTypeHint(cast.return_type, type_ir);
    out.clear();
    AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::CAST));
    AppendBytes(out, type_ir);
    AppendLenPrefixed(out, child_ir);
    return;
  }
  case ExpressionClass::BOUND_FUNCTION: {
    auto &fn = expr.Cast<BoundFunctionExpression>();
    if (fn.children.size() == 2) {
      auto lowered = StringUtil::Lower(fn.function.name);
      if (lowered == "+" || lowered == "add" || lowered == "-" ||
          lowered == "subtract" || lowered == "*" || lowered == "multiply" ||
          lowered == "/" || lowered == "divide") {
        string lhs_ir, rhs_ir;
        EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                             *fn.children[0], lhs_ir);
        EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                             *fn.children[1], rhs_ir);
        return EncodeBinaryNode(MapBinaryOpName(fn.function.name), lhs_ir,
                                rhs_ir, out);
      }
    }

    vector<string> args;
    args.reserve(fn.children.size());
    for (auto &child : fn.children) {
      if (!child) {
        throw InternalException(
            "Lance UPDATE function expression has null child");
      }
      string child_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *child,
                           child_ir);
      args.push_back(std::move(child_ir));
    }
    return EncodeFunctionNode(fn.function.name, args, out);
  }
  case ExpressionClass::BOUND_OPERATOR: {
    auto &op = expr.Cast<BoundOperatorExpression>();
    if (op.type == ExpressionType::OPERATOR_NOT) {
      if (op.children.size() != 1 || !op.children[0]) {
        throw InternalException("Lance UPDATE NOT expression is malformed");
      }
      string child_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *op.children[0], child_ir);
      return EncodeUnaryNode(LanceUpdateExprIRTag::NOT, child_ir, out);
    }
    if (op.type == ExpressionType::OPERATOR_IS_NULL ||
        op.type == ExpressionType::OPERATOR_IS_NOT_NULL) {
      if (op.children.size() != 1 || !op.children[0]) {
        throw InternalException("Lance UPDATE IS NULL expression is malformed");
      }
      string child_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *op.children[0], child_ir);
      return EncodeUnaryNode(op.type == ExpressionType::OPERATOR_IS_NULL
                                 ? LanceUpdateExprIRTag::IS_NULL
                                 : LanceUpdateExprIRTag::IS_NOT_NULL,
                             child_ir, out);
    }
    if (op.type == ExpressionType::COMPARE_IN ||
        op.type == ExpressionType::COMPARE_NOT_IN) {
      if (op.children.size() < 2 || !op.children[0]) {
        throw InternalException("Lance UPDATE IN expression is malformed");
      }
      string lhs_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *op.children[0], lhs_ir);
      out.clear();
      AppendU8(out, static_cast<uint8_t>(LanceUpdateExprIRTag::IN_LIST));
      AppendU8(out, op.type == ExpressionType::COMPARE_NOT_IN ? 1 : 0);
      AppendLenPrefixed(out, lhs_ir);
      AppendU32(out, static_cast<uint32_t>(op.children.size() - 1));
      for (idx_t i = 1; i < op.children.size(); i++) {
        if (!op.children[i]) {
          throw InternalException(
              "Lance UPDATE IN expression has null list element");
        }
        string item_ir;
        EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                             *op.children[i], item_ir);
        AppendLenPrefixed(out, item_ir);
      }
      return;
    }
    throw NotImplementedException(
        "Lance UPDATE does not support operator %s in SET expressions",
        ExpressionTypeToString(op.type));
  }
  case ExpressionClass::BOUND_COMPARISON: {
    auto &cmp = expr.Cast<BoundComparisonExpression>();
    string lhs_ir, rhs_ir;
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *cmp.left,
                         lhs_ir);
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *cmp.right,
                         rhs_ir);
    uint8_t op = 0;
    EncodeComparisonOp(cmp.type, op);
    return EncodeComparisonNode(op, lhs_ir, rhs_ir, out);
  }
  case ExpressionClass::BOUND_CONJUNCTION: {
    auto &conj = expr.Cast<BoundConjunctionExpression>();
    LanceUpdateExprIRConjunctionOp op;
    if (conj.type == ExpressionType::CONJUNCTION_AND) {
      op = LanceUpdateExprIRConjunctionOp::AND;
    } else if (conj.type == ExpressionType::CONJUNCTION_OR) {
      op = LanceUpdateExprIRConjunctionOp::OR;
    } else {
      throw NotImplementedException(
          "Lance UPDATE does not support conjunction %s in SET expressions",
          ExpressionTypeToString(conj.type));
    }
    vector<string> children;
    children.reserve(conj.children.size());
    for (auto &child : conj.children) {
      if (!child) {
        throw InternalException(
            "Lance UPDATE conjunction expression has null child");
      }
      string child_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *child,
                           child_ir);
      children.push_back(std::move(child_ir));
    }
    return EncodeConjunctionNode(op, children, out);
  }
  case ExpressionClass::BOUND_CASE: {
    auto &case_expr = expr.Cast<BoundCaseExpression>();
    vector<std::pair<string, string>> checks;
    checks.reserve(case_expr.case_checks.size());
    for (auto &check : case_expr.case_checks) {
      if (!check.when_expr || !check.then_expr) {
        throw InternalException("Lance UPDATE CASE expression has null branch");
      }
      string when_ir, then_ir;
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *check.when_expr, when_ir);
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *check.then_expr, then_ir);
      checks.emplace_back(std::move(when_ir), std::move(then_ir));
    }
    string else_ir_storage;
    string *else_ir = nullptr;
    if (case_expr.else_expr) {
      EncodeUpdateExprNode(scan_get, scan_bind, expression_scope,
                           *case_expr.else_expr, else_ir_storage);
      else_ir = &else_ir_storage;
    }
    return EncodeCaseNode(checks, else_ir, out);
  }
  case ExpressionClass::BOUND_BETWEEN: {
    auto &between = expr.Cast<BoundBetweenExpression>();
    string input_ir, lower_ir, upper_ir;
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *between.input,
                         input_ir);
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *between.lower,
                         lower_ir);
    EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, *between.upper,
                         upper_ir);
    uint8_t lower_op = 0;
    uint8_t upper_op = 0;
    EncodeComparisonOp(between.LowerComparisonType(), lower_op);
    EncodeComparisonOp(between.UpperComparisonType(), upper_op);
    string lower_cmp_ir, upper_cmp_ir;
    EncodeComparisonNode(lower_op, input_ir, lower_ir, lower_cmp_ir);
    EncodeComparisonNode(upper_op, input_ir, upper_ir, upper_cmp_ir);
    vector<string> children;
    children.push_back(std::move(lower_cmp_ir));
    children.push_back(std::move(upper_cmp_ir));
    return EncodeConjunctionNode(LanceUpdateExprIRConjunctionOp::AND, children,
                                 out);
  }
  default:
    throw NotImplementedException(
        "Lance UPDATE does not support expression class %s in SET expressions",
        ExpressionClassToString(expr.GetExpressionClass()));
  }
}

void EncodeLanceExprIR(const LogicalGet &scan_get,
                       const LanceScanBindData &scan_bind,
                       const LogicalOperator &expression_scope,
                       const Expression &expr, string &out_ir) {
  string root;
  EncodeUpdateExprNode(scan_get, scan_bind, expression_scope, expr, root);
  out_ir = EncodeMessage(root);
}

} // namespace duckdb
