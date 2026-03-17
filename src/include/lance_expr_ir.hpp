#pragma once

#include "duckdb.hpp"

namespace duckdb {

class LogicalGet;
class LogicalOperator;
struct LanceScanBindData;

bool LanceExprIRSupportsLogicalType(const LogicalType &type);

bool TryEncodeLanceExprIRColumnRef(const string &name, string &out_ir);
bool TryEncodeLanceExprIRColumnRef(const vector<string> &segments,
                                   string &out_ir);
bool TryEncodeLanceExprIRLiteral(const Value &value, string &out_ir);
bool TryEncodeLanceExprIRComparisonOp(ExpressionType type, uint8_t &out_op);
bool TryEncodeLanceExprIRAnd(const vector<string> &children, string &out_ir);
bool TryEncodeLanceExprIROr(const vector<string> &children, string &out_ir);
bool TryEncodeLanceExprIRNot(const string &child, string &out_ir);
bool TryEncodeLanceExprIRIsNull(const string &child, string &out_ir);
bool TryEncodeLanceExprIRIsNotNull(const string &child, string &out_ir);
bool TryEncodeLanceExprIRComparison(uint8_t op, const string &left,
                                    const string &right, string &out_ir);
bool TryEncodeLanceExprIRInList(bool negated, const string &expr,
                                const vector<string> &list, string &out_ir);
bool TryEncodeLanceExprIRScalarFunction(const string &name,
                                        const vector<string> &args,
                                        string &out_ir);
bool TryEncodeLanceExprIRLike(bool case_insensitive, bool has_escape,
                              uint8_t escape_char, const string &expr,
                              const string &pattern, string &out_ir);
bool TryEncodeLanceExprIRRegexp(uint8_t mode, bool has_flags,
                                const string &expr, const string &pattern,
                                const string &flags, string &out_ir);

bool TryWrapLanceExprIRMessage(const string &root_node, string &out_msg);

static constexpr uint8_t LANCE_EXPR_IR_REGEXP_MODE_PARTIAL_MATCH = 0;
static constexpr uint8_t LANCE_EXPR_IR_REGEXP_MODE_FULL_MATCH = 1;

void EncodeLanceExprIR(const LogicalGet &scan_get,
                       const LanceScanBindData &scan_bind,
                       const LogicalOperator &expression_scope,
                       const Expression &expr, string &out_ir);

} // namespace duckdb
