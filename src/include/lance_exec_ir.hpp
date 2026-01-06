#pragma once

#include "duckdb.hpp"

#include <cstdint>
#include <string>

namespace duckdb {

class LogicalAggregate;
class LogicalGet;
struct LanceScanBindData;

// ExecIR v1 is an internal, versioned binary message used by the optimizer
// rewrite to describe a small, pushdownable logical subtree executed in Rust.
//
// Magic: "LEX1"
// Version: 1
//
// Current v1 scope (P0):
// - scan_projection (by physical column name)
// - filter_ir bytes (reuses existing FilterIR message)
// - global aggregates (no GROUP BY)
// - aggregate argument expressions: ColumnRef, Constant, Binary(+,-,*,/)
bool TryEncodeLanceExecIRv1(const LogicalGet &scan_get,
                            const LanceScanBindData &scan_bind,
                            const string &filter_ir_msg,
                            const vector<idx_t> &extra_scan_col_ids,
                            const vector<unique_ptr<Expression>> *projection_exprs,
                            const LogicalAggregate &aggregate,
                            string &out_exec_ir);

} // namespace duckdb
