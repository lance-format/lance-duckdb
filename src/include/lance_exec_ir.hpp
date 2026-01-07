#pragma once

#include "duckdb.hpp"

#include <cstdint>
#include <string>

namespace duckdb {

class LogicalAggregate;
class LogicalGet;
class LogicalOrder;
class LogicalProjection;
struct LanceScanBindData;

// ExecIR is an internal, versioned binary message used by the optimizer
// rewrite to describe a small, pushdownable logical subtree executed in Rust.
//
// Magic: "LEX1"
//
// Version 3 scope:
// - scan_projection (by physical column name)
// - filter_ir bytes (reuses existing FilterIR message)
// - group keys: ColumnRef only
// - aggregates: sum/min/max/avg/count/count(*) (no DISTINCT, FILTER, ORDER
// BY-in-agg)
// - ORDER BY on output columns (group keys / aggregates)
// - expressions: ColumnRef, Constant, Binary(+,-,*,/), Cast(type_hint, expr)
//
bool TryEncodeLanceExecIRv1(
    const LogicalGet &scan_get, const LanceScanBindData &scan_bind,
    const string &filter_ir_msg, const vector<idx_t> &extra_scan_col_ids,
    const vector<const vector<unique_ptr<Expression>> *> &projection_stack,
    const LogicalAggregate &aggregate, const LogicalOrder *order,
    const LogicalProjection *post_projection, string &out_exec_ir);

} // namespace duckdb
