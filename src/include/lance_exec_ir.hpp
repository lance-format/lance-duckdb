#pragma once

#include "duckdb.hpp"

#include <cstdint>
#include <string>

namespace duckdb {

class ClientContext;
class LogicalAggregate;
class LogicalGet;
class LogicalOrder;
class LogicalProjection;
struct LanceScanBindData;

// An aggregate whose result the exec plan cannot compute with DuckDB's type
// and rounding. Lance returns exact intermediate columns instead, and DuckDB
// finishes them (BuildLanceExecFinish).
struct LanceExecIRFinish {
  enum class Kind : uint8_t {
    // SUM over SMALLINT, INTEGER or BIGINT. Lance sums in DECIMAL(38, 0)
    // (DataFusion's integer SUM wraps on overflow), one column.
    INTEGER_SUM,
    // AVG over SMALLINT, INTEGER, BIGINT or DECIMAL(p <= 18) that DuckDB keeps
    // as AVG. DataFusion's avg rounds a DECIMAL(p, s) mean to s + 4 digits
    // and sums integers in Float64, so Lance returns the exact sum and the
    // count, two columns, and DuckDB divides them as AVG's finalize does.
    AVG,
  };
  Kind kind;
  // Index into LogicalAggregate::expressions.
  idx_t aggregate_index;
  // The type of the aggregate's argument.
  LogicalType input_type;
  // DECIMAL(38, s): the type of the sum column the exec plan returns.
  LogicalType sum_type;
};

// ExecIR is an internal, versioned binary message used by the optimizer
// rewrite to describe a small, pushdownable logical subtree executed in Rust.
//
// Magic: "LEX1"
//
// Version 3 scope:
// - scan_projection (by physical column name)
// - filter_ir bytes (reuses existing FilterIR message)
// - group keys: ColumnRef only, no FLOAT/DOUBLE keys
// - aggregates: sum/min/max/avg/count/count(*) (no DISTINCT, FILTER, ORDER
// BY-in-agg, no SUM/AVG over DECIMAL(p > 18), whose sum can overflow, and
// integer SUM/AVG only over plain columns, since DataFusion's integer
// arithmetic wraps). `out_finishes` lists the aggregates encoded as exact
// intermediates; an AVG takes two output columns.
// - ORDER BY on output columns (group keys / aggregates other than AVGs in
// `out_finishes`)
// - expressions: ColumnRef, Constant, Binary(+,-,*,/), Cast(type_hint, expr)
//
bool TryEncodeLanceExecIRv1(
    const LogicalGet &scan_get, const LanceScanBindData &scan_bind,
    const string &filter_ir_msg, const vector<idx_t> &extra_scan_col_ids,
    const vector<const vector<unique_ptr<Expression>> *> &projection_stack,
    const LogicalAggregate &aggregate, const LogicalOrder *order,
    const LogicalProjection *post_projection, string &out_exec_ir,
    vector<LanceExecIRFinish> &out_finishes);

// The aggregate result DuckDB computes from the exec plan's columns: `sum`
// (`finish.sum_type`) and, for an AVG, `count` (BIGINT).
unique_ptr<Expression> BuildLanceExecFinish(ClientContext &context,
                                            const LanceExecIRFinish &finish,
                                            unique_ptr<Expression> sum,
                                            unique_ptr<Expression> count);

} // namespace duckdb
