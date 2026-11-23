#pragma once

#include "duckdb.hpp"

namespace duckdb {

struct TableFunctionInitInput;
class LogicalGet;
class Expression;

bool LanceFilterIRSupportsLogicalType(const LogicalType &type);

struct LanceFilterIRBuildResult {
  vector<string> parts;
  bool all_filters_pushed = true;
  bool all_prefilterable_filters_pushed = true;
  //! Set when the scan has dynamic filters that a join attached at execution
  //! time and `exclude_runtime_filters` left them out of `parts`. Callers must
  //! then keep DuckDB's residual filter evaluation enabled so those filters
  //! still run after the scan's output is produced.
  bool has_runtime_filters = false;
};

// `json_columns` names top-level fields whose DuckDB VARCHAR view is logical
// JSON text. Filters that touch one of these fields must remain in DuckDB.
//
// `include_runtime_hints` also encodes the `col IN (...)` list a hash join
// derived from its build side, which DuckDB marks optional. Such a hint only
// narrows what the scan reads and never counts as a pushed filter
// (`all_filters_pushed` stays false). Do not enable it for search functions: a
// prefilter changes which rows a top-k search returns.
//
// `exclude_runtime_filters` builds only from the filters DuckDB planned for the
// scan, leaving out the dynamic filters a hash join attaches at execution time.
// Search functions must set it: a join range applied as a prefilter changes
// which rows enter the top-k. The caller still evaluates `input.filters` above
// the scan when `result.has_runtime_filters` is set.
LanceFilterIRBuildResult BuildLanceTableFilterIRParts(
    const vector<string> &names, const vector<LogicalType> &types,
    const vector<string> &json_columns, const TableFunctionInitInput &input,
    bool exclude_computed_columns, bool include_runtime_hints,
    bool exclude_runtime_filters = false);

// Probe whether every filter attached to `get.table_filters` can be encoded
// into Lance filter IR. Owns the invariant "filters encodable → limit/offset
// pushdown is safe" used by both LanceLimitOffsetPushdown and
// LanceExecPushdown, so those passes cannot diverge and fall out of sync
// with the runtime scan init.
//
// Returns `all_filters_pushed=true` with empty `parts` if the filter set
// is empty. Callers that need the encoded parts (e.g., to build an exec
// IR message) can consume `parts` directly.
LanceFilterIRBuildResult
ProbeLanceTableFilterIR(LogicalGet &get, const vector<string> &names,
                        const vector<LogicalType> &types,
                        const vector<string> &json_columns);

bool TryBuildLanceExprFilterIR(const LogicalGet &get,
                               const vector<string> &names,
                               const vector<LogicalType> &types,
                               const vector<string> &json_columns,
                               bool exclude_computed_columns,
                               const Expression &expr, string &out_ir);

bool TryEncodeLanceFilterIRMessage(const vector<string> &parts,
                                   string &out_msg);

} // namespace duckdb
