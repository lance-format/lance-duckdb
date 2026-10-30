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
};

// `json_columns` names top-level fields whose DuckDB VARCHAR view is logical
// JSON text. Filters that touch one of these fields must remain in DuckDB.
LanceFilterIRBuildResult BuildLanceTableFilterIRParts(
    const vector<string> &names, const vector<LogicalType> &types,
    const vector<string> &json_columns, const TableFunctionInitInput &input,
    bool exclude_computed_columns);

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
