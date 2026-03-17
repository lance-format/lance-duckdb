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

LanceFilterIRBuildResult BuildLanceTableFilterIRParts(
    const vector<string> &names, const vector<LogicalType> &types,
    const TableFunctionInitInput &input, bool exclude_computed_columns);

bool TryBuildLanceExprFilterIR(const LogicalGet &get,
                               const vector<string> &names,
                               const vector<LogicalType> &types,
                               bool exclude_computed_columns,
                               const Expression &expr, string &out_ir);

bool TryBuildLanceTableFilterIRParts(const vector<string> &names,
                                     const vector<LogicalType> &types,
                                     const TableFilterSet &filters,
                                     vector<string> &out_parts);

bool TryBuildLanceTableFilterIRMessage(const vector<string> &names,
                                       const vector<LogicalType> &types,
                                       const TableFilterSet &filters,
                                       string &out_msg);

bool TryEncodeLanceFilterIRMessage(const vector<string> &parts,
                                   string &out_msg);

} // namespace duckdb
