#pragma once

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  bool explain_verbose = false;
  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;
  vector<string> duckdb_pushed_filter_sql_parts;

  bool limit_offset_pushed_down = false;
  optional_idx pushed_limit = optional_idx::Invalid();
  idx_t pushed_offset = 0;

  ~LanceScanBindData() override;
};

} // namespace duckdb
