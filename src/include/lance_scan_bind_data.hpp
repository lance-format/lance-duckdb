#pragma once

#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/optional_idx.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"

#include <cstdint>

namespace duckdb {

struct LanceScanBindData : public TableFunctionData {
  string file_path;
  bool explain_verbose = false;
  void *dataset = nullptr;
  ArrowSchemaWrapper schema_root;
  ArrowSchemaWrapper scan_schema_root;
  ArrowTableSchema arrow_table;
  ArrowTableSchema scan_arrow_table;
  vector<string> names;
  vector<LogicalType> types;
  vector<string> lance_pushed_filter_ir_parts;
  vector<string> duckdb_pushed_filter_sql_parts;

  bool sampling_pushed_down = false;
  double sample_percentage = 0.0;
  int64_t sample_seed = -1;
  bool sample_repeatable = false;
  vector<uint64_t> take_row_ids;

  bool limit_offset_pushed_down = false;
  optional_idx pushed_limit = optional_idx::Invalid();
  idx_t pushed_offset = 0;

  ~LanceScanBindData() override;
};

} // namespace duckdb
