#pragma once

#include "duckdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"

namespace duckdb {

class LanceDatasetCacheEntry;
class TableCatalogEntry;

// The post-bind carrier supplies the columns consumed by DuckDB's RETURNING
// projection. Unreferenced fields keep their type but need no materialization.
class LanceReturningOperator {
public:
  explicit LanceReturningOperator(idx_t column_count)
      : returning_needed(column_count, true) {}
  virtual ~LanceReturningOperator() = default;

  void SetReturningNeeded(vector<bool> needed) {
    returning_needed = std::move(needed);
  }

  vector<bool> returning_needed;
};

class LanceReturningSourceState : public GlobalSourceState {
public:
  explicit LanceReturningSourceState(optional_ptr<ColumnDataCollection> rows);
  SourceResultType Scan(DataChunk &chunk);

private:
  optional_ptr<ColumnDataCollection> rows;
  ColumnDataScanState scan;
};

// Build a row image from evaluated assignments and requested unchanged fields
// of the pinned dataset. An empty action produces ordinary DML rows; MERGE
// appends its action column. Values are buffered before transaction
// publication.
void LanceAppendReturningRows(ClientContext &context,
                              ColumnDataCollection &collection,
                              const TableCatalogEntry &table,
                              const vector<bool> &needed, DataChunk &values,
                              const vector<idx_t> &assigned_columns,
                              optional_ptr<LanceDatasetCacheEntry> dataset,
                              const vector<uint64_t> &row_ids,
                              const string &action = string());

void LanceWrapReturningOperator(LogicalOperator &plan);

} // namespace duckdb
