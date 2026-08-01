#pragma once

#include "lance_dataset_cache.hpp"

#include "duckdb.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/table_function.hpp"

#include <cstdint>

namespace duckdb {

class ExtensionLoader;
class TableCatalogEntry;

// Bind data for the internal __lance_knn_scan table function. The
// optimizer constructs this directly via the rewrite pass; the rebuild
// path (prepared statements / EXPLAIN ANALYZE replanning) reconstructs
// the same shape from the function's BLOB parameters.
struct LanceKnnScanBindData : public TableFunctionData {
  string file_path;
  string vector_column;
  vector<float> query;
  uint64_t k = 0;
  // Distance metric inferred from the rewritten distance function:
  // "l2" / "cosine" / "dot". Lance itself reads the metric from the
  // selected index; this string is informational (EXPLAIN clarity and
  // optimizer / index-metric consistency checks).
  string metric;
  uint64_t nprobes = 0;
  uint64_t refine_factor = 0;
  uint64_t ef = 0;
  bool prefilter = true;
  bool use_index = true;

  // Encoded Lance Filter IR (LFT1) bytes, possibly empty.
  string filter_ir_msg;

  // Dataset handle is always accessed through dataset_entry->Handle().
  // We intentionally do not cache the raw pointer separately — keeping
  // two fields in sync (and making sure neither outlives the other) was a
  // foot-gun: a future copy-then-move of bind_data would leave the raw
  // pointer dangling while dataset_entry was nulled out. shared_ptr ref-
  // count is the single source of truth for dataset lifetime.
  shared_ptr<LanceDatasetCacheEntry> dataset_entry;
  bool dataset_cache_hit = false;

  // Lance-side scan schema (matches the bound LogicalGet return types).
  ArrowSchemaWrapper schema_root;
  ArrowTableSchema arrow_table;
  vector<string> names;
  vector<LogicalType> types;
};

TableFunction LanceKnnScanFunction();
void RegisterLanceKNNScan(ExtensionLoader &loader);

// Populates the bind_data's arrow_table / schema_root / names / types by
// asking Lance for the KNN scan schema. The optimizer-built rewrite path
// uses this to bootstrap the bind data it just constructed (the normal
// Bind callback does the same thing internally on re-bind).
void LanceKnnScanPopulateSchema(ClientContext &context,
                                LanceKnnScanBindData &bind_data);

} // namespace duckdb
