#pragma once

namespace duckdb {

class DBConfig;

// Registers the KNN rewrite optimizer extension. The pass converts a
// LOGICAL_TOP_N (ORDER BY array_distance(vec, const_query) LIMIT k) sitting
// over a Lance scan (with an optional pushable LOGICAL_FILTER in between)
// into a single LOGICAL_GET against the internal __lance_knn_scan table
// function, so the request is served by Lance's vector index instead of by
// a full table scan + DuckDB in-memory sort.
//
// All fallback conditions (missing index, metric mismatch, OFFSET, DESC,
// multi-order-by, non-constant query vector, non-pushable WHERE, ...) keep
// the plan unchanged — the original semantics always remain correct.
void RegisterLanceKNNOptimizer(DBConfig &config);

} // namespace duckdb
