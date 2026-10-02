#pragma once

#include "duckdb/common/arrow/arrow.hpp"

#include <string>
#include <vector>

namespace duckdb {

// Compatibility shim between the Arrow types Lance emits and the subset
// DuckDB's bundled Arrow consumer supports.
//
// Some Lance datasets carry Arrow types that DuckDB cannot represent natively
// (today: FloatingPoint(HALF); future additions slot in without touching call
// sites). The helpers below "coerce" such types to the nearest DuckDB-supported
// shape at the reader boundary:
//
//   * `LanceCoerceArrowSchemaForDuckDB` — in-place rewrite of the Arrow schema
//     so DuckDB's type mapper accepts it. Returns the top-level column names
//     whose declared type had to be coerced (empty if none). Callers should
//     store this list on the catalog entry so write paths can refuse to
//     silently re-encode.
//
//   * `LanceCoerceArrowArrayForDuckDB` — per-batch mirror: rewrite buffers of
//     the same coerced fields so values DuckDB reads match the coerced schema.
//     `schema` MUST still carry the original (un-coerced) formats — call this
//     BEFORE `LanceCoerceArrowSchemaForDuckDB` on the same schema.
//
// Ownership: both helpers install wrapping release callbacks so their new
// allocations are freed when the producer's release fires. Pass the ROOT
// ArrowSchema / ArrowArray; children are released transitively.

// Umbrella helpers — call these at every Lance → DuckDB boundary.
std::vector<std::string> LanceCoerceArrowSchemaForDuckDB(ArrowSchema *schema);

void LanceCoerceArrowArrayForDuckDB(const ArrowSchema *schema,
                                    ArrowArray *array);

// Quick predicate — true when the schema contains any type that the coercion
// helpers above would rewrite.
bool LanceArrowSchemaNeedsCoercion(const ArrowSchema *schema);

// DuckDB's Arrow exporter names list child fields `l`; Arrow (and therefore
// Lance, pyarrow and DataFusion) call them `item`. Lance persists the name, and
// DataFusion's array-function coercion rewrites any list argument to an `item`
// field — which wraps a non-`item` column in a cast and hides it from Lance's
// scalar-index planner, so `array_has_any` / `array_has_all` can never reach a
// LABEL_LIST index. Rewrite the name on every DuckDB → Lance write boundary.
//
// Call on the ROOT ArrowSchema before handing it to Lance. Names are rewritten
// to static storage, so no allocation is added to the producer's lifetime.
void LanceNormalizeArrowListFieldNames(ArrowSchema *schema);

} // namespace duckdb
