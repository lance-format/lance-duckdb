#pragma once

#include "duckdb/common/arrow/arrow_wrapper.hpp"

#include <string>
#include <vector>

namespace duckdb {

// Compatibility shim between the Arrow types Lance emits and the subset
// DuckDB's bundled Arrow consumer supports.
//
// Lance emits logical JSON text with the canonical `arrow.json` extension and
// some physical types DuckDB cannot represent natively (today:
// FloatingPoint(HALF)). The helpers below normalize those fields at the reader
// boundary:
//
//   * `LanceCoerceArrowSchemaForDuckDB` — in-place rewrite of the Arrow schema
//     so DuckDB's type mapper accepts it. It reports storage-coerced column
//     names and JSON-containing top-level columns separately. Callers retain
//     both lists because their write and pushdown rules differ.
//
//   * `LanceCoerceArrowArrayForDuckDB` — per-batch mirror: rewrite buffers of
//     storage-coerced fields so values DuckDB reads match the coerced schema.
//     JSON needs metadata normalization only; its text buffers are not
//     rewritten. `schema` MUST still carry the original formats and metadata.
//
// Ownership: both helpers install wrapping release callbacks so their new
// allocations are freed when the producer's release fires. Pass the ROOT
// ArrowSchema / ArrowArray; children are released transitively.

struct LanceArrowSchemaCoercionResult {
  std::vector<std::string> storage_coerced_columns;
  std::vector<std::string> json_columns;
};

// Umbrella helpers — call these at every Lance → DuckDB boundary.
LanceArrowSchemaCoercionResult
LanceCoerceArrowSchemaForDuckDB(ArrowSchema *schema);

void LanceCoerceArrowArrayForDuckDB(const ArrowSchema *schema,
                                    ArrowArray *array);

// Consumes the opaque Rust batch, including when export or coercion throws.
// The schema retains its original formats for column mapping. Arrays have
// already been widened for DuckDB; callers that consume the schema must coerce
// it separately. Keep array alive while DuckDB vectors reference its buffers.
struct LanceArrowBatch {
  explicit LanceArrowBatch(
      void *batch,
      const char *export_error =
          "Failed to export Lance RecordBatch to Arrow C Data Interface");
  LanceArrowBatch(const LanceArrowBatch &) = delete;
  LanceArrowBatch &operator=(const LanceArrowBatch &) = delete;

  ArrowSchemaWrapper schema;
  shared_ptr<ArrowArrayWrapper> array;
};

// Quick predicate — true when the schema needs format or metadata
// normalization.
bool LanceArrowSchemaNeedsCoercion(const ArrowSchema *schema);

// True when this field, or any nested child, carries a Lance blob marker
// (Blob v1 or Blob v2). Ordinary Arrow binary fields do not match.
bool LanceArrowFieldContainsBlob(const ArrowSchema &field);

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
