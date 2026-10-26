#pragma once

#include "duckdb.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

class ClientContext;

// The variants of the SQL write input type. The order is the UNION tag order
// and is part of the value contract shared by every Lance write path.
enum class LanceBlobKind : uint8_t {
  BYTES = 0,
  REFERENCE = 1,
  INGEST = 2,
};

// A LANCE_BLOB_INPUT value after write-boundary validation.
//
// `is_null` represents the normalized outer NULL; non-NULL values have either
// `bytes` or `uri` set. `has_range` is true only when `position`/`size` were
// supplied together; `position` is then the zero-based offset and `size` the
// positive length within the external object.
struct LanceBlobInput {
  bool is_null = true;
  LanceBlobKind kind = LanceBlobKind::BYTES;
  string bytes;
  string uri;
  bool has_range = false;
  uint64_t position = 0;
  uint64_t size = 0;
};

// The named write input type:
// UNION(bytes BLOB, reference STRUCT(uri, position, size),
//        ingest STRUCT(uri, position, size))
LogicalType LanceBlobInputType();

// True when `type` is the Blob write input type. The alias is preferred, but
// DuckDB expression composition may drop it, so the member names and types are
// also matched structurally.
bool LanceIsBlobInput(const LogicalType &type);

// Validate and normalize a SQL value into the shared write input
// representation. Returns false and fills `error` for values that are not a
// valid LANCE_BLOB_INPUT.
bool LanceNormalizeBlobInput(const Value &value, LanceBlobInput &out,
                             string &error);

// The per-batch ingest materialization budget in bytes, derived from DuckDB's
// memory limit (one eighth).
idx_t LanceBlobIngestBatchBudget(ClientContext &context);

// Source storage options for the distinct ingest URIs in one batch, resolved
// by DuckDB's native secret lookup. `uris[i]` matches the exact URI carried by
// the SQL value; `keys[i]`/`values[i]` hold its resolved Lance storage
// options.
struct LanceBatchSourceOptions {
  vector<string> uris;
  vector<vector<string>> keys;
  vector<vector<string>> values;
};

// Scan `chunk` for LANCE_BLOB_INPUT ingest values and resolve each distinct
// source URI against the connection's DuckDB secrets. Non-ingest rows are
// skipped; malformed values are left for the write boundary to reject.
void LanceCollectIngestSourceOptions(ClientContext &context, DataChunk &chunk,
                                     LanceBatchSourceOptions &out);

// Flattened C-pointer form of one batch's per-URI source options. Lives here
// so every Lance batch sink (INSERT, UPDATE, MERGE) passes ingest credentials
// through the same shape.
struct LanceBlobSourceOptionArrays {
  vector<const char *> uris;
  vector<const char *> all_keys;
  vector<const char *> all_values;
  vector<const char **> key_offsets;
  vector<const char **> value_offsets;
  vector<size_t> counts;

  const char **Uris() const {
    return uris.empty() ? nullptr : const_cast<const char **>(uris.data());
  }
  const char ***KeyOffsets() const {
    return key_offsets.empty() ? nullptr
                               : const_cast<const char ***>(key_offsets.data());
  }
  const char ***ValueOffsets() const {
    return value_offsets.empty()
               ? nullptr
               : const_cast<const char ***>(value_offsets.data());
  }
  const size_t *Counts() const {
    return counts.empty() ? nullptr : counts.data();
  }
};

void BuildLanceBlobSourceOptionArrays(
    const LanceBatchSourceOptions &source_options,
    LanceBlobSourceOptionArrays &out);

// Write one Arrow batch to the Lance writer together with the per-URI source
// storage options resolved for its ingest values.
int32_t LanceWriteBlobBatch(void *writer, ArrowArray *array,
                            const LanceBatchSourceOptions &source_options);

// Per-column flags passed to the Rust writer: 1 for top-level columns whose
// DuckDB SQL type is BLOB (or the Blob write input). Arrow alone cannot tell
// BLOB apart from e.g. BIT with lossless conversion disabled.
vector<uint8_t> LanceBlobFlagsForTypes(const vector<LogicalType> &types);

// Build a `bytes` member LANCE_BLOB_INPUT value from already-materialized data.
Value LanceMakeBytesBlobInputValue(const string &bytes);

// Materialize every ingest row of the chunk in place, so the same batch can be
// shared by the writer and RETURNING without reading the source twice.
// Reference rows are left untouched. Returns the number of replaced rows.
idx_t NormalizeIngestRowsInChunk(ClientContext &context, DataChunk &chunk);

void RegisterLanceBlobInput(ExtensionLoader &loader);
void RegisterLanceBlobPlannerExtension(DBConfig &config);

} // namespace duckdb
