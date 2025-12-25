# Lance DuckDB Extension

Query [Lance](https://github.com/lance-format/lance/) datasets directly from DuckDB.

Lance is a modern columnar data format optimized for ML/AI workloads, with native cloud storage support. This extension brings Lance into a familiar SQL workflow.


## Usage

### Query a `*.lance` path

```sql
-- local file
SELECT *
  FROM 'path/to/dataset.lance'
  LIMIT 10;
-- s3
SELECT *
  FROM 's3://bucket/path/to/dataset.lance'
  LIMIT 10;
```

### S3 authentication via DuckDB Secrets

For `s3://` paths, the extension can use DuckDB's native Secrets mechanism to obtain credentials:

```sql
CREATE SECRET (TYPE S3, provider credential_chain);

SELECT *
  FROM 's3://bucket/path/to/dataset.lance'
  LIMIT 10;
```

### Search (vector / FTS / hybrid)

The extension exposes three table functions:
- `lance_vector_search(...)` for vector search (KNN / ANN)
- `lance_fts(...)` for full-text search (FTS)
- `lance_hybrid_search(...)` for hybrid search (vector + FTS)

Note: DuckDB treats `column` as a keyword in some contexts. Prefer `text_column` / `vector_column`.

#### Full-text search (FTS)

```sql
-- Search a text column, returning BM25-like scores in `_score`
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', 'text', 'puppy', k = 10, prefilter = true)
ORDER BY _score DESC;
```

#### Vector search

```sql
-- Search a vector column, returning distances in `_distance` (smaller is closer)
SELECT id, label, _distance
FROM lance_vector_search('path/to/dataset.lance', 'vec', [0.1, 0.2, 0.3, 0.4]::FLOAT[],
                         k = 5, prefilter = true)
ORDER BY _distance ASC;
```

#### Hybrid search (vector + FTS)

```sql
-- Combine vector and text scores, returning `_hybrid_score` in addition to `_distance` / `_score`
SELECT id, _hybrid_score, _distance, _score
FROM lance_hybrid_search('path/to/dataset.lance',
                         'vec', [0.1, 0.2, 0.3, 0.4]::FLOAT[],
                         'text', 'puppy',
                         k = 10, prefilter = false,
                         alpha = 0.5, oversample_factor = 4)
ORDER BY _hybrid_score DESC;
```

## Install

### Install from DuckDB Community Extensions (recommended)

If you just want to use the extension, install it directly from DuckDB's community extensions repository:

```sql
INSTALL lance FROM community;
LOAD lance;

SELECT *
  FROM 'path/to/dataset.lance'
  LIMIT 1;
```

See DuckDB's extension page for `lance` for the latest details: https://duckdb.org/community_extensions/extensions/lance

### Build from source (development)

This repository focuses on source builds for development and CI.

1. Initialize submodules:

```bash
git submodule update --init --recursive
```

2. Build:

```bash
GEN=ninja make release
```

3. Load the extension from a standalone DuckDB binary (local builds typically require unsigned extensions):

```bash
duckdb -unsigned -c "LOAD 'build/release/extension/lance/lance.duckdb_extension'; SELECT 1;"
```

## Contributing

Issues and PRs are welcome. High-impact areas include pushdown, parallelism/performance, type coverage, and better diagnostics.

## License

Apache License 2.0.
