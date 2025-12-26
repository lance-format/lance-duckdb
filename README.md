# Lance DuckDB Extension

Query [Lance](https://github.com/lance-format/lance/) datasets directly from DuckDB.

Lance is a modern columnar data format optimized for ML/AI workloads, with native cloud storage support. This extension brings Lance into a familiar SQL workflow.


## Usage

### Query a Lance dataset

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

To read `s3://` paths, the extension can use DuckDB's native Secrets mechanism to obtain credentials:

```sql
CREATE SECRET (TYPE S3, provider credential_chain);

SELECT *
  FROM 's3://bucket/path/to/dataset.lance'
  LIMIT 10;
```

### Write a Lance dataset

Use DuckDB's `COPY ... TO ...` to materialize query results as a Lance dataset.

```sql
-- Create/overwrite a Lance dataset from a query
COPY (
  SELECT 1::BIGINT AS id, 'a'::VARCHAR AS s
  UNION ALL
  SELECT 2::BIGINT AS id, 'b'::VARCHAR AS s
) TO 'path/to/out.lance' (FORMAT lance, mode 'overwrite');

-- Read it back via the replacement scan
SELECT count(*) FROM 'path/to/out.lance';

-- Append more rows to an existing dataset
COPY (
  SELECT 3::BIGINT AS id, 'c'::VARCHAR AS s
) TO 'path/to/out.lance' (FORMAT lance, mode 'append');

-- Optionally create an empty dataset (schema only)
COPY (
  SELECT 1::BIGINT AS id, 'x'::VARCHAR AS s
  LIMIT 0
) TO 'path/to/empty.lance' (FORMAT lance, mode 'overwrite', write_empty_file true);
```

To write to `s3://...` paths, load DuckDB's `httpfs` extension and configure an S3 secret:

```sql
INSTALL httpfs;
LOAD httpfs;

CREATE SECRET (TYPE S3, provider credential_chain);

COPY (SELECT 1 AS id) TO 's3://bucket/path/to/out.lance' (FORMAT lance, mode 'overwrite');
```

### Create a Lance dataset via `CREATE TABLE` (directory namespace)

When you `ATTACH` a directory as a Lance namespace, you can create new datasets using `CREATE TABLE` (schema-only)
or `CREATE TABLE AS SELECT` (CTAS). The dataset is written to `<namespace_root>/<table_name>.lance`.

```sql
ATTACH 'path/to/dir' AS lance_ns (TYPE LANCE);

-- Schema-only (creates an empty dataset)
CREATE TABLE lance_ns.main.my_empty (id BIGINT, s VARCHAR);

-- CTAS (writes query results)
CREATE TABLE lance_ns.main.my_dataset AS
  SELECT 1::BIGINT AS id, 'a'::VARCHAR AS s
  UNION ALL
  SELECT 2::BIGINT AS id, 'b'::VARCHAR AS s;

SELECT count(*) FROM lance_ns.main.my_dataset;
```

### Vector search

```sql
-- Search a vector column, returning distances in `_distance` (smaller is closer)
SELECT id, label, _distance
FROM lance_vector_search('path/to/dataset.lance', 'vec', [0.1, 0.2, 0.3, 0.4]::FLOAT[],
                         k = 5, prefilter = true)
ORDER BY _distance ASC;
```

- Signature: `lance_vector_search(uri, vector_column, query_vector, ...)`
- Positional arguments:
  - `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
  - `vector_column` (VARCHAR): Vector column name.
  - `query_vector` (FLOAT[] or DOUBLE[]): Query vector (must be non-empty; values are cast to float32).
- Named parameters:
  - `k` (BIGINT, default `10`): Number of results to return.
  - `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
  - `use_index` (BOOLEAN, default `true`): If `true`, allow ANN index usage when available.
  - `explain_verbose` (BOOLEAN, default `false`): Emit a more verbose Lance plan in `EXPLAIN` output.
- Output:
  - Dataset columns plus `_distance` (smaller is closer).

### Full-text search (FTS)

```sql
-- Search a text column, returning BM25-like scores in `_score`
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', 'text', 'puppy', k = 10, prefilter = true)
ORDER BY _score DESC;
```

- Signature: `lance_fts(uri, text_column, query, ...)`
- Positional arguments:
  - `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
  - `text_column` (VARCHAR): Text column name.
  - `query` (VARCHAR): Query string.
- Named parameters:
  - `k` (BIGINT, default `10`): Number of results to return.
  - `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- Output:
  - Dataset columns plus `_score` (larger is better).

### Hybrid search (vector + FTS)

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

- Signature: `lance_hybrid_search(uri, vector_column, query_vector, text_column, query, ...)`
- Positional arguments:
  - `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
  - `vector_column` (VARCHAR): Vector column name.
  - `query_vector` (FLOAT[] or DOUBLE[]): Query vector (must be non-empty; values are cast to float32).
  - `text_column` (VARCHAR): Text column name.
  - `query` (VARCHAR): Query string.
- Named parameters:
  - `k` (BIGINT, default `10`): Number of results to return.
  - `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
  - `alpha` (FLOAT, default `0.5`): Vector/text mixing weight.
  - `oversample_factor` (INTEGER, default `4`): Oversample factor for candidate generation (larger can improve recall at higher cost).
- Output:
  - Dataset columns plus `_hybrid_score` (larger is better), `_distance`, and `_score`.

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
