# SQL Reference

This document lists the SQL surface currently supported by the `lance` DuckDB extension, with short examples.

## Loading

```sql
INSTALL lance FROM community;
LOAD lance;
```

For local development builds, load the extension artifact directly:

```sql
LOAD 'build/release/extension/lance/lance.duckdb_extension';
```

## Scan

Query a dataset by selecting from its URI directly:

```sql
SELECT *
FROM 'path/to/dataset.lance'
LIMIT 10;
```

## Search

### Vector search: `lance_vector_search`

```sql
-- Search a vector column, returning distances in `_distance` (smaller is closer)
SELECT id, label, _distance
FROM lance_vector_search(
  'path/to/dataset.lance',
  'vec',
  [0.1, 0.2, 0.3, 0.4]::FLOAT[4],
  k = 5,
  use_index = true,
  nprobs = 4,
  refine_factor = 2,
  prefilter = true
)
ORDER BY _distance ASC;
```

Signature: `lance_vector_search(uri, vector_column, query_vector, ...)`

Positional arguments:
- `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
- `vector_column` (VARCHAR): Vector column name.
- `query_vector` (FLOAT[dim] or DOUBLE[dim], preferred): Query vector (must be non-empty; values are cast to float32). `FLOAT[]` / `DOUBLE[]` are also accepted.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `use_index` (BOOLEAN, default `true`): If `true`, allow ANN index usage when available.
- `nprobs` (BIGINT, optional): Number of IVF partitions to probe when using a vector index. Must be > 0. Only affects IVF-based vector indices.
- `refine_factor` (BIGINT, optional): Over-fetch factor for re-ranking using original vectors. Must be > 0. A value of `1` still enables re-ranking.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `explain_verbose` (BOOLEAN, default `false`): Emit a more verbose Lance plan in `EXPLAIN` output.

Output:
- Dataset columns plus `_distance` (smaller is closer).

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.

### Full-text search: `lance_fts`

```sql
-- Search a text column, returning BM25-like scores in `_score` (larger is better)
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', 'text', 'puppy', k = 10, prefilter = true)
ORDER BY _score DESC;
```

Signature: `lance_fts(uri, text_column, query, ...)`

Positional arguments:
- `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.

Output:
- Dataset columns plus `_score` (larger is better).

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.

### Hybrid search: `lance_hybrid_search`

```sql
-- Combine vector and text scores, returning `_hybrid_score` (larger is better)
SELECT id, _hybrid_score, _distance, _score
FROM lance_hybrid_search(
  'path/to/dataset.lance',
  'vec',
  [0.1, 0.2, 0.3, 0.4]::FLOAT[4],
  'text',
  'puppy',
  k = 10,
  prefilter = false,
  alpha = 0.5,
  oversample_factor = 4
)
ORDER BY _hybrid_score DESC;
```

Signature: `lance_hybrid_search(uri, vector_column, query_vector, text_column, query, ...)`

Positional arguments:
- `uri` (VARCHAR): Dataset root path or object store URI (e.g. `s3://...`).
- `vector_column` (VARCHAR): Vector column name.
- `query_vector` (FLOAT[dim] or DOUBLE[dim], preferred): Query vector (must be non-empty; values are cast to float32). `FLOAT[]` / `DOUBLE[]` are also accepted.
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `alpha` (FLOAT, default `0.5`): Vector/text mixing weight. Larger values weigh vector similarity more heavily.
- `oversample_factor` (INTEGER, default `4`): Oversample factor for candidate generation. If provided, must be > 0.

Output:
- Dataset columns plus `_hybrid_score` (larger is better), `_distance`, and `_score`.

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.

## Namespaces

Namespaces let you treat a directory (or a remote namespace service) as a database catalog and access datasets as tables.

### Directory namespace

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

-- A dataset stored at path/to/dir/my_table.lance becomes ns.main.my_table
SELECT count(*) FROM ns.main.my_table;
SHOW TABLES FROM ns.main;

DETACH ns;
```

### REST namespace

```sql
ATTACH 'namespace_id' AS ns (TYPE LANCE, ENDPOINT 'http://127.0.0.1:2333');

SHOW TABLES FROM ns.main;
SELECT count(*) FROM ns.main.some_table;

DETACH ns;
```

## Write datasets

### `COPY ... TO ... (FORMAT lance, ...)`

Write a new dataset (or overwrite an existing one):

```sql
COPY (
  SELECT 1::BIGINT AS id, 'a'::VARCHAR AS s
  UNION ALL
  SELECT 2::BIGINT AS id, 'b'::VARCHAR AS s
) TO 'path/to/out.lance' (FORMAT lance, mode 'overwrite');
```

Append to an existing dataset:

```sql
COPY (SELECT 3::BIGINT AS id, 'c'::VARCHAR AS s)
TO 'path/to/out.lance' (FORMAT lance, mode 'append');
```

Create an empty dataset (schema only):

```sql
COPY (
  SELECT 1::BIGINT AS id, 'x'::VARCHAR AS s
  LIMIT 0
) TO 'path/to/empty.lance' (FORMAT lance, mode 'overwrite', write_empty_file true);
```

Notes:
- `mode` supports at least `overwrite` and `append`.
- `write_empty_file` controls whether an empty dataset is materialized when the input produces zero rows.

### `CREATE TABLE` / `CTAS` in an attached namespace

When a directory is attached as a namespace, `CREATE TABLE` and `CREATE TABLE AS SELECT` write datasets into the namespace root.

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

-- Schema-only (creates an empty dataset)
CREATE OR REPLACE TABLE ns.main.my_empty (id BIGINT, s VARCHAR);

-- CTAS (writes query results)
CREATE OR REPLACE TABLE ns.main.my_dataset AS
  SELECT 1::BIGINT AS id, 'a'::VARCHAR AS s
  UNION ALL
  SELECT 2::BIGINT AS id, 'b'::VARCHAR AS s;

SELECT count(*) FROM ns.main.my_dataset;
DETACH ns;
```

## DML on attached tables

These statements apply to tables inside an attached namespace (e.g. `ns.main.my_table`).

### `INSERT`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

INSERT INTO ns.main.my_table VALUES (3::BIGINT, 'c'::VARCHAR);
INSERT INTO ns.main.my_table SELECT 4::BIGINT AS id, 'd'::VARCHAR AS s;

DETACH ns;
```

### `UPDATE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

UPDATE ns.main.my_table SET s = 'bb' WHERE id = 2;
UPDATE ns.main.my_table SET s = 'x'; -- update all rows
UPDATE ns.main.my_table SET s = DEFAULT WHERE id = 2;

DETACH ns;
```

### `DELETE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

DELETE FROM ns.main.my_table WHERE id <= 2;
DELETE FROM ns.main.my_table; -- delete all rows

DETACH ns;
```

### `MERGE INTO`

`MERGE` is supported on Lance tables, including:
- `WHEN MATCHED` with `UPDATE` / `DELETE` / `DO NOTHING` / `ERROR`
- `WHEN NOT MATCHED` (`BY TARGET`) with `INSERT` / `DO NOTHING`
- `WHEN NOT MATCHED BY SOURCE` with `UPDATE` / `DELETE` / `DO NOTHING`
- `RETURNING` (including DuckDB's `merge_action` virtual column)

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

MERGE INTO ns.main.my_table AS t
USING (
  SELECT 2::BIGINT AS id, 'bb'::VARCHAR AS s
  UNION ALL
  SELECT 3::BIGINT AS id, 'c'::VARCHAR AS s
) AS src
ON t.id = src.id
WHEN MATCHED THEN UPDATE SET s = src.s
WHEN NOT MATCHED THEN INSERT (id, s) VALUES (src.id, src.s);

MERGE INTO ns.main.my_table AS t
USING (SELECT 3::BIGINT AS id) AS src
ON t.id = src.id
WHEN MATCHED THEN DELETE
RETURNING merge_action, id, s;

DETACH ns;
```

Notes:
- Mutating the same target row more than once in a single `MERGE` raises a constraint error.
- `MERGE` uses a single Lance transaction per statement.

### `TRUNCATE TABLE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

TRUNCATE TABLE ns.main.my_table;

DETACH ns;
```

## DDL on attached tables

### `DROP TABLE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

DROP TABLE ns.main.my_table;
DROP TABLE IF EXISTS ns.main.my_table;

DETACH ns;
```

Notes:
- Directory namespaces reject unsafe dataset names that attempt path traversal.

### `ALTER TABLE`

Schema evolution:

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

ALTER TABLE ns.main.my_table
  ADD COLUMN age_plus_one BIGINT DEFAULT (age + 1);

ALTER TABLE ns.main.my_table RENAME COLUMN score TO score2;
ALTER TABLE ns.main.my_table ALTER COLUMN age TYPE BIGINT;
ALTER TABLE ns.main.my_table ALTER COLUMN embedding TYPE FLOAT[128];
ALTER TABLE ns.main.my_table DROP COLUMN score2;

DETACH ns;
```

Table and column comments:

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

COMMENT ON TABLE ns.main.my_table IS 'table comment';
COMMENT ON COLUMN ns.main.my_table.age_plus_one IS 'col comment';

DETACH ns;
```

Notes:
- `ALTER COLUMN ... SET NOT NULL` is currently not supported.

## Index DDL

The extension supports index DDL targeting a dataset path.

### `CREATE INDEX`

Vector ANN index:

```sql
CREATE INDEX vec_idx ON 'path/to/dataset.lance' (vec)
USING IVF_FLAT WITH (num_partitions=1, metric_type='l2');
```

Scalar index:

```sql
CREATE INDEX label_idx ON 'path/to/dataset.lance' (label)
USING BTREE;
```

Full-text index:

```sql
CREATE INDEX text_idx ON 'path/to/dataset.lance' (text)
USING INVERTED;
```

Notes:
- `CREATE INDEX` currently supports a single column.
- Vector indices require a fixed-size vector column. If a dataset was written with
  `FLOAT[]` / `DOUBLE[]`, first cast it with `ALTER TABLE ... ALTER COLUMN ... TYPE FLOAT[N]`
  (or `DOUBLE[N]`) and then create the index.

### `SHOW INDEXES`

```sql
SHOW INDEXES ON 'path/to/dataset.lance';
```

### `DROP INDEX`

```sql
DROP INDEX vec_idx ON 'path/to/dataset.lance';
```

### `ALTER INDEX ... OPTIMIZE`

```sql
ALTER INDEX vec_idx
ON 'path/to/dataset.lance'
OPTIMIZE WITH (mode = 'append');
```

Supported `mode` values:
- `append`
- `merge` (`num_indices_to_merge` is supported)
- `retrain`

## Maintenance SQL

Maintenance statements accept either:
- A dataset path string literal (for example `'path/to/dataset.lance'`)
- An attached Lance table name (for example `ns.main.my_table`)

### `OPTIMIZE`

```sql
OPTIMIZE 'path/to/dataset.lance' WITH (
  target_rows_per_fragment = 1048576,
  max_rows_per_group = 1024,
  max_bytes_per_file = 0,
  materialize_deletions = true,
  materialize_deletions_threshold = 0.1,
  num_threads = 0,
  batch_size = 0,
  defer_index_remap = false
);
```

Returns:
- `Operation` (`compact`)
- `Target`
- `MetricsJSON` (compaction metrics)

### `VACUUM LANCE`

```sql
VACUUM LANCE 'path/to/dataset.lance' WITH (
  older_than_seconds = 1209600,
  delete_unverified = false,
  error_if_tagged_old_versions = true,
  retain_n_versions = 3
);
```

Returns:
- `Operation` (`cleanup`)
- `Target`
- `MetricsJSON` (cleanup metrics such as removed bytes / versions)

### `ALTER INDEX ... OPTIMIZE`

```sql
ALTER INDEX vec_idx
ON 'path/to/dataset.lance'
OPTIMIZE WITH (
  mode = 'merge',
  num_indices_to_merge = 4
);
```

Supported `mode` values:
- `append`
- `merge` (`num_indices_to_merge` is supported)
- `retrain`

Returns:
- `Operation` (`optimize_index`)
- `Target`
- `MetricsJSON`

### Auto cleanup configuration

```sql
ALTER TABLE 'path/to/dataset.lance'
SET AUTO_CLEANUP WITH (interval = 1, older_than = '1h', retain_versions = 3);

ALTER TABLE 'path/to/dataset.lance' UNSET AUTO_CLEANUP;

SHOW MAINTENANCE ON 'path/to/dataset.lance';
```

`SHOW MAINTENANCE` returns key/value rows including:
- `enabled`
- `interval` (if configured)
- `older_than` (if configured)
- `retain_versions` (if configured)
