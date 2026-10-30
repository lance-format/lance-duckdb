# SQL Reference

This document lists the SQL surface currently supported by the `lance` DuckDB extension, with short examples.

## Loading

```sql
INSTALL lance;
LOAD lance;
```

For local development builds, start the matching DuckDB binary with `-unsigned`
(e.g. `./build/release/duckdb -unsigned`) and load the extension artifact directly:

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

Float16 columns are exposed as `FLOAT`, including values nested in arrays, lists,
and structs. `INSERT`, `UPDATE`, and `MERGE` reject tables with coerced columns
to avoid silently changing their on-disk types. For a scalar float16 column,
`ALTER TABLE ... ALTER COLUMN ... TYPE FLOAT` explicitly migrates it to float32
and makes it writable.

## JSON columns

Lance JSON extension fields scan as JSON text with SQL type `VARCHAR`, including
JSON fields nested in structs or lists. SQL predicates and aggregates use that
text representation. Search predicates on JSON columns require
`prefilter := false` when they cannot be pushed to Lance; they then filter the
top-k results.

`INSERT` and `MERGE INSERT` do not support tables containing JSON columns.
`UPDATE` and `MERGE UPDATE` can change other columns, but cannot assign a
JSON-containing column. `DELETE` remains supported. JSON columns can be renamed
or dropped, but `ALTER COLUMN ... TYPE` rejects a JSON source. To populate a
derived text column, add the column and then use a separate `UPDATE` statement.

`COPY` uses its input Arrow types. To create or append native JSON, use DuckDB
`JSON` input values with `arrow_lossless_conversion = true`. Appending ordinary
`VARCHAR` or bytes to an existing JSON field fails Lance's schema check.
Overwrite mode can replace the dataset schema.

## Blob columns and write inputs

SQL `BLOB` columns read and write real bytes. In addition, a write can carry
an explicit external-reference or managed-ingest description through the
`LANCE_BLOB_INPUT` value type:

```sql
SELECT typeof(lance_blob_ref('file:///data/a.jpg'));
-- LANCE_BLOB_INPUT
```

For partial reads without materializing whole values, see
[Blob range reads](#blob-range-reads).

### Encoding defaults

- New top-level `BLOB` fields in datasets written with an effective file
  version `>= 2.2` are stored as Lance Blob v2. Lance chooses inline, packed,
  or dedicated storage; the SQL surface does not expose layout knobs.
  `ALTER TABLE ... ADD COLUMN` applies the same default: a new top-level BLOB
  column becomes Blob v2 at file version `>= 2.2` and stays plain binary
  below it. Non-BLOB binary-typed columns such as `BIT` are never re-encoded.
- `ALTER TABLE ... ADD COLUMN ... DEFAULT` back-fills every existing row. BLOB
  defaults accept constant-foldable bytes and `NULL`; URI inputs are a
  DML value contract and are rejected as stored defaults, and non-foldable
  expressions such as column references or volatile functions fail with an
  explicit error. The stored `duckdb_default_expr` metadata is persisted by a
  separate best-effort commit after the schema evolution; this does not create
  a new atomic DDL guarantee.
- Datasets written with an older file version keep their existing binary
  encoding.
- Existing plain binary and legacy Blob columns keep their encoding. They
  still accept ordinary bytes, but a URI input is rejected instead of
  silently migrating the column.
- URI inputs are only supported for top-level `BLOB` columns. Nested
  `LANCE_BLOB_INPUT` values (inside `STRUCT`, `LIST`, or `MAP`) are rejected;
  ordinary nested `BLOB` values keep their existing behavior.

### Constructors

```sql
lance_blob_ref(uri)
lance_blob_ref(uri, offset, length)

lance_blob_ingest(uri)
lance_blob_ingest(uri, offset, length)
```

- `uri` is `VARCHAR` and must be absolute with a scheme, for example
  `file:///path/object.bin` or `s3://bucket/key`. Relative paths are
  rejected; they are never resolved against the process working directory.
- Without `offset`/`length`, the whole object is referenced. With a range,
  `offset` is a zero-based `BIGINT >= 0` and `length` is a `BIGINT > 0`.
  Both range arguments must be non-NULL and the range must not overflow.
- A NULL URI produces a NULL blob. NULL is distinct from an empty BLOB; an
  empty object can be ingested or referenced as a whole object.
- Constructors only build and validate the value. They perform no HEAD, GET,
  or ingest I/O; I/O happens only when a row reaches the writer.

### Reference vs. ingest

| Variant | Source I/O at write time | Ownership after commit |
| --- | --- | --- |
| `lance_blob_ref` | None | The external owner keeps the object. Later reads follow the URI; deleting or changing the source affects those reads. |
| `lance_blob_ingest` | Reads the selected object or range | Lance manages the stored bytes. The source object can be deleted after a successful commit. |

Source storage is resolved per source URI through DuckDB secrets, separately
from the target dataset. Creating a reference does not validate that the
external object exists; reference reads and ingest writes report missing
objects or invalid ranges as errors.

### INSERT, COPY, and CTAS

Ordinary `BLOB` values and both URI constructors share one value contract.
Each top-level value keeps its own mode, including values selected by `CASE`.

```sql
ATTACH 'path/to/namespace' AS ns (TYPE LANCE);

INSERT INTO ns.main.media (id, payload)
VALUES (1, 'plain bytes'::BLOB),
       (2, lance_blob_ref('s3://source/a.jpg')),
       (3, lance_blob_ingest('s3://source/b.jpg', 1024, 4096)),
       (4, NULL);

COPY (
  SELECT id, lance_blob_ingest(uri) AS payload
  FROM incoming_media
) TO 'path/to/media.lance' (FORMAT lance);

CREATE TABLE ns.main.media_copy AS
SELECT id,
       CASE WHEN keep_reference THEN lance_blob_ref(uri)
            ELSE lance_blob_ingest(uri) END AS payload
FROM incoming_media;
```

`UPDATE` and `MERGE` accept the same values:

```sql
UPDATE ns.main.media
SET payload = lance_blob_ingest('s3://source/replacement.jpg')
WHERE id = 1;

MERGE INTO ns.main.media AS target
USING incoming_media AS source ON target.id = source.id
WHEN MATCHED THEN
  UPDATE SET payload = lance_blob_ingest(source.uri)
WHEN NOT MATCHED THEN
  INSERT (id, payload) VALUES (source.id, lance_blob_ref(source.uri));
```

Ingest is materialized with a per-batch budget derived from DuckDB's
`memory_limit` (`memory_limit / 8`, shared by all blob columns and all ingest
rows of one batch). An object or range larger than that budget fails with an
explicit error; use `lance_blob_ref` for large external objects or raise
`memory_limit`. The budget bounds one batch conversion, not the total query
or process RSS: buffered batches and downstream operators can hold additional
memory.

Existing SQL capability boundaries apply. `MERGE ... RETURNING` is supported
and returns the action row's bytes; `INSERT ... RETURNING` and
`UPDATE ... RETURNING` remain rejected by the extension.

## Blob range reads

`lance_read_blob_ranges` reads byte ranges inside top-level blob values without
materializing whole blobs in SQL:

```sql
SELECT request_id, version, row_address, data
FROM lance_read_blob_ranges(
  'path/to/dataset.lance',
  'payload',
  7,               -- dataset version, NULL selects the latest version
  'row_address',   -- or 'row_id'
  (
    SELECT request_id, row_value, offset, length
    FROM requests
  )
);
```

Positional arguments:

- `dataset` (VARCHAR): Dataset root path or object store URI. Storage
  credentials resolve through the same secrets and session as scans.
- `column` (VARCHAR): A top-level blob column.
- `version` (BIGINT): Dataset version to pin. `NULL` means the latest version
  and is only allowed for `row_id` requests against a dataset whose manifest
  enables stable row ids. Without stable row ids `_rowid` is the physical row
  address, so both selectors require an explicit version there.
- `selector` (VARCHAR): One selector kind for the whole call. `row_address`
  addresses the physical `row_address` value reported by a previous call at the
  same version. `row_id` addresses the dataset row id exposed as `_rowid`:
  stable row ids when the dataset has them enabled, the physical row address
  otherwise.
- `requests` (TABLE): A relation with one row per request. It must provide
  `request_id` (any type; echoed back), `row_value`, `offset`, and `length`
  (integer types). Extra columns are ignored.

The function returns one row per request:

- `request_index` (UBIGINT): A processing ordinal unique within one call, in
  no guaranteed relation order. The request relation has no intrinsic order
  and independent pipelines (for example `UNION ALL` branches) may process
  requests in any order, so use `request_id` to correlate results and `ORDER
  BY` to choose display order. Repeated executions are not promised to
  assign the same ordinals.
- `request_id`: The caller's `request_id`, unchanged. Duplicate ids stay
  duplicated.
- `row_address` (UBIGINT): Physical row address of the selected blob at the
  pinned version.
- `data` (BLOB): The requested bytes.
- `version` (UBIGINT): The dataset version the handle was actually pinned to.

`offset` and `length` are relative to the logical blob value, and ranges of
blobs that reference an external source are translated by Lance. A NULL blob
returns NULL `data`; a zero-length range on a non-NULL blob returns an empty
BLOB. Out-of-bounds ranges, offset/length overflow, missing rows, missing
physical fragments and source I/O failures raise errors instead of returning
NULL or truncated bytes.

Lance serves the requested ranges directly and may combine nearby requests
into fewer physical reads; the extension never reads a whole blob and slices
it afterwards. Requests are processed in windows sized by DuckDB's current
`memory_limit` (window budget = `memory_limit / 8`), bounding each native read
call: a single range larger than the budget fails with an explicit error, so
raise `memory_limit` or split the range. The budget bounds a single call, not
the whole query - concurrent request pipelines and downstream SQL operators
can still hold more materialized bytes in total. The function does not accept
nested blob columns.

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
- `uri` (VARCHAR): Dataset root path, object store URI (e.g. `s3://...`), or an attached Lance table name.
- `vector_column` (VARCHAR): Vector column name.
- `query_vector` (FLOAT[dim] or DOUBLE[dim], preferred): Query vector (must be non-empty; values are cast to float32). `FLOAT[]` / `DOUBLE[]` are also accepted.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `use_index` (BOOLEAN, default `true`): If `true`, allow ANN index usage when available.
- `nprobs` (BIGINT, optional): Number of IVF partitions to probe when using a vector index. Must be > 0. Only affects IVF-based vector indices.
- `refine_factor` (BIGINT, optional): Over-fetch factor for re-ranking using original vectors. Must be > 0. A value of `1` still enables re-ranking.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `explain_verbose` (BOOLEAN, default `false`): Emit a more verbose Lance plan in `EXPLAIN` output.
- `filter` (VARCHAR, optional): Namespace `query_table` filter expression. Only supported when `uri` resolves to an attached Lance namespace table. Combined with pushed `WHERE` predicates using `AND`.

Output:
- Dataset columns plus `_distance` (smaller is closer).

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.
- `WHERE` predicates push down for both dataset paths and attached namespace tables. For namespace tables, pushed predicates are translated to a Lance Namespace `query_table` filter; a pushed predicate and an explicit `filter` combine with `AND`.

### Full-text search: `lance_fts`

```sql
-- Search a text column, returning BM25-like scores in `_score` (larger is better)
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', 'text', 'puppy', k = 10, prefilter = true)
ORDER BY _score DESC;
```

Signature: `lance_fts(uri, text_column, query, ...)`

Positional arguments:
- `uri` (VARCHAR): Dataset root path, object store URI (e.g. `s3://...`), or an attached Lance table name.
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `filter` (VARCHAR, optional): Namespace `query_table` filter expression. Only supported when `uri` resolves to an attached Lance namespace table. Combined with pushed `WHERE` predicates using `AND`.

Output:
- Dataset columns plus `_score` (larger is better).

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.
- `WHERE` predicates push down for both dataset paths and attached namespace tables. For namespace tables, pushed predicates are translated to a Lance Namespace `query_table` filter; a pushed predicate and an explicit `filter` combine with `AND`.

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
- `uri` (VARCHAR): Dataset root path, object store URI (e.g. `s3://...`), or an attached Lance table name.
- `vector_column` (VARCHAR): Vector column name.
- `query_vector` (FLOAT[dim] or DOUBLE[dim], preferred): Query vector (must be non-empty; values are cast to float32). `FLOAT[]` / `DOUBLE[]` are also accepted.
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `nprobs` (BIGINT, optional): Number of IVF partitions to probe when using a vector index. Must be > 0. Only affects IVF-based vector indices.
- `refine_factor` (BIGINT, optional): Refine factor for the vector branch. Must be > 0.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `use_index` (BOOLEAN, default `true`): If `true`, allow ANN index usage for the vector branch when available. If `false`, the vector branch runs exact KNN.
- `alpha` (FLOAT, default `0.5`): Vector/text mixing weight. Larger values weigh vector similarity more heavily.
- `oversample_factor` (INTEGER, default `4`): Oversample factor for candidate generation. If provided, must be > 0.

Output:
- Dataset columns plus `_hybrid_score` (larger is better), `_distance`, and `_score`.

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.
- `WHERE` predicates push down for both dataset paths and attached namespace tables.

### Filter pushdown notes

These notes apply to all three search functions.

- Comparison predicates (`=`, `>`, numeric `IN`, ...) and string predicates (`starts_with`, `LIKE`, `regexp_matches`) push down into Lance.
- Containment predicates over list columns — `array_has_any` and `array_has_all` — push down. With `prefilter=true` and a `LABEL_LIST` index on the list column, Lance answers them from the index; without one, Lance evaluates them with a scan. The index is used for `VARCHAR[]` and `BIGINT[]` element types; narrower integer widths still push and prefilter, but scan.
- `IS NULL` / `IS NOT NULL` on list columns push down. No scalar index answers them.
- Containment forms that cannot push — empty or non-constant needles, needles containing `NULL`, `array_has` / `array_contains`, and the `&&` / `@>` / `<@` aliases — are evaluated by DuckDB after the search.
- Datasets whose list columns have a child field named `l` (written by earlier builds of this extension) push and prefilter correctly, but the `LABEL_LIST` index is not used. A metadata-only rename fixes this without rewriting data: `ALTER TABLE ns.main.t RENAME COLUMN "tags.l" TO item;`
- Known limitation: `IN` over a `VARCHAR` column with two or more values currently raises an error from the search functions when it must prefilter. Apply it after the search, or use a scan.

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

Scans of REST namespace tables use the Lance Namespace `query_table` API.
Projection, supported filters, and `LIMIT`/`OFFSET` pairs are pushed into the
request; unsupported filters and standalone `OFFSET` operations remain in
DuckDB. Directory namespace scans continue to read the Lance dataset directly.
Set `lance_namespace_query_table = false` to have REST namespace scans open
the underlying dataset directly as well, e.g. when the namespace server does
not implement `query_table`.

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
- `mode` accepts `create` (the default), `overwrite`, or `append`.
  `create` fails if the dataset already exists.
- `data_storage_version` defaults to `2.2` when `COPY` creates or overwrites a dataset. Appends preserve the existing storage version.
- `write_empty_file` defaults to `true` and controls whether an empty dataset is materialized when the input produces zero rows.
- Floating-point list columns with consistent, nonzero dimensions are written as
  fixed-size vectors. Prefer explicit `FLOAT[N]` / `DOUBLE[N]` types when writing
  vectors; inconsistent list dimensions raise an error.

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

`CREATE TABLE` and CTAS default to data storage version `2.2`. Override it with
`WITH (data_storage_version = '<version>')` when another version is required.

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
DELETE FROM ns.main.my_table WHERE id IN (SELECT id FROM staging);
DELETE FROM ns.main.my_table USING staging s WHERE my_table.id = s.id;
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
  ADD COLUMN age_plus_one BIGINT;
UPDATE ns.main.my_table SET age_plus_one = age + 1;

ALTER TABLE ns.main.my_table RENAME COLUMN score TO score2;
ALTER TABLE ns.main.my_table ALTER COLUMN age TYPE BIGINT;
ALTER TABLE ns.main.my_table ALTER COLUMN embedding TYPE FLOAT[128];
ALTER TABLE ns.main.my_table DROP COLUMN score2;
ALTER TABLE ns.main.my_table ALTER COLUMN age SET NOT NULL;
ALTER TABLE ns.main.my_table ALTER COLUMN age DROP NOT NULL;

DETACH ns;
```

Stored defaults must pass DuckDB's default binder; they cannot reference other
columns or use DataFusion-only functions. Use literal or cast-literal defaults
with the current DuckDB target. DuckDB rewrites other `ADD COLUMN ... DEFAULT`
forms into multiple statements ending in `ALTER COLUMN ... SET DEFAULT`, which
Lance does not support. Such a statement can fail after the column is added.

Table and column comments:

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

COMMENT ON TABLE ns.main.my_table IS 'table comment';
COMMENT ON COLUMN ns.main.my_table.age_plus_one IS 'col comment';

DETACH ns;
```

Notes:
- `ALTER COLUMN ... SET NOT NULL` and `DROP NOT NULL` update Lance nullability metadata.
  `SET NOT NULL` rejects a column containing NULL values; subsequent writes enforce the constraint.

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
- `MetricsJSON` (`bytes_removed`, `old_versions`, `failed_deletes`)

Notes:
- Cleanup is best-effort: a file that cannot be deleted is counted in `failed_deletes` and left in place for a later run instead of failing the statement.
- `delete_unverified = true` removes unreferenced data and index files regardless of age, including those of writes that have not committed, so use it only when nothing else is writing to the dataset. It is rejected while a transaction in this database holds Lance writes it has not committed, though a write or `COMMIT` that is still running is not detected. The rejection aborts the transaction it runs in, so a following `COMMIT` rolls that transaction back.

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
