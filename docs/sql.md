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

`INSERT`, `UPDATE`, `MERGE`, and `COPY ... (mode 'append')` write values as the
Arrow types the table stores, so tables written by other Lance writers (for
example with `large_string`, dictionary, `date64`, or millisecond timestamp
columns) accept DuckDB values. A value the stored type cannot hold exactly,
such as a fractional second in a seconds column, an `INTERVAL` with days in a
`duration` column, or more distinct strings in a batch than a dictionary's key
type can index, fails the statement instead of being truncated or stored as
NULL.

### Aggregate memory

Pushed aggregates account their tracked intermediate state in DuckDB's shared
`memory_limit` budget and use the connection's `operator_memory_limit` at
execution time, including prepared statements. Reservations appear under
`EXTENSION` in `duckdb_memory()`.

Pushed aggregates fail with a memory error when admission fails; they do not
spill to disk. Scan batches and Lance caches are outside this accounting, so
these settings are not a limit on total process memory.

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

`COPY` creates datasets with its input Arrow types. To create or append native JSON, use DuckDB
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

`INSERT`, `UPDATE`, `DELETE`, and `MERGE` support `RETURNING`. Returned BLOB
columns contain the row's bytes. Unreferenced external payloads are not read
solely to construct the returned row. Reading a returned reference uses the
connection's source access policy and scoped Lance secrets.

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
- `ef` (BIGINT, optional): Search-time candidate list size for HNSW sub-indexes (for example `IVF_HNSW_SQ`). Must be > 0 and at least `k * refine_factor`, otherwise Lance fails the query with `ef must be greater than or equal to k`. The Lance default is `1.5 * k * refine_factor`, which can give poor recall on large partitions; raise it (for example `ef = 200`) to trade latency for recall. Ignored by index types without an HNSW graph.
- `distance_type` (VARCHAR, optional): Distance metric of `_distance`: `l2` (or `euclidean`), `cosine`, `dot`, or `hamming`, case-insensitive. If unset, a search that uses a vector index ranks by the index's metric, while an exact search (`use_index = false`, or no index) uses Lance's default for the column type (`l2` for floating-point vectors). Set it to the index's `metric_type` to compare ANN results with exact ones. Lance does not use an index built with a different metric; it runs an exact search instead.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `explain_verbose` (BOOLEAN, default `false`): Emit a more verbose Lance plan in `EXPLAIN` output.
- `filter` (VARCHAR, optional): Namespace `query_table` filter expression. Only supported when `uri` resolves to an attached Lance namespace table. Combined with pushed `WHERE` predicates using `AND`.

Output:
- Dataset columns plus `_distance` (smaller is closer).
- `_rowid`, a virtual column as in a table scan: the row id of each hit, equal to the row id a
  scan of the same dataset version returns (`_rowid` on a dataset path, `rowid` on an attached
  table; the physical row address unless the dataset enables stable row ids). Select or filter it
  by name; `SELECT *` leaves it out, so alias it to keep it in written results, for example
  `SELECT *, _rowid AS hit_rowid`. Inside a `DELETE`, a search of the attached target table reads
  the version the `DELETE` scans, so
  `DELETE FROM ns.main.t WHERE rowid IN (SELECT _rowid FROM lance_fts('ns.main.t', ...))` deletes
  the hits. This does not hold inside `UPDATE` or `MERGE` yet.

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

Signatures:
- `lance_fts(uri, text_column, query, ...)`: matches any term of `query` against `text_column`.
- `lance_fts(uri, query, ...)`: `query` is a structured query (see below).

Positional arguments:
- `uri` (VARCHAR): Dataset root path, object store URI (e.g. `s3://...`), or an attached Lance table name.
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string, or a structured query in the two-argument form.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `filter` (VARCHAR, optional): Namespace `query_table` filter expression. Only supported when `uri` resolves to an attached Lance namespace table. Combined with pushed `WHERE` predicates using `AND`.

Output:
- Dataset columns plus `_score` (larger is better).
- `_rowid`, a virtual column, as for `lance_vector_search`.

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.
- `WHERE` predicates push down for both dataset paths and attached namespace tables. For namespace tables, pushed predicates are translated to a Lance Namespace `query_table` filter; a pushed predicate and an explicit `filter` combine with `AND`.

Structured queries:

```sql
-- Exact phrase; the INVERTED index must be built WITH (with_position = true)
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', '{"phrase": {"column": "text", "terms": "lazy dog"}}');

-- Every term, allowing one edit per term
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance',
  '{"match": {"column": "text", "terms": "lazy dgo", "operator": "And", "fuzziness": 1}}');
```

- A structured query is a Lance `FtsQuery` in its JSON form, the format of LanceDB's `fts` table function: `match` (`terms`, `operator` `"And"` or `"Or"`, `fuzziness`, `max_expansions`, `prefix_length`, `boost`), `phrase` (`terms`, `slop`), `multi_match` (`match_queries`), `boolean` (`should`, `must`, `must_not`), `boost` (`positive`, `negative`, `negative_boost`) and `combined_fields`. Defaults are Lance's; Lance currently treats an omitted `fuzziness` as `0` (exact terms), so set a positive `fuzziness` to match misspellings.
- Fuzzy terms are expanded from the INVERTED index, so rows appended after the index was built match them only once the index covers those rows (`ALTER INDEX ... OPTIMIZE`).
- A query without `column` searches every column with an INVERTED index.
- Invalid JSON, an unknown column, or a phrase query on an index without positions fails the query.
- On attached namespace tables the query is sent as the `query_table` `structured_query`. That API cannot express `combined_fields`, `document_granularity`, or a `match` without `fuzziness`, so these fail; set `fuzziness` (`0` for exact terms).

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

Signatures:
- `lance_hybrid_search(uri, vector_column, query_vector, text_column, query, ...)`
- `lance_hybrid_search(uri, vector_column, query_vector, query, ...)`: `query` is a structured query, as for `lance_fts`.

Positional arguments:
- `uri` (VARCHAR): Dataset root path, object store URI (e.g. `s3://...`), or an attached Lance table name.
- `vector_column` (VARCHAR): Vector column name.
- `query_vector` (FLOAT[dim] or DOUBLE[dim], preferred): Query vector (must be non-empty; values are cast to float32). `FLOAT[]` / `DOUBLE[]` are also accepted.
- `text_column` (VARCHAR): Text column name.
- `query` (VARCHAR): Query string, or a structured query in the four-argument form.

Named parameters:
- `k` (BIGINT, default `10`): Number of results to return. Must be > 0.
- `nprobs` (BIGINT, optional): Number of IVF partitions to probe when using a vector index. Must be > 0. Only affects IVF-based vector indices.
- `refine_factor` (BIGINT, optional): Refine factor for the vector branch. Must be > 0.
- `ef` (BIGINT, optional): HNSW search-time candidate list size for the vector branch. Must be > 0 and at least `k * oversample_factor` (times `refine_factor` if set), otherwise Lance fails the query with `ef must be greater than or equal to k`.
- `distance_type` (VARCHAR, optional): Distance metric of the vector branch, as for `lance_vector_search`.
- `prefilter` (BOOLEAN, default `false`): If `true`, filters are applied before top-k selection.
- `use_index` (BOOLEAN, default `true`): If `true`, allow ANN index usage for the vector branch when available. If `false`, the vector branch runs exact KNN.
- `alpha` (FLOAT, default `0.5`): Vector/text mixing weight. Larger values weigh vector similarity more heavily.
- `oversample_factor` (INTEGER, default `4`): Oversample factor for candidate generation. If provided, must be > 0.

Output:
- Dataset columns plus `_hybrid_score` (larger is better), `_distance`, and `_score`.
- `_rowid`, a virtual column, as for `lance_vector_search`.
- Among the candidates, rows that tie on `_hybrid_score` at the `k` boundary are ranked by `_rowid` (smaller first). Ties in the vector or text score at the `k * oversample_factor` candidate cut are decided by Lance.

Filter semantics:
- If `prefilter=false`, filter pushdown is best-effort. If pushdown fails, the query is retried without pushed filters and DuckDB applies filters for correctness.
- If `prefilter=true`, prefilterable filters must be pushed down, otherwise the query fails with an error.
- `WHERE` predicates push down for both dataset paths and attached namespace tables.

### Filter pushdown notes

These notes apply to all three search functions.

- Comparison predicates (`=`, `>`, numeric `IN`, ...) and string predicates (`starts_with`, `LIKE`, `regexp_matches`) push down into Lance.
- `ILIKE`, `lower`/`upper`, and regular expressions that use `\d`, `\w`, `\s` or `\b` are evaluated by DuckDB, because Lance applies Unicode rules to case mapping and to these classes, so its results would differ from DuckDB's. With `prefilter=true` they raise an error; apply them after the search with `prefilter := false`.
- Containment predicates over list columns — `array_has_any` and `array_has_all` — push down. With `prefilter=true` and a `LABEL_LIST` index on the list column, Lance answers them from the index; without one, Lance evaluates them with a scan. The index is used for `VARCHAR[]` and `BIGINT[]` element types; narrower integer widths still push and prefilter, but scan.
- `IS NULL` / `IS NOT NULL` on list columns push down. No scalar index answers them.
- Containment forms that cannot push — empty or non-constant needles, needles containing `NULL`, `array_has` / `array_contains`, and the `&&` / `@>` / `<@` aliases — are evaluated by DuckDB after the search.
- Datasets whose list columns have a child field named `l` (written by earlier builds of this extension) push and prefilter correctly, but the `LABEL_LIST` index is not used. A metadata-only rename fixes this without rewriting data: `ALTER TABLE ns.main.t RENAME COLUMN "tags.l" TO item;`
- `IN` lists and `OR` of equalities on one column prefilter like any other pushed predicate. A join or `IN (SELECT ...)` is applied after the search, whatever `prefilter` is.

## Namespaces

Namespaces let you treat a directory (or a remote namespace service) as a database catalog and access datasets as tables.

### Directory namespace

SQL identifiers follow DuckDB's case-insensitive matching while dataset names
retain their physical spelling. `CREATE OR REPLACE`, CTAS, and `DROP TABLE` use
the resolved dataset name. Discovery reports an ambiguity error if physical
names such as `Foo.lance` and `foo.lance` have the same DuckDB identifier.
Rename one dataset outside DuckDB to resolve that ambiguity.

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
- `enable_stable_row_ids` (BOOLEAN) creates the dataset with stable row ids, so `_rowid` stays the same
  across compaction; without them `_rowid` is the physical row address and changes when rows move. The
  setting takes effect when the dataset is created. Appending to or overwriting a dataset with a
  different setting is an error, and omitting the option keeps the dataset's setting.
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

`WITH (enable_stable_row_ids = true)` creates the table with stable row ids, as for `COPY`, including for
tables created through a REST namespace. In a directory namespace, `CREATE OR REPLACE TABLE` keeps the
existing table's setting and fails when the option asks for another one; in a REST namespace it drops and
recreates the table, so the option (default off) applies to the new table. `CREATE TABLE` and CTAS reject
any other `WITH` option.

```sql
CREATE TABLE ns.main.my_stable WITH (enable_stable_row_ids = true) AS
  SELECT 1::BIGINT AS id, 'a'::VARCHAR AS s;
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

An `UPDATE` commits against the table version its scan read. `COMMIT` fails with a retryable
conflict when another writer, or an earlier statement of the same transaction, updated or deleted
rows of a fragment it updates; changes to other fragments do not conflict. See
[Transactions](#transactions) for what one transaction may contain.

### `DELETE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

DELETE FROM ns.main.my_table WHERE id <= 2;
DELETE FROM ns.main.my_table WHERE id IN (SELECT id FROM staging);
DELETE FROM ns.main.my_table USING staging s WHERE my_table.id = s.id;
DELETE FROM ns.main.my_table; -- delete all rows

DETACH ns;
```

DELETEs of different rows both commit in concurrent connections, even when the rows share a
fragment. `COMMIT` still fails with a retryable conflict when another writer deleted one of the same
rows or rewrote one of the fragments (`UPDATE`, `MERGE`, compaction). See
[Transactions](#transactions) for what one transaction may contain.

### `RETURNING`

`INSERT` and `UPDATE` return the new row values, including evaluated defaults
and assignments; `DELETE` returns the old values. Expressions, aliases, and
`RETURNING *` use DuckDB's normal binding rules. `UPDATE ... FROM` and
`DELETE ... USING` return each affected target row once even when several
source rows match it.

```sql
INSERT INTO ns.main.my_table VALUES (5, 'e') RETURNING id, s;
UPDATE ns.main.my_table SET s = upper(s) WHERE id = 5 RETURNING *;
DELETE FROM ns.main.my_table WHERE id = 5 RETURNING id, s;
```

Returned rows have no guaranteed order. They describe the statement's affected
rows; inside an explicit transaction they do not imply that `COMMIT` has
succeeded. The existing [transaction restrictions](#transactions) still apply.

A single `INSERT`, `UPDATE`, or `DELETE` CTE can feed a `SELECT`. A statement
containing a modifying CTE and a Lance write cannot combine multiple writes,
including other DML operations, `CREATE TABLE AS`, `COPY TO`, or query-backed
`CREATE TYPE`. These combinations fail before execution; ordinary CTAS and COPY
with read-only inputs still work.

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
- `MERGE` uses a single Lance transaction per statement, committed against the table version its
  scan read, with the same conflicts as `UPDATE`.

### `TRUNCATE TABLE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

TRUNCATE TABLE ns.main.my_table;

DETACH ns;
```

### Transactions

Every `INSERT`, `UPDATE`, `DELETE` and `MERGE` on an attached Lance table builds its own Lance
transaction against the table version its scan read. `COMMIT` publishes them one at a time, each
as its own Lance commit. Lance has no commit that spans several statements or several datasets, and
a statement cannot read the uncommitted writes of an earlier statement of the same transaction.

Two such commits on the same table conflict unless they are appends or deletes of different rows.
To keep `COMMIT` from failing half-way, with the earlier statements already published, or from
silently dropping a change, a `DELETE`, `UPDATE` or `MERGE` that cannot follow the table's earlier
write in the transaction fails with a `Not implemented Error`. The statement is not executed;
`ROLLBACK` discards the transaction's earlier writes.

| Statements on one table in one transaction | Supported |
|---|---|
| `INSERT` after any write | yes |
| `DELETE` after `DELETE` | yes, if they delete different rows; a shared row still fails `COMMIT` with a retryable conflict |
| `UPDATE` or `MERGE` after any write | no |
| `DELETE` after `INSERT`, `UPDATE` or `MERGE` | no |
| `DELETE FROM t` without a filter, or any `DELETE` after it | no |

Limits that remain:
- Reads inside the transaction see the table as of the start of each statement, not the
  transaction's own earlier writes. `INSERT ... SELECT` from the table being written does not see
  the transaction's earlier `INSERT`.
- A transaction that writes several Lance tables commits them one after another. If a later commit
  fails (for example because of a concurrent writer), the earlier tables stay committed.

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

`ALTER TABLE ... OPTIMIZE INDEXES` is described in the Maintenance SQL section.

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
- Option names in `WITH (...)` are case-insensitive, and each option may be given once.
- Vector indices accept these options; any other option, or a value of the wrong type, is an error:
  - every type: `metric_type` (`l2` by default, `euclidean`, `cosine`, `dot`, `hamming`),
    `num_partitions` (default 256), `version` (`v3`; `legacy` only for `IVF_PQ`)
  - `IVF_PQ`, `IVF_HNSW_PQ`: `num_bits` (default 8), `num_sub_vectors` (16), `max_iterations` (50)
  - `IVF_SQ`, `IVF_HNSW_SQ`: `num_bits` (default 8), `sample_rate` (256)
  - `IVF_RQ`: `num_bits` (default 8)
  - `IVF_HNSW_*`: `hnsw_m`, `hnsw_ef_construction`, `hnsw_max_level`, `hnsw_prefetch_distance`
    (Lance's defaults). To disable prefetching, give the full option set as `params` JSON with
    `"hnsw_prefetch_distance": null`.
- Scalar and full-text index options are passed to Lance with lower-cased names. Lance validates
  their values and ignores names it does not recognize.
- `params = '<json object>'` supplies the complete option set, with names as written, and cannot
  be combined with other options except `replace` and `train`.
- For optional vector parameters, JSON `null` means omission, except that
  `hnsw_prefetch_distance: null` disables prefetching.
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

Use `ALTER TABLE ... OPTIMIZE INDEXES` for every index; see the Maintenance SQL section for the modes.

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

### `ALTER INDEX ... OPTIMIZE` and `ALTER TABLE ... OPTIMIZE INDEXES`

```sql
ALTER INDEX vec_idx
ON 'path/to/dataset.lance'
OPTIMIZE WITH (
  mode = 'merge',
  num_indices_to_merge = 4
);
```

```sql
ALTER TABLE 'path/to/dataset.lance' OPTIMIZE INDEXES [WITH (...)];
```

`ALTER TABLE ... OPTIMIZE INDEXES` optimizes every index of the table in one statement and accepts the same options (`index_name` is `null` in the metrics; a table without indices is a no-op). Lance may skip an index it cannot rewrite (for example one of an unknown type or a newer segment version) and only logs a warning that this extension does not surface; the metrics do not list skipped indices, so check `SHOW INDEXES` when it matters.

Supported `mode` values:
- `append` (default for `ALTER INDEX`): index new rows as one more segment; IVF partitions are never rebalanced
- `auto` (default for `ALTER TABLE ... OPTIMIZE INDEXES`): Lance's default merge count. A vector index appends a segment (always, if its segments use different IVF models), or merges all segments when a partition splits or joins, and with no new rows it may rebalance one segment or do nothing; scalar indices (BTree, Bitmap, NGram, INVERTED and the others) replace their newest segment with a merge of it and the new rows instead of adding one; with stable row ids, older segments that cover fragments with deletions are rewritten too
- `merge` (`num_indices_to_merge` defaults to 1; `enable_partition_rebalancing = true` also splits or joins partitions inside the merged segments)
- `retrain`: retrain vector indices over all rows (Lance supports it for v3 vector indices only; scalar indices only merge their newest segment, as under `auto`); rejects `num_indices_to_merge`

`enable_partition_rebalancing` is only valid with `mode = 'merge'`. Under `auto` a vector index gains one segment per run unless a partition splits or joins. A scheduled job that must keep the segment count flat should use `mode = 'merge', num_indices_to_merge = 1, enable_partition_rebalancing = true`; a larger `num_indices_to_merge` consolidates segments that have already accumulated.

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
