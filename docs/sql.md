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

DETACH ns;
```

Notes:
- `UPDATE` requires a `WHERE` predicate (full-table `UPDATE` is not supported).
- `SET column = DEFAULT` is not supported.

### `DELETE`

```sql
ATTACH 'path/to/dir' AS ns (TYPE LANCE);

DELETE FROM ns.main.my_table WHERE id <= 2;
DELETE FROM ns.main.my_table; -- delete all rows

DETACH ns;
```

Notes:
- Explicit transactions for `DELETE` are currently not supported.

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
- Use `REINDEX ... RETRAIN` to retrain an untrained index (if created with training disabled).

### `SHOW INDEXES`

```sql
SHOW INDEXES ON 'path/to/dataset.lance';
```

### `DROP INDEX`

```sql
DROP INDEX vec_idx ON 'path/to/dataset.lance';
```
