# Lance DuckDB Extension

[Lance](https://github.com/lance-format/lance/) is a modern columnar data format optimized for ML/AI workloads, with native cloud storage support. This extension will make `Lance` the best file/table/lakehouse formats on DuckDB.

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

See DuckDB's extension page for `lance` for the latest release details: https://duckdb.org/community_extensions/extensions/lance

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

## Usage

- Full SQL reference: [`docs/sql.md`](./docs/sql.md)
- Cloud storage reference: [`docs/cloud.md`](./docs/cloud.md)

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

To access object store URIs (e.g. `s3://...`), configure a `TYPE LANCE` secret (see [`docs/cloud.md`](./docs/cloud.md)).

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER credential_chain,
  SCOPE 's3://bucket/'
);

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

To write to `s3://...` paths, configure a `TYPE LANCE` secret for that scope (see [`docs/cloud.md`](./docs/cloud.md)).

```sql
CREATE SECRET (
  TYPE LANCE,
  PROVIDER credential_chain,
  SCOPE 's3://bucket/'
);

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
FROM lance_vector_search('path/to/dataset.lance', 'vec', [0.1, 0.2, 0.3, 0.4]::FLOAT[4],
                         k = 5, prefilter = true)
ORDER BY _distance ASC;
```

See the SQL reference for full parameter documentation: [docs/sql.md#search](docs/sql.md#search).

### Full-text search (FTS)

```sql
-- Search a text column, returning BM25-like scores in `_score`
SELECT id, text, _score
FROM lance_fts('path/to/dataset.lance', 'text', 'puppy', k = 10, prefilter = true)
ORDER BY _score DESC;
```

See the SQL reference for full parameter documentation: [docs/sql.md#search](docs/sql.md#search).

### Hybrid search (vector + FTS)

```sql
-- Combine vector and text scores, returning `_hybrid_score` in addition to `_distance` / `_score`
SELECT id, _hybrid_score, _distance, _score
FROM lance_hybrid_search('path/to/dataset.lance',
                         'vec', [0.1, 0.2, 0.3, 0.4]::FLOAT[4],
                         'text', 'puppy',
                         k = 10, prefilter = false,
                         alpha = 0.5, oversample_factor = 4)
ORDER BY _hybrid_score DESC;
```

See the SQL reference for full parameter documentation: [docs/sql.md#search](docs/sql.md#search).

## Contributing

Issues and PRs are welcome. High-impact areas include pushdown, parallelism/performance, type coverage, and better diagnostics.

### Automation: Lance dependency bumps

This repository includes a GitHub Actions workflow pair that opens a PR when a newer tag is detected in `lance-format/lance`:

- `.github/workflows/lance-release-timer.yml`: polls for a newer Lance tag and triggers the Codex workflow.
- `.github/workflows/codex-update-lance-dependency.yml`: uses Codex CLI to update `Cargo.toml` / `Cargo.lock`, run basic Rust checks, and open a PR.

Required repository secrets:

- `LANCE_RELEASE_TOKEN`: a GitHub token that can read tags and create PRs in this repository.
- `CODEX_TOKEN`: an OpenAI API key used by Codex CLI.

## License

Apache License 2.0.
