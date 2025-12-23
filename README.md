# Lance DuckDB Extension

Query Lance datasets directly from DuckDB.

Lance is a modern columnar data format optimized for ML/AI workloads, with native cloud storage support. This extension brings Lance into a familiar SQL workflow.

## What you can do today

- Scan a Lance dataset root via `lance_scan(path)`.
- Query a `*.lance` dataset path directly via replacement scan:
  - `SELECT * FROM '.../dataset.lance'` is rewritten into `lance_scan('.../dataset.lance')`.

## Usage

### Scan via table function

```sql
SELECT *
FROM lance_scan('path/to/dataset.lance')
LIMIT 10;
```

### Scan a `*.lance` path directly

```sql
SELECT count(*)
FROM 'path/to/dataset.lance';
```

## Install

### Install from DuckDB Community Extensions (recommended)

If you just want to use the extension, install it directly from DuckDB's community extensions repository:

```sql
INSTALL lance FROM community;
LOAD lance;
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

## Notes and limitations

- Projection and filter pushdown are currently disabled in `lance_scan`.
- Stream consumption is serialized with a global mutex, which limits parallelism.

## Contributing

Issues and PRs are welcome. High-impact areas include pushdown, parallelism/performance, type coverage, and better diagnostics.

## License

Apache License 2.0.
