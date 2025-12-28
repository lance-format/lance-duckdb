# SQLLogicTest Conventions

This directory contains SQLLogicTest suites for the `lance` DuckDB extension.

## File Naming

Tests are grouped by prefix to keep related coverage discoverable:

- `scan_*`: Scan/replacement scan behavior and rowid semantics
- `pushdown_*`: Filter/predicate pushdown and related diagnostics
- `search_*`: `lance_*_search` and `lance_fts` SQL surface
- `index_*`: Index DDL and index-backed query paths
- `optimizer_*`: Optimizer/statistics integration
- `namespace_*`: `ATTACH ... (TYPE LANCE)` namespace mapping and table discovery
- `dml_*`: Write-path SQL (COPY/INSERT/UPDATE/DELETE/TRUNCATE/DROP/ALTER)
- `s3_*`: End-to-end S3 tests (MinIO), gated by `require-env`
- `tpch.test`: TPC-H correctness suite
- `bench_*`: Larger correctness suites and fixtures (e.g., BigANN)

## Header and Layout

Each file starts with:

- `# name: ...`
- `# description: ...`
- `# group: [sql]`

For environment-gated tests, use this order:

1. `require-env ...`
2. `test-env ...` (separate each entry with an empty line)
3. `require lance`

Prefer small, focused files with section comments for setup, correctness, and diagnostics.
