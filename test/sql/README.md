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
- `s3_*`: End-to-end S3 tests (MinIO), gated by `require-env` and using a `TYPE LANCE` secret
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

## Temporary data and fixtures

Use `{TEST_DIR}` for generated files and namespace roots. DuckDB creates a private
directory per test, so tests can run repeatedly or concurrently without sharing
Lance manifests. Use `{variable}` for loop and environment substitutions.

Treat `test/data/` fixtures as read-only. Tests that must evolve an on-disk Arrow
type copy the fixture bytes into `{TEST_DIR}` using DuckDB's `read_blob` and
`COPY ... (FORMAT BLOB)` before attaching it. Do not clone these fixtures through
`COPY ... (FORMAT lance)`: the reader widens float16, and the writer normalizes
floating-point lists to fixed-size vectors, erasing the schema being tested.

Build the desired configuration before running `make test_debug` or
`make test_release`; those targets run tests without rebuilding the extension.
To include the MinIO suites, set `LANCE_TEST_S3=1` and configure the test
environment variables listed at the top of each S3 test. REST namespace tests
require a separate server and their own `LANCE_NAMESPACE_*` variables.
