# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

## [0.4.0] - 2025-12-29

### Added

- Transactional `DELETE` support (honors `BEGIN`/`COMMIT`/`ROLLBACK`).
- Support `UPDATE` without a `WHERE` clause to update all rows.
- Predicate pushdown for string functions: `starts_with`, `ends_with`, `contains`.
- Support `CREATE INDEX ... ON ns.main.tbl(col) USING BTREE` on attached namespaces.

### Changed

- **Breaking:** Object store credential/configuration now uses `CREATE SECRET (TYPE LANCE, ...)` (instead of `TYPE S3`).
- Aligned directory and REST namespace behavior.

### Fixed

- `UPDATE ... SET col = DEFAULT` now works correctly.

## [0.3.0] - 2025-12-28

### Added

- `COPY ... TO ... (FORMAT lance, ...)` writer for creating and appending Lance datasets.
- Directory and REST namespaces via `ATTACH ... (TYPE LANCE, ...)`.
- DDL support for attached namespaces: `CREATE TABLE`, `CTAS`, `DROP TABLE`, `ALTER TABLE` (schema evolution and comments).
- DML support for attached namespaces: `INSERT`, `UPDATE`, `DELETE`, `TRUNCATE TABLE`.
- Index DDL on dataset URIs: `CREATE INDEX`, `SHOW INDEXES`, `DROP INDEX` (vector, scalar, and full-text).
- Additional scan pushdown and planning improvements: `LIMIT`/`OFFSET`, `TABLESAMPLE SYSTEM`, `LIKE`, `regexp_matches`,
  `IS (NOT) DISTINCT FROM`, and expanded type coverage (including timestamp/decimal/struct filters).
- DuckDB `rowid` / Lance `_rowid` support for scans, enabling point-lookup and rowid-based optimizations.
- Basic scan statistics for improved planning.

### Changed

- Scan and maintenance helper table functions are internal-only; the user-facing surface is replacement scan, namespaces,
  search functions, and the `lance` copy format.
- Improved parallelism for writing datasets.
- Refined test coverage and organization for sqllogictests.

## [0.2.0] - 2025-12-25

### Added

- Filter and projection pushdown for scans on Lance datasets.
- S3 authentication via DuckDB Secrets for `s3://...` dataset paths.
- `EXPLAIN (FORMAT JSON)` diagnostics for Lance scans (bind-time and runtime plan details).
- `lance_vector_search(path, vector_column, vector, ...)` table function.
- `lance_fts(path, text_column, query, ...)` table function.
- `lance_hybrid_search(path, vector_column, vector, text_column, query, ...)` table function.

### Changed

- Optimized `SELECT COUNT(*)` on Lance datasets.
- Improved error propagation across the Rust FFI boundary.
- Upgraded the Lance dependency to `v1.0.0` (including Hugging Face backend support).

## [0.1.0] - 2025-12-22

### Added

- DuckDB extension `lance` for scanning Lance datasets.
- `lance_scan(path)` table function.
- Replacement scan to enable `SELECT ... FROM 'path/to/dataset.lance'`.
- Rust FFI bridge backed by the Lance Rust crate and Arrow C Data Interface.
- Fragment-level parallelism for scanning.

### Changed

- Build and CI target DuckDB `v1.4.3`.
