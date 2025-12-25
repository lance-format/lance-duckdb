# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

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
