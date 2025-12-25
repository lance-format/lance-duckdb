# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Added

- `lance_vector_search(path, vector_column, vector, ...)` table function.
- `lance_fts(path, text_column, query, ...)` table function.
- `lance_hybrid_search(path, vector_column, vector, text_column, query, ...)` table function.

### Removed

- `lance_search(...)` table function.

## [0.1.0] - 2025-12-22

### Added

- DuckDB extension `lance` for scanning Lance datasets.
- `lance_scan(path)` table function.
- Replacement scan to enable `SELECT ... FROM 'path/to/dataset.lance'`.
- Rust FFI bridge backed by the Lance Rust crate and Arrow C Data Interface.
- Fragment-level parallelism for scanning.

### Changed

- Build and CI target DuckDB `v1.4.3`.
