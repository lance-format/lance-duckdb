# AGENTS.md

This file provides guidance to coding agents working in this repository, including a project overview, common commands, and key architecture notes.

## Project Overview

This repository contains a DuckDB extension for querying Lance format datasets. The DuckDB integration is implemented in C++ (under `src/`) and links a Rust static library (`lance_duckdb_ffi`) that uses the Lance Rust crate and exports data via the Arrow C Data Interface.

## Documentation Language

All documentation in this repository (including `README.md` and files under `docs/`) must be written in English.

## Essential Commands

### Building
```bash
# Initial setup (only needed once)
git submodule update --init --recursive

# Build commands (provided by DuckDB extension tooling from `extension-ci-tools`)
make
GEN=ninja make release
GEN=ninja make debug
GEN=ninja make clean
GEN=ninja make clean_all

# Rust-only checks (without a full DuckDB/CMake build)
cargo check --manifest-path Cargo.toml
cargo clippy --manifest-path Cargo.toml --all-targets
```

### Testing

The `release` build can be slow. For fast iteration, prefer `test_debug` when available.

```bash
# Run all tests (builds and runs sqllogictest)
GEN=ninja make test

# Run with specific build
GEN=ninja make test_debug     # Test with debug build
GEN=ninja make test_release   # Test with release build

# Run DuckDB with extension for manual testing
./build/release/duckdb -c "SELECT * FROM 'test/test_data.lance' LIMIT 1;"

# Or load the loadable extension from a standalone DuckDB binary
duckdb -unsigned -c "LOAD 'build/release/extension/lance/lance.duckdb_extension'; SELECT * FROM 'test/test_data.lance' LIMIT 1;"
```

### Development Iteration
```bash
# Fast iteration cycle
GEN=ninja make debug && GEN=ninja make test_debug

# Check for issues without full build
cargo clippy --manifest-path Cargo.toml --all-targets
```

## Architecture & Key Design Decisions

### Extension Architecture

#### Source layout

- Primary C++ extension sources are under `src/` (see `CMakeLists.txt` for `EXTENSION_SOURCES`).

#### DuckDB layer (C++)

- `src/lance_extension.cpp`
  - Defines the `lance_init` entry point and the `LanceExtension` class.
  - Registers the `lance_scan` table function and a replacement scan.

- `src/lance_scan.cpp`
  - Implements the `lance_scan(path)` table function.
  - `bind`: opens the dataset via Rust FFI, exports schema via Arrow C Data Interface, and lets DuckDB derive output types.
  - `init`: creates a streaming scanner via Rust FFI.
  - `func`: pulls RecordBatches, exports each batch via Arrow C Data Interface, and converts Arrow -> DuckDB via `ArrowTableFunction::ArrowToDuckDB`.
  - Currently disables projection and filter pushdown and serializes stream access via a global mutex.

- `src/lance_replacement.cpp`
  - Enables `SELECT * FROM '.../dataset.lance'` by rewriting it into `lance_scan('.../dataset.lance')`.

#### Rust FFI layer

- `rust/lib.rs`: C ABI for opening datasets, exporting schema and batches via Arrow C Data Interface, and streaming RecordBatches.
- `rust/scanner.rs`: wraps Lance scanning as a `RecordBatch` stream.

#### Naming Strategy

The project uses different names to avoid conflicts:
- **Extension name**: `lance` (user-facing)
- **Rust crate/staticlib**: `lance_duckdb_ffi` (linked into the extension)

## Test Data & Testing Conventions

### Test Dataset

Location: `test/test_data.lance`

### Test Format

Uses DuckDB's `sqllogictest` format in `test/sql/`:
- `statement ok/error`: Test statement execution
- `query <types>`: Test queries with expected results (`I`=int, `T`=text, `R`=real)
- `require lance`: Load the extension

## Common Issues

### Extension Loading
```sql
-- Always use -unsigned flag for local builds
duckdb -unsigned
LOAD 'build/release/extension/lance/lance.duckdb_extension';
```
