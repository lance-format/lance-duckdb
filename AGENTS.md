# AGENTS.md

This file provides guidance to coding agents working in this repository, including a project overview, common commands, and key architecture notes.

## Project Overview

This repository contains a DuckDB extension for querying Lance format datasets (including scan, vector search, and full-text search). The DuckDB integration is implemented in C++ (under `src/`) and links a Rust static library (`lance_duckdb_ffi`) that uses the Lance Rust crate and exports data via the Arrow C Data Interface.

## Documentation Language

All documentation in this repository (including `README.md` and files under `docs/`) must be written in English.

## Deliverable & Compatibility Policy

This project is delivered as a fully self-contained DuckDB extension artifact (statically linked in our distribution). All APIs/ABIs/formats in this repository (C++/Rust FFI boundaries, internal encodings, file/IPC formats, etc.) are strictly internal implementation details:

- They are not intended to be directly consumed by end users.
- There are no external downstream users that depend on them.
- When planning/implementing/refactoring, prioritize first principles and the most direct correct design; do not optimize for migrations or compatibility unless explicitly requested.

## SQL Export Policy (Release Requirement)

For releases, keep the exported function surface minimal. Only the search table functions and the `COPY ... (FORMAT lance)` writer should be user-visible:

- Exported (user-facing):
  - `lance_vector_search`
  - `lance_fts`
  - `lance_hybrid_search`
  - `COPY ... TO ... (FORMAT lance, ...)` (registered as the `lance` copy format)
- Not exported (must be internal-only), including but not limited to:
  - `lance_scan` (and any scan/namespace helper table functions)
  - all metadata/maintenance table functions (e.g. `lance_*metadata`, `lance_*config`, `lance_*indices`, compaction/cleanup)

Prefer exposing capabilities via DuckDB standard SQL mechanisms (replacement scan, `ATTACH ... (TYPE LANCE)`, standard DDL/DML) rather than new user-facing functions.

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

### Formatting

This repository is configured with `uv`, so you can run formatting via:

```bash
uv run make format
```

PR requirement: Before submitting a PR, run `uv run make format` and commit any resulting formatting changes (as a separate commit if it helps review).

### Testing

The `release` build can be slow. For fast iteration, prefer `test_debug` when available.

```bash
# Run all tests (builds and runs sqllogictest)
GEN=ninja make test

# Run with specific build
GEN=ninja make test_debug     # Test with debug build
GEN=ninja make test_release   # Test with release build

# Run DuckDB with extension for manual testing
./build/release/duckdb -c "SELECT * FROM 'test/data/test_data.lance' LIMIT 1;"

# Or load the loadable extension from a standalone DuckDB binary
duckdb -unsigned -c "LOAD 'build/release/extension/lance/lance.duckdb_extension'; SELECT * FROM 'test/data/test_data.lance' LIMIT 1;"
```

### Development Iteration
```bash
# Fast iteration cycle
GEN=ninja make debug && GEN=ninja make test_debug

# Check for issues without full build
cargo clippy --manifest-path Cargo.toml --all-targets
```

## Architecture & Key Design Decisions

#### Naming Strategy

The project uses different names to avoid conflicts:
- **Extension name**: `lance` (user-facing)
- **Rust crate/staticlib**: `lance_duckdb_ffi` (linked into the extension)

## Test Data & Testing Conventions

### Test Format

Uses DuckDB's `sqllogictest` format in `test/sql/`:
- `statement ok/error`: Test statement execution
- `query <types>`: Test queries with expected results (`I`=int, `T`=text, `R`=real)
- `require lance`: Load the extension

Key test files:
- `test/sql/scan_smoke.test` (scan smoke + replacement scan + error handling)
- `test/sql/scan_limit_offset_sampling.test` (LIMIT/OFFSET and TABLESAMPLE pushdown)
- `test/sql/scan_filter_pushdown.test` (filter/projection pushdown + explain diagnostics)
- `test/sql/search_functions.test` (vector/FTS/hybrid search surface)
- `test/sql/index_ddl.test` (index DDL + index-backed query paths)
- `test/sql/s3_scan_minio.test` (S3 scan/search via DuckDB secrets, gated by `LANCE_TEST_S3=1`)

## Common Issues

### Extension Loading
```sql
-- Always use -unsigned flag for local builds
duckdb -unsigned
LOAD 'build/release/extension/lance/lance.duckdb_extension';
```

## DuckDB C++ Implementation Guidelines

### C++ Guidelines

- Do not use `malloc`, prefer the use of smart pointers. Keywords `new` and `delete` are a code smell.
- Strongly prefer the use of `unique_ptr` over `shared_ptr`, only use `shared_ptr` if you absolutely have to.
- Use `const` whenever possible.
- Do not import namespaces (e.g. `using std`).
- All functions in source files in the core (`src` directory) should be part of the `duckdb` namespace.
- When overriding a virtual method, avoid repeating `virtual` and always use `override` or `final`.
- Use `[u]int(8|16|32|64)_t` instead of `int`, `long`, `uint` etc. Use `idx_t` instead of `size_t` for offsets/indices/counts of any kind.
- Prefer using references over pointers as arguments.
- Use `const` references for arguments of non-trivial objects (e.g. `std::vector`, ...).
- Use C++11 for loops when possible: `for (const auto& item : items) {...}`.
- Use braces for `if` statements and loops. Avoid single-line `if` statements and loops, especially nested ones.
- Class layout should follow this structure:

```cpp
class MyClass {
public:
	MyClass();

	int my_public_variable;

public:
	void MyFunction();

private:
	void MyPrivateFunction();

private:
	int my_private_variable;
};
```

- Avoid unnamed magic numbers. Use named variables stored in a `constexpr`.
- Return early and avoid deep nested branches.
- Do not include commented-out code blocks in pull requests.

### Error Handling

- Use exceptions only when an error terminates a query (e.g. parser error, table not found).
- For expected/non-fatal errors, prefer return values over exceptions.
- Try to add test cases that trigger exceptions. If an exception cannot be easily triggered by tests, it may be an assertion instead.
- Use `D_ASSERT` to assert. Use `assert` only for programmer errors.
- Assertions should never be triggerable by user input.
- Avoid assertions without context like `D_ASSERT(a > b + 3);` unless accompanied by clear context/comments.
- Assert liberally, and make clear (with concise comments when needed) what invariant failed.

### Naming Conventions

- Choose descriptive names. Avoid single-letter variable names.
- Files: lowercase with underscores, e.g. `abstract_operator.cpp`.
- Types (classes, structs, enums, typedefs, `using`): `CamelCase` starting with uppercase, e.g. `BaseColumn`.
- Variables: lowercase with underscores, e.g. `chunk_size`.
- Functions: `CamelCase` starting with uppercase, e.g. `GetChunk`.
- Avoid `i`, `j`, etc. in nested loops. Prefer names like `column_idx`, `check_idx`.
- In non-nested loops, `i` is permissible as iterator index.
- These rules are partially enforced by `clang-tidy`.
