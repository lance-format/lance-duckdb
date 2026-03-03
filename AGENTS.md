# AGENTS.md

## Project Overview

This repository contains a DuckDB extension for querying Lance format datasets (including scan, vector search, and full-text search). The DuckDB integration is implemented in C++ (under `src/`) and links a Rust static library (`lance_duckdb_ffi`) that uses the Lance Rust crate and exports data via the Arrow C Data Interface.

## Documentation Language

All documentation in this repository (including `README.md` and files under `docs/`) must be written in English.

## Constraint Hierarchy (Highest Priority)

For any planning, implementation, review, or refactor work in this repository:

- The only hard constraints are upstream DuckDB requirements and upstream Lance requirements.
- If any repository-local guidance conflicts with DuckDB or Lance behavior/contracts, follow DuckDB/Lance.
- Any decision made inside `lance-duckdb` (APIs/ABIs, FFI boundaries, internal formats, module boundaries, naming, testing conventions, or implementation strategies) is mutable and may be changed when a better design is found.
- Treat repository-local rules in this document as default guidance, not immutable policy.

## Deliverable & Compatibility Policy

This project is delivered as a fully self-contained DuckDB extension artifact (statically linked in our distribution). Low-level APIs/ABIs/formats in this repository (C++/Rust FFI boundaries, internal encodings, file/IPC formats, etc.) are strictly internal implementation details:

- They are not intended to be directly consumed by end users.
- There are no external downstream users that depend on them.
- Prioritize first principles and the most direct correct design; do not optimize for migrations or compatibility unless explicitly requested.
- Compatibility commitments are driven by DuckDB/Lance behavior, not by historical repository-local decisions.
- User-facing contracts should be defined at the SQL surface.

## Public Surface Policy (SQL-First)

Expose user-facing capabilities primarily through SQL, not internal helper functions.

- Default: expose features via DuckDB SQL mechanisms (replacement scan, `ATTACH ... (TYPE LANCE)`, standard DDL/DML, and `COPY ... (FORMAT lance)`).
- Internal functions may exist for implementation/composition, but should not be the primary user-facing entry points.
- Narrow exceptions are allowed when a dedicated function is the clearest SQL contract (for example, `lance_fts` and similar search entry points).

When in doubt, prefer SQL surface area that matches DuckDB idioms over extension-specific internal entry points.

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

### Commit & PR Conventions

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type[(scope)]: short description
```

- **Common types**: `feat`, `fix`, `refactor`, `docs`, `ci`, `test`, `chore`.
- **Scope** is optional. When present, use the affected module (e.g. `search`, `resolver`, `scan`).
- Use `!` suffix for breaking changes (e.g. `refactor!: ...`).
- PR numbers are appended automatically by GitHub on merge (e.g. `(#153)`).

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

## Architecture & Key Design Decisions

### Data Flow & FFI Boundary

C++ and Rust communicate through the Arrow C Data Interface:

```
DuckDB SQL → C++ Extension (src/) → FFI calls → Rust (rust/) → Lance crate
                                   ← Arrow C Data Interface (schema + batches) ←
```

See [docs/rust_guidelines.md](docs/rust_guidelines.md) for the FFI function patterns, error propagation, and memory ownership conventions.

### Key Concepts: Filter IR & Exec IR

The extension uses two custom binary wire formats to push work from C++ into Rust:

- **Filter IR** (magic `LFT1`): Serializes DuckDB bound filter expressions into a compact binary format that Rust deserializes into DataFusion `Expr`. This avoids passing expression trees through C function signatures.
- **Exec IR** (magic `LEX1`): Encodes aggregate/projection plans so that Rust can execute them entirely inside Lance/DataFusion, returning only pre-aggregated results to C++.

When modifying filter or aggregate pushdown, changes typically span both sides: C++ serialization (`lance_filter_ir.cpp` / `lance_exec_ir.cpp`) and Rust deserialization (`rust/filter_ir.rs` / `rust/exec_ir.rs`). The magic bytes and version numbers must stay in sync.

### Namespace & Catalog Architecture

`ATTACH ... (TYPE LANCE)` supports two namespace backends (directory for local, REST for remote LanceDB). Both lazily discover tables and auto-create catalog entries on first access.

## Testing Conventions

Tests use DuckDB's `sqllogictest` format in `test/sql/`. S3/MinIO tests (`s3_*.test`) are gated by `LANCE_TEST_S3=1` and require a running MinIO instance.

## Common Gotchas

- **Extension loading**: Local builds are unsigned. Always use `duckdb -unsigned` when loading the extension manually.
- **Debug vs release**: `D_ASSERT` checks are stripped in release builds. A test that passes in one configuration but fails in the other usually indicates an assertion violation or uninitialized-variable issue.
- **Rust build diagnostics**: When Cargo fails inside the CMake build, error output is hard to read. Run `cargo check --manifest-path Cargo.toml` in isolation for clearer diagnostics.

## Implementation Guidelines

- **C++**: See [docs/cpp_guidelines.md](docs/cpp_guidelines.md) for coding style, error handling, and naming conventions.
- **Rust**: See [docs/rust_guidelines.md](docs/rust_guidelines.md) for unsafe/FFI conventions and error handling.
