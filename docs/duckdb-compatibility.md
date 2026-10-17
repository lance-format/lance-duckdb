# DuckDB compatibility and release branches

## Version and release policy

Lance is a DuckDB core extension. **lance-duckdb has no independent product
version, release tags, or release cadence.** It ships through DuckDB's core
extension release process, in step with DuckDB releases. DuckDB's official build
selects this repository's latest `main` for the active target and owns artifact
publication. Merging a change here does not by itself make it available through
`INSTALL lance`.

Compatibility follows the target **DuckDB commit**. This extension uses DuckDB's
C++ API, so an artifact must match the DuckDB build and platform that load it.
The `duckdb` submodule gitlink records the exact commit used for local and CI
validation; a `.gitmodules` branch annotation alone does not update that pin.
Official builds use the DuckDB commit selected by DuckDB's build process.

Use these identifiers explicitly in bug reports, validation records, and release
coordination:

| Identifier | Meaning |
| --- | --- |
| DuckDB release and commit | The runtime and compatibility target |
| lance-duckdb commit | The extension source revision included in a build |
| `extension-ci-tools` commit | The tooling used for a repository build |
| Lance Rust dependency version | The upstream Lance library used internally |

The internal `lance_duckdb_ffi` Cargo package version is build metadata, not an
extension release number. The crate is not published. DuckDB may also expose an
`extension_version` value supplied by its build tooling; keep that provenance
metadata without interpreting it as an independent lance-duckdb release.
Lance dataset versions, storage-format versions, and internal IR versions have
their own meanings and do not define this project's release cycle.

Describe changes as “Lance support for DuckDB 2.0” or “lance-duckdb commit
`<sha>` built against DuckDB commit `<sha>`”. Do not plan a lance-duckdb version
bump, tag, or GitHub release as a delivery step. Updating the Lance Rust dependency
also does not create a new lance-duckdb release.

## Branches

Branch names identify DuckDB compatibility lines, not lance-duckdb versions.

| Branch | DuckDB target | Policy |
| --- | --- | --- |
| `main` | `v2.0-cyanoptera` | Feature development and fixes compatible with DuckDB 2.0 |
| `v1.5-variegata` | `v1.5-variegata` | Selected correctness, security, and build fixes for DuckDB 1.5 |
| `v1.4-andium` | Existing 1.4 target | Historical maintenance line; retain its existing scope |

Main targets DuckDB 2.0; use `v1.5-variegata` for DuckDB 1.5 builds.
Official artifact availability is separate from source compatibility.

Use DuckDB's release branch for compatibility work. DuckDB `main` may already
target the next release. Update both the actual `duckdb` and
`extension-ci-tools` gitlinks and their `.gitmodules` branch annotations.

Do not create a permanent 2.0 maintenance branch while main targets 2.0.
Create `v2.0-cyanoptera` when main moves to the next DuckDB release or needs
changes unsuitable for 2.0. Follow DuckDB's support lifecycle rather than
promising an independent support window.

Apply shared fixes to main and selectively backport them to maintained release
branches. Adapt each fix to that branch's DuckDB API; do not merge whole branches
across an API transition. Keep routine Lance upgrades on main unless a specific
maintenance fix needs a dependency update.

## Acceptance and publication

1. Build and link the complete extension against the selected DuckDB commit.
2. Run the SQL suite, including assertion-enabled catalog/DML tests and the S3
   suite against MinIO. Keep external REST namespace coverage explicit when its
   service is unavailable. A skipped `require lance` is a failed acceptance.
3. Preserve pushdown and residual-filter correctness, prepared-statement schema
   refresh, defaults, parser statement boundaries, and Arrow ownership. Check
   existing EXPLAIN assertions as well as query results.
4. Run the matching extension-ci-tools distribution workflow on source and
   gitlink changes. It uses the checked-out DuckDB gitlink, a Rust toolchain,
   and the selected platforms. Our Actions artifacts are build evidence;
   DuckDB owns official deployment.
5. Merge only after the required behavior and platform checks pass. Give the
   DuckDB maintainers the accepted main SHA, both dependency pins, and remaining
   coverage limitations. Confirm Lance is included in the official build.
6. Validate installation, loading, and representative SQL against the official
   staging or release artifact. Source compatibility alone does not establish
   official availability.

Keep upstream defects visible. Temporary patches may isolate a failure, but
results from a patched DuckDB checkout must be labeled as conditional. Do not
commit a misleading gitlink, suppress failed SQL, or bypass binary version checks.
If a release cutoff is missed, coordinate a later extension deployment with
DuckDB through its core-extension publication process.

See DuckDB's [extension release cycle](https://duckdb.org/docs/preview/dev/release_cycle#the-main-extension-release-cycle)
for source selection and release-branch conventions.
