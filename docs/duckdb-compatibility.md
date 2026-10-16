# DuckDB compatibility and release branches

Lance is a DuckDB core extension. The official core-extension build checks out
this repository's latest main branch and publishes the extension. A lance-duckdb version or release tag is not a
prerequisite for DuckDB support. Compatibility includes the SQL behavior of the
extension, its C++ integration, and the official build configuration.

## Branches

| Branch | DuckDB target | Policy |
| --- | --- | --- |
| `main` | `v2.0-cyanoptera` | Feature development and fixes compatible with DuckDB 2.0 |
| `v1.5-variegata` | `v1.5-variegata` | Selected correctness, security, and build fixes for DuckDB 1.5 |
| `v1.4-andium` | Existing 1.4 target | Historical maintenance line; retain its existing scope |

The 1.5 branch preserves the source tree from the last verified pre-migration
main commit, `c886612ab517e7821487b209804f3a23dfe15ac6`. Main targets DuckDB 2.0;
use `v1.5-variegata` for DuckDB 1.5 builds. Official artifact availability is
tracked separately from source compatibility in issue #265.

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
   coverage limitations. Confirm they re-enable Lance in the official build.
6. Validate installation, loading, and representative SQL against the official
   staging or release artifact. Source compatibility alone does not establish
   official availability.

Keep upstream defects visible. Temporary patches may isolate a failure, but
results from a patched DuckDB checkout must be labeled as conditional. Do not
commit a misleading gitlink, suppress failed SQL, or bypass binary version checks.
If a release cutoff is missed, coordinate a later extension deployment with
DuckDB rather than tying delivery to an unrelated lance-duckdb tag.

See DuckDB's [extension release cycle](https://duckdb.org/docs/preview/dev/release_cycle#the-main-extension-release-cycle)
for source selection and release-branch conventions.

Track upstream fixes and release acceptance in [issue #265](https://github.com/lance-format/lance-duckdb/issues/265).
