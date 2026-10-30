# This file is included by DuckDB's build system. It specifies which extension to load

# Extension from this repo
duckdb_extension_load(lance
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# Keep the local CLI and test runner self-contained.
duckdb_extension_statically_link(lance)

# Any extra extensions that should be built
# e.g.: duckdb_extension_load(json)
