PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=lance
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

CORE_EXTENSIONS=''

# Default build: skip DuckDB's parquet extension.
# Override by providing EXT_FLAGS that already contains -DSKIP_EXTENSIONS=...
ifeq (,$(findstring -DSKIP_EXTENSIONS=,$(EXT_FLAGS)))
	EXT_FLAGS += -DSKIP_EXTENSIONS=parquet
endif

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

.PHONY: configure_ci
configure_ci:
	@bash scripts/configure_ci.sh
