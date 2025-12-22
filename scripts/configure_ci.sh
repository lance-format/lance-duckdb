#!/usr/bin/env bash
set -euo pipefail

mkdir -p .ci

duckdb_platform="${DUCKDB_PLATFORM:-}"
vcpkg_target_triplet="${VCPKG_TARGET_TRIPLET:-}"

if [[ -z "${vcpkg_target_triplet}" ]]; then
  echo "VCPKG_TARGET_TRIPLET is required to locate protoc from vcpkg." >&2
  exit 1
fi

build_type="release"

is_windows=0
case "${duckdb_platform}" in
  windows_*) is_windows=1 ;;
esac

protoc_exe="protoc"
if [[ "${is_windows}" -eq 1 ]]; then
  protoc_exe="protoc.exe"
fi

protoc_rel="build/${build_type}/vcpkg_installed/${vcpkg_target_triplet}/tools/protobuf/${protoc_exe}"
link_path=".ci/${protoc_exe}"

# Create a stable path for protoc. The target may not exist yet (vcpkg can install during build).
ln -sf "../${protoc_rel}" "${link_path}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  repo_root="$(pwd)"
  printf "PROTOC=%s\n" "${repo_root}/${link_path}" >> "${GITHUB_ENV}"
fi
