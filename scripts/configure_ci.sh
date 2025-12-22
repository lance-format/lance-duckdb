#!/usr/bin/env bash
set -euo pipefail

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
repo_root="$(pwd)"
if [[ "${is_windows}" -eq 1 ]] && command -v cygpath >/dev/null 2>&1; then
  repo_root="$(cygpath -m "${repo_root}")"
fi

protoc_abs="${repo_root}/${protoc_rel}"

write_github_env() {
  if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf "PROTOC=%s\n" "${protoc_abs}" >> "${GITHUB_ENV}"
  fi
}

write_docker_env_file() {
  local env_file="${repo_root}/docker_env.txt"
  if [[ "${LINUX_CI_IN_DOCKER:-0}" == "1" ]] && [[ -f "${env_file}" ]]; then
    # Make PROTOC available to subsequent docker runs that consume --env-file=docker_env.txt.
    # Replace any existing PROTOC entry to avoid ambiguity.
    local tmp_file="${env_file}.tmp"
    grep -v '^PROTOC=' "${env_file}" > "${tmp_file}" || true
    printf "PROTOC=%s\n" "${protoc_abs}" >> "${tmp_file}"
    mv "${tmp_file}" "${env_file}"
  fi
}

write_github_env
write_docker_env_file
