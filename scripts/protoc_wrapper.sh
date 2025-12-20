#!/usr/bin/env sh
set -eu

if [ -n "${PROTOC_REAL:-}" ] && [ -x "${PROTOC_REAL}" ]; then
  exec "${PROTOC_REAL}" "$@"
fi

vcpkg_root="${VCPKG_ROOT:-}"
installed_root="${VCPKG_INSTALLED_DIR:-}"
host_triplet="${VCPKG_HOST_TRIPLET:-${VCPKG_TARGET_TRIPLET:-}}"

if [ -n "${vcpkg_root}" ] && [ -n "${host_triplet}" ]; then
  if [ -z "${installed_root}" ]; then
    installed_root="${vcpkg_root}/installed"
  fi
  for candidate in \
    "${installed_root}/${host_triplet}/tools/protobuf/protoc" \
    "${installed_root}/${host_triplet}/tools/protobuf/protoc-"* \
    "${vcpkg_root}/packages/protobuf_${host_triplet}/tools/protobuf/protoc" \
    "${vcpkg_root}/packages/protobuf_${host_triplet}/tools/protobuf/protoc-"*; do
    if [ -x "${candidate}" ]; then
      exec "${candidate}" "$@"
    fi
  done
fi

if command -v protoc >/dev/null 2>&1; then
  exec protoc "$@"
fi

echo "protoc_wrapper: protoc not found (VCPKG_ROOT=${vcpkg_root} VCPKG_HOST_TRIPLET=${host_triplet})" >&2
exit 127
