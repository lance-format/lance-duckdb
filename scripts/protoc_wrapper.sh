#!/usr/bin/env sh
set -eu

if [ -n "${PROTOC_REAL:-}" ] && [ -x "${PROTOC_REAL}" ]; then
  exec "${PROTOC_REAL}" "$@"
fi

vcpkg_root="${VCPKG_ROOT:-}"
host_triplet="${VCPKG_HOST_TRIPLET:-${VCPKG_TARGET_TRIPLET:-}}"

if [ -n "${vcpkg_root}" ] && [ -n "${host_triplet}" ]; then
  for candidate in \
    "${vcpkg_root}/installed/${host_triplet}/tools/protobuf/protoc" \
    "${vcpkg_root}/installed/${host_triplet}/tools/protobuf/protoc-"* \
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
