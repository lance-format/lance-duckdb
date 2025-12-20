cmake_minimum_required(VERSION 3.5)

# Set extension name here
set(TARGET_NAME lance)

# No external dependencies for now

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)

set(EXTENSION_SOURCES src/lance_extension.cpp src/lance_scan.cpp
                      src/lance_replacement.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Platform libs required by Rust deps (e.g., TLS, timezone)
if(APPLE)
  if(NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(CMAKE_OSX_DEPLOYMENT_TARGET
        13.3
        CACHE STRING "Minimum OS X deployment version" FORCE)
  endif()
  set(PLATFORM_LIBS
      "-framework CoreFoundation"
      "-framework SystemConfiguration"
      "-framework Security")
elseif(UNIX)
  set(PLATFORM_LIBS m c)
elseif(WIN32)
  set(PLATFORM_LIBS ws2_32 userenv advapi32)
endif()

# Build and link Rust staticlib (lance_duckdb_ffi)
set(RUST_PLATFORM_TARGET "")

set(DUCKDB_PLATFORM "$ENV{DUCKDB_PLATFORM}")
if(DUCKDB_PLATFORM STREQUAL "linux_amd64_musl")
  set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-musl")
elseif(DUCKDB_PLATFORM STREQUAL "linux_arm64_musl")
  set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-musl")
endif()

if(RUST_PLATFORM_TARGET STREQUAL "")
  if("${OS_NAME}" STREQUAL "linux")
    if("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-gnu")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-gnu")
    endif()
  elseif("${OS_NAME}" STREQUAL "osx")
    if("${OSX_BUILD_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    elseif("${OSX_BUILD_ARCH}" STREQUAL "x86_64")
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    elseif("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    endif()
  elseif("${OS_NAME}" STREQUAL "windows")
    if("${OS_ARCH}" STREQUAL "arm64")
      set(RUST_PLATFORM_TARGET "aarch64-pc-windows-msvc")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-pc-windows-msvc")
    endif()
  endif()
endif()

if(RUST_PLATFORM_TARGET STREQUAL "")
  if(APPLE)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(RUST_PLATFORM_TARGET "aarch64-apple-darwin")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-apple-darwin")
    endif()
  elseif(UNIX)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
      set(RUST_PLATFORM_TARGET "aarch64-unknown-linux-gnu")
    else()
      set(RUST_PLATFORM_TARGET "x86_64-unknown-linux-gnu")
    endif()
  elseif(WIN32)
    set(RUST_PLATFORM_TARGET "x86_64-pc-windows-msvc")
  endif()
endif()

if(WIN32)
  set(RUST_STATICLIB_NAME "lance_duckdb_ffi.lib")
else()
  set(RUST_STATICLIB_NAME "liblance_duckdb_ffi.a")
endif()

set(RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
set(RUST_DEBUG_LIB "${RUST_TARGET_DIR}/${RUST_PLATFORM_TARGET}/debug/${RUST_STATICLIB_NAME}")
set(RUST_RELEASE_LIB "${RUST_TARGET_DIR}/${RUST_PLATFORM_TARGET}/release/${RUST_STATICLIB_NAME}")

set(RUST_FFI_DEPENDS
    ${CMAKE_CURRENT_LIST_DIR}/Cargo.toml
    ${CMAKE_CURRENT_LIST_DIR}/Cargo.lock
    ${CMAKE_CURRENT_LIST_DIR}/rust/lib.rs
    ${CMAKE_CURRENT_LIST_DIR}/rust/scanner.rs
    ${CMAKE_CURRENT_LIST_DIR}/rust/types.rs
    ${CMAKE_CURRENT_LIST_DIR}/rust/writer.rs
    ${CMAKE_CURRENT_LIST_DIR}/scripts/protoc_wrapper.sh
    ${CMAKE_CURRENT_LIST_DIR}/scripts/protoc_wrapper.cmd)

set(PROTOC_BIN "")
if(WIN32)
  set(PROTOC_BIN "${CMAKE_CURRENT_LIST_DIR}/scripts/protoc_wrapper.cmd")
else()
  set(PROTOC_BIN "${CMAKE_CURRENT_LIST_DIR}/scripts/protoc_wrapper.sh")
endif()

set(RUST_CARGO_ENV
    CARGO_TARGET_DIR=${RUST_TARGET_DIR}
    MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
list(APPEND RUST_CARGO_ENV PROTOC=${PROTOC_BIN})
if(DEFINED VCPKG_ROOT)
  list(APPEND RUST_CARGO_ENV VCPKG_ROOT=${VCPKG_ROOT})
endif()
if(DEFINED VCPKG_INSTALLED_DIR)
  list(APPEND RUST_CARGO_ENV VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR})
endif()
if(DEFINED VCPKG_HOST_TRIPLET)
  list(APPEND RUST_CARGO_ENV VCPKG_HOST_TRIPLET=${VCPKG_HOST_TRIPLET})
endif()
if(DEFINED VCPKG_TARGET_TRIPLET)
  list(APPEND RUST_CARGO_ENV VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET})
endif()

# Building both debug and release Rust artifacts is expensive and can trigger OOMs in CI.
# For single-config generators (Ninja/Makefiles), only build the active configuration's artifact.
if(CMAKE_CONFIGURATION_TYPES)
  add_custom_command(
    OUTPUT "${RUST_DEBUG_LIB}" "${RUST_RELEASE_LIB}"
    COMMAND
      ${CMAKE_COMMAND} -E env
      ${RUST_CARGO_ENV}
      cargo build --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml --target=${RUST_PLATFORM_TARGET}
    COMMAND
      ${CMAKE_COMMAND} -E env
      ${RUST_CARGO_ENV}
      cargo build --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml --release --target=${RUST_PLATFORM_TARGET}
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
    DEPENDS ${RUST_FFI_DEPENDS}
    VERBATIM)

  add_custom_target(lance_duckdb_ffi_build DEPENDS "${RUST_DEBUG_LIB}" "${RUST_RELEASE_LIB}")
else()
  if(NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(
      OUTPUT "${RUST_DEBUG_LIB}"
      COMMAND
        ${CMAKE_COMMAND} -E env
        ${RUST_CARGO_ENV}
        cargo build --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml --target=${RUST_PLATFORM_TARGET}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      DEPENDS ${RUST_FFI_DEPENDS}
      VERBATIM)

    add_custom_target(lance_duckdb_ffi_build DEPENDS "${RUST_DEBUG_LIB}")
  else()
    add_custom_command(
      OUTPUT "${RUST_RELEASE_LIB}"
      COMMAND
        ${CMAKE_COMMAND} -E env
        ${RUST_CARGO_ENV}
        cargo build --manifest-path=${CMAKE_CURRENT_LIST_DIR}/Cargo.toml --release --target=${RUST_PLATFORM_TARGET}
      WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
      DEPENDS ${RUST_FFI_DEPENDS}
      VERBATIM)

    add_custom_target(lance_duckdb_ffi_build DEPENDS "${RUST_RELEASE_LIB}")
  endif()
endif()

target_link_libraries(
  ${EXTENSION_NAME}
  debug
  "${RUST_DEBUG_LIB}"
  optimized
  "${RUST_RELEASE_LIB}"
  ${PLATFORM_LIBS})
add_dependencies(${EXTENSION_NAME} lance_duckdb_ffi_build)

target_link_libraries(
  ${LOADABLE_EXTENSION_NAME}
  debug
  "${RUST_DEBUG_LIB}"
  optimized
  "${RUST_RELEASE_LIB}"
  ${PLATFORM_LIBS})
add_dependencies(${LOADABLE_EXTENSION_NAME} lance_duckdb_ffi_build)

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
