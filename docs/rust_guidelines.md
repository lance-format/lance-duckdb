# Rust Implementation Guidelines

Unless Lance upstream conventions dictate otherwise, follow the defaults below. These are guidelines, not immutable constraints.

## Unsafe & FFI Conventions

- Every `unsafe` block must have a `// SAFETY:` comment. Minimize unsafe scope.
- All public FFI functions follow a two-layer pattern: an outer `#[no_mangle] pub unsafe extern "C"` wrapper that converts between C types and error codes, and an inner function returning `FfiResult<T>`. See existing functions in `rust/ffi/` for the canonical pattern.
- Return `0` for success, `-1` for error. On error, store details via `set_last_error(code, message)` into thread-local storage. C++ retrieves errors via `lance_last_error_code()` / `lance_last_error_message()`.
- Transfer ownership to C with `Box::into_raw()`; reclaim with `Box::from_raw()` in the corresponding `lance_free_*()` function.
- When adding new error variants, append to the `ErrorCode` enum in `rust/error.rs` rather than reusing existing codes.
