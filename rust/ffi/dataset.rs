use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::refs::Ref;
use lance::dataset::statistics::DatasetStatisticsExt;

use crate::constants::{
    ROW_ADDR_COLUMN, ROW_CREATED_AT_VERSION_COLUMN, ROW_ID_COLUMN,
    ROW_LAST_UPDATED_AT_VERSION_COLUMN,
};
use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::record_dataset_open;
use super::source_resolver::activate_source_resolution_for_session;
use super::types::DatasetHandle;
use super::util::{
    cstr_to_str, dataset_handle, optional_session_handle, slice_from_ptr, FfiError, FfiResult,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LanceFieldStats {
    pub field_id: u32,
    pub bytes_on_disk: u64,
}

/// Manifest string of the legacy (V1) data storage format.
const LEGACY_FILE_FORMAT_VERSION: &str = "0.1";

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LanceFragmentStats {
    pub fragment_id: u64,
    /// Number of rows in the fragment. `-1` means unknown.
    pub num_rows: i64,
    /// Physical row slots, deleted rows included. `-1` means unknown or that the
    /// dataset's storage format cannot scan a fragment in physical slices.
    pub physical_rows: i64,
    /// Sum of known data file sizes in bytes. Missing/unknown sizes are treated as 0.
    pub bytes_on_disk: u64,
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset(path: *const c_char) -> *mut c_void {
    match open_dataset_inner(path, ptr::null_mut()) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_session(
    path: *const c_char,
    session: *mut c_void,
) -> *mut c_void {
    match open_dataset_inner(path, session) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn open_dataset_inner(path: *const c_char, session: *mut c_void) -> FfiResult<DatasetHandle> {
    let session_ptr = session;
    let path_str = unsafe { cstr_to_str(path, "path")? };
    let session = unsafe { optional_session_handle(session)? };
    let dataset = match runtime::block_on(async {
        let mut builder = DatasetBuilder::from_uri(path_str);
        if let Some(session) = session {
            builder = builder.with_session(session.clone());
        }
        builder.load().await
    }) {
        Ok(Ok(ds)) => Arc::new(ds),
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetOpen,
                format!("dataset open '{path_str}': {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };
    unsafe {
        activate_source_resolution_for_session(session_ptr, &dataset);
    }
    record_dataset_open();
    Ok(DatasetHandle::new(dataset))
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> *mut c_void {
    match open_dataset_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        ptr::null_mut(),
        None,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_storage_options_and_session(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
) -> *mut c_void {
    match open_dataset_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        None,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

/// Open a dataset checked out at an explicit version.
///
/// `has_version` distinguishes "latest" from a real version number so callers
/// never need to reserve a sentinel version value.
#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_storage_options_and_version(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    has_version: u8,
    version: u64,
) -> *mut c_void {
    let requested_version = if has_version != 0 {
        Some(version)
    } else {
        None
    };
    match open_dataset_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        requested_version,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn open_dataset_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    version: Option<u64>,
) -> FfiResult<DatasetHandle> {
    let session_ptr = session;
    let path_str = unsafe { cstr_to_str(path, "path")? };
    let session = unsafe { optional_session_handle(session)? };

    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }

    let dataset = match runtime::block_on(async {
        let mut builder = DatasetBuilder::from_uri(path_str).with_storage_options(storage_options);
        if let Some(version) = version {
            builder = builder.with_version(version);
        }
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        builder.load().await
    }) {
        Ok(Ok(ds)) => Arc::new(ds),
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetOpen,
                format!("dataset open '{path_str}': {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    unsafe {
        activate_source_resolution_for_session(session_ptr, &dataset);
    }
    record_dataset_open();
    Ok(DatasetHandle::new(dataset))
}

/// Report the version the dataset handle is pinned to.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_version(dataset: *mut c_void, out_version: *mut u64) -> i32 {
    if out_version.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_version must not be null");
        return -1;
    }
    match unsafe { dataset_handle(dataset) } {
        Ok(handle) => {
            clear_last_error();
            unsafe {
                *out_version = handle.dataset.manifest.version;
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn newest_version_at_timestamp(
    versions: &[lance::dataset::Version],
    timestamp_micros: i64,
) -> Option<u64> {
    let cutoff = (
        timestamp_micros.div_euclid(1_000_000),
        timestamp_micros.rem_euclid(1_000_000) as u32 * 1_000,
    );
    versions
        .iter()
        .filter(|version| {
            (
                version.timestamp.timestamp(),
                version.timestamp.timestamp_subsec_nanos(),
            ) <= cutoff
        })
        .map(|version| version.version)
        .max()
}

/// Resolve the newest version committed at or before `timestamp_micros`
/// (microseconds since the Unix epoch, UTC).
///
/// Returns `0` and writes `out_version` when such a version exists, `1` when
/// every version is newer than the timestamp (`out_version` is untouched), and
/// `-1` on error. Reads every manifest of the dataset.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_version_at_timestamp(
    dataset: *mut c_void,
    timestamp_micros: i64,
    out_version: *mut u64,
) -> i32 {
    if out_version.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_version must not be null");
        return -1;
    }
    let handle = match unsafe { dataset_handle(dataset) } {
        Ok(handle) => handle,
        Err(err) => {
            set_last_error(err.code, err.message);
            return -1;
        }
    };
    let versions = match runtime::block_on(handle.dataset.versions()) {
        Ok(Ok(versions)) => versions,
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::DatasetOpen,
                format!("list dataset versions: {err}"),
            );
            return -1;
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return -1;
        }
    };
    clear_last_error();
    match newest_version_at_timestamp(&versions, timestamp_micros) {
        Some(version) => {
            unsafe {
                *out_version = version;
            }
            0
        }
        None => 1,
    }
}

/// Check out a version from the same dataset handle, preserving its location,
/// branch, object store and session instead of resolving the namespace again.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_checkout_version(
    dataset: *mut c_void,
    version: u64,
    out_dataset: *mut *mut c_void,
) -> i32 {
    // SAFETY: The caller supplies a live dataset handle and a writable output slot.
    unsafe { checkout_ref(dataset, Ok(Ref::from(version)), out_dataset) }
}

fn parse_named_ref(reference: &str) -> FfiResult<Ref> {
    if let Some(tag) = reference.strip_prefix("tag:") {
        return Ok(Ref::from(tag));
    }
    let (branch, version) = reference
        .strip_prefix("branch:")
        .and_then(|selector| selector.split_once(':'))
        .filter(|(branch, _)| !branch.is_empty())
        .ok_or_else(|| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                "expected branch:<name>:<version>",
            )
        })?;
    // Only one decimal spelling per version, since the SQL reference also keys
    // the statement's pinned dataset.
    let version = match version.as_bytes().first() {
        Some(b'1'..=b'9') => version.parse::<u64>().ok(),
        _ => None,
    }
    .ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "branch version must be a positive decimal integer without leading zeros",
        )
    })?;
    Ok(Ref::from((branch, version)))
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_checkout_named_ref(
    dataset: *mut c_void,
    reference: *const c_char,
    out_dataset: *mut *mut c_void,
) -> i32 {
    // SAFETY: The caller supplies a NUL-terminated reference for the duration of this call.
    let reference =
        unsafe { cstr_to_str(reference, "snapshot reference") }.and_then(parse_named_ref);
    // SAFETY: Both handles follow the same contract as numeric checkout.
    unsafe { checkout_ref(dataset, reference, out_dataset) }
}

unsafe fn checkout_ref(
    dataset: *mut c_void,
    reference: FfiResult<Ref>,
    out_dataset: *mut *mut c_void,
) -> i32 {
    if out_dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_dataset must not be null");
        return -1;
    }
    // SAFETY: The non-null output slot is writable even on failure.
    unsafe {
        ptr::write_unaligned(out_dataset, ptr::null_mut());
    }
    // SAFETY: The caller keeps the input handle alive throughout checkout.
    let handle = match unsafe { dataset_handle(dataset) } {
        Ok(handle) => handle,
        Err(err) => {
            set_last_error(err.code, err.message);
            return -1;
        }
    };
    let reference = match reference {
        Ok(reference) => reference,
        Err(err) => {
            set_last_error(err.code, err.message);
            return -1;
        }
    };
    let checked_out = match runtime::block_on(handle.dataset.checkout_version(reference)) {
        Ok(Ok(dataset)) => dataset,
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::DatasetOpen,
                format!("checkout dataset reference: {err}"),
            );
            return -1;
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return -1;
        }
    };
    let new_handle =
        Box::into_raw(Box::new(DatasetHandle::new(Arc::new(checked_out)))) as *mut c_void;
    // SAFETY: Transfer the new handle into the validated output slot.
    unsafe {
        ptr::write_unaligned(out_dataset, new_handle);
    }
    clear_last_error();
    0
}

/// Report whether the dataset manifest enables stable row ids.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_uses_stable_row_ids(
    dataset: *mut c_void,
    out_uses_stable_row_ids: *mut u8,
) -> i32 {
    if out_uses_stable_row_ids.is_null() {
        set_last_error(
            ErrorCode::InvalidArgument,
            "out_uses_stable_row_ids must not be null",
        );
        return -1;
    }
    match unsafe { dataset_handle(dataset) } {
        Ok(handle) => {
            clear_last_error();
            unsafe {
                *out_uses_stable_row_ids = u8::from(handle.dataset.manifest.uses_stable_row_ids());
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

/// Refresh a dataset handle to the latest committed version if it is stale.
///
/// Compares the handle's checked-out manifest identity against the latest
/// committed manifest on storage (a metadata-only lookup). The identity is
/// version + naming scheme + manifest e-tag, mirroring Lance's own
/// `already_checked_out` semantics: a bare version-id comparison would treat
/// a dataset that was dropped and re-created at the same URI (whose history
/// restarts at the same version id) as fresh. A missing e-tag on either side
/// is conservatively treated as stale.
///
/// If the cached manifest is stale, a *new* dataset handle checked out at the
/// latest version is written to `out_new_dataset`; the input handle is left
/// untouched so that concurrent readers of the old handle stay valid. If the
/// handle is already at the latest version, `out_new_dataset` is set to null.
///
/// Returns `0` on success and `-1` on error.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_checkout_latest_if_stale(
    dataset: *mut c_void,
    out_new_dataset: *mut *mut c_void,
) -> i32 {
    match checkout_latest_if_stale_inner(dataset, out_new_dataset) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

/// Refresh `dataset` to the latest committed version if its checked-out
/// manifest is no longer the current one. Shared by the plain and the
/// namespace-aware revalidation FFI entry points.
///
/// Returns `Ok(None)` when the dataset is already at the latest version and
/// `Ok(Some(refreshed))` when a newer (or re-created) manifest was found.
pub(crate) async fn refresh_dataset_if_stale(
    dataset: &Arc<lance::Dataset>,
) -> Result<Option<lance::Dataset>, String> {
    let (_, latest_location) = dataset
        .latest_manifest()
        .await
        .map_err(|err| format!("latest manifest lookup: {err}"))?;
    let cached_location = dataset.manifest_location();
    // Compare the full manifest identity, not just the version id: a
    // dataset dropped and re-created at the same URI restarts its history
    // at the same version id, so only the e-tag distinguishes it from the
    // cached manifest. Like Lance's own `already_checked_out`, a missing
    // e-tag on either side means the identity cannot be confirmed and the
    // entry is treated as stale.
    let is_current = latest_location.version == cached_location.version
        && latest_location.naming_scheme == cached_location.naming_scheme
        && latest_location.e_tag.as_ref().is_some_and(|latest_e_tag| {
            cached_location
                .e_tag
                .as_ref()
                .is_some_and(|cached_e_tag| latest_e_tag == cached_e_tag)
        });
    if is_current {
        return Ok(None);
    }
    // A different manifest was committed, possibly by an external writer.
    // Check out the latest version on a clone so the original handle
    // remains usable by existing holders. `checkout_latest` re-reads the
    // manifest keyed by (version, e-tag), so re-created datasets are
    // loaded fresh rather than served from the metadata cache.
    let mut refreshed_dataset = (**dataset).clone();
    refreshed_dataset
        .checkout_latest()
        .await
        .map_err(|err| format!("checkout latest: {err}"))?;
    Ok(Some(refreshed_dataset))
}

fn checkout_latest_if_stale_inner(
    dataset: *mut c_void,
    out_new_dataset: *mut *mut c_void,
) -> FfiResult<()> {
    if out_new_dataset.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_new_dataset is null",
        ));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let refreshed = match runtime::block_on(refresh_dataset_if_stale(&handle.dataset)) {
        Ok(Ok(v)) => v,
        Ok(Err(message)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetCheckoutLatest,
                format!("dataset checkout latest if stale: {message}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let new_handle = match refreshed {
        Some(refreshed_dataset) => {
            Box::into_raw(Box::new(DatasetHandle::new(Arc::new(refreshed_dataset)))) as *mut c_void
        }
        None => ptr::null_mut(),
    };

    // SAFETY: `out_new_dataset` was null-checked above and is provided by the
    // caller as a valid output location.
    unsafe {
        std::ptr::write_unaligned(out_new_dataset, new_handle);
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_dataset(dataset: *mut c_void) {
    if !dataset.is_null() {
        unsafe {
            let _ = Box::from_raw(dataset as *mut DatasetHandle);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_count_rows(dataset: *mut c_void) -> i64 {
    match dataset_count_rows_inner(dataset) {
        Ok(v) => {
            clear_last_error();
            v
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn dataset_count_rows_inner(dataset: *mut c_void) -> FfiResult<i64> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let rows = match runtime::block_on(handle.dataset.count_rows(None)) {
        Ok(Ok(rows)) => rows,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetCountRows,
                format!("dataset count_rows: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    i64::try_from(rows)
        .map_err(|_| FfiError::new(ErrorCode::DatasetCountRows, "row count overflow"))
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_schema(dataset: *mut c_void) -> *mut c_void {
    match get_schema_inner(dataset) {
        Ok(schema) => {
            clear_last_error();
            Box::into_raw(Box::new(schema)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn get_schema_inner(dataset: *mut c_void) -> FfiResult<super::types::SchemaHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    Ok(handle.read_schema.clone())
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_schema_for_scan(dataset: *mut c_void) -> *mut c_void {
    match get_schema_for_scan_inner(dataset) {
        Ok(schema) => {
            clear_last_error();
            Box::into_raw(Box::new(schema)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn get_schema_for_scan_inner(dataset: *mut c_void) -> FfiResult<super::types::SchemaHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let mut schema: Schema = (*handle.read_schema).clone();
    let has_row_id = schema.fields.iter().any(|f| f.name() == ROW_ID_COLUMN);
    if !has_row_id {
        let mut fields = schema.fields.iter().cloned().collect::<Vec<_>>();
        fields.push(Arc::new(Field::new(ROW_ID_COLUMN, DataType::UInt64, false)));
        // Row address and row lineage columns follow `_rowid`, in the order the C++ scan
        // expects: internal column indexes are the table width plus 0..=3.
        for name in [
            ROW_ADDR_COLUMN,
            ROW_CREATED_AT_VERSION_COLUMN,
            ROW_LAST_UPDATED_AT_VERSION_COLUMN,
        ] {
            fields.push(Arc::new(Field::new(name, DataType::UInt64, true)));
        }
        schema.fields = fields.into();
    }

    Ok(Arc::new(schema))
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_fragments(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut u64 {
    match dataset_list_fragments_inner(dataset, out_len) {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn dataset_list_fragments_inner(dataset: *mut c_void, out_len: *mut usize) -> FfiResult<*mut u64> {
    if out_len.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "out_len is null"));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ids: Vec<u64> = handle.dataset.fragments().iter().map(|f| f.id).collect();

    let mut boxed = ids.into_boxed_slice();
    let len = boxed.len();
    let data = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        std::ptr::write_unaligned(out_len, len);
    }
    Ok(data)
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_fragment_stats(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut LanceFragmentStats {
    match dataset_list_fragment_stats_inner(dataset, out_len) {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn dataset_list_fragment_stats_inner(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> FfiResult<*mut LanceFragmentStats> {
    if out_len.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "out_len is null"));
    }
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    // Lance rejects physical slices on legacy (V1) storage.
    let sliceable = handle
        .dataset
        .manifest
        .data_storage_format
        .lance_file_format()
        .to_manifest_string()
        != LEGACY_FILE_FORMAT_VERSION;

    let mut out: Vec<LanceFragmentStats> = Vec::with_capacity(handle.dataset.fragments().len());
    for frag in handle.dataset.fragments().iter() {
        let mut bytes_on_disk = 0u64;
        for file in frag.files.iter() {
            if let Some(sz) = file.file_size_bytes.get() {
                bytes_on_disk = bytes_on_disk.saturating_add(sz.get());
            }
        }
        let num_rows = match frag.num_rows() {
            Some(v) => i64::try_from(v).unwrap_or(-1),
            None => -1,
        };
        let physical_rows = match frag.physical_rows {
            Some(v) if sliceable => i64::try_from(v).unwrap_or(-1),
            _ => -1,
        };
        out.push(LanceFragmentStats {
            fragment_id: frag.id,
            num_rows,
            physical_rows,
            bytes_on_disk,
        });
    }

    let mut boxed = out.into_boxed_slice();
    let len = boxed.len();
    let data = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        std::ptr::write_unaligned(out_len, len);
    }
    Ok(data)
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_fragment_list(ptr: *mut u64, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        let _ = Box::<[u64]>::from_raw(slice);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_fragment_stats_list(ptr: *mut LanceFragmentStats, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        let _ = Box::<[LanceFragmentStats]>::from_raw(slice);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_field_stats(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut LanceFieldStats {
    match dataset_list_field_stats_inner(dataset, out_len) {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

/// Transfer a boxed slice to C, writing its length to `out_len`.
unsafe fn boxed_slice_to_c<T>(items: Vec<T>, out_len: *mut usize) -> *mut T {
    let mut boxed = items.into_boxed_slice();
    let len = boxed.len();
    let data = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    std::ptr::write_unaligned(out_len, len);
    data
}

macro_rules! fetch_data_stats {
    ($handle:expr) => {
        match runtime::block_on($handle.dataset.calculate_data_stats()) {
            Ok(Ok(stats)) => stats,
            Ok(Err(err)) => {
                return Err(FfiError::new(
                    ErrorCode::DatasetCalculateDataStats,
                    format!("dataset calculate_data_stats: {err}"),
                ))
            }
            Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    };
}

fn dataset_list_field_stats_inner(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> FfiResult<*mut LanceFieldStats> {
    if out_len.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "out_len is null"));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let stats = fetch_data_stats!(handle);

    let out: Vec<LanceFieldStats> = stats
        .fields
        .into_iter()
        .map(|field| LanceFieldStats {
            field_id: field.id,
            bytes_on_disk: field.bytes_on_disk,
        })
        .collect();

    Ok(unsafe { boxed_slice_to_c(out, out_len) })
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_field_stats_list(ptr: *mut LanceFieldStats, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        let _ = Box::<[LanceFieldStats]>::from_raw(slice);
    }
}

#[repr(C)]
pub struct LanceNamedFieldStats {
    pub name: *const c_char,
    pub bytes_on_disk: u64,
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_named_field_stats(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut LanceNamedFieldStats {
    match dataset_list_named_field_stats_inner(dataset, out_len) {
        Ok(ptr) => {
            clear_last_error();
            ptr
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn dataset_list_named_field_stats_inner(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> FfiResult<*mut LanceNamedFieldStats> {
    if out_len.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "out_len is null"));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let stats = fetch_data_stats!(handle);

    // Build field_id → name map from lance schema.
    let lance_schema = handle.dataset.schema();
    let mut id_to_name: HashMap<i32, String> = HashMap::new();
    fn collect_field_names(
        fields: &[lance_core::datatypes::Field],
        map: &mut HashMap<i32, String>,
    ) {
        for f in fields {
            map.insert(f.id, f.name.clone());
            collect_field_names(&f.children, map);
        }
    }
    collect_field_names(&lance_schema.fields, &mut id_to_name);

    let out: Vec<LanceNamedFieldStats> = stats
        .fields
        .into_iter()
        .filter_map(|field| {
            let name = id_to_name.get(&(field.id as i32))?;
            Some(LanceNamedFieldStats {
                name: super::util::to_c_string(name).into_raw(),
                bytes_on_disk: field.bytes_on_disk,
            })
        })
        .collect();

    Ok(unsafe { boxed_slice_to_c(out, out_len) })
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_named_field_stats_list(
    ptr: *mut LanceNamedFieldStats,
    len: usize,
) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        for item in slice.iter() {
            if !item.name.is_null() {
                let _ = std::ffi::CString::from_raw(item.name as *mut c_char);
            }
        }
        let boxed = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        drop(boxed);
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::fs;

    use arrow_array::{Int32Array, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance::dataset::{WriteMode, WriteParams};
    use lance::Dataset;

    use super::*;

    /// Commit a batch to `uri` through the Lance Rust API, bypassing any FFI
    /// dataset handle (i.e. acting as an external writer).
    fn external_write(uri: &str, ids: Vec<i32>, mode: WriteMode) {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = arrow_array::RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(ids))],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let params = WriteParams {
            mode,
            ..Default::default()
        };
        runtime::block_on(Dataset::write(reader, uri, Some(params)))
            .unwrap()
            .unwrap();
    }

    #[test]
    fn test_checkout_latest_if_stale_sees_external_commits() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();
        let dataset_dir =
            std::env::temp_dir().join(format!("ffi-dataset-revalidate-{}", rand::random::<u64>()));
        let uri = dataset_dir.to_string_lossy().to_string();

        external_write(&uri, vec![1, 2, 3], WriteMode::Create);

        unsafe {
            let uri_c = CString::new(uri.clone()).unwrap();
            let handle = lance_open_dataset(uri_c.as_ptr());
            assert!(!handle.is_null());
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // Already at the latest version: no refreshed handle is produced.
            let mut refreshed: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(handle, &mut refreshed),
                0
            );
            assert!(refreshed.is_null());

            // External append committed behind the open handle's back.
            external_write(&uri, vec![4, 5], WriteMode::Append);

            // The stale handle still serves the old version...
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // ...but revalidation yields a refreshed handle at the latest
            // version while leaving the original handle untouched.
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(handle, &mut refreshed),
                0
            );
            assert!(!refreshed.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed), 5);
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // External delete on top of the refreshed handle.
            {
                let mut ds = runtime::block_on(Dataset::open(&uri)).unwrap().unwrap();
                runtime::block_on(ds.delete("id = 1")).unwrap().unwrap();
            }
            let mut refreshed_again: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(refreshed, &mut refreshed_again),
                0
            );
            assert!(!refreshed_again.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed_again), 4);

            lance_close_dataset(refreshed_again);
            lance_close_dataset(refreshed);
            lance_close_dataset(handle);
        }

        let _ = fs::remove_dir_all(dataset_dir);
    }

    #[test]
    fn test_checkout_latest_if_stale_sees_recreated_dataset_with_same_version_id() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();
        let dataset_dir = std::env::temp_dir().join(format!(
            "ffi-dataset-revalidate-recreate-{}",
            rand::random::<u64>()
        ));
        let uri = dataset_dir.to_string_lossy().to_string();

        external_write(&uri, vec![1, 2, 3], WriteMode::Create);

        unsafe {
            let uri_c = CString::new(uri.clone()).unwrap();
            let handle = lance_open_dataset(uri_c.as_ptr());
            assert!(!handle.is_null());
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // External drop + re-create at the same URI: the new dataset's
            // history restarts at the same version id as the cached handle,
            // so only the manifest e-tag distinguishes the two tables.
            fs::remove_dir_all(&dataset_dir).unwrap();
            external_write(&uri, vec![10, 20], WriteMode::Create);

            let mut refreshed: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(handle, &mut refreshed),
                0
            );
            assert!(!refreshed.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed), 2);

            lance_close_dataset(refreshed);
            lance_close_dataset(handle);
        }

        let _ = fs::remove_dir_all(dataset_dir);
    }

    #[test]
    fn test_checkout_latest_if_stale_rejects_invalid_arguments() {
        unsafe {
            // Null output pointer.
            let mut refreshed: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(ptr::null_mut(), &mut refreshed),
                -1
            );
            // Null dataset handle.
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(ptr::null_mut(), ptr::null_mut()),
                -1
            );
        }
    }

    #[test]
    fn test_checkout_latest_if_stale_errors_when_dataset_removed() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();
        let dataset_dir = std::env::temp_dir().join(format!(
            "ffi-dataset-revalidate-removed-{}",
            rand::random::<u64>()
        ));
        let uri = dataset_dir.to_string_lossy().to_string();

        external_write(&uri, vec![1, 2, 3], WriteMode::Create);

        unsafe {
            let uri_c = CString::new(uri.clone()).unwrap();
            let handle = lance_open_dataset(uri_c.as_ptr());
            assert!(!handle.is_null());

            // The dataset vanishing from storage must surface as an error,
            // not as a silent stale read.
            fs::remove_dir_all(&dataset_dir).unwrap();
            let mut refreshed: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_latest_if_stale(handle, &mut refreshed),
                -1
            );
            assert!(refreshed.is_null());

            lance_close_dataset(handle);
        }
    }

    #[test]
    fn test_version_at_timestamp_preserves_submicrosecond_precision() {
        let version = |version: u64, seconds: i64, nanos: u32| lance::dataset::Version {
            version,
            timestamp: chrono::DateTime::from_timestamp(seconds, nanos).unwrap(),
            metadata: Default::default(),
        };
        let versions = vec![version(1, 0, 0), version(2, 1, 999), version(3, 2, 0)];

        // Version 2 commits 999ns after this cutoff and must not be selected.
        assert_eq!(newest_version_at_timestamp(&versions, 1_000_000), Some(1));
        assert_eq!(newest_version_at_timestamp(&versions, 1_000_001), Some(2));
        assert_eq!(newest_version_at_timestamp(&versions, 2_000_000), Some(3));

        // Euclidean cutoff decomposition also keeps pre-epoch timestamps exact.
        let pre_epoch = vec![version(1, -1, 999_999_999)]; // one ns before epoch
        assert_eq!(newest_version_at_timestamp(&pre_epoch, -1), None);
        assert_eq!(newest_version_at_timestamp(&pre_epoch, 0), Some(1));
    }

    #[test]
    fn test_checkout_version_preserves_source_dataset_identity() {
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();
        let dataset_dir = std::env::temp_dir().join(format!(
            "ffi-dataset-at-ts-identity-{}",
            rand::random::<u64>()
        ));
        let uri = dataset_dir.to_string_lossy().to_string();

        external_write(&uri, vec![1, 2, 3], WriteMode::Create);
        external_write(&uri, vec![4], WriteMode::Append);
        let source = runtime::block_on(Dataset::open(&uri)).unwrap().unwrap();
        let versions = runtime::block_on(source.versions()).unwrap().unwrap();
        let first_timestamp = versions
            .iter()
            .find(|version| version.version == 1)
            .unwrap()
            .timestamp;
        let mut first_cutoff = first_timestamp.timestamp_micros();
        if first_timestamp.timestamp_subsec_nanos() % 1_000 != 0 {
            first_cutoff += 1;
        }
        drop(versions);
        drop(source);

        unsafe {
            let uri_c = CString::new(uri.clone()).unwrap();
            let handle = lance_open_dataset(uri_c.as_ptr());
            assert!(!handle.is_null());
            let mut version = 0u64;
            assert_eq!(
                lance_dataset_version_at_timestamp(handle, first_cutoff, &mut version),
                0
            );
            assert_eq!(version, 1);

            let mut checked_out: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_checkout_version(handle, version, &mut checked_out),
                0
            );
            assert!(!checked_out.is_null());
            assert_eq!(lance_dataset_count_rows(checked_out), 3);
            assert_eq!(lance_dataset_count_rows(handle), 4);

            let source_handle = dataset_handle(handle).unwrap();
            let checked_out_handle = dataset_handle(checked_out).unwrap();
            assert_eq!(
                checked_out_handle.dataset.uri(),
                source_handle.dataset.uri()
            );
            assert_eq!(
                checked_out_handle.dataset.manifest.branch,
                source_handle.dataset.manifest.branch
            );
            assert!(Arc::ptr_eq(
                &checked_out_handle.dataset.session(),
                &source_handle.dataset.session()
            ));

            lance_close_dataset(checked_out);
            lance_close_dataset(handle);
        }

        let _ = fs::remove_dir_all(dataset_dir);
    }

    #[test]
    fn test_version_at_timestamp_picks_newest_version_not_after_timestamp() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();
        let dataset_dir =
            std::env::temp_dir().join(format!("ffi-dataset-at-ts-{}", rand::random::<u64>()));
        let uri = dataset_dir.to_string_lossy().to_string();

        external_write(&uri, vec![1], WriteMode::Create);
        std::thread::sleep(std::time::Duration::from_millis(20));
        external_write(&uri, vec![2], WriteMode::Append);
        std::thread::sleep(std::time::Duration::from_millis(20));
        external_write(&uri, vec![3], WriteMode::Append);

        let timestamp = |version: u64| {
            let ds = runtime::block_on(lance::Dataset::open(&uri))
                .unwrap()
                .unwrap();
            let versions = runtime::block_on(ds.versions()).unwrap().unwrap();
            versions
                .iter()
                .find(|v| v.version == version)
                .unwrap()
                .timestamp
        };
        let (commit1, commit2, commit3) = (timestamp(1), timestamp(2), timestamp(3));
        let inclusive_micros = |timestamp: chrono::DateTime<chrono::Utc>| {
            timestamp.timestamp_micros()
                + i64::from(timestamp.timestamp_subsec_nanos() % 1_000 != 0)
        };
        let (ts1, ts2, ts3) = (
            commit1.timestamp_micros(),
            inclusive_micros(commit2),
            commit3.timestamp_micros(),
        );
        assert!(ts1 < ts2 && ts2 < ts3);

        unsafe {
            let uri_c = CString::new(uri.clone()).unwrap();
            let handle = lance_open_dataset(uri_c.as_ptr());
            assert!(!handle.is_null());

            let resolve = |ts: i64| {
                let mut version = 0u64;
                let rc = lance_dataset_version_at_timestamp(handle, ts, &mut version);
                (rc, version)
            };
            // The first microsecond containing the full commit time includes
            // it; one microsecond earlier does not, even within the same
            // truncated microsecond as the commit.
            assert_eq!(resolve(ts2), (0, 2));
            assert_eq!(resolve(ts2 - 1), (0, 1));
            assert_eq!(resolve(ts3 + 1_000_000), (0, 3));
            // Every version is newer than the timestamp: no error, no version.
            assert_eq!(resolve(ts1 - 1), (1, 0));

            let mut version = 0u64;
            assert_eq!(
                lance_dataset_version_at_timestamp(handle, ts2, ptr::null_mut()),
                -1
            );
            assert_eq!(
                lance_dataset_version_at_timestamp(ptr::null_mut(), ts2, &mut version),
                -1
            );

            lance_close_dataset(handle);
        }

        let _ = fs::remove_dir_all(dataset_dir);
    }
}
