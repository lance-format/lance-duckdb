use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion::scalar::ScalarValue;
use datafusion_sql::unparser::expr_to_sql;
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::statistics::DatasetStatisticsExt;
use lance::dataset::transaction::{Operation, Transaction};
use roaring::RoaringTreemap;

use crate::constants::ROW_ID_COLUMN;
use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::record_dataset_open;
use super::types::DatasetHandle;
use super::update::{apply_deletions, build_row_id_index, CapturedRowIds};
use super::util::{
    cstr_to_str, optional_session_handle, parse_optional_filter_ir, slice_from_ptr, FfiError,
    FfiResult,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LanceFieldStats {
    pub field_id: u32,
    pub bytes_on_disk: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LanceFragmentStats {
    pub fragment_id: u64,
    /// Number of rows in the fragment. `-1` means unknown.
    pub num_rows: i64,
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
    let path_str = unsafe { cstr_to_str(path, "path")? };
    let session = unsafe { optional_session_handle(session)? };
    let dataset = match runtime::block_on(async {
        let mut builder = DatasetBuilder::from_uri(path_str);
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
) -> FfiResult<DatasetHandle> {
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

    record_dataset_open();
    Ok(DatasetHandle::new(dataset))
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
    Ok(handle.arrow_schema.clone())
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

    let mut schema: Schema = (*handle.arrow_schema).clone();
    let has_row_id = schema.fields.iter().any(|f| f.name() == ROW_ID_COLUMN);
    if !has_row_id {
        let mut fields = schema.fields.iter().cloned().collect::<Vec<_>>();
        fields.push(Arc::new(Field::new(ROW_ID_COLUMN, DataType::UInt64, false)));
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
        out.push(LanceFragmentStats {
            fragment_id: frag.id,
            num_rows,
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

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_delete(
    dataset: *mut c_void,
    filter_ir: *const u8,
    filter_ir_len: usize,
    out_deleted_rows: *mut i64,
) -> i32 {
    match dataset_delete_inner(dataset, filter_ir, filter_ir_len, out_deleted_rows) {
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

#[no_mangle]
pub unsafe extern "C" fn lance_delete_transaction_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    session: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> i32 {
    match delete_transaction_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        filter_ir,
        filter_ir_len,
        session,
        out_transaction,
        out_deleted_rows,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn delete_transaction_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    session: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> FfiResult<()> {
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }
    if out_deleted_rows.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_deleted_rows is null",
        ));
    }

    let path_str = unsafe { cstr_to_str(path, "path")? };
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

    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::DatasetDelete,
            "delete filter_ir",
        )?
    };
    let predicate = match filter {
        Some(expr) => expr_to_sql(&expr)
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetDelete, format!("predicate sql: {err}"))
            })?
            .to_string(),
        None => "true".to_string(),
    };
    let session = unsafe { optional_session_handle(session)? };

    let (maybe_txn, deleted_rows) = match runtime::block_on(async {
        let mut builder = DatasetBuilder::from_uri(path_str).with_storage_options(storage_options);
        if let Some(session) = session.clone() {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|e| e.to_string())?;
        let dataset = Arc::new(dataset);

        let mut scanner = dataset.scan();
        scanner
            .with_row_id()
            .project(&[lance_core::ROW_ID])
            .map_err(|e| e.to_string())?
            .filter(&predicate)
            .map_err(|e| e.to_string())?;

        let Some(filter_expr) = scanner.get_expr_filter().map_err(|e| e.to_string())? else {
            return Ok::<_, String>((None, 0_i64));
        };

        if matches!(
            filter_expr,
            Expr::Literal(ScalarValue::Boolean(Some(false)), _)
        ) {
            return Ok::<_, String>((None, 0_i64));
        }

        let (updated_fragments, deleted_fragment_ids, deleted_rows) = if matches!(
            filter_expr,
            Expr::Literal(ScalarValue::Boolean(Some(true)), _)
        ) {
            let deleted_fragment_ids = dataset
                .get_fragments()
                .iter()
                .map(|f| f.id() as u64)
                .collect::<Vec<_>>();
            let deleted_rows = dataset.count_rows(None).await.map_err(|e| e.to_string())?;
            let deleted_rows = i64::try_from(deleted_rows)
                .map_err(|_| "deleted row count overflow".to_string())?;
            (Vec::new(), deleted_fragment_ids, deleted_rows)
        } else {
            let stable_row_ids = dataset.manifest.uses_stable_row_ids();
            let mut captured_row_ids = CapturedRowIds::new(stable_row_ids);

            let stream: SendableRecordBatchStream = scanner
                .try_into_stream()
                .await
                .map_err(|e| e.to_string())?
                .into();

            futures::pin_mut!(stream);
            while let Some(batch) = stream.try_next().await.map_err(|e| e.to_string())? {
                let row_ids = batch
                    .column_by_name(lance_core::ROW_ID)
                    .ok_or_else(|| "missing _rowid column".to_string())?
                    .as_primitive::<UInt64Type>()
                    .values();
                captured_row_ids
                    .capture(row_ids)
                    .map_err(|e| e.to_string())?;
            }

            let deleted_rows = captured_row_ids.len();
            let deleted_rows_i64 = i64::try_from(deleted_rows)
                .map_err(|_| "deleted row count overflow".to_string())?;

            let row_addrs = match &captured_row_ids {
                CapturedRowIds::AddressStyle(addrs) => addrs.clone(),
                CapturedRowIds::SequenceStyle(sequence) => {
                    let row_id_index = build_row_id_index(dataset.as_ref()).await?;
                    let mut addrs = RoaringTreemap::new();
                    for row_id in sequence.iter() {
                        let addr = row_id_index
                            .get(row_id)
                            .ok_or_else(|| format!("row id missing from row id index: {row_id}"))?;
                        addrs.insert(u64::from(addr));
                    }
                    addrs
                }
            };

            let (fragments, deleted_ids) = apply_deletions(dataset.as_ref(), &row_addrs).await?;
            (fragments, deleted_ids, deleted_rows_i64)
        };

        if updated_fragments.is_empty() && deleted_fragment_ids.is_empty() {
            return Ok::<_, String>((None, deleted_rows));
        }

        let operation = Operation::Delete {
            updated_fragments,
            deleted_fragment_ids,
            predicate,
        };
        let txn = Transaction::new(dataset.manifest.version, operation, None);
        Ok::<_, String>((Some(txn), deleted_rows))
    }) {
        Ok(Ok(v)) => v,
        Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetDelete, message)),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("runtime: {err}"),
            ))
        }
    };

    unsafe {
        std::ptr::write_unaligned(out_deleted_rows, deleted_rows);
        std::ptr::write_unaligned(out_transaction, std::ptr::null_mut());
    }

    if let Some(txn) = maybe_txn {
        let boxed = Box::new(txn);
        unsafe {
            std::ptr::write_unaligned(out_transaction, Box::into_raw(boxed) as *mut c_void);
        }
    }

    Ok(())
}

fn dataset_delete_inner(
    dataset: *mut c_void,
    filter_ir: *const u8,
    filter_ir_len: usize,
    out_deleted_rows: *mut i64,
) -> FfiResult<()> {
    if out_deleted_rows.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_deleted_rows is null",
        ));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::DatasetDelete,
            "delete filter_ir",
        )?
    };
    let predicate = match filter {
        Some(expr) => expr_to_sql(&expr)
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetDelete, format!("predicate sql: {err}"))
            })?
            .to_string(),
        None => "true".to_string(),
    };

    let mut ds = (*handle.dataset).clone();

    let deleted_rows = match runtime::block_on(ds.delete(&predicate)) {
        Ok(Ok(result)) => result.num_deleted_rows,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("dataset delete: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };
    let deleted_rows_i64 = i64::try_from(deleted_rows)
        .map_err(|_| FfiError::new(ErrorCode::DatasetDelete, "deleted row count overflow"))?;

    unsafe {
        std::ptr::write_unaligned(out_deleted_rows, deleted_rows_i64);
    }
    Ok(())
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
}
