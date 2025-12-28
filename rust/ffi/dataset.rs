use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use datafusion_sql::unparser::expr_to_sql;
use lance::dataset::statistics::DatasetStatisticsExt;
use lance::dataset::builder::DatasetBuilder;
use lance::Dataset;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::DatasetHandle;
use super::util::{cstr_to_str, parse_optional_filter_ir, slice_from_ptr, FfiError, FfiResult};

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
    match open_dataset_inner(path) {
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

fn open_dataset_inner(path: *const c_char) -> FfiResult<DatasetHandle> {
    let path_str = unsafe { cstr_to_str(path, "path")? };
    let dataset = match runtime::block_on(Dataset::open(path_str)) {
        Ok(Ok(ds)) => Arc::new(ds),
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetOpen,
                format!("dataset open '{path_str}': {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };
    Ok(DatasetHandle::new(dataset))
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> *mut c_void {
    match open_dataset_with_storage_options_inner(path, option_keys, option_values, options_len) {
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
) -> FfiResult<DatasetHandle> {
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

    let dataset = match runtime::block_on(async {
        DatasetBuilder::from_uri(path_str)
            .with_storage_options(storage_options)
            .load()
            .await
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

    Ok(DatasetHandle::new(dataset))
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

fn dataset_list_field_stats_inner(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> FfiResult<*mut LanceFieldStats> {
    if out_len.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "out_len is null"));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let stats = match runtime::block_on(handle.dataset.calculate_data_stats()) {
        Ok(Ok(stats)) => stats,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetCalculateDataStats,
                format!("dataset calculate_data_stats: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let mut out: Vec<LanceFieldStats> = Vec::with_capacity(stats.fields.len());
    for field in stats.fields {
        out.push(LanceFieldStats {
            field_id: field.id,
            bytes_on_disk: field.bytes_on_disk,
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
pub unsafe extern "C" fn lance_free_field_stats_list(ptr: *mut LanceFieldStats, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = std::ptr::slice_from_raw_parts_mut(ptr, len);
        let _ = Box::<[LanceFieldStats]>::from_raw(slice);
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

    let before_rows = match runtime::block_on(ds.count_rows(None)) {
        Ok(Ok(rows)) => rows,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("dataset count_rows(before): {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    match runtime::block_on(ds.delete(&predicate)) {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("dataset delete: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let after_rows = match runtime::block_on(ds.count_rows(None)) {
        Ok(Ok(rows)) => rows,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("dataset count_rows(after): {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let deleted_rows = before_rows.saturating_sub(after_rows);
    let deleted_rows_i64 = i64::try_from(deleted_rows)
        .map_err(|_| FfiError::new(ErrorCode::DatasetDelete, "deleted row count overflow"))?;

    unsafe {
        std::ptr::write_unaligned(out_deleted_rows, deleted_rows_i64);
    }
    Ok(())
}
