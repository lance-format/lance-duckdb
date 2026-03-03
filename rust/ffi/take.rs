use std::ffi::{c_char, c_void};
use std::ptr;

use lance::dataset::ProjectionRequest;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::StreamHandle;
use super::util::{optional_cstr_array, slice_from_ptr, FfiError, FfiResult};

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_take_stream(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
) -> *mut c_void {
    match create_dataset_take_stream_inner(
        dataset,
        row_ids,
        row_ids_len,
        columns,
        columns_len,
        true,
    ) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_take_stream_unfiltered(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
) -> *mut c_void {
    match create_dataset_take_stream_inner(
        dataset,
        row_ids,
        row_ids_len,
        columns,
        columns_len,
        false,
    ) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn create_dataset_take_stream_inner(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
    filter_out_of_range: bool,
) -> FfiResult<StreamHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let row_ids = if row_ids_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? }
    };
    let row_ids_filtered;
    let row_ids = if !filter_out_of_range || row_ids.is_empty() {
        row_ids
    } else {
        let max_row_id = if handle.dataset.manifest.uses_stable_row_ids() {
            handle.dataset.manifest.next_row_id
        } else {
            handle
                .dataset
                .manifest
                .fragments
                .iter()
                .map(|fragment| fragment.num_rows().unwrap_or_default() as u64)
                .sum::<u64>()
        };
        if row_ids.iter().all(|id| *id < max_row_id) {
            row_ids
        } else {
            row_ids_filtered = row_ids
                .iter()
                .copied()
                .filter(|id| *id < max_row_id)
                .collect::<Vec<_>>();
            row_ids_filtered.as_slice()
        }
    };

    let projection_cols = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    let projection = if projection_cols.is_empty() {
        ProjectionRequest::from_schema(handle.dataset.schema().clone())
    } else {
        ProjectionRequest::from_columns(
            projection_cols.iter().map(|s| s.as_str()),
            handle.dataset.schema(),
        )
    };

    let batch = match runtime::block_on(handle.dataset.take_rows(row_ids, projection)) {
        Ok(Ok(batch)) => batch,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetTake,
                format!("dataset take_rows: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    Ok(StreamHandle::Batches(vec![batch].into_iter()))
}
