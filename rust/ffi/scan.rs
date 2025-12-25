use std::ffi::{c_char, c_void};
use std::ptr;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use crate::scanner::LanceStream;

use super::types::StreamHandle;
use super::util::{
    optional_cstr_array, parse_optional_filter_ir, to_c_string, u64_to_usize, FfiError, FfiResult,
};

#[no_mangle]
pub unsafe extern "C" fn lance_create_fragment_stream_ir(
    dataset: *mut c_void,
    fragment_id: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> *mut c_void {
    match create_fragment_stream_ir_inner(
        dataset,
        fragment_id,
        columns,
        columns_len,
        filter_ir,
        filter_ir_len,
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

fn create_fragment_stream_ir_inner(
    dataset: *mut c_void,
    fragment_id: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> FfiResult<StreamHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let fragment_id_usize = u64_to_usize(fragment_id, "fragment_id")?;

    let fragment = handle
        .dataset
        .get_fragment(fragment_id_usize)
        .ok_or_else(|| {
            FfiError::new(
                ErrorCode::FragmentScan,
                format!("fragment not found: {fragment_id}"),
            )
        })?;

    let mut scan = fragment.scan();

    let projection = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    if !projection.is_empty() {
        scan.project(&projection).map_err(|err| {
            FfiError::new(
                ErrorCode::FragmentScan,
                format!("fragment scan project: {err}"),
            )
        })?;
    }

    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::FragmentScan,
            "fragment filter_ir",
        )?
    };
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }

    scan.scan_in_order(false);
    let stream = LanceStream::from_scanner(scan)
        .map_err(|err| FfiError::new(ErrorCode::StreamCreate, format!("stream create: {err}")))?;
    Ok(StreamHandle::Lance(stream))
}

#[no_mangle]
pub unsafe extern "C" fn lance_explain_dataset_scan_ir(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    verbose: u8,
) -> *const c_char {
    match explain_dataset_scan_ir_inner(
        dataset,
        columns,
        columns_len,
        filter_ir,
        filter_ir_len,
        verbose,
    ) {
        Ok(plan) => {
            clear_last_error();
            to_c_string(plan).into_raw() as *const c_char
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

fn explain_dataset_scan_ir_inner(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    verbose: u8,
) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let mut scan = handle.dataset.scan();

    let projection = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    if !projection.is_empty() {
        scan.project(&projection).map_err(|err| {
            FfiError::new(
                ErrorCode::ExplainPlan,
                format!("dataset scan project: {err}"),
            )
        })?;
    }

    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::ExplainPlan,
            "dataset scan filter_ir",
        )?
    };
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }

    scan.scan_in_order(false);
    match runtime::block_on(scan.explain_plan(verbose != 0)) {
        Ok(Ok(plan)) => Ok(plan),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::ExplainPlan,
            format!("dataset scan explain_plan: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}
