use std::ffi::{c_char, c_void};
use std::ptr;

use arrow_array::Float32Array;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use crate::scanner::LanceStream;

use super::projection;
use super::types::{SchemaHandle, StreamHandle};
use super::util::{
    cstr_to_str, nonzero_u64_to_usize, parse_optional_filter_ir, slice_from_ptr, to_c_string,
    FfiError, FfiResult,
};

#[no_mangle]
pub unsafe extern "C" fn lance_get_knn_schema(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    prefilter: u8,
    use_index: u8,
) -> *mut c_void {
    match get_knn_schema_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        k,
        prefilter,
        use_index,
    ) {
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

fn get_knn_schema_inner(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    prefilter: u8,
    use_index: u8,
) -> FfiResult<SchemaHandle> {
    if query_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }
    let vector_column = unsafe { cstr_to_str(vector_column, "vector_column")? };
    let query_values = unsafe { slice_from_ptr(query_values, query_len, "query_values")? };
    let k_usize = nonzero_u64_to_usize(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let projection = projection::build_knn_projection(&handle.base_projection, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema nearest: {err}")))?;
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref())
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema project: {err}")))?;
    scan.scan_in_order(false);

    let schema = LanceStream::from_scanner(scan)
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema: {err}")))?
        .schema();
    Ok(schema)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_knn_stream_ir(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
) -> *mut c_void {
    match create_knn_stream_ir_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        k,
        filter_ir,
        filter_ir_len,
        prefilter,
        use_index,
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

#[allow(clippy::too_many_arguments)]
fn create_knn_stream_ir_inner(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
) -> FfiResult<StreamHandle> {
    if query_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }
    let vector_column = unsafe { cstr_to_str(vector_column, "vector_column")? };
    let query_values = unsafe { slice_from_ptr(query_values, query_len, "query_values")? };
    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::KnnStreamCreate,
            "knn filter_ir",
        )?
    };
    let k_usize = nonzero_u64_to_usize(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let projection = projection::build_knn_projection(&handle.base_projection, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| {
            FfiError::new(
                ErrorCode::KnnStreamCreate,
                format!("knn scan nearest: {err}"),
            )
        })?;
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref()).map_err(|err| {
        FfiError::new(
            ErrorCode::KnnStreamCreate,
            format!("knn scan project: {err}"),
        )
    })?;
    scan.scan_in_order(false);

    let stream = LanceStream::from_scanner(scan).map_err(|err| {
        FfiError::new(
            ErrorCode::KnnStreamCreate,
            format!("knn stream create: {err}"),
        )
    })?;
    Ok(StreamHandle::Lance(stream))
}

#[no_mangle]
pub unsafe extern "C" fn lance_explain_knn_scan_ir(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
    verbose: u8,
) -> *const c_char {
    match explain_knn_scan_ir_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        k,
        filter_ir,
        filter_ir_len,
        prefilter,
        use_index,
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

#[allow(clippy::too_many_arguments)]
fn explain_knn_scan_ir_inner(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    use_index: u8,
    verbose: u8,
) -> FfiResult<String> {
    if query_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }
    let vector_column = unsafe { cstr_to_str(vector_column, "vector_column")? };
    let query_values = unsafe { slice_from_ptr(query_values, query_len, "query_values")? };
    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::ExplainPlan,
            "knn filter_ir",
        )?
    };
    let k_usize = nonzero_u64_to_usize(k, "k")?;

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let projection = projection::build_knn_projection(&handle.base_projection, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| FfiError::new(ErrorCode::ExplainPlan, format!("knn scan nearest: {err}")))?;
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref())
        .map_err(|err| FfiError::new(ErrorCode::ExplainPlan, format!("knn scan project: {err}")))?;
    scan.scan_in_order(false);

    match runtime::block_on(scan.explain_plan(verbose != 0)) {
        Ok(Ok(plan)) => Ok(plan),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::ExplainPlan,
            format!("knn scan explain_plan: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}
