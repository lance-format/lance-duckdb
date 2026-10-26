use std::ffi::{c_char, c_void};
use std::ptr;

use arrow_array::Float32Array;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use crate::scanner::LanceStream;

use super::projection;
use super::types::{SchemaHandle, StreamHandle};
use super::util::{
    apply_scanner_filter, cstr_to_str, nonzero_u64_to_usize, optional_cstr_array,
    optional_cstr_to_str, parse_optional_filter_ir, slice_from_ptr, to_c_string, FfiError,
    FfiResult,
};

#[no_mangle]
pub unsafe extern "C" fn lance_get_knn_schema(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    nprobes: u64,
    refine_factor: u64,
    prefilter: u8,
    use_index: u8,
) -> *mut c_void {
    match get_knn_schema_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        k,
        nprobes,
        refine_factor,
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

#[allow(clippy::too_many_arguments)]
fn get_knn_schema_inner(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    nprobes: u64,
    refine_factor: u64,
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
    let projection = projection::build_knn_projection(&handle.base_projection);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    scan.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema nearest: {err}")))?;
    if nprobes != 0 {
        let nprobes_usize = nonzero_u64_to_usize(nprobes, "nprobes")?;
        scan.nprobes(nprobes_usize);
    }
    if refine_factor != 0 {
        let refine_factor_u32: u32 = refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in u32")
        })?;
        scan.refine(refine_factor_u32);
    }
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    scan.project(projection.as_ref())
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema project: {err}")))?;
    scan.scan_in_order(false);

    let schema = runtime::block_on(scan.schema())
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("knn schema runtime: {err}")))?
        .map_err(|err| FfiError::new(ErrorCode::KnnSchema, format!("knn schema: {err}")))?;
    Ok(schema)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_knn_stream_ir(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    nprobes: u64,
    refine_factor: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    filter_sql: *const c_char,
    prefilter: u8,
    use_index: u8,
) -> *mut c_void {
    match create_knn_stream_ir_inner(
        dataset,
        vector_column,
        query_values,
        query_len,
        k,
        nprobes,
        refine_factor,
        columns,
        columns_len,
        filter_ir,
        filter_ir_len,
        filter_sql,
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
    nprobes: u64,
    refine_factor: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    filter_sql: *const c_char,
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
    let mut projection_columns = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
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
    let filter_sql = unsafe { optional_cstr_to_str(filter_sql, "filter_sql")? };
    projection_columns.push(vector_column.to_string());
    projection_columns.push(crate::constants::DISTANCE_COLUMN.to_string());

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    scan.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| {
            FfiError::new(
                ErrorCode::KnnStreamCreate,
                format!("knn scan nearest: {err}"),
            )
        })?;
    if nprobes != 0 {
        let nprobes_usize = nonzero_u64_to_usize(nprobes, "nprobes")?;
        scan.nprobes(nprobes_usize);
    }
    if refine_factor != 0 {
        let refine_factor_u32: u32 = refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in u32")
        })?;
        scan.refine(refine_factor_u32);
    }
    scan.use_index(use_index != 0);
    let filter = apply_scanner_filter(
        &mut scan,
        filter,
        filter_sql,
        ErrorCode::KnnStreamCreate,
        "knn filter_sql",
    )?;
    if let Some(filter) = &filter {
        projection_columns.extend(lance::io::exec::Planner::column_names_in_expr(filter));
    }
    projection_columns.sort();
    projection_columns.dedup();
    scan.disable_scoring_autoprojection();
    scan.project(&projection_columns).map_err(|err| {
        FfiError::new(
            ErrorCode::KnnStreamCreate,
            format!("knn scan project: {err}"),
        )
    })?;
    scan.scan_in_order(false);

    let full_schema = projection::build_knn_schema(&handle.read_schema);
    let stream = LanceStream::from_scanner_with_schema(scan, full_schema).map_err(|err| {
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
    nprobes: u64,
    refine_factor: u64,
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
        nprobes,
        refine_factor,
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
    nprobes: u64,
    refine_factor: u64,
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
    let projection = projection::build_knn_projection(&handle.base_projection);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    scan.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    scan.nearest(vector_column, &query, k_usize)
        .map_err(|err| FfiError::new(ErrorCode::ExplainPlan, format!("knn scan nearest: {err}")))?;
    if nprobes != 0 {
        let nprobes_usize = nonzero_u64_to_usize(nprobes, "nprobes")?;
        scan.nprobes(nprobes_usize);
    }
    if refine_factor != 0 {
        let refine_factor_u32: u32 = refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in u32")
        })?;
        scan.refine(refine_factor_u32);
    }
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
