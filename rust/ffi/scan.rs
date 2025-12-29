use std::ffi::{c_char, c_void};
use std::ptr;

use crate::constants::ROW_ID_COLUMN;
use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;
use crate::scanner::{LanceStream, LanceTakeStream};

use super::types::StreamHandle;
use super::util::{
    optional_cstr_array, parse_optional_filter_ir, to_c_string, u64_to_usize, FfiError, FfiResult,
};

use lance::dataset::ProjectionRequest;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;

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
        if projection.iter().any(|c| c == ROW_ID_COLUMN) {
            scan.with_row_id();
        }
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
pub unsafe extern "C" fn lance_create_dataset_stream_ir(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
) -> *mut c_void {
    match create_dataset_stream_ir_inner(
        dataset,
        columns,
        columns_len,
        filter_ir,
        filter_ir_len,
        limit,
        offset,
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

fn create_dataset_stream_ir_inner(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
) -> FfiResult<StreamHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    if offset < 0 {
        return Err(FfiError::new(
            ErrorCode::DatasetScan,
            "offset must be non-negative".to_string(),
        ));
    }
    if limit < -1 {
        return Err(FfiError::new(
            ErrorCode::DatasetScan,
            "limit must be >= -1".to_string(),
        ));
    }

    let mut scan = handle.dataset.scan();

    let projection = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    if !projection.is_empty() {
        if projection.iter().any(|c| c == ROW_ID_COLUMN) {
            scan.with_row_id();
        }
        scan.project(&projection).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetScan,
                format!("dataset scan project: {err}"),
            )
        })?;
    }

    let filter = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::DatasetScan,
            "dataset scan filter_ir",
        )?
    };
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }

    if limit != -1 || offset != 0 {
        let limit_opt = if limit == -1 { None } else { Some(limit) };
        let offset_opt = if offset == 0 { None } else { Some(offset) };
        scan.limit(limit_opt, offset_opt).map_err(|err| {
            FfiError::new(ErrorCode::DatasetScan, format!("dataset scan limit: {err}"))
        })?;
    }

    scan.scan_in_order(false);
    let stream = LanceStream::from_scanner(scan)
        .map_err(|err| FfiError::new(ErrorCode::StreamCreate, format!("stream create: {err}")))?;
    Ok(StreamHandle::Lance(stream))
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_sample_stream_ir(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    sample_percentage: f64,
    seed: i64,
    repeatable: u8,
) -> *mut c_void {
    match create_dataset_sample_stream_ir_inner(
        dataset,
        columns,
        columns_len,
        sample_percentage,
        seed,
        repeatable,
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

fn create_dataset_sample_stream_ir_inner(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    sample_percentage: f64,
    seed: i64,
    _repeatable: u8,
) -> FfiResult<StreamHandle> {
    const DEFAULT_TAKE_BATCH_SIZE: usize = 8192;

    if !sample_percentage.is_finite() {
        return Err(FfiError::new(
            ErrorCode::DatasetScan,
            "sample_percentage must be finite".to_string(),
        ));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let total_rows = match runtime::block_on(handle.dataset.count_rows(None)) {
        Ok(Ok(rows)) => rows,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetScan,
                format!("dataset count_rows: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let mut pct = sample_percentage / 100.0;
    if pct < 0.0 {
        pct = 0.0;
    } else if pct > 1.0 {
        pct = 1.0;
    }
    let mut target = ((total_rows as f64) * pct).floor() as usize;
    if target > total_rows {
        target = total_rows;
    }

    if target == 0 {
        return Ok(StreamHandle::Batches(Vec::new().into_iter()));
    }

    let projection = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    let dataset_schema = handle.dataset.schema();
    let projection = if projection.is_empty() {
        ProjectionRequest::from_schema(dataset_schema.clone())
    } else {
        ProjectionRequest::from_columns(projection.iter(), dataset_schema)
    };

    let row_indices: Vec<u64> = if target == total_rows {
        (0..total_rows as u64).collect()
    } else {
        let mut rng = if seed >= 0 {
            StdRng::seed_from_u64(seed as u64)
        } else {
            StdRng::from_entropy()
        };
        sample(&mut rng, total_rows, target)
            .into_vec()
            .into_iter()
            .map(|v| v as u64)
            .collect()
    };

    let stream = LanceTakeStream::try_new(
        handle.dataset.clone(),
        projection,
        row_indices,
        DEFAULT_TAKE_BATCH_SIZE,
    )
    .map_err(|err| FfiError::new(ErrorCode::StreamCreate, format!("stream create: {err}")))?;
    Ok(StreamHandle::Take(stream))
}

#[no_mangle]
pub unsafe extern "C" fn lance_explain_dataset_scan_ir(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
    verbose: u8,
) -> *const c_char {
    match explain_dataset_scan_ir_inner(
        dataset,
        columns,
        columns_len,
        filter_ir,
        filter_ir_len,
        limit,
        offset,
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
    limit: i64,
    offset: i64,
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

    if offset < 0 {
        return Err(FfiError::new(
            ErrorCode::ExplainPlan,
            "offset must be non-negative".to_string(),
        ));
    }
    if limit < -1 {
        return Err(FfiError::new(
            ErrorCode::ExplainPlan,
            "limit must be >= -1".to_string(),
        ));
    }
    if limit != -1 || offset != 0 {
        let limit_opt = if limit == -1 { None } else { Some(limit) };
        let offset_opt = if offset == 0 { None } else { Some(offset) };
        scan.limit(limit_opt, offset_opt).map_err(|err| {
            FfiError::new(ErrorCode::ExplainPlan, format!("dataset scan limit: {err}"))
        })?;
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
