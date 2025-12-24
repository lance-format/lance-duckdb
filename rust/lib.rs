#![allow(clippy::missing_safety_doc)]

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Schema};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::builder::Float32Builder;
use arrow_array::Float32Array;
use lance::dataset::builder::DatasetBuilder;
use lance::Dataset;

mod error;
mod filter_ir;
mod runtime;
mod scanner;

use error::{clear_last_error, set_last_error, ErrorCode};
use scanner::LanceStream;

// FFI ownership contract (Arrow C Data Interface):
// - All `*_open/create/get` functions return opaque handles owned by the caller,
//   which must be released exactly once via the matching `*_close/free` call.
// - `lance_schema_to_arrow` transfers ownership of the populated ArrowSchema to
//   the caller. The caller must call `release` exactly once on success.
// - `lance_batch_to_arrow` transfers ownership of the populated ArrowArray and
//   ArrowSchema to the caller. The caller must call `release` exactly once on
//   each on success.
// - On error (non-zero return), output ArrowSchema/ArrowArray are left
//   untouched and must not be released unless the caller initialized them to a
//   valid value before calling into this library.

// Dataset operations - just holds the dataset
struct DatasetHandle {
    dataset: Arc<Dataset>,
}

const DISTANCE_COLUMN: &str = "_distance";

fn normalize_distance_column(batch: &RecordBatch) -> Result<RecordBatch, String> {
    let schema = batch.schema();
    let idx = match schema.index_of(DISTANCE_COLUMN) {
        Ok(v) => v,
        Err(_) => return Ok(batch.clone()),
    };

    let col = batch.column(idx);
    if col.data_type() != &DataType::Float32 {
        return Ok(batch.clone());
    }

    let col = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| "distance column is not Float32Array".to_string())?;

    // Ensure the exported distance buffer has a simple, owned layout.
    let mut builder = Float32Builder::with_capacity(col.len());
    for i in 0..col.len() {
        if col.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(col.value(i));
        }
    }
    let normalized = Arc::new(builder.finish()) as Arc<dyn Array>;

    let mut cols: Vec<Arc<dyn Array>> = batch.columns().iter().cloned().collect();
    cols[idx] = normalized;

    RecordBatch::try_new(schema.clone(), cols).map_err(|e| format!("{e}"))
}

fn cstr_to_str<'a>(ptr: *const c_char, what: &'static str) -> Result<&'a str, ()> {
    if ptr.is_null() {
        set_last_error(ErrorCode::InvalidArgument, format!("{what} is null"));
        return Err(());
    }
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(v) => Ok(v),
        Err(err) => {
            set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
            Err(())
        }
    }
}

fn slice_from_ptr<'a, T>(ptr: *const T, len: usize, what: &'static str) -> Result<&'a [T], ()> {
    if ptr.is_null() {
        set_last_error(ErrorCode::InvalidArgument, format!("{what} is null"));
        return Err(());
    }
    // SAFETY: Caller guarantees ptr points to at least len elements.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn build_default_knn_projection(dataset: &Dataset, vector_column: &str) -> Arc<[String]> {
    let schema: Schema = dataset.schema().into();
    // Exclude the vector column from the output by default. DuckDB's Arrow
    // conversion can mis-handle FixedSizeList columns.
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if field.name() == vector_column {
            continue;
        }
        cols.push(field.name().to_string());
    }
    cols.push(DISTANCE_COLUMN.to_string());
    cols.into()
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "path is null");
        return ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                return ptr::null_mut();
            }
        }
    };

    let dataset = match runtime::block_on(Dataset::open(path_str)) {
        Ok(Ok(ds)) => Arc::new(ds),
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::DatasetOpen,
                format!("dataset open '{path_str}': {err}"),
            );
            return ptr::null_mut();
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return ptr::null_mut();
        }
    };

    let handle = Box::new(DatasetHandle { dataset });
    clear_last_error();

    Box::into_raw(handle) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> *mut c_void {
    if path.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "path is null");
        return ptr::null_mut();
    }
    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        set_last_error(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        );
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(err) => {
            set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
            return ptr::null_mut();
        }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for i in 0..options_len {
        let key_ptr = *option_keys.add(i);
        let value_ptr = *option_values.add(i);
        if key_ptr.is_null() || value_ptr.is_null() {
            set_last_error(ErrorCode::InvalidArgument, "option key/value is null");
            return ptr::null_mut();
        }
        let key = match CStr::from_ptr(key_ptr).to_str() {
            Ok(s) => s,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode key: {err}"));
                return ptr::null_mut();
            }
        };
        let value = match CStr::from_ptr(value_ptr).to_str() {
            Ok(s) => s,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode value: {err}"));
                return ptr::null_mut();
            }
        };
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
            set_last_error(
                ErrorCode::DatasetOpen,
                format!("dataset open '{path_str}': {err}"),
            );
            return ptr::null_mut();
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return ptr::null_mut();
        }
    };

    let handle = Box::new(DatasetHandle { dataset });
    clear_last_error();

    Box::into_raw(handle) as *mut c_void
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
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return -1;
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };

    match runtime::block_on(handle.dataset.count_rows(None)) {
        Ok(Ok(rows)) => {
            clear_last_error();
            match i64::try_from(rows) {
                Ok(v) => v,
                Err(_) => {
                    set_last_error(ErrorCode::DatasetCountRows, "row count overflow");
                    -1
                }
            }
        }
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::DatasetCountRows,
                format!("dataset count_rows: {err}"),
            );
            -1
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            -1
        }
    }
}

// Schema operations
#[no_mangle]
pub unsafe extern "C" fn lance_get_schema(dataset: *mut c_void) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let schema = handle.dataset.schema();

    let arrow_schema: Schema = schema.into();

    clear_last_error();
    Box::into_raw(Box::new(Arc::new(arrow_schema))) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_schema(schema: *mut c_void) {
    if !schema.is_null() {
        unsafe {
            let _ = Box::from_raw(schema as *mut Arc<Schema>);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_schema_to_arrow(
    schema: *mut c_void,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    if schema.is_null() || out_schema.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "schema or out_schema is null");
        return -1;
    }

    let schema = unsafe { &*(schema as *const Arc<Schema>) };
    let data_type = DataType::Struct(schema.fields().clone());

    let ffi_schema = match FFI_ArrowSchema::try_from(&data_type) {
        Ok(schema) => schema,
        Err(err) => {
            set_last_error(ErrorCode::SchemaExport, format!("schema export: {err}"));
            return -1;
        }
    };

    std::ptr::write_unaligned(out_schema, ffi_schema);
    clear_last_error();
    0
}

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
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }
    if query_len == 0 {
        set_last_error(ErrorCode::InvalidArgument, "query vector must be non-empty");
        return ptr::null_mut();
    }

    let vector_column = match cstr_to_str(vector_column, "vector_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query_values = match slice_from_ptr(query_values, query_len, "query_values") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_knn_projection(&handle.dataset, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    let k_usize = match usize::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };
    if let Err(err) = scan.nearest(vector_column, &query, k_usize) {
        set_last_error(ErrorCode::KnnSchema, format!("knn schema nearest: {err}"));
        return ptr::null_mut();
    }
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(ErrorCode::KnnSchema, format!("knn schema project: {err}"));
        return ptr::null_mut();
    }
    scan.scan_in_order(false);

    let schema = match LanceStream::from_scanner(scan) {
        Ok(stream) => stream.schema(),
        Err(err) => {
            set_last_error(ErrorCode::KnnSchema, format!("knn schema: {err}"));
            return ptr::null_mut();
        }
    };

    clear_last_error();
    Box::into_raw(Box::new(schema)) as *mut c_void
}

// Stream operations
#[no_mangle]
pub unsafe extern "C" fn lance_create_stream(dataset: *mut c_void) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };

    let scanner = handle.dataset.scan();
    match LanceStream::from_scanner(scanner) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(ErrorCode::StreamCreate, format!("stream create: {err}"));
            ptr::null_mut()
        }
    }
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
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let mut scan = handle.dataset.scan();

    if !columns.is_null() && columns_len > 0 {
        let mut projection = Vec::with_capacity(columns_len);
        for idx in 0..columns_len {
            let col_ptr = unsafe { *columns.add(idx) };
            if col_ptr.is_null() {
                set_last_error(ErrorCode::InvalidArgument, "column name is null");
                return ptr::null();
            }
            let col_name = match unsafe { CStr::from_ptr(col_ptr) }.to_str() {
                Ok(v) => v,
                Err(err) => {
                    set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                    return ptr::null();
                }
            };
            projection.push(col_name.to_string());
        }
        if let Err(err) = scan.project(&projection) {
            set_last_error(
                ErrorCode::ExplainPlan,
                format!("dataset scan project: {err}"),
            );
            return ptr::null();
        }
    }

    if !filter_ir.is_null() && filter_ir_len > 0 {
        let bytes = unsafe { std::slice::from_raw_parts(filter_ir, filter_ir_len) };
        let expr = match crate::filter_ir::parse_filter_ir(bytes) {
            Ok(v) => v,
            Err(err) => {
                set_last_error(
                    ErrorCode::ExplainPlan,
                    format!("dataset scan filter_ir: {err}"),
                );
                return ptr::null();
            }
        };
        scan.filter_expr(expr);
    }

    scan.scan_in_order(false);

    let plan = match runtime::block_on(scan.explain_plan(verbose != 0)) {
        Ok(Ok(plan)) => plan,
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::ExplainPlan,
                format!("dataset scan explain_plan: {err}"),
            );
            return ptr::null();
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return ptr::null();
        }
    };

    let out = match std::ffi::CString::new(plan.as_str()) {
        Ok(v) => v,
        Err(_) => std::ffi::CString::new(plan.replace('\0', "\\0"))
            .unwrap_or_else(|_| std::ffi::CString::new("invalid plan").unwrap()),
    };
    clear_last_error();
    out.into_raw() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn lance_explain_knn_scan(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_sql: *const c_char,
    prefilter: u8,
    use_index: u8,
    verbose: u8,
) -> *const c_char {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null();
    }
    if query_len == 0 {
        set_last_error(ErrorCode::InvalidArgument, "query vector must be non-empty");
        return ptr::null();
    }

    let vector_column = match cstr_to_str(vector_column, "vector_column") {
        Ok(v) => v,
        Err(()) => return ptr::null(),
    };
    let query_values = match slice_from_ptr(query_values, query_len, "query_values") {
        Ok(v) => v,
        Err(()) => return ptr::null(),
    };

    let filter = if filter_sql.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(filter_sql) }.to_str() {
            Ok(v) if !v.is_empty() => Some(v),
            Ok(_) => None,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                return ptr::null();
            }
        }
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_knn_projection(&handle.dataset, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        if let Err(err) = scan.filter(filter) {
            set_last_error(ErrorCode::ExplainPlan, format!("knn scan filter: {err}"));
            return ptr::null();
        }
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    let k_usize = match usize::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null();
        }
    };
    if let Err(err) = scan.nearest(vector_column, &query, k_usize) {
        set_last_error(ErrorCode::ExplainPlan, format!("knn scan nearest: {err}"));
        return ptr::null();
    }
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(ErrorCode::ExplainPlan, format!("knn scan project: {err}"));
        return ptr::null();
    }
    scan.scan_in_order(false);

    let plan = match runtime::block_on(scan.explain_plan(verbose != 0)) {
        Ok(Ok(plan)) => plan,
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::ExplainPlan,
                format!("knn scan explain_plan: {err}"),
            );
            return ptr::null();
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return ptr::null();
        }
    };

    let out = match std::ffi::CString::new(plan.as_str()) {
        Ok(v) => v,
        Err(_) => std::ffi::CString::new(plan.replace('\0', "\\0"))
            .unwrap_or_else(|_| std::ffi::CString::new("invalid plan").unwrap()),
    };
    clear_last_error();
    out.into_raw() as *const c_char
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_knn_stream(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    k: u64,
    filter_sql: *const c_char,
    prefilter: u8,
    use_index: u8,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }
    if query_len == 0 {
        set_last_error(ErrorCode::InvalidArgument, "query vector must be non-empty");
        return ptr::null_mut();
    }

    let vector_column = match cstr_to_str(vector_column, "vector_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query_values = match slice_from_ptr(query_values, query_len, "query_values") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let filter = if filter_sql.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(filter_sql) }.to_str() {
            Ok(v) if !v.is_empty() => Some(v),
            Ok(_) => None,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                return ptr::null_mut();
            }
        }
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_knn_projection(&handle.dataset, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        if let Err(err) = scan.filter(filter) {
            set_last_error(
                ErrorCode::KnnStreamCreate,
                format!("knn scan filter: {err}"),
            );
            return ptr::null_mut();
        }
    }
    let query = Float32Array::from_iter_values(query_values.iter().copied());
    let k_usize = match usize::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };
    if let Err(err) = scan.nearest(vector_column, &query, k_usize) {
        set_last_error(
            ErrorCode::KnnStreamCreate,
            format!("knn scan nearest: {err}"),
        );
        return ptr::null_mut();
    }
    scan.use_index(use_index != 0);
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(
            ErrorCode::KnnStreamCreate,
            format!("knn scan project: {err}"),
        );
        return ptr::null_mut();
    }
    scan.scan_in_order(false);

    match LanceStream::from_scanner(scan) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(
                ErrorCode::KnnStreamCreate,
                format!("knn stream create: {err}"),
            );
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_fragments(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut u64 {
    if dataset.is_null() || out_len.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset or out_len is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let ids: Vec<u64> = handle.dataset.fragments().iter().map(|f| f.id).collect();

    let mut boxed = ids.into_boxed_slice();
    let len = boxed.len();
    let data = boxed.as_mut_ptr();
    std::mem::forget(boxed);

    unsafe {
        ptr::write_unaligned(out_len, len);
    }
    clear_last_error();
    data
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
pub unsafe extern "C" fn lance_create_fragment_stream(
    dataset: *mut c_void,
    fragment_id: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_sql: *const c_char,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let fragment_id_usize = match usize::try_from(fragment_id) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(
                ErrorCode::InvalidArgument,
                format!("invalid fragment id: {err}"),
            );
            return ptr::null_mut();
        }
    };

    let fragment = match handle.dataset.get_fragment(fragment_id_usize) {
        Some(f) => f,
        None => {
            set_last_error(
                ErrorCode::FragmentScan,
                format!("fragment not found: {fragment_id}"),
            );
            return ptr::null_mut();
        }
    };

    let mut scan = fragment.scan();

    if !columns.is_null() && columns_len > 0 {
        let mut projection = Vec::with_capacity(columns_len);
        for idx in 0..columns_len {
            let col_ptr = unsafe { *columns.add(idx) };
            if col_ptr.is_null() {
                set_last_error(ErrorCode::InvalidArgument, "column name is null");
                return ptr::null_mut();
            }
            let col_name = match unsafe { CStr::from_ptr(col_ptr) }.to_str() {
                Ok(v) => v,
                Err(err) => {
                    set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                    return ptr::null_mut();
                }
            };
            projection.push(col_name.to_string());
        }
        if let Err(err) = scan.project(&projection) {
            set_last_error(
                ErrorCode::FragmentScan,
                format!("fragment scan project: {err}"),
            );
            return ptr::null_mut();
        }
    }

    if !filter_sql.is_null() {
        let filter = match unsafe { CStr::from_ptr(filter_sql) }.to_str() {
            Ok(v) => v,
            Err(err) => {
                set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                return ptr::null_mut();
            }
        };
        if !filter.is_empty() {
            if let Err(err) = scan.filter(filter) {
                set_last_error(
                    ErrorCode::FragmentScan,
                    format!("fragment scan filter: {err}"),
                );
                return ptr::null_mut();
            }
        }
    }
    scan.scan_in_order(false);

    match LanceStream::from_scanner(scan) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(ErrorCode::StreamCreate, format!("stream create: {err}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_fragment_stream_ir(
    dataset: *mut c_void,
    fragment_id: u64,
    columns: *const *const c_char,
    columns_len: usize,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let fragment_id_usize = match usize::try_from(fragment_id) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(
                ErrorCode::InvalidArgument,
                format!("invalid fragment id: {err}"),
            );
            return ptr::null_mut();
        }
    };

    let fragment = match handle.dataset.get_fragment(fragment_id_usize) {
        Some(f) => f,
        None => {
            set_last_error(
                ErrorCode::FragmentScan,
                format!("fragment not found: {fragment_id}"),
            );
            return ptr::null_mut();
        }
    };

    let mut scan = fragment.scan();

    if !columns.is_null() && columns_len > 0 {
        let mut projection = Vec::with_capacity(columns_len);
        for idx in 0..columns_len {
            let col_ptr = unsafe { *columns.add(idx) };
            if col_ptr.is_null() {
                set_last_error(ErrorCode::InvalidArgument, "column name is null");
                return ptr::null_mut();
            }
            let col_name = match unsafe { CStr::from_ptr(col_ptr) }.to_str() {
                Ok(v) => v,
                Err(err) => {
                    set_last_error(ErrorCode::Utf8, format!("utf8 decode: {err}"));
                    return ptr::null_mut();
                }
            };
            projection.push(col_name.to_string());
        }
        if let Err(err) = scan.project(&projection) {
            set_last_error(
                ErrorCode::FragmentScan,
                format!("fragment scan project: {err}"),
            );
            return ptr::null_mut();
        }
    }

    if !filter_ir.is_null() && filter_ir_len > 0 {
        let bytes = unsafe { std::slice::from_raw_parts(filter_ir, filter_ir_len) };
        let expr = match crate::filter_ir::parse_filter_ir(bytes) {
            Ok(v) => v,
            Err(err) => {
                set_last_error(
                    ErrorCode::FragmentScan,
                    format!("fragment scan filter_ir: {err}"),
                );
                return ptr::null_mut();
            }
        };
        scan.filter_expr(expr);
    }

    scan.scan_in_order(false);

    match LanceStream::from_scanner(scan) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(ErrorCode::StreamCreate, format!("stream create: {err}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_stream_next(
    stream: *mut c_void,
    out_batch: *mut *mut c_void,
) -> i32 {
    if out_batch.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_batch is null");
        return -1;
    }
    unsafe {
        ptr::write_unaligned(out_batch, ptr::null_mut());
    }

    if stream.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "stream is null");
        return -1;
    }

    let stream = unsafe { &mut *(stream as *mut LanceStream) };

    match stream.next() {
        Ok(Some(batch)) => {
            clear_last_error();
            let batch_ptr = Box::into_raw(Box::new(batch)) as *mut c_void;
            unsafe {
                ptr::write_unaligned(out_batch, batch_ptr);
            }
            0
        }
        Ok(None) => {
            clear_last_error();
            1
        }
        Err(err) => {
            set_last_error(ErrorCode::StreamNext, format!("stream next: {err}"));
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_stream(stream: *mut c_void) {
    if !stream.is_null() {
        unsafe {
            let _ = Box::from_raw(stream as *mut LanceStream);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_batch(batch: *mut c_void) {
    if !batch.is_null() {
        unsafe {
            let _ = Box::from_raw(batch as *mut RecordBatch);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_batch_num_rows(batch: *mut c_void) -> i64 {
    if batch.is_null() {
        return 0;
    }

    let batch = unsafe { &*(batch as *const RecordBatch) };
    batch.num_rows() as i64
}

// Export RecordBatch to Arrow C Data Interface
#[no_mangle]
pub unsafe extern "C" fn lance_batch_to_arrow(
    batch: *mut c_void,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    if batch.is_null() || out_array.is_null() || out_schema.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "batch or out pointer is null");
        return -1;
    }

    let batch = unsafe { &*(batch as *const RecordBatch) };
    let batch = match normalize_distance_column(batch) {
        Ok(b) => b,
        Err(err) => {
            set_last_error(ErrorCode::BatchExport, format!("batch export: {err}"));
            return -1;
        }
    };

    // Convert RecordBatch to StructArray for FFI export
    let struct_array: Arc<dyn Array> = Arc::new(StructArray::from(batch));

    let data = struct_array.to_data();
    let array = FFI_ArrowArray::new(&data);
    let schema = match FFI_ArrowSchema::try_from(data.data_type()) {
        Ok(schema) => schema,
        Err(err) => {
            set_last_error(ErrorCode::BatchExport, format!("batch export: {err}"));
            return -1;
        }
    };

    std::ptr::write_unaligned(out_array, array);
    std::ptr::write_unaligned(out_schema, schema);

    clear_last_error();
    0
}
