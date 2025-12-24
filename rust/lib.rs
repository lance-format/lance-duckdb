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
use lance::dataset::ProjectionRequest;
use lance::Dataset;
use lance_index::scalar::FullTextSearchQuery;
use std::cmp::Ordering;

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
const SCORE_COLUMN: &str = "_score";
const HYBRID_SCORE_COLUMN: &str = "_hybrid_score";
const ROW_ID_COLUMN: &str = "_rowid";

enum StreamHandle {
    Lance(LanceStream),
    Batches(Vec<RecordBatch>),
}

impl StreamHandle {
    fn next(&mut self) -> Result<Option<RecordBatch>, String> {
        match self {
            StreamHandle::Lance(stream) => stream.next().map_err(|e| format!("{e}")),
            StreamHandle::Batches(batches) => {
                if batches.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(batches.remove(0)))
                }
            }
        }
    }
}

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

    let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
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

fn parse_optional_filter_ir(
    filter_ir: *const u8,
    filter_ir_len: usize,
    code: ErrorCode,
    what: &'static str,
) -> Result<Option<datafusion_expr::Expr>, ()> {
    if filter_ir_len == 0 {
        return Ok(None);
    }
    if filter_ir.is_null() {
        set_last_error(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        );
        return Err(());
    }

    let bytes = unsafe { std::slice::from_raw_parts(filter_ir, filter_ir_len) };
    match crate::filter_ir::parse_filter_ir(bytes) {
        Ok(v) => Ok(Some(v)),
        Err(err) => {
            set_last_error(code, format!("{what} parse: {err}"));
            Err(())
        }
    }
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

fn is_default_excluded_field(field: &arrow::datatypes::Field) -> bool {
    matches!(field.data_type(), DataType::FixedSizeList(_, _))
}

fn build_default_fts_projection(dataset: &Dataset) -> Arc<[String]> {
    let schema: Schema = dataset.schema().into();
    let mut cols = Vec::with_capacity(schema.fields().len() + 1);
    for field in schema.fields() {
        if is_default_excluded_field(field) {
            continue;
        }
        cols.push(field.name().to_string());
    }
    cols.push(SCORE_COLUMN.to_string());
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
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
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

    let filter = match parse_optional_filter_ir(filter_ir, filter_ir_len, ErrorCode::ExplainPlan, "knn filter_ir") {
        Ok(v) => v,
        Err(()) => return ptr::null(),
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_knn_projection(&handle.dataset, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
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
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
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

    let filter = match parse_optional_filter_ir(
        filter_ir,
        filter_ir_len,
        ErrorCode::KnnStreamCreate,
        "knn filter_ir",
    ) {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_knn_projection(&handle.dataset, vector_column);

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
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
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
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
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
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
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
        }
        Err(err) => {
            set_last_error(ErrorCode::StreamCreate, format!("stream create: {err}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_fts_schema(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    prefilter: u8,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let text_column = match cstr_to_str(text_column, "text_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query = match cstr_to_str(query, "query") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let k_i64 = match i64::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_fts_projection(&handle.dataset);

    let fts_query = match FullTextSearchQuery::new(query.to_string()).with_column(text_column.to_string()) {
        Ok(v) => v.limit(Some(k_i64)),
        Err(err) => {
            set_last_error(ErrorCode::FtsSchema, format!("fts query: {err}"));
            return ptr::null_mut();
        }
    };

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Err(err) = scan.full_text_search(fts_query) {
        set_last_error(ErrorCode::FtsSchema, format!("fts schema search: {err}"));
        return ptr::null_mut();
    }
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(ErrorCode::FtsSchema, format!("fts schema project: {err}"));
        return ptr::null_mut();
    }
    scan.scan_in_order(false);

    let schema = match LanceStream::from_scanner(scan) {
        Ok(stream) => stream.schema(),
        Err(err) => {
            set_last_error(ErrorCode::FtsSchema, format!("fts schema: {err}"));
            return ptr::null_mut();
        }
    };

    clear_last_error();
    Box::into_raw(Box::new(schema)) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_fts_stream(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    filter_sql: *const c_char,
    prefilter: u8,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let text_column = match cstr_to_str(text_column, "text_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query = match cstr_to_str(query, "query") {
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

    let k_i64 = match i64::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_fts_projection(&handle.dataset);

    let fts_query = match FullTextSearchQuery::new(query.to_string()).with_column(text_column.to_string()) {
        Ok(v) => v.limit(Some(k_i64)),
        Err(err) => {
            set_last_error(ErrorCode::FtsStreamCreate, format!("fts query: {err}"));
            return ptr::null_mut();
        }
    };

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        if let Err(err) = scan.filter(filter) {
            set_last_error(
                ErrorCode::FtsStreamCreate,
                format!("fts scan filter: {err}"),
            );
            return ptr::null_mut();
        }
    }
    if let Err(err) = scan.full_text_search(fts_query) {
        set_last_error(
            ErrorCode::FtsStreamCreate,
            format!("fts scan search: {err}"),
        );
        return ptr::null_mut();
    }
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(
            ErrorCode::FtsStreamCreate,
            format!("fts scan project: {err}"),
        );
        return ptr::null_mut();
    }
    scan.scan_in_order(false);

    match LanceStream::from_scanner(scan) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
        }
        Err(err) => {
            set_last_error(
                ErrorCode::FtsStreamCreate,
                format!("fts stream create: {err}"),
            );
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_fts_stream_ir(
    dataset: *mut c_void,
    text_column: *const c_char,
    query: *const c_char,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "dataset is null");
        return ptr::null_mut();
    }

    let text_column = match cstr_to_str(text_column, "text_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query = match cstr_to_str(query, "query") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let filter = match parse_optional_filter_ir(
        filter_ir,
        filter_ir_len,
        ErrorCode::FtsStreamCreate,
        "fts filter_ir",
    ) {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let k_i64 = match i64::try_from(k) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let projection = build_default_fts_projection(&handle.dataset);

    let fts_query =
        match FullTextSearchQuery::new(query.to_string()).with_column(text_column.to_string()) {
            Ok(v) => v.limit(Some(k_i64)),
            Err(err) => {
                set_last_error(ErrorCode::FtsStreamCreate, format!("fts query: {err}"));
                return ptr::null_mut();
            }
        };

    let mut scan = handle.dataset.scan();
    scan.prefilter(prefilter != 0);
    if let Some(filter) = filter {
        scan.filter_expr(filter);
    }
    if let Err(err) = scan.full_text_search(fts_query) {
        set_last_error(
            ErrorCode::FtsStreamCreate,
            format!("fts scan search: {err}"),
        );
        return ptr::null_mut();
    }
    scan.disable_scoring_autoprojection();
    if let Err(err) = scan.project(projection.as_ref()) {
        set_last_error(
            ErrorCode::FtsStreamCreate,
            format!("fts scan project: {err}"),
        );
        return ptr::null_mut();
    }
    scan.scan_in_order(false);

    match LanceStream::from_scanner(scan) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(StreamHandle::Lance(stream))) as *mut c_void
        }
        Err(err) => {
            set_last_error(
                ErrorCode::FtsStreamCreate,
                format!("fts stream create: {err}"),
            );
            ptr::null_mut()
        }
    }
}

fn normalize_range(values: &[f32]) -> (f32, f32) {
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    for &v in values {
        if !v.is_finite() {
            continue;
        }
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    if !min_v.is_finite() {
        (0.0, 0.0)
    } else {
        (min_v, max_v)
    }
}

fn normalize_value(v: f32, min_v: f32, max_v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    if (max_v - min_v).abs() < f32::EPSILON {
        return 0.5;
    }
    ((v - min_v) / (max_v - min_v)).clamp(0.0, 1.0)
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_hybrid_schema(dataset: *mut c_void) -> *mut c_void {
    if dataset.is_null() {
        set_last_error(ErrorCode::HybridSchema, "dataset is null");
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let schema: Schema = handle.dataset.schema().into();

    let mut fields = Vec::with_capacity(schema.fields().len() + 3);
    for field in schema.fields() {
        if is_default_excluded_field(field) {
            continue;
        }
        fields.push(field.clone());
    }
    fields.push(Arc::new(arrow::datatypes::Field::new(
        DISTANCE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        SCORE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        HYBRID_SCORE_COLUMN,
        DataType::Float32,
        true,
    )));

    clear_last_error();
    Box::into_raw(Box::new(Arc::new(Schema::new(fields)))) as *mut c_void
}

enum HybridFilter {
    Sql(String),
    Expr(datafusion_expr::Expr),
}

fn apply_hybrid_filter(
    scan: &mut lance::dataset::scanner::Scanner,
    filter: Option<&HybridFilter>,
    context: &'static str,
) -> Result<(), ()> {
    let Some(filter) = filter else {
        return Ok(());
    };

    match filter {
        HybridFilter::Sql(sql) => match scan.filter(sql.as_str()) {
            Ok(_) => Ok(()),
            Err(err) => {
                set_last_error(ErrorCode::HybridStreamCreate, format!("{context} filter: {err}"));
                Err(())
            }
        },
        HybridFilter::Expr(expr) => {
            scan.filter_expr(expr.clone());
            Ok(())
        }
    }
}

fn create_hybrid_stream_impl(
    handle: &DatasetHandle,
    vector_column: &str,
    query_values: &[f32],
    text_column: &str,
    text_query: &str,
    k: u64,
    filter: Option<HybridFilter>,
    prefilter: u8,
    alpha: f32,
    oversample_factor: u32,
) -> *mut c_void {
    let k_usize = match usize::try_from(k) {
        Ok(v) if v > 0 => v,
        Ok(_) => {
            set_last_error(ErrorCode::InvalidArgument, "k must be > 0");
            return ptr::null_mut();
        }
        Err(err) => {
            set_last_error(ErrorCode::InvalidArgument, format!("invalid k: {err}"));
            return ptr::null_mut();
        }
    };

    let query = Float32Array::from_iter_values(query_values.iter().copied());
    let oversample = k_usize
        .saturating_mul(oversample_factor.max(1) as usize)
        .max(k_usize);

    let filter = filter.as_ref();

    let mut vector_scan = handle.dataset.scan();
    vector_scan.prefilter(prefilter != 0);
    if apply_hybrid_filter(&mut vector_scan, filter, "hybrid vector").is_err() {
        return ptr::null_mut();
    }
    if let Err(err) = vector_scan.nearest(vector_column, &query, oversample) {
        set_last_error(
            ErrorCode::HybridStreamCreate,
            format!("hybrid vector nearest: {err}"),
        );
        return ptr::null_mut();
    }
    vector_scan.use_index(false);
    vector_scan.with_row_id();
    vector_scan.disable_scoring_autoprojection();
    if let Err(err) = vector_scan.project(&[ROW_ID_COLUMN, DISTANCE_COLUMN]) {
        set_last_error(
            ErrorCode::HybridStreamCreate,
            format!("hybrid vector project: {err}"),
        );
        return ptr::null_mut();
    }
    vector_scan.scan_in_order(false);

    let mut vector_stream = match LanceStream::from_scanner(vector_scan) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(
                ErrorCode::HybridStreamCreate,
                format!("hybrid vector stream: {err}"),
            );
            return ptr::null_mut();
        }
    };

    let mut vector_rows = Vec::<(u64, f32)>::new();
    loop {
        match vector_stream.next() {
            Ok(Some(batch)) => {
                let idx_rowid = match batch.schema().index_of(ROW_ID_COLUMN) {
                    Ok(v) => v,
                    Err(_) => {
                        set_last_error(
                            ErrorCode::HybridStreamCreate,
                            "hybrid vector batch missing _rowid",
                        );
                        return ptr::null_mut();
                    }
                };
                let idx_dist = match batch.schema().index_of(DISTANCE_COLUMN) {
                    Ok(v) => v,
                    Err(_) => {
                        set_last_error(
                            ErrorCode::HybridStreamCreate,
                            "hybrid vector batch missing _distance",
                        );
                        return ptr::null_mut();
                    }
                };
                let rowids = batch
                    .column(idx_rowid)
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>();
                let dists = batch
                    .column(idx_dist)
                    .as_any()
                    .downcast_ref::<Float32Array>();
                let (Some(rowids), Some(dists)) = (rowids, dists) else {
                    set_last_error(
                        ErrorCode::HybridStreamCreate,
                        "hybrid vector batch has unexpected column types",
                    );
                    return ptr::null_mut();
                };
                for i in 0..batch.num_rows() {
                    if rowids.is_null(i) || dists.is_null(i) {
                        continue;
                    }
                    vector_rows.push((rowids.value(i), dists.value(i)));
                }
            }
            Ok(None) => break,
            Err(err) => {
                set_last_error(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid vector next: {err}"),
                );
                return ptr::null_mut();
            }
        }
    }

    let fts_query =
        match FullTextSearchQuery::new(text_query.to_string()).with_column(text_column.to_string()) {
            Ok(v) => v.limit(Some(oversample as i64)),
            Err(err) => {
                set_last_error(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid fts query: {err}"),
                );
                return ptr::null_mut();
            }
        };

    let mut fts_scan = handle.dataset.scan();
    fts_scan.prefilter(prefilter != 0);
    if apply_hybrid_filter(&mut fts_scan, filter, "hybrid fts").is_err() {
        return ptr::null_mut();
    }
    if let Err(err) = fts_scan.full_text_search(fts_query) {
        set_last_error(
            ErrorCode::HybridStreamCreate,
            format!("hybrid fts search: {err}"),
        );
        return ptr::null_mut();
    }
    fts_scan.with_row_id();
    fts_scan.disable_scoring_autoprojection();
    if let Err(err) = fts_scan.project(&[ROW_ID_COLUMN, SCORE_COLUMN]) {
        set_last_error(
            ErrorCode::HybridStreamCreate,
            format!("hybrid fts project: {err}"),
        );
        return ptr::null_mut();
    }
    fts_scan.scan_in_order(false);

    let mut fts_stream = match LanceStream::from_scanner(fts_scan) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(
                ErrorCode::HybridStreamCreate,
                format!("hybrid fts stream: {err}"),
            );
            return ptr::null_mut();
        }
    };

    let mut fts_rows = Vec::<(u64, f32)>::new();
    loop {
        match fts_stream.next() {
            Ok(Some(batch)) => {
                let idx_rowid = match batch.schema().index_of(ROW_ID_COLUMN) {
                    Ok(v) => v,
                    Err(_) => {
                        set_last_error(
                            ErrorCode::HybridStreamCreate,
                            "hybrid fts batch missing _rowid",
                        );
                        return ptr::null_mut();
                    }
                };
                let idx_score = match batch.schema().index_of(SCORE_COLUMN) {
                    Ok(v) => v,
                    Err(_) => {
                        set_last_error(
                            ErrorCode::HybridStreamCreate,
                            "hybrid fts batch missing _score",
                        );
                        return ptr::null_mut();
                    }
                };
                let rowids = batch
                    .column(idx_rowid)
                    .as_any()
                    .downcast_ref::<arrow_array::UInt64Array>();
                let scores = batch
                    .column(idx_score)
                    .as_any()
                    .downcast_ref::<Float32Array>();
                let (Some(rowids), Some(scores)) = (rowids, scores) else {
                    set_last_error(
                        ErrorCode::HybridStreamCreate,
                        "hybrid fts batch has unexpected column types",
                    );
                    return ptr::null_mut();
                };
                for i in 0..batch.num_rows() {
                    if rowids.is_null(i) || scores.is_null(i) {
                        continue;
                    }
                    fts_rows.push((rowids.value(i), scores.value(i)));
                }
            }
            Ok(None) => break,
            Err(err) => {
                set_last_error(
                    ErrorCode::HybridStreamCreate,
                    format!("hybrid fts next: {err}"),
                );
                return ptr::null_mut();
            }
        }
    }

    let vector_values = vector_rows.iter().map(|(_, d)| *d).collect::<Vec<_>>();
    let fts_values = fts_rows.iter().map(|(_, s)| *s).collect::<Vec<_>>();
    let (dist_min, dist_max) = normalize_range(&vector_values);
    let (score_min, score_max) = normalize_range(&fts_values);

    let alpha = alpha.clamp(0.0, 1.0);
    let mut merged = std::collections::HashMap::<u64, (Option<f32>, Option<f32>, f32)>::new();
    for (rowid, dist) in vector_rows {
        let dist_norm = 1.0 - normalize_value(dist, dist_min, dist_max);
        merged.insert(rowid, (Some(dist), None, alpha * dist_norm));
    }
    for (rowid, score) in fts_rows {
        let score_norm = normalize_value(score, score_min, score_max);
        merged
            .entry(rowid)
            .and_modify(|e| {
                e.1 = Some(score);
                e.2 += (1.0 - alpha) * score_norm;
            })
            .or_insert((None, Some(score), (1.0 - alpha) * score_norm));
    }

    let mut ranked: Vec<(u64, Option<f32>, Option<f32>, f32)> = merged
        .into_iter()
        .map(|(rowid, (dist, score, hybrid))| (rowid, dist, score, hybrid))
        .collect();

    ranked.sort_by(|a, b| {
        b.3.partial_cmp(&a.3).unwrap_or_else(|| {
            if b.3.is_nan() && a.3.is_nan() {
                Ordering::Equal
            } else if b.3.is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
    });
    ranked.truncate(k_usize);

    let row_ids: Vec<u64> = ranked.iter().map(|(rowid, _, _, _)| *rowid).collect();
    let projection_cols = {
        let schema: Schema = handle.dataset.schema().into();
        schema
            .fields()
            .iter()
            .filter(|f| !is_default_excluded_field(f))
            .map(|f| f.name().to_string())
            .collect::<Vec<_>>()
    };

    let projection = ProjectionRequest::from_columns(&projection_cols, handle.dataset.schema());
    let rows = match runtime::block_on(handle.dataset.take_rows(&row_ids, projection)) {
        Ok(Ok(batch)) => batch,
        Ok(Err(err)) => {
            set_last_error(
                ErrorCode::HybridStreamCreate,
                format!("hybrid take_rows: {err}"),
            );
            return ptr::null_mut();
        }
        Err(err) => {
            set_last_error(ErrorCode::Runtime, format!("runtime: {err}"));
            return ptr::null_mut();
        }
    };

    let mut dist_builder = Float32Builder::with_capacity(rows.num_rows());
    let mut score_builder = Float32Builder::with_capacity(rows.num_rows());
    let mut hybrid_builder = Float32Builder::with_capacity(rows.num_rows());
    for (_, dist, score, hybrid) in &ranked {
        match dist {
            Some(v) => dist_builder.append_value(*v),
            None => dist_builder.append_null(),
        }
        match score {
            Some(v) => score_builder.append_value(*v),
            None => score_builder.append_null(),
        }
        if hybrid.is_finite() {
            hybrid_builder.append_value(*hybrid);
        } else {
            hybrid_builder.append_null();
        }
    }

    let mut cols: Vec<Arc<dyn Array>> = rows.columns().to_vec();
    cols.push(Arc::new(dist_builder.finish()) as Arc<dyn Array>);
    cols.push(Arc::new(score_builder.finish()) as Arc<dyn Array>);
    cols.push(Arc::new(hybrid_builder.finish()) as Arc<dyn Array>);

    let mut fields = rows.schema().fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        DISTANCE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        SCORE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        HYBRID_SCORE_COLUMN,
        DataType::Float32,
        true,
    )));

    let out_schema = Arc::new(arrow::datatypes::Schema::new(fields));
    let rows = match RecordBatch::try_new(out_schema, cols) {
        Ok(v) => v,
        Err(err) => {
            set_last_error(
                ErrorCode::HybridStreamCreate,
                format!("hybrid batch: {err}"),
            );
            return ptr::null_mut();
        }
    };

    clear_last_error();
    Box::into_raw(Box::new(StreamHandle::Batches(vec![rows]))) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_hybrid_stream(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    text_column: *const c_char,
    text_query: *const c_char,
    k: u64,
    filter_sql: *const c_char,
    prefilter: u8,
    alpha: f32,
    oversample_factor: u32,
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
    let text_column = match cstr_to_str(text_column, "text_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let text_query = match cstr_to_str(text_query, "text_query") {
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
    let filter_hybrid = filter.map(|v| HybridFilter::Sql(v.to_string()));
    return create_hybrid_stream_impl(
        handle,
        vector_column,
        query_values,
        text_column,
        text_query,
        k,
        filter_hybrid,
        prefilter,
        alpha,
        oversample_factor,
    );
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_hybrid_stream_ir(
    dataset: *mut c_void,
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    text_column: *const c_char,
    text_query: *const c_char,
    k: u64,
    filter_ir: *const u8,
    filter_ir_len: usize,
    prefilter: u8,
    alpha: f32,
    oversample_factor: u32,
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
    let text_column = match cstr_to_str(text_column, "text_column") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let text_query = match cstr_to_str(text_query, "text_query") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };
    let query_values = match slice_from_ptr(query_values, query_len, "query_values") {
        Ok(v) => v,
        Err(()) => return ptr::null_mut(),
    };

    let filter = match parse_optional_filter_ir(
        filter_ir,
        filter_ir_len,
        ErrorCode::HybridStreamCreate,
        "hybrid filter_ir",
    ) {
        Ok(v) => v.map(HybridFilter::Expr),
        Err(()) => return ptr::null_mut(),
    };

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    create_hybrid_stream_impl(
        handle,
        vector_column,
        query_values,
        text_column,
        text_query,
        k,
        filter,
        prefilter,
        alpha,
        oversample_factor,
    )
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

    let stream = unsafe { &mut *(stream as *mut StreamHandle) };

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
            let _ = Box::from_raw(stream as *mut StreamHandle);
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
