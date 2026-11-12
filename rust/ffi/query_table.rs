use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::io::Cursor;
use std::ptr;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::{cast_with_options, CastOptions};
use arrow::datatypes::{DataType, FieldRef, Schema};
use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::record_batch::RecordBatchReader;
use datafusion_expr::Expr;
use datafusion_sql::unparser::dialect::CustomDialectBuilder;
use datafusion_sql::unparser::Unparser;
use lance_namespace::models::{
    QueryTableRequest, QueryTableRequestColumns, QueryTableRequestFullTextQuery,
    QueryTableRequestVector, StringFtsQuery,
};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::{DirectoryNamespaceBuilder, RestNamespaceBuilder};

use crate::constants::ROW_ID_COLUMN;
use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::StreamHandle;
use super::util::{
    cstr_to_str, optional_distance_type, parse_optional_filter_ir, slice_from_ptr, FfiError,
    FfiResult,
};

const NAMESPACE_KIND_DIRECTORY: u8 = 0;
const NAMESPACE_KIND_REST: u8 = 1;

#[repr(C)]
pub struct LanceNamespaceQueryConfig {
    namespace_kind: u8,
    root: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    columns: *const *const c_char,
    columns_len: usize,
    filter: *const c_char,
    k: u64,
    prefilter: u8,
}

#[repr(C)]
pub struct LanceNamespaceVectorSearchOptions {
    vector_column: *const c_char,
    query_values: *const f32,
    query_len: usize,
    nprobes: u64,
    refine_factor: u64,
    ef: u64,
    distance_type: *const c_char,
    use_index: u8,
    with_row_id: u8,
}

#[repr(C)]
pub struct LanceNamespaceFtsSearchOptions {
    text_column: *const c_char,
    query: *const c_char,
    with_row_id: u8,
}

enum NamespaceBackend {
    Directory {
        root: String,
        storage_options: HashMap<String, String>,
    },
    Rest {
        endpoint: String,
        bearer_token: Option<String>,
        api_key: Option<String>,
        delimiter: Option<String>,
        headers: Vec<(String, String)>,
    },
}

struct ParsedNamespaceQueryConfig {
    backend: NamespaceBackend,
    table_id: String,
    columns: Vec<String>,
    filter: Option<String>,
    k: i32,
    prefilter: bool,
}

unsafe fn optional_cstr_to_string(
    ptr: *const c_char,
    what: &'static str,
) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { cstr_to_str(ptr, what)? };
    if s.is_empty() {
        Ok(None)
    } else {
        Ok(Some(s.to_string()))
    }
}

unsafe fn parse_string_array(
    ptr: *const *const c_char,
    len: usize,
    what: &'static str,
) -> FfiResult<Vec<String>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        ));
    }

    let values = unsafe { slice_from_ptr(ptr, len, what)? };
    let mut out = Vec::with_capacity(len);
    for (idx, &value_ptr) in values.iter().enumerate() {
        if value_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("{what}[{idx}] is null"),
            ));
        }
        let value = unsafe { CStr::from_ptr(value_ptr) }
            .to_str()
            .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what}[{idx}] utf8: {err}")))?;
        out.push(value.to_string());
    }
    Ok(out)
}

unsafe fn parse_storage_options(
    keys: *const *const c_char,
    values: *const *const c_char,
    len: usize,
) -> FfiResult<HashMap<String, String>> {
    let keys = unsafe { parse_string_array(keys, len, "option_keys")? };
    let values = unsafe { parse_string_array(values, len, "option_values")? };
    if keys.len() != values.len() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option key/value length mismatch",
        ));
    }
    Ok(keys.into_iter().zip(values).collect())
}

fn parse_headers_tsv(headers_tsv: Option<&str>) -> Vec<(String, String)> {
    headers_tsv
        .map(|tsv| {
            tsv.lines()
                .filter_map(|line| {
                    let mut parts = line.splitn(2, '\t');
                    match (parts.next(), parts.next()) {
                        (Some(k), Some(v)) if !k.is_empty() => Some((k.to_string(), v.to_string())),
                        _ => None,
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

unsafe fn parse_config(
    config: *const LanceNamespaceQueryConfig,
) -> FfiResult<ParsedNamespaceQueryConfig> {
    if config.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace query config is null",
        ));
    }
    let config = unsafe { &*config };
    let table_id = unsafe { cstr_to_str(config.table_id, "table_id")? }.to_string();
    let columns = unsafe { parse_string_array(config.columns, config.columns_len, "columns")? };
    let filter = unsafe { optional_cstr_to_string(config.filter, "filter")? };
    let k = if config.k > i32::MAX as u64 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "k must fit in i32",
        ));
    } else {
        config.k as i32
    };

    let backend = match config.namespace_kind {
        NAMESPACE_KIND_DIRECTORY => {
            let root = unsafe { cstr_to_str(config.root, "root")? }.to_string();
            let storage_options = unsafe {
                parse_storage_options(config.option_keys, config.option_values, config.options_len)?
            };
            NamespaceBackend::Directory {
                root,
                storage_options,
            }
        }
        NAMESPACE_KIND_REST => {
            let endpoint = unsafe { cstr_to_str(config.endpoint, "endpoint")? }.to_string();
            let bearer_token =
                unsafe { optional_cstr_to_string(config.bearer_token, "bearer_token")? };
            let api_key = unsafe { optional_cstr_to_string(config.api_key, "api_key")? };
            let delimiter = unsafe { optional_cstr_to_string(config.delimiter, "delimiter")? };
            let headers_tsv =
                unsafe { optional_cstr_to_string(config.headers_tsv, "headers_tsv")? };
            let headers = parse_headers_tsv(headers_tsv.as_deref());
            NamespaceBackend::Rest {
                endpoint,
                bearer_token,
                api_key,
                delimiter,
                headers,
            }
        }
        other => {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("unknown namespace kind: {other}"),
            ))
        }
    };

    Ok(ParsedNamespaceQueryConfig {
        backend,
        table_id,
        columns,
        filter,
        k,
        prefilter: config.prefilter != 0,
    })
}

fn apply_base_request(config: &ParsedNamespaceQueryConfig, request: &mut QueryTableRequest) {
    request.id = Some(match &config.backend {
        NamespaceBackend::Rest { delimiter, .. } => {
            let delimiter = delimiter.as_deref().unwrap_or("$");
            config
                .table_id
                .split(delimiter)
                .map(str::to_string)
                .collect()
        }
        NamespaceBackend::Directory { .. } => vec![config.table_id.clone()],
    });
    request.prefilter = Some(config.prefilter);
    if !config.columns.is_empty() {
        let mut columns = QueryTableRequestColumns::new();
        columns.column_names = Some(config.columns.clone());
        request.columns = Some(Box::new(columns));
    }
    if let Some(filter) = &config.filter {
        request.filter = Some(filter.clone());
    }
}

async fn execute_query_table(
    config: ParsedNamespaceQueryConfig,
    request: QueryTableRequest,
) -> FfiResult<Vec<u8>> {
    match config.backend {
        NamespaceBackend::Directory {
            root,
            storage_options,
        } => {
            let mut builder = DirectoryNamespaceBuilder::new(&root).manifest_enabled(false);
            if !storage_options.is_empty() {
                builder = builder.storage_options(storage_options);
            }
            let namespace = builder.build().await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!("dir namespace build '{root}': {err}"),
                )
            })?;
            namespace
                .query_table(request)
                .await
                .map(|bytes| bytes.to_vec())
                .map_err(|err| {
                    FfiError::new(
                        ErrorCode::NamespaceQueryTable,
                        format!("dir namespace query_table: {err}"),
                    )
                })
        }
        NamespaceBackend::Rest {
            endpoint,
            bearer_token,
            api_key,
            delimiter,
            headers,
        } => {
            let mut builder = RestNamespaceBuilder::new(&endpoint);
            if let Some(token) = bearer_token {
                builder = builder.header("Authorization", format!("Bearer {token}"));
            }
            if let Some(key) = api_key {
                builder = builder.header("x-api-key", key);
            }
            for (key, value) in headers {
                builder = builder.header(key, value);
            }
            if let Some(delimiter) = delimiter {
                builder = builder.delimiter(delimiter);
            }
            let namespace = builder.build();
            namespace
                .query_table(request)
                .await
                .map(|bytes| bytes.to_vec())
                .map_err(|err| {
                    FfiError::new(
                        ErrorCode::NamespaceQueryTable,
                        format!("namespace query_table: {err}"),
                    )
                })
        }
    }
}

fn query_field_storage(field: &FieldRef, in_array: bool) -> FieldRef {
    let data_type = query_dictionary_storage(field.data_type(), in_array);
    if &data_type == field.data_type() {
        return field.clone();
    }
    Arc::new(field.as_ref().clone().with_data_type(data_type))
}

fn query_dictionary_storage(data_type: &DataType, in_array: bool) -> DataType {
    match data_type {
        DataType::Dictionary(_, value) if in_array || value.is_nested() => {
            query_dictionary_storage(value, in_array)
        }
        DataType::FixedSizeList(field, size) => {
            DataType::FixedSizeList(query_field_storage(field, true), *size)
        }
        DataType::List(field) => DataType::List(query_field_storage(field, true)),
        DataType::LargeList(field) => DataType::LargeList(query_field_storage(field, true)),
        DataType::ListView(field) => DataType::ListView(query_field_storage(field, true)),
        DataType::LargeListView(field) => DataType::LargeListView(query_field_storage(field, true)),
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| query_field_storage(field, in_array))
                .collect(),
        ),
        DataType::Map(field, sorted) => {
            DataType::Map(query_field_storage(field, in_array), *sorted)
        }
        DataType::Union(fields, mode) => DataType::Union(
            fields
                .iter()
                .map(|(id, field)| (id, query_field_storage(field, in_array)))
                .collect(),
            *mode,
        ),
        DataType::RunEndEncoded(ends, values) => {
            DataType::RunEndEncoded(ends.clone(), query_field_storage(values, in_array))
        }
        _ => data_type.clone(),
    }
}

fn decode_query_dictionaries(
    batch: RecordBatch,
    requested: &HashSet<String>,
) -> FfiResult<RecordBatch> {
    let schema = batch.schema();
    let fields = schema
        .fields()
        .iter()
        .map(|field| {
            if requested.contains(field.name()) {
                query_field_storage(field, false)
            } else {
                field.clone()
            }
        })
        .collect::<Vec<_>>();
    if fields.iter().zip(schema.fields()).all(|(a, b)| a == b) {
        return Ok(batch);
    }
    // DuckDB cannot safely decode dictionary values that are containers, or
    // dictionaries inside lists/arrays. Use Arrow's lossless dictionary
    // decoding for only those layouts; ordinary batches retain their buffers.
    let options = CastOptions {
        safe: false,
        ..Default::default()
    };
    let columns = batch
        .columns()
        .iter()
        .zip(&fields)
        .map(|(column, field)| {
            if column.data_type() == field.data_type() {
                Ok(column.clone())
            } else {
                cast_with_options(column.as_ref(), field.data_type(), &options)
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("decode Arrow dictionary storage: {err}"),
            )
        })?;
    RecordBatch::try_new(
        Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
        columns,
    )
    .map_err(|err| {
        FfiError::new(
            ErrorCode::NamespaceQueryTable,
            format!("decoded Arrow dictionary batch: {err}"),
        )
    })
}

fn collect_ipc_batches(
    reader: impl RecordBatchReader,
    encoding: &str,
    requested: &HashSet<String>,
) -> FfiResult<Vec<RecordBatch>> {
    let schema = reader.schema();
    let mut batches = reader.collect::<Result<Vec<_>, _>>().map_err(|err| {
        FfiError::new(
            ErrorCode::NamespaceQueryTable,
            format!("read Arrow IPC {encoding}: {err}"),
        )
    })?;
    // Keep the response schema observable by the decoder even with no rows.
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(schema));
    }
    batches
        .into_iter()
        .map(|batch| decode_query_dictionaries(batch, requested))
        .collect()
}

fn ipc_bytes_to_batches(
    bytes: Vec<u8>,
    requested: &HashSet<String>,
) -> FfiResult<Vec<RecordBatch>> {
    match FileReader::try_new(Cursor::new(bytes.clone()), None) {
        Ok(reader) => collect_ipc_batches(reader, "file", requested),
        Err(file_err) => {
            let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(|stream_err| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!(
                        "read Arrow IPC response: file reader failed: {file_err}; stream reader failed: {stream_err}"
                    ),
                )
            })?;
            collect_ipc_batches(reader, "stream", requested)
        }
    }
}

fn execute_to_stream(
    config: ParsedNamespaceQueryConfig,
    request: QueryTableRequest,
) -> FfiResult<StreamHandle> {
    let batches = execute_to_batches(config, request)?;
    Ok(StreamHandle::Batches(batches.into_iter()))
}

fn execute_to_batches(
    config: ParsedNamespaceQueryConfig,
    request: QueryTableRequest,
) -> FfiResult<Vec<RecordBatch>> {
    let requested = config.columns.iter().cloned().collect();
    let bytes = runtime::block_on(execute_query_table(config, request))
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
    ipc_bytes_to_batches(bytes, &requested)
}

/// Runs a search request. The search schema ends with its metric column and
/// then `_rowid`, while query_table places `_rowid` where the server chooses,
/// so it is moved last.
fn execute_search_to_stream(
    config: ParsedNamespaceQueryConfig,
    mut request: QueryTableRequest,
    with_row_id: bool,
) -> FfiResult<StreamHandle> {
    // Search responses still contain full rows. Keep the requested columns for
    // local decoding without changing the namespace request's projection.
    request.columns = None;
    if with_row_id {
        request.with_row_id = Some(true);
    }
    let batches = execute_to_batches(config, request)?
        .into_iter()
        .map(|batch| move_row_id_last(batch, with_row_id))
        .collect::<FfiResult<Vec<_>>>()?;
    Ok(StreamHandle::Batches(batches.into_iter()))
}

fn move_row_id_last(batch: RecordBatch, required: bool) -> FfiResult<RecordBatch> {
    let schema = batch.schema();
    let Ok(index) = schema.index_of(ROW_ID_COLUMN) else {
        if required {
            return Err(FfiError::new(
                ErrorCode::NamespaceQueryTable,
                "namespace query_table response has no _rowid column for with_row_id",
            ));
        }
        return Ok(batch);
    };
    // The bind schema declares `_rowid` as UInt64 and DuckDB reads the buffer
    // as that type, so any other type is rejected rather than misread.
    let data_type = schema.field(index).data_type();
    if required && data_type != &DataType::UInt64 {
        return Err(FfiError::new(
            ErrorCode::NamespaceQueryTable,
            format!(
                "namespace query_table response has _rowid of type {data_type}, expected UInt64"
            ),
        ));
    }
    let mut order = (0..batch.num_columns())
        .filter(|&column| column != index)
        .collect::<Vec<_>>();
    order.push(index);
    batch.project(&order).map_err(|err| {
        FfiError::new(
            ErrorCode::NamespaceQueryTable,
            format!("reorder _rowid: {err}"),
        )
    })
}

fn filter_expr_to_sql(expr: &Expr) -> FfiResult<String> {
    let dialect = CustomDialectBuilder::new()
        .with_identifier_quote_style('`')
        .build();
    Unparser::new(&dialect)
        .expr_to_sql(expr)
        .map(|sql| sql.to_string())
        .map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("unparse namespace filter: {err}"),
            )
        })
}

// A pushed filter arrives as a DataFusion `Expr`, but namespace query_table only accepts a
// filter string, so it is unparsed here and ANDed with whatever the caller passed as the
// explicit `filter :=` argument.
fn apply_filter_expr(request: &mut QueryTableRequest, filter_expr: Option<&Expr>) -> FfiResult<()> {
    let Some(filter_expr) = filter_expr else {
        return Ok(());
    };
    let filter_sql = filter_expr_to_sql(filter_expr)?;
    request.filter = Some(match request.filter.take() {
        Some(existing) => format!("({existing}) AND ({filter_sql})"),
        None => filter_sql,
    });
    Ok(())
}

fn build_namespace_scan_request(
    config: &ParsedNamespaceQueryConfig,
    filter_expr: Option<&Expr>,
    limit: i64,
    offset: i64,
    with_row_id: bool,
) -> FfiResult<QueryTableRequest> {
    if limit < -1 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan limit must be >= -1",
        ));
    }
    if offset < 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan offset must be non-negative",
        ));
    }
    if limit == -1 && offset != 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace query_table cannot apply OFFSET without LIMIT",
        ));
    }

    let k = if limit == -1 {
        0
    } else {
        i32::try_from(limit).map_err(|_| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                "namespace scan limit must fit in i32",
            )
        })?
    };
    let offset = i32::try_from(offset).map_err(|_| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace scan offset must fit in i32",
        )
    })?;

    let mut request = QueryTableRequest::new(k, QueryTableRequestVector::new());
    apply_base_request(config, &mut request);
    request.prefilter = None;
    if offset != 0 {
        request.offset = Some(offset);
    }
    if with_row_id {
        request.with_row_id = Some(true);
        let mut clear_columns = false;
        if let Some(columns) = request.columns.as_mut() {
            if let Some(column_names) = columns.column_names.as_mut() {
                column_names.retain(|name| name != "_rowid");
                clear_columns = column_names.is_empty();
            }
        }
        if clear_columns {
            request.columns = None;
        }
    }

    apply_filter_expr(&mut request, filter_expr)?;
    Ok(request)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_scan_stream_ir(
    config: *const LanceNamespaceQueryConfig,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
    with_row_id: u8,
) -> *mut c_void {
    match unsafe {
        create_namespace_scan_stream_ir_inner(
            config,
            filter_ir,
            filter_ir_len,
            limit,
            offset,
            with_row_id,
        )
    } {
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

unsafe fn create_namespace_scan_stream_ir_inner(
    config: *const LanceNamespaceQueryConfig,
    filter_ir: *const u8,
    filter_ir_len: usize,
    limit: i64,
    offset: i64,
    with_row_id: u8,
) -> FfiResult<StreamHandle> {
    let config = unsafe { parse_config(config)? };
    let filter_expr = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::NamespaceQueryTable,
            "namespace scan filter_ir",
        )?
    };
    let request = build_namespace_scan_request(
        &config,
        filter_expr.as_ref(),
        limit,
        offset,
        with_row_id != 0,
    )?;
    if limit == 0 {
        return Ok(StreamHandle::Batches(Vec::new().into_iter()));
    }
    execute_to_stream(config, request)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_vector_search_stream(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceVectorSearchOptions,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> *mut c_void {
    match unsafe {
        create_namespace_vector_search_stream_inner(config, options, filter_ir, filter_ir_len)
    } {
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

unsafe fn create_namespace_vector_search_stream_inner(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceVectorSearchOptions,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace vector search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
    if config.k == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace vector search k must be positive",
        ));
    }
    let options = unsafe { &*options };
    let vector_column = unsafe { cstr_to_str(options.vector_column, "vector_column")? };
    let query_values =
        unsafe { slice_from_ptr(options.query_values, options.query_len, "query_values")? };
    if query_values.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }

    let mut vector = QueryTableRequestVector::new();
    vector.single_vector = Some(query_values.to_vec());
    let filter_expr = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::NamespaceQueryTable,
            "namespace vector search filter_ir",
        )?
    };

    let mut request = QueryTableRequest::new(config.k, vector);
    apply_base_request(&config, &mut request);
    apply_filter_expr(&mut request, filter_expr.as_ref())?;
    request.vector_column = Some(vector_column.to_string());
    apply_vector_search_options(&mut request, options)?;

    execute_search_to_stream(config, request, options.with_row_id != 0)
}

fn apply_vector_search_options(
    request: &mut QueryTableRequest,
    options: &LanceNamespaceVectorSearchOptions,
) -> Result<(), FfiError> {
    let to_i32 = |value: u64, name: &str| {
        i32::try_from(value).map_err(|_| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("{name} must fit in i32"),
            )
        })
    };
    if options.nprobes != 0 {
        request.nprobes = Some(to_i32(options.nprobes, "nprobes")?);
    }
    if options.refine_factor != 0 {
        request.refine_factor = Some(to_i32(options.refine_factor, "refine_factor")?);
    }
    if options.ef != 0 {
        request.ef = Some(to_i32(options.ef, "ef")?);
    }
    if let Some(distance_type) = unsafe { optional_distance_type(options.distance_type)? } {
        request.distance_type = Some(distance_type.to_string());
    }
    request.bypass_vector_index = Some(options.use_index == 0);
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_fts_search_stream(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceFtsSearchOptions,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> *mut c_void {
    match unsafe {
        create_namespace_fts_search_stream_inner(config, options, filter_ir, filter_ir_len)
    } {
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

unsafe fn create_namespace_fts_search_stream_inner(
    config: *const LanceNamespaceQueryConfig,
    options: *const LanceNamespaceFtsSearchOptions,
    filter_ir: *const u8,
    filter_ir_len: usize,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace FTS search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
    if config.k == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace FTS search k must be positive",
        ));
    }
    let options = unsafe { &*options };
    let text_column = unsafe { cstr_to_str(options.text_column, "text_column")? };
    let query = unsafe { cstr_to_str(options.query, "query")? };

    let filter_expr = unsafe {
        parse_optional_filter_ir(
            filter_ir,
            filter_ir_len,
            ErrorCode::NamespaceQueryTable,
            "namespace FTS search filter_ir",
        )?
    };

    let mut request = QueryTableRequest::new(config.k, QueryTableRequestVector::new());
    apply_base_request(&config, &mut request);
    apply_filter_expr(&mut request, filter_expr.as_ref())?;

    let mut string_query = StringFtsQuery::new(query.to_string());
    string_query.columns = Some(vec![text_column.to_string()]);
    let mut fts_query = QueryTableRequestFullTextQuery::new();
    fts_query.string_query = Some(Box::new(string_query));
    request.full_text_query = Some(Box::new(fts_query));

    execute_search_to_stream(config, request, options.with_row_id != 0)
}

#[cfg(test)]
mod tests {
    use datafusion_expr::{col, lit};

    use super::*;

    fn rest_config(columns: &[&str]) -> ParsedNamespaceQueryConfig {
        ParsedNamespaceQueryConfig {
            backend: NamespaceBackend::Rest {
                endpoint: "http://localhost:8000".to_string(),
                bearer_token: None,
                api_key: None,
                delimiter: Some("$".to_string()),
                headers: Vec::new(),
            },
            table_id: "parent$child$table".to_string(),
            columns: columns.iter().map(|column| (*column).to_string()).collect(),
            filter: None,
            k: 0,
            prefilter: true,
        }
    }

    #[test]
    fn namespace_scan_request_pushes_query_shape() {
        let config = rest_config(&["name", "_rowid"]);
        let filter = col("age").gt(lit(21));
        let request = build_namespace_scan_request(&config, Some(&filter), 5, 2, true).unwrap();

        assert_eq!(
            request.id,
            Some(vec![
                "parent".to_string(),
                "child".to_string(),
                "table".to_string()
            ])
        );
        assert_eq!(request.k, 5);
        assert_eq!(request.offset, Some(2));
        assert_eq!(request.with_row_id, Some(true));
        assert_eq!(request.prefilter, None);
        assert_eq!(
            request.columns.unwrap().column_names,
            Some(vec!["name".to_string()])
        );
        let filter = request.filter.unwrap();
        assert!(filter.contains("age"));
        assert!(filter.contains("21"));
    }

    fn vector_options(ef: u64) -> LanceNamespaceVectorSearchOptions {
        LanceNamespaceVectorSearchOptions {
            vector_column: std::ptr::null(),
            query_values: std::ptr::null(),
            query_len: 0,
            nprobes: 3,
            refine_factor: 2,
            ef,
            distance_type: std::ptr::null(),
            use_index: 1,
            with_row_id: 0,
        }
    }

    #[test]
    fn namespace_vector_request_forwards_ef() {
        let mut request = QueryTableRequest::new(5, QueryTableRequestVector::new());
        apply_vector_search_options(&mut request, &vector_options(64)).unwrap();
        assert_eq!(request.ef, Some(64));
        assert_eq!(request.nprobes, Some(3));
        assert_eq!(request.refine_factor, Some(2));
        assert_eq!(request.bypass_vector_index, Some(false));

        let mut request = QueryTableRequest::new(5, QueryTableRequestVector::new());
        apply_vector_search_options(&mut request, &vector_options(0)).unwrap();
        assert_eq!(request.ef, None);
    }

    #[test]
    fn namespace_vector_request_forwards_distance_type() {
        let mut request = QueryTableRequest::new(5, QueryTableRequestVector::new());
        apply_vector_search_options(&mut request, &vector_options(0)).unwrap();
        assert_eq!(request.distance_type, None);

        let cosine = std::ffi::CString::new("COSINE").unwrap();
        let mut options = vector_options(0);
        options.distance_type = cosine.as_ptr();
        apply_vector_search_options(&mut request, &options).unwrap();
        assert_eq!(request.distance_type.as_deref(), Some("cosine"));

        let unknown = std::ffi::CString::new("manhattan").unwrap();
        options.distance_type = unknown.as_ptr();
        let error = apply_vector_search_options(&mut request, &options).unwrap_err();
        assert!(error
            .message
            .contains("Metric type 'manhattan' is not supported"));
    }

    #[test]
    fn namespace_vector_request_rejects_oversized_ef() {
        let mut request = QueryTableRequest::new(5, QueryTableRequestVector::new());
        let error =
            apply_vector_search_options(&mut request, &vector_options(u64::MAX)).unwrap_err();
        assert!(error.message.contains("ef must fit in i32"));
    }

    #[test]
    fn namespace_scan_request_rejects_offset_without_limit() {
        let config = rest_config(&["name"]);
        let error = build_namespace_scan_request(&config, None, -1, 1, false).unwrap_err();
        assert!(error.message.contains("cannot apply OFFSET without LIMIT"));
    }

    fn batch(columns: &[&str]) -> RecordBatch {
        use arrow::array::{ArrayRef, UInt64Array};
        use arrow::datatypes::{Field, Schema};
        let fields = columns
            .iter()
            .map(|name| Field::new(*name, DataType::UInt64, false))
            .collect::<Vec<_>>();
        let arrays = (0..columns.len() as u64)
            .map(|value| std::sync::Arc::new(UInt64Array::from(vec![value])) as ArrayRef)
            .collect();
        RecordBatch::try_new(std::sync::Arc::new(Schema::new(fields)), arrays).unwrap()
    }

    fn column_names(batch: &RecordBatch) -> Vec<String> {
        let schema = batch.schema();
        schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect()
    }

    #[test]
    fn search_response_puts_row_id_last() {
        let original = batch(&["id", "_rowid", "_distance"]);
        let moved = move_row_id_last(original.clone(), true).unwrap();
        assert_eq!(column_names(&moved), ["id", "_distance", "_rowid"]);
        assert_eq!(moved.column(2), original.column(1));

        let unchanged = move_row_id_last(batch(&["id", "_distance"]), false).unwrap();
        assert_eq!(column_names(&unchanged), ["id", "_distance"]);

        let error = move_row_id_last(batch(&["id", "_distance"]), true).unwrap_err();
        assert!(error.message.contains("no _rowid column"));

        let int32_row_id = RecordBatch::try_from_iter([(
            "_rowid",
            std::sync::Arc::new(arrow::array::Int32Array::from(vec![1])) as arrow::array::ArrayRef,
        )])
        .unwrap();
        let error = move_row_id_last(int32_row_id.clone(), true).unwrap_err();
        assert!(error
            .message
            .contains("_rowid of type Int32, expected UInt64"));
        assert!(move_row_id_last(int32_row_id, false).is_ok());
    }
}
