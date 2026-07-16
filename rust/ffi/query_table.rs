use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::io::Cursor;
use std::ptr;

use arrow::array::RecordBatch;
use arrow::ipc::reader::{FileReader, StreamReader};
use lance_namespace::models::{
    QueryTableRequest, QueryTableRequestColumns, QueryTableRequestFullTextQuery,
    QueryTableRequestVector, StringFtsQuery,
};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::{DirectoryNamespaceBuilder, RestNamespaceBuilder};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::StreamHandle;
use super::util::{cstr_to_str, slice_from_ptr, FfiError, FfiResult};

const NAMESPACE_KIND_DIRECTORY: u8 = 0;
const NAMESPACE_KIND_REST: u8 = 1;

#[repr(C)]
pub struct LanceNamespaceSearchConfig {
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
    use_index: u8,
}

#[repr(C)]
pub struct LanceNamespaceFtsSearchOptions {
    text_column: *const c_char,
    query: *const c_char,
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

struct ParsedConfig {
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

unsafe fn parse_config(config: *const LanceNamespaceSearchConfig) -> FfiResult<ParsedConfig> {
    if config.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace search config is null",
        ));
    }
    let config = unsafe { &*config };
    let table_id = unsafe { cstr_to_str(config.table_id, "table_id")? }.to_string();
    let columns = unsafe { parse_string_array(config.columns, config.columns_len, "columns")? };
    let filter = unsafe { optional_cstr_to_string(config.filter, "filter")? };
    let k = if config.k == 0 || config.k > i32::MAX as u64 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "k must be in the range 1..=i32::MAX",
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

    Ok(ParsedConfig {
        backend,
        table_id,
        columns,
        filter,
        k,
        prefilter: config.prefilter != 0,
    })
}

fn apply_base_request(config: &ParsedConfig, request: &mut QueryTableRequest) {
    // A qualified REST table id (e.g. "parent$child$tbl") must be sent as its
    // multi-segment namespace path, not a single segment: query_table POSTs
    // the id array in the request body, so the segment count is observable
    // server-side. This differs from GET operations like list_tables, whose
    // URL path re-joins the id with the delimiter and is therefore
    // wire-identical for one segment or many.
    // Directory namespaces have single-level ids and no delimiter (a
    // directory name may legitimately contain '$'), so their id deliberately
    // stays one segment.
    request.id = Some(match &config.backend {
        NamespaceBackend::Rest { delimiter, .. } => {
            // Must match the delimiter the REST client is built with
            // (RestNamespaceBuilder's default delimiter is "$").
            let delimiter = delimiter.as_deref().unwrap_or("$");
            config
                .table_id
                .split(delimiter)
                .map(|s| s.to_string())
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
    config: ParsedConfig,
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

fn ipc_bytes_to_batches(bytes: Vec<u8>) -> FfiResult<Vec<RecordBatch>> {
    match FileReader::try_new(Cursor::new(bytes.clone()), None) {
        Ok(reader) => reader.collect::<Result<Vec<_>, _>>().map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceQueryTable,
                format!("read Arrow IPC file: {err}"),
            )
        }),
        Err(file_err) => {
            let reader = StreamReader::try_new(Cursor::new(bytes), None).map_err(|stream_err| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!(
                        "read Arrow IPC response: file reader failed: {file_err}; stream reader failed: {stream_err}"
                    ),
                )
            })?;
            reader.collect::<Result<Vec<_>, _>>().map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceQueryTable,
                    format!("read Arrow IPC stream: {err}"),
                )
            })
        }
    }
}

fn execute_to_stream(config: ParsedConfig, request: QueryTableRequest) -> FfiResult<StreamHandle> {
    let bytes = runtime::block_on(execute_query_table(config, request))
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;
    let batches = ipc_bytes_to_batches(bytes)?;
    Ok(StreamHandle::Batches(batches.into_iter()))
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_vector_search_stream(
    config: *const LanceNamespaceSearchConfig,
    options: *const LanceNamespaceVectorSearchOptions,
) -> *mut c_void {
    match create_namespace_vector_search_stream_inner(config, options) {
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
    config: *const LanceNamespaceSearchConfig,
    options: *const LanceNamespaceVectorSearchOptions,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace vector search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
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
    let mut request = QueryTableRequest::new(config.k, vector);
    apply_base_request(&config, &mut request);
    request.vector_column = Some(vector_column.to_string());
    if options.nprobes != 0 {
        request.nprobes =
            Some(options.nprobes.try_into().map_err(|_| {
                FfiError::new(ErrorCode::InvalidArgument, "nprobes must fit in i32")
            })?);
    }
    if options.refine_factor != 0 {
        request.refine_factor = Some(options.refine_factor.try_into().map_err(|_| {
            FfiError::new(ErrorCode::InvalidArgument, "refine_factor must fit in i32")
        })?);
    }
    request.bypass_vector_index = Some(options.use_index == 0);

    execute_to_stream(config, request)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_namespace_fts_search_stream(
    config: *const LanceNamespaceSearchConfig,
    options: *const LanceNamespaceFtsSearchOptions,
) -> *mut c_void {
    match create_namespace_fts_search_stream_inner(config, options) {
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
    config: *const LanceNamespaceSearchConfig,
    options: *const LanceNamespaceFtsSearchOptions,
) -> FfiResult<StreamHandle> {
    if options.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "namespace FTS search options is null",
        ));
    }
    let config = unsafe { parse_config(config)? };
    let options = unsafe { &*options };
    let text_column = unsafe { cstr_to_str(options.text_column, "text_column")? };
    let query = unsafe { cstr_to_str(options.query, "query")? };

    let mut request = QueryTableRequest::new(config.k, QueryTableRequestVector::new());
    apply_base_request(&config, &mut request);

    let mut string_query = StringFtsQuery::new(query.to_string());
    string_query.columns = Some(vec![text_column.to_string()]);
    let mut fts_query = QueryTableRequestFullTextQuery::new();
    fts_query.string_query = Some(Box::new(string_query));
    request.full_text_query = Some(Box::new(fts_query));

    execute_to_stream(config, request)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use super::*;

    /// (endpoint, stop flag, log of raw requests) handed out by
    /// `spawn_query_server`.
    type MockServer = (String, Arc<Mutex<bool>>, Arc<Mutex<Vec<String>>>);

    /// Minimal in-process REST namespace mock: answers every request whose
    /// JSON body `id` array equals `expected_id` with a 200 and a
    /// deliberately non-Arrow JSON body; every other request gets a
    /// spec-shaped 404 error body (numeric code 4 = TableNotFound). The REST
    /// client joins the id segments with the delimiter in the URL path
    /// (identical for one segment or many), so the body `id` array is where a
    /// qualified id sent as a single segment shows up. Every raw request
    /// (head + body) is appended to the returned log so tests can assert on
    /// what actually went over the wire.
    fn spawn_query_server(expected_id: Vec<&'static str>) -> MockServer {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(Mutex::new(false));
        let stop_flag = stop.clone();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_log = requests.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if *stop_flag.lock().unwrap() {
                    break;
                }
                let Ok(mut stream) = stream else {
                    continue;
                };
                // Read the request head and the content-length body so the
                // client sees a complete exchange.
                let mut buf = Vec::new();
                let mut chunk = [0u8; 1024];
                let header_end = loop {
                    let Ok(n) = stream.read(&mut chunk) else {
                        break None;
                    };
                    if n == 0 {
                        break None;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(pos + 4);
                    }
                };
                let Some(header_end) = header_end else {
                    continue;
                };
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let content_length = head
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                while buf.len() < header_end + content_length {
                    let Ok(n) = stream.read(&mut chunk) else {
                        break;
                    };
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }

                let request_body = String::from_utf8_lossy(&buf[header_end..]).to_string();
                requests_log
                    .lock()
                    .unwrap()
                    .push(format!("{head}{request_body}"));

                let id_matches = serde_json::from_str::<serde_json::Value>(&request_body)
                    .ok()
                    .and_then(|request| request.get("id").and_then(|id| id.as_array()).cloned())
                    .is_some_and(|segments| {
                        segments
                            .iter()
                            .map(|segment| segment.as_str().unwrap_or_default())
                            .eq(expected_id.iter().copied())
                    });
                let response = if id_matches {
                    let body = "{\"not\": \"arrow\"}";
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else {
                    // Spec-shaped error body: the numeric `code` field drives
                    // client-side error classification, and 4 is the spec's
                    // TableNotFound code.
                    let error = "{\"code\": 4, \"error\": \"table not found\"}";
                    format!(
                        "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                        error.len(),
                        error
                    )
                };
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (endpoint, stop, requests)
    }

    #[test]
    fn test_namespace_vector_search_splits_multi_level_table_id() {
        // query_table POSTs the id array in its request body (unlike
        // list_tables, whose GET path re-joins the id), so the segment count
        // is observable server-side. The mock only answers for the parsed
        // 3-segment id: a qualified id sent as a single segment gets a 404
        // before the response body ever matters.
        let (endpoint, stop, requests) = spawn_query_server(vec!["parent", "child", "tbl"]);

        let endpoint_c = CString::new(endpoint).unwrap();
        let table_id_c = CString::new("parent$child$tbl").unwrap();
        let vector_column_c = CString::new("vec").unwrap();
        let query = [0.0f32, 1.0f32];

        let config = LanceNamespaceSearchConfig {
            namespace_kind: NAMESPACE_KIND_REST,
            root: ptr::null(),
            option_keys: ptr::null(),
            option_values: ptr::null(),
            options_len: 0,
            endpoint: endpoint_c.as_ptr(),
            table_id: table_id_c.as_ptr(),
            bearer_token: ptr::null(),
            api_key: ptr::null(),
            delimiter: ptr::null(), // default "$" must drive the split
            headers_tsv: ptr::null(),
            columns: ptr::null(),
            columns_len: 0,
            filter: ptr::null(),
            k: 3,
            prefilter: 0,
        };
        let options = LanceNamespaceVectorSearchOptions {
            vector_column: vector_column_c.as_ptr(),
            query_values: query.as_ptr(),
            query_len: query.len(),
            nprobes: 0,
            refine_factor: 0,
            use_index: 1,
        };

        let stream = unsafe { lance_create_namespace_vector_search_stream(&config, &options) };
        // The mock accepted the multi-segment id and returned 200 with a
        // deliberately non-Arrow body: the only permitted failure is the IPC
        // parse. A single-segment id would instead fail with the mock's 404
        // ("namespace query_table: ..." message).
        assert!(stream.is_null());
        let message = crate::error::lance_last_error_message();
        assert!(!message.is_null());
        let message_str = unsafe { std::ffi::CStr::from_ptr(message) }
            .to_string_lossy()
            .to_string();
        unsafe { crate::error::lance_free_string(message) };
        assert!(
            message_str.contains("read Arrow IPC"),
            "expected to fail only on the fake response body, got: {message_str}"
        );

        // The request hit the query route with the percent-encoded joined id
        // in the path and the split id array in the POST body.
        let log = requests.lock().unwrap();
        assert_eq!(log.len(), 1);
        assert!(
            log[0].contains("/v1/table/parent%24child%24tbl/query"),
            "request: {}",
            log[0]
        );
        let request_body = log[0].split("\r\n\r\n").nth(1).unwrap_or_default();
        let body_json: serde_json::Value = serde_json::from_str(request_body).unwrap();
        let id_segments = body_json
            .get("id")
            .and_then(|id| id.as_array())
            .map(|segments| {
                segments
                    .iter()
                    .map(|s| s.as_str().unwrap_or_default().to_string())
                    .collect::<Vec<_>>()
            });
        assert_eq!(
            id_segments,
            Some(vec![
                "parent".to_string(),
                "child".to_string(),
                "tbl".to_string()
            ])
        );

        *stop.lock().unwrap() = true;
    }
}
