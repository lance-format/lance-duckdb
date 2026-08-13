use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Arc;

use arrow::ffi::FFI_ArrowSchema;
use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;

use lance_namespace::models::{
    CreateNamespaceRequest, DeclareTableRequest, DescribeTableRequest, DropNamespaceRequest,
    DropTableRequest, ListNamespacesRequest, ListTablesRequest,
};
use lance_namespace::schema::convert_json_arrow_schema;
use lance_namespace::{ErrorCode as NamespaceErrorCode, LanceNamespace, NamespaceError};
use lance_namespace_impls::RestNamespaceBuilder;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::session::{record_dataset_open, record_namespace_describe};
use super::types::DatasetHandle;
use super::util::{
    cstr_to_str, optional_session_handle, schema_to_ffi_arrow_schema, to_c_string, FfiError,
    FfiResult,
};

unsafe fn optional_cstr_to_string(
    ptr: *const c_char,
    what: &'static str,
) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    let s = unsafe { cstr_to_str(ptr, what)? };
    if s.is_empty() {
        return Ok(None);
    }
    Ok(Some(s.to_string()))
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

fn build_config(
    endpoint: &str,
    bearer_token: Option<&str>,
    api_key: Option<&str>,
    headers_tsv: Option<&str>,
) -> RestNamespaceBuilder {
    let mut builder = RestNamespaceBuilder::new(endpoint);
    if let Some(token) = bearer_token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    if let Some(key) = api_key {
        builder = builder.header("x-api-key", key.to_string());
    }
    // Add custom headers from TSV
    for (key, value) in parse_headers_tsv(headers_tsv) {
        builder = builder.header(key, value);
    }
    builder
}

fn storage_options_to_tsv(storage_options: std::collections::HashMap<String, String>) -> String {
    if storage_options.is_empty() {
        return String::new();
    }
    let mut items: Vec<(String, String)> = storage_options.into_iter().collect();
    items.sort_by(|(a, _), (b, _)| a.cmp(b));
    items
        .into_iter()
        .map(|(k, v)| format!("{k}\t{v}"))
        .collect::<Vec<_>>()
        .join("\n")
}

const STRING_LIST_PREFIX: &str = "LID1;";

fn encode_string_list(values: &[String]) -> String {
    let mut encoded = format!("{STRING_LIST_PREFIX}{};", values.len());
    for value in values {
        encoded.push_str(&format!("{}:", value.len()));
        encoded.push_str(value);
    }
    encoded
}

pub(crate) fn decode_id(id: &str, delimiter: &str) -> FfiResult<Vec<String>> {
    let Some(encoded) = id.strip_prefix(STRING_LIST_PREFIX) else {
        return Ok(if id.is_empty() {
            Vec::new()
        } else {
            id.split(delimiter).map(ToString::to_string).collect()
        });
    };

    let (count_text, mut encoded) = encoded.split_once(';').ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "invalid Lance identifier list encoding",
        )
    })?;
    let count = count_text.parse::<usize>().map_err(|_| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            "invalid Lance identifier list count",
        )
    })?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let colon = encoded.find(':').ok_or_else(|| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                "invalid Lance identifier list encoding",
            )
        })?;
        let length = encoded[..colon].parse::<usize>().map_err(|_| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                "invalid Lance identifier length",
            )
        })?;
        encoded = &encoded[colon + 1..];
        if length > encoded.len() || !encoded.is_char_boundary(length) {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "invalid Lance identifier length",
            ));
        }
        values.push(encoded[..length].to_string());
        encoded = &encoded[length..];
    }
    if !encoded.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "trailing data in Lance identifier list",
        ));
    }
    Ok(values)
}

fn namespace_operation_config(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(impl LanceNamespace, Vec<String>)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let namespace_id = unsafe { cstr_to_str(namespace_id, "namespace_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? }
        .unwrap_or_else(|| "$".to_string());
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };
    let id = decode_id(namespace_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();
    Ok((namespace, id))
}

fn is_unsupported_namespace_operation(error: &LanceError) -> bool {
    matches!(error, LanceError::NotSupported { .. })
        || match error {
            LanceError::Namespace { source, .. } => source
                .downcast_ref::<NamespaceError>()
                .is_some_and(|error| error.code() == NamespaceErrorCode::Unsupported),
            _ => false,
        }
        // lance-namespace 9.0.1 maps an empty HTTP 501 response to Internal
        // because there is no structured error code to preserve.
        || error.to_string().contains("status=501 Not Implemented")
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_list_namespaces(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> *const c_char {
    let result = (|| {
        let (namespace, id) = namespace_operation_config(
            endpoint,
            namespace_id,
            bearer_token,
            api_key,
            delimiter,
            headers_tsv,
        )?;
        runtime::block_on(async move {
            let mut out = Vec::new();
            let mut page_token = None;
            loop {
                let mut request = ListNamespacesRequest::new();
                request.id = Some(id.clone());
                request.page_token = page_token.clone();
                request.limit = Some(1000);
                let response = match namespace.list_namespaces(request).await {
                    Ok(response) => response,
                    Err(err) if is_unsupported_namespace_operation(&err) => break,
                    Err(err) => {
                        return Err(FfiError::new(
                            ErrorCode::NamespaceListNamespaces,
                            format!("namespace list_namespaces: {err}"),
                        ));
                    }
                };
                out.extend(response.namespaces);
                match response.page_token {
                    Some(token) if !token.is_empty() => page_token = Some(token),
                    _ => break,
                }
            }
            Ok::<_, FfiError>(out)
        })
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
    })();
    match result {
        Ok(namespaces) => {
            clear_last_error();
            to_c_string(encode_string_list(&namespaces)).into_raw() as *const c_char
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_create_namespace(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    mode: *const c_char,
) -> i32 {
    let result = (|| {
        let (namespace, id) = namespace_operation_config(
            endpoint,
            namespace_id,
            bearer_token,
            api_key,
            delimiter,
            headers_tsv,
        )?;
        let mode = unsafe { optional_cstr_to_string(mode, "mode")? };
        runtime::block_on(async move {
            let mut request = CreateNamespaceRequest::new();
            request.id = Some(id);
            request.mode = mode;
            namespace.create_namespace(request).await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceCreateNamespace,
                    format!("namespace create_namespace: {err}"),
                )
            })?;
            Ok::<_, FfiError>(())
        })
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
    })();
    match result {
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

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_drop_namespace(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    behavior: *const c_char,
) -> i32 {
    let result = (|| {
        let (namespace, id) = namespace_operation_config(
            endpoint,
            namespace_id,
            bearer_token,
            api_key,
            delimiter,
            headers_tsv,
        )?;
        let behavior = unsafe { optional_cstr_to_string(behavior, "behavior")? };
        runtime::block_on(async move {
            let mut request = DropNamespaceRequest::new();
            request.id = Some(id);
            request.behavior = behavior;
            namespace.drop_namespace(request).await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceDropNamespace,
                    format!("namespace drop_namespace: {err}"),
                )
            })?;
            Ok::<_, FfiError>(())
        })
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
    })();
    match result {
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

fn list_tables_inner(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<Vec<String>> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let namespace_id = unsafe { cstr_to_str(namespace_id, "namespace_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace_id = decode_id(namespace_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let tables = runtime::block_on(async move {
        let mut out = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut req = ListTablesRequest::new();
            req.id = Some(namespace_id.clone());
            req.page_token = page_token.clone();
            req.limit = Some(1000);
            let resp = namespace.list_tables(req).await.map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceListTables,
                    format!("namespace list_tables: {err}"),
                )
            })?;
            out.extend(resp.tables);
            match resp.page_token {
                Some(token) if !token.is_empty() => page_token = Some(token),
                _ => break,
            }
        }
        Ok::<_, FfiError>(out)
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(tables)
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_list_tables(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> *const c_char {
    match list_tables_inner(
        endpoint,
        namespace_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(tables) => {
            clear_last_error();
            to_c_string(encode_string_list(&tables)).into_raw() as *const c_char
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

fn describe_table_info_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(String, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let table_id_segments = decode_id(table_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (location, storage_options_tsv) = runtime::block_on(async move {
        record_namespace_describe();
        let mut req = DescribeTableRequest::new();
        req.id = Some(table_id_segments);
        req.with_table_uri = Some(true);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                format!("namespace describe_table: {err}"),
            )
        })?;
        let location = resp.table_uri.or(resp.location).ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                "namespace describe_table: missing location and table_uri",
            )
        })?;
        let storage_options_tsv = storage_options_to_tsv(resp.storage_options.unwrap_or_default());
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((location, storage_options_tsv))
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    if !out_location.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_location, ptr::null());
        }
    }
    if !out_storage_options_tsv.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
        }
    }

    match describe_table_info_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            if !out_location.is_null() {
                let c = CString::new(location).unwrap_or_else(|_| to_c_string("invalid location"));
                unsafe {
                    std::ptr::write_unaligned(out_location, c.into_raw() as *const c_char);
                }
            }
            if !out_storage_options_tsv.is_null() {
                let c = CString::new(storage_options_tsv)
                    .unwrap_or_else(|_| to_c_string("invalid storage options"));
                unsafe {
                    std::ptr::write_unaligned(
                        out_storage_options_tsv,
                        c.into_raw() as *const c_char,
                    );
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn create_empty_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<(String, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let table_id_segments = decode_id(table_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (location, storage_options_tsv) = runtime::block_on(async move {
        let mut req = DeclareTableRequest::new();
        req.id = Some(table_id_segments);
        let resp = namespace.declare_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                format!("namespace declare_table: {err}"),
            )
        })?;
        let location = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                "namespace declare_table: missing location",
            )
        })?;
        let storage_options_tsv = storage_options_to_tsv(resp.storage_options.unwrap_or_default());
        Ok::<_, FfiError>((location, storage_options_tsv))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((location, storage_options_tsv))
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_create_empty_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_location: *mut *const c_char,
    out_storage_options_tsv: *mut *const c_char,
) -> i32 {
    if !out_location.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_location, ptr::null());
        }
    }
    if !out_storage_options_tsv.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_storage_options_tsv, ptr::null());
        }
    }

    match create_empty_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok((location, storage_options_tsv)) => {
            clear_last_error();
            if !out_location.is_null() {
                let c = CString::new(location).unwrap_or_else(|_| to_c_string("invalid location"));
                unsafe {
                    std::ptr::write_unaligned(out_location, c.into_raw() as *const c_char);
                }
            }
            if !out_storage_options_tsv.is_null() {
                let c = CString::new(storage_options_tsv)
                    .unwrap_or_else(|_| to_c_string("invalid storage options"));
                unsafe {
                    std::ptr::write_unaligned(
                        out_storage_options_tsv,
                        c.into_raw() as *const c_char,
                    );
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn drop_table_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<()> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let table_id_segments = decode_id(table_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    runtime::block_on(async move {
        let mut req = DropTableRequest::new();
        req.id = Some(table_id_segments);
        match namespace.drop_table(req).await {
            Ok(_) => Ok(()),
            Err(LanceError::NotFound { .. }) => Ok(()),
            Err(err) => Err(FfiError::new(
                ErrorCode::NamespaceDropTable,
                format!("namespace drop_table '{table_id}': {err}"),
            )),
        }
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_drop_table(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> i32 {
    match drop_table_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
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

/// Describe a table with `load_detailed_metadata=true` and return the schema
/// as a JSON string. This avoids opening the dataset from S3.
fn describe_table_with_schema_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
) -> FfiResult<String> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let table_id_segments = decode_id(table_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let schema_json = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        req.id = Some(table_id_segments);
        req.with_table_uri = Some(true);
        req.load_detailed_metadata = Some(true);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table: {err}"),
            )
        })?;

        let schema = resp.schema.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table: missing schema in response",
            )
        })?;

        serde_json::to_string(&schema).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to serialize schema: {err}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(schema_json)
}

#[no_mangle]
pub unsafe extern "C" fn lance_namespace_describe_table_with_schema(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_schema_json: *mut *const c_char,
) -> i32 {
    if !out_schema_json.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_schema_json, ptr::null());
        }
    }

    match describe_table_with_schema_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
    ) {
        Ok(schema_json) => {
            clear_last_error();
            if !out_schema_json.is_null() {
                let c = CString::new(schema_json)
                    .unwrap_or_else(|_| to_c_string("invalid schema json"));
                unsafe {
                    std::ptr::write_unaligned(out_schema_json, c.into_raw() as *const c_char);
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn open_dataset_in_namespace_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
) -> FfiResult<(DatasetHandle, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let table_id_segments = decode_id(table_id, &delimiter)?;
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();
    let session = unsafe { optional_session_handle(session)? };

    let (dataset, table_uri) = runtime::block_on(async move {
        record_namespace_describe();
        let mut builder = DatasetBuilder::from_namespace(Arc::new(namespace), table_id_segments)
            .await
            .map_err(|err| {
                FfiError::new(
                    ErrorCode::NamespaceDescribeTable,
                    format!("namespace describe_table: {err}"),
                )
            })?;
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetOpen,
                format!("namespace dataset open: {err}"),
            )
        })?;
        let table_uri = dataset.uri().to_string();
        Ok::<_, FfiError>((Arc::new(dataset), table_uri))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    record_dataset_open();
    Ok((DatasetHandle::new(dataset), table_uri))
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_namespace(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
        ptr::null_mut(),
    ) {
        Ok((handle, table_uri)) => {
            clear_last_error();
            if !out_table_uri.is_null() {
                let uri_c = CString::new(table_uri).unwrap_or_else(|_| to_c_string("invalid uri"));
                unsafe {
                    std::ptr::write_unaligned(out_table_uri, uri_c.into_raw() as *const c_char);
                }
            }
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_namespace_with_session(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
        session,
    ) {
        Ok((handle, table_uri)) => {
            clear_last_error();
            if !out_table_uri.is_null() {
                let uri_c = CString::new(table_uri).unwrap_or_else(|_| to_c_string("invalid uri"));
                unsafe {
                    std::ptr::write_unaligned(out_table_uri, uri_c.into_raw() as *const c_char);
                }
            }
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

/// Convert a JSON Arrow schema string to Arrow C Data Interface ArrowSchema.
#[no_mangle]
pub unsafe extern "C" fn lance_json_arrow_schema_to_c(
    json_schema: *const c_char,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    let result = (|| -> FfiResult<()> {
        let json_str = unsafe { cstr_to_str(json_schema, "json_schema")? };
        let json_arrow: lance_namespace::models::JsonArrowSchema = serde_json::from_str(json_str)
            .map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to parse JSON arrow schema: {err}"),
            )
        })?;
        let arrow_schema = convert_json_arrow_schema(&json_arrow).map_err(|err| {
            FfiError::new(
                ErrorCode::SchemaExport,
                format!("failed to convert JSON arrow schema: {err}"),
            )
        })?;
        let ffi_schema = schema_to_ffi_arrow_schema(&arrow_schema)?;
        unsafe {
            std::ptr::write_unaligned(out_schema, ffi_schema);
        }
        Ok(())
    })();

    match result {
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

#[cfg(test)]
mod tests {
    use super::{decode_id, encode_string_list};

    #[test]
    fn identifier_list_round_trips_delimiters_and_newlines() {
        let values = vec![
            "default".to_string(),
            "a$b".to_string(),
            "a\nb".to_string(),
            "销售".to_string(),
        ];
        let encoded = encode_string_list(&values);
        assert_eq!(decode_id(&encoded, "$").unwrap(), values);
    }

    #[test]
    fn legacy_identifier_still_uses_configured_delimiter() {
        assert_eq!(
            decode_id("default$schema$table", "$").unwrap(),
            vec!["default", "schema", "table"]
        );
    }
}
