use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Arc;

use arrow::ffi::FFI_ArrowSchema;
use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;

use lance_namespace::models::{
    DeclareTableRequest, DescribeTableRequest, DropTableRequest, ListTablesRequest,
};
use lance_namespace::schema::convert_json_arrow_schema;
use lance_namespace::LanceNamespace;
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
            req.id = Some(if namespace_id.is_empty() {
                Vec::new()
            } else {
                vec![namespace_id.to_string()]
            });
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
            let joined = tables.join("\n");
            to_c_string(joined).into_raw() as *const c_char
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
        req.id = Some(vec![table_id.to_string()]);
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
        req.id = Some(vec![table_id.to_string()]);
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
        req.id = Some(vec![table_id.to_string()]);
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
        req.id = Some(vec![table_id.to_string()]);
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
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();
    let session = unsafe { optional_session_handle(session)? };

    let (dataset, table_uri, storage_options) = runtime::block_on(async move {
        // Capture the namespace-vended storage options with an explicit
        // describe so revalidation can later detect option changes (e.g.
        // rotated credentials). `DatasetBuilder::from_namespace` consumes its
        // own describe internally, so this costs one extra namespace round
        // trip per open — opens only happen on cache misses.
        record_namespace_describe();
        let describe_request = DescribeTableRequest {
            id: Some(vec![table_id.to_string()]),
            with_table_uri: Some(true),
            ..Default::default()
        };
        let describe_response =
            namespace
                .describe_table(describe_request)
                .await
                .map_err(|err| {
                    FfiError::new(
                        ErrorCode::NamespaceDescribeTable,
                        format!("namespace describe_table: {err}"),
                    )
                })?;
        let storage_options = describe_response.storage_options;

        record_namespace_describe();
        let mut builder =
            DatasetBuilder::from_namespace(Arc::new(namespace), vec![table_id.to_string()])
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
        Ok::<_, FfiError>((Arc::new(dataset), table_uri, storage_options))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    record_dataset_open();
    Ok((
        DatasetHandle::new(dataset).with_namespace_storage_options(storage_options),
        table_uri,
    ))
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

/// Refresh a namespace-backed dataset handle if the table moved or is stale.
///
/// Namespace tables are cached by endpoint/table id rather than by physical
/// URI, so revalidating only the already-resolved handle would miss an
/// external drop/re-create that re-points the table to a new location. This
/// entry point first re-describes the table through the namespace (one
/// namespace round trip — the same order of cost as the namespace open path,
/// which itself starts with a describe) and reopens through the namespace when
/// the resolved location changed. When the location is unchanged, it falls
/// back to the manifest-identity revalidation used for plain datasets, which
/// catches both ordinary new commits and same-location re-creates (e-tag).
///
/// On success writes the refreshed handle (or null when the cached handle is
/// current) to `out_new_dataset`. `out_table_uri` receives the newly resolved
/// table URI only when the table moved; the caller frees it with
/// `lance_free_string`. Returns `0` on success and `-1` on error.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn lance_dataset_namespace_checkout_latest_if_stale(
    dataset: *mut c_void,
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
    out_new_dataset: *mut *mut c_void,
    out_table_uri: *mut *const c_char,
) -> i32 {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }
    match namespace_checkout_latest_if_stale_inner(
        dataset,
        endpoint,
        table_id,
        bearer_token,
        api_key,
        delimiter,
        headers_tsv,
        session,
        out_new_dataset,
        out_table_uri,
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

#[allow(clippy::too_many_arguments)]
fn namespace_checkout_latest_if_stale_inner(
    dataset: *mut c_void,
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
    session: *mut c_void,
    out_new_dataset: *mut *mut c_void,
    out_table_uri: *mut *const c_char,
) -> FfiResult<()> {
    if out_new_dataset.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_new_dataset is null",
        ));
    }

    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };
    let headers_tsv = unsafe { optional_cstr_to_string(headers_tsv, "headers_tsv")? };
    let session = unsafe { optional_session_handle(session)? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(
        endpoint,
        bearer_token.as_deref(),
        api_key.as_deref(),
        headers_tsv.as_deref(),
    )
    .delimiter(delimiter)
    .build();

    let (refreshed, moved_uri) = runtime::block_on(async move {
        // Re-resolve the table location through the namespace before trusting
        // the cached handle: an external drop/re-create can re-point the table
        // to a different physical URI.
        record_namespace_describe();
        let request = DescribeTableRequest {
            id: Some(vec![table_id.to_string()]),
            // Mirror the other describe paths: request the complete table URI
            // so namespaces that only report `table_uri` (the response model
            // allows omitting `location`) can still be revalidated.
            with_table_uri: Some(true),
            ..Default::default()
        };
        let response = namespace.describe_table(request).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table: {err}"),
            )
        })?;
        let location = response.location;
        let table_uri = response.table_uri;
        if location.is_none() && table_uri.is_none() {
            return Err(FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "table location not found in namespace response",
            ));
        }

        // Treat the cached handle as current when either reported form
        // matches its URI: `DatasetBuilder::from_namespace` derives
        // `dataset.uri()` from `location`, so requiring the preferred
        // `table_uri` form to match would flag a perpetual (false) move on
        // servers that report both fields in different spellings.
        let cached_uri = handle.dataset.uri();
        let location_matches =
            location.as_deref() == Some(cached_uri) || table_uri.as_deref() == Some(cached_uri);

        // The namespace-vended storage options must match too: a namespace
        // can re-describe the same physical URI with different options (e.g.
        // rotated object-store credentials), and the cached handle's object
        // store would keep using the stale ones. Namespaces that vend
        // per-describe temporary credentials will reopen on every hit — the
        // cost of always reading with the freshest credentials.
        let current_storage_options = response.storage_options;
        let storage_options_match = handle.namespace_storage_options == current_storage_options;

        if !location_matches || !storage_options_match {
            // The table was re-pointed to a new location or its storage
            // options changed: reopen through the namespace path so managed
            // versioning and namespace-provided storage options are
            // re-applied (this re-describes internally; the extra round trip
            // only happens on this path).
            let mut builder =
                DatasetBuilder::from_namespace(Arc::new(namespace), vec![table_id.to_string()])
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
            let reopened = builder.load().await.map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetOpen,
                    format!("namespace dataset open: {err}"),
                )
            })?;
            record_dataset_open();
            let uri = reopened.uri().to_string();
            let moved = !location_matches;
            return Ok::<_, FfiError>((
                Some((reopened, current_storage_options)),
                moved.then_some(uri),
            ));
        }

        // Same location and options: fall back to the manifest-identity
        // revalidation.
        let refreshed = super::dataset::refresh_dataset_if_stale(&handle.dataset)
            .await
            .map_err(|message| {
                FfiError::new(
                    ErrorCode::DatasetCheckoutLatest,
                    format!("dataset checkout latest if stale: {message}"),
                )
            })?;
        Ok((
            refreshed.map(|dataset| (dataset, current_storage_options)),
            None,
        ))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    let new_handle = match refreshed {
        Some((refreshed_dataset, storage_options)) => Box::into_raw(Box::new(
            DatasetHandle::new(Arc::new(refreshed_dataset))
                .with_namespace_storage_options(storage_options),
        )) as *mut c_void,
        None => ptr::null_mut(),
    };

    // SAFETY: `out_new_dataset` was null-checked above and is provided by the
    // caller as a valid output location.
    unsafe {
        std::ptr::write_unaligned(out_new_dataset, new_handle);
    }
    if let (Some(uri), false) = (moved_uri, out_table_uri.is_null()) {
        let uri_c = CString::new(uri).unwrap_or_else(|_| to_c_string("invalid uri"));
        // SAFETY: `out_table_uri` was null-checked in the tuple condition.
        unsafe {
            std::ptr::write_unaligned(out_table_uri, uri_c.into_raw() as *const c_char);
        }
    }
    Ok(())
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
    use std::ffi::CString;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema as ArrowSchema};
    use lance::dataset::{WriteMode, WriteParams};
    use lance::Dataset;

    use super::super::dataset::{lance_close_dataset, lance_dataset_count_rows};
    use super::*;
    use crate::runtime;

    /// Commit a batch to `uri` through the Lance Rust API, bypassing any FFI
    /// dataset handle (i.e. acting as an external writer).
    fn external_write(uri: &str, ids: Vec<i32>, mode: WriteMode) {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(ids))]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);
        let params = WriteParams {
            mode,
            ..Default::default()
        };
        runtime::block_on(Dataset::write(reader, uri, Some(params)))
            .unwrap()
            .unwrap();
    }

    /// Minimal REST namespace mock: answers every request with the
    /// `describe_table`-shaped JSON body currently stored in `body`, so tests
    /// can switch between location-only, table_uri-only, and re-pointed
    /// responses (emulating an external drop/re-create that moves the table).
    fn spawn_describe_server(body: Arc<Mutex<String>>) -> (String, Arc<Mutex<bool>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let stop = Arc::new(Mutex::new(false));
        let stop_flag = stop.clone();
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

                let body = body.lock().unwrap().clone();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (endpoint, stop)
    }

    #[test]
    fn test_namespace_checkout_latest_if_stale_rejects_invalid_arguments() {
        unsafe {
            let endpoint = CString::new("http://127.0.0.1:1").unwrap();
            let table_id = CString::new("t").unwrap();
            let mut refreshed: *mut c_void = ptr::null_mut();
            // Null output pointer.
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    ptr::null_mut(),
                    endpoint.as_ptr(),
                    table_id.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                -1
            );
            // Null dataset handle.
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    ptr::null_mut(),
                    endpoint.as_ptr(),
                    table_id.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    ptr::null_mut(),
                ),
                -1
            );
        }
    }

    #[test]
    fn test_namespace_checkout_latest_if_stale_reopens_on_storage_option_change() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();

        let base = std::env::temp_dir().join(format!("ffi-ns-opts-{}", rand::random::<u64>()));
        let table = base.join("t.lance");
        let uri = table.to_string_lossy().to_string();
        external_write(&uri, vec![1, 2, 3], WriteMode::Create);

        let body = Arc::new(Mutex::new(format!("{{\"location\": \"{uri}\"}}")));
        let (endpoint, stop) = spawn_describe_server(body.clone());

        unsafe {
            let endpoint_c = CString::new(endpoint).unwrap();
            let table_id_c = CString::new("t").unwrap();

            let mut opened_uri: *const c_char = ptr::null();
            let handle = lance_open_dataset_in_namespace(
                endpoint_c.as_ptr(),
                table_id_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut opened_uri,
            );
            assert!(!handle.is_null());
            if !opened_uri.is_null() {
                crate::error::lance_free_string(opened_uri);
            }
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // Same location and options: current.
            let mut refreshed: *mut c_void = ptr::null_mut();
            let mut moved_uri: *const c_char = ptr::null();
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    handle,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    &mut moved_uri,
                ),
                0
            );
            assert!(refreshed.is_null());
            assert!(moved_uri.is_null());

            // The namespace now vends different storage options for the same
            // location (e.g. rotated credentials): the handle must be
            // reopened even though the URI is unchanged.
            *body.lock().unwrap() = format!(
                "{{\"location\": \"{uri}\", \"storage_options\": {{\"test_option\": \"v1\"}}}}"
            );
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    handle,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    &mut moved_uri,
                ),
                0
            );
            assert!(!refreshed.is_null());
            // Location unchanged: the display URI must not be reported as moved.
            assert!(moved_uri.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed), 3);
            let reopened_v1 = refreshed;

            // Unchanged options on the reopened handle: current again.
            let mut refreshed_again: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    reopened_v1,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed_again,
                    &mut moved_uri,
                ),
                0
            );
            assert!(refreshed_again.is_null());

            // Another rotation: reopen again.
            *body.lock().unwrap() = format!(
                "{{\"location\": \"{uri}\", \"storage_options\": {{\"test_option\": \"v2\"}}}}"
            );
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    reopened_v1,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed_again,
                    &mut moved_uri,
                ),
                0
            );
            assert!(!refreshed_again.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed_again), 3);

            lance_close_dataset(refreshed_again);
            lance_close_dataset(reopened_v1);
            lance_close_dataset(handle);
        }

        *stop.lock().unwrap() = true;
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn test_namespace_checkout_latest_if_stale_follows_moved_table() {
        // Opening datasets mutates the process-global debug counters, which
        // other tests assert on; serialize with them.
        let _counter_guard = crate::ffi::session::debug_counter_test_lock();

        let base =
            std::env::temp_dir().join(format!("ffi-ns-revalidate-{}", rand::random::<u64>()));
        let table_a = base.join("a.lance");
        let table_b = base.join("b.lance");
        let uri_a = table_a.to_string_lossy().to_string();
        let uri_b = table_b.to_string_lossy().to_string();
        external_write(&uri_a, vec![1, 2, 3], WriteMode::Create);
        external_write(&uri_b, vec![10, 20], WriteMode::Create);

        let body = Arc::new(Mutex::new(format!("{{\"location\": \"{uri_a}\"}}")));
        let (endpoint, stop) = spawn_describe_server(body.clone());

        unsafe {
            let endpoint_c = CString::new(endpoint).unwrap();
            let table_id_c = CString::new("t").unwrap();

            let mut opened_uri: *const c_char = ptr::null();
            let handle = lance_open_dataset_in_namespace(
                endpoint_c.as_ptr(),
                table_id_c.as_ptr(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                &mut opened_uri,
            );
            assert!(!handle.is_null());
            if !opened_uri.is_null() {
                crate::error::lance_free_string(opened_uri);
            }
            assert_eq!(lance_dataset_count_rows(handle), 3);

            // Same location, no external commit: the cached handle is current.
            let mut refreshed: *mut c_void = ptr::null_mut();
            let mut moved_uri: *const c_char = ptr::null();
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    handle,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    &mut moved_uri,
                ),
                0
            );
            assert!(refreshed.is_null());
            assert!(moved_uri.is_null());

            // A namespace that reports only `table_uri` (no `location`) must
            // also revalidate cleanly instead of failing with "table location
            // not found".
            *body.lock().unwrap() = format!("{{\"table_uri\": \"{uri_a}\"}}");
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    handle,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    &mut moved_uri,
                ),
                0
            );
            assert!(refreshed.is_null());
            assert!(moved_uri.is_null());
            *body.lock().unwrap() = format!("{{\"location\": \"{uri_a}\"}}");

            // Same location, external append: the manifest-identity fallback
            // must produce a refreshed handle.
            external_write(&uri_a, vec![4, 5], WriteMode::Append);
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    handle,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut refreshed,
                    &mut moved_uri,
                ),
                0
            );
            assert!(!refreshed.is_null());
            assert!(moved_uri.is_null());
            assert_eq!(lance_dataset_count_rows(refreshed), 5);
            let refreshed_same_location = refreshed;

            // The namespace re-points the table to a new location (external
            // drop/re-create): revalidation must reopen through the namespace
            // and observe the new table, not checkout the old URI.
            *body.lock().unwrap() = format!("{{\"location\": \"{uri_b}\"}}");
            let mut moved: *mut c_void = ptr::null_mut();
            assert_eq!(
                lance_dataset_namespace_checkout_latest_if_stale(
                    refreshed_same_location,
                    endpoint_c.as_ptr(),
                    table_id_c.as_ptr(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null_mut(),
                    &mut moved,
                    &mut moved_uri,
                ),
                0
            );
            assert!(!moved.is_null());
            assert_eq!(lance_dataset_count_rows(moved), 2);
            assert!(!moved_uri.is_null());
            let moved_uri_str = std::ffi::CStr::from_ptr(moved_uri)
                .to_string_lossy()
                .to_string();
            assert!(moved_uri_str.contains("b.lance"), "uri: {moved_uri_str}");
            crate::error::lance_free_string(moved_uri);

            lance_close_dataset(moved);
            lance_close_dataset(refreshed_same_location);
            lance_close_dataset(handle);
        }

        *stop.lock().unwrap() = true;
        let _ = std::fs::remove_dir_all(base);
    }
}
