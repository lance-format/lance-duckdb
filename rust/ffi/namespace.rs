use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Arc;

use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;

use lance_namespace::models::{
    CreateEmptyTableRequest, DescribeTableRequest, DropTableRequest, ListTablesRequest,
};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::RestNamespaceBuilder;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::DatasetHandle;
use super::util::{cstr_to_str, to_c_string, FfiError, FfiResult};

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
    match list_tables_inner(endpoint, namespace_id, bearer_token, api_key, delimiter, headers_tsv) {
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
        let mut req = DescribeTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                format!("namespace describe_table: {err}"),
            )
        })?;
        let location = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTableInfo,
                "namespace describe_table: missing location",
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

    match describe_table_info_inner(endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv) {
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
        let mut req = CreateEmptyTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        let resp = namespace.create_empty_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                format!("namespace create_empty_table: {err}"),
            )
        })?;
        let location = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceCreateEmptyTable,
                "namespace create_empty_table: missing location",
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

    match create_empty_table_inner(endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv) {
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
    match drop_table_inner(endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv) {
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

fn open_dataset_in_namespace_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
    headers_tsv: *const c_char,
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

    let (dataset, table_uri) = runtime::block_on(async move {
        let mut req = DescribeTableRequest::new();
        req.id = Some(vec![table_id.to_string()]);
        let resp = namespace.describe_table(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                format!("namespace describe_table: {err}"),
            )
        })?;

        let table_uri = resp.location.ok_or_else(|| {
            FfiError::new(
                ErrorCode::NamespaceDescribeTable,
                "namespace describe_table: missing location",
            )
        })?;
        let storage_options = resp.storage_options.unwrap_or_default();

        let dataset = DatasetBuilder::from_uri(&table_uri)
            .with_storage_options(storage_options)
            .load()
            .await
            .map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetOpen,
                    format!("dataset open '{table_uri}': {err}"),
                )
            })?;
        Ok::<_, FfiError>((Arc::new(dataset), table_uri))
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

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

    match open_dataset_in_namespace_inner(endpoint, table_id, bearer_token, api_key, delimiter, headers_tsv) {
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
