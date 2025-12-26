use std::ffi::{c_char, c_void, CString};
use std::ptr;
use std::sync::Arc;

use lance::dataset::builder::DatasetBuilder;

use lance_namespace::models::{DescribeTableRequest, ListTablesRequest};
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

fn build_config(
    endpoint: &str,
    bearer_token: Option<&str>,
    api_key: Option<&str>,
) -> RestNamespaceBuilder {
    let mut builder = RestNamespaceBuilder::new(endpoint);
    if let Some(token) = bearer_token {
        builder = builder.header("Authorization", format!("Bearer {token}"));
    }
    if let Some(key) = api_key {
        builder = builder.header("x-api-key", key.to_string());
    }
    builder
}

fn list_tables_inner(
    endpoint: *const c_char,
    namespace_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
) -> FfiResult<Vec<String>> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let namespace_id = unsafe { cstr_to_str(namespace_id, "namespace_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(endpoint, bearer_token.as_deref(), api_key.as_deref())
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
) -> *const c_char {
    match list_tables_inner(endpoint, namespace_id, bearer_token, api_key, delimiter) {
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

fn open_dataset_in_namespace_inner(
    endpoint: *const c_char,
    table_id: *const c_char,
    bearer_token: *const c_char,
    api_key: *const c_char,
    delimiter: *const c_char,
) -> FfiResult<(DatasetHandle, String)> {
    let endpoint = unsafe { cstr_to_str(endpoint, "endpoint")? };
    let table_id = unsafe { cstr_to_str(table_id, "table_id")? };
    let delimiter = unsafe { optional_cstr_to_string(delimiter, "delimiter")? };
    let bearer_token = unsafe { optional_cstr_to_string(bearer_token, "bearer_token")? };
    let api_key = unsafe { optional_cstr_to_string(api_key, "api_key")? };

    let delimiter = delimiter.unwrap_or_else(|| "$".to_string());
    let namespace = build_config(endpoint, bearer_token.as_deref(), api_key.as_deref())
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
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_namespace_inner(endpoint, table_id, bearer_token, api_key, delimiter) {
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
