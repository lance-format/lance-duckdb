use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::Arc;

use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;
use lance_namespace::models::{DropTableRequest, ListTablesRequest};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::DirectoryNamespaceBuilder;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::DatasetHandle;
use super::util::{cstr_to_str, slice_from_ptr, to_c_string, FfiError, FfiResult};

fn parse_storage_options(
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<HashMap<String, String>> {
    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }
    Ok(storage_options)
}

fn dir_namespace_list_tables_inner(
    root: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<Vec<String>> {
    let root = unsafe { cstr_to_str(root, "root")? };
    let storage_options = parse_storage_options(option_keys, option_values, options_len)?;

    let tables = runtime::block_on(async move {
        let mut builder = DirectoryNamespaceBuilder::new(root).manifest_enabled(false);
        if !storage_options.is_empty() {
            builder = builder.storage_options(storage_options);
        }
        let namespace = builder.build().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DirNamespaceListTables,
                format!("dir namespace build '{root}': {err}"),
            )
        })?;

        let mut req = ListTablesRequest::new();
        req.id = Some(Vec::new());
        let resp = namespace.list_tables(req).await.map_err(|err| {
            FfiError::new(
                ErrorCode::DirNamespaceListTables,
                format!("dir namespace list_tables '{root}': {err}"),
            )
        })?;
        Ok::<_, FfiError>(resp.tables)
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok(tables)
}

#[no_mangle]
pub unsafe extern "C" fn lance_dir_namespace_list_tables(
    root: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> *const c_char {
    match dir_namespace_list_tables_inner(root, option_keys, option_values, options_len) {
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

fn open_dataset_in_dir_namespace_inner(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<(DatasetHandle, String)> {
    let root = unsafe { cstr_to_str(root, "root")? }
        .trim_end_matches('/')
        .to_string();
    let table_name = unsafe { cstr_to_str(table_name, "table_name")? };
    let storage_options = parse_storage_options(option_keys, option_values, options_len)?;

    let table_uri = format!("{root}/{table_name}.lance");
    let dataset = runtime::block_on(async {
        let mut builder = DatasetBuilder::from_uri(&table_uri);
        if !storage_options.is_empty() {
            builder = builder.with_storage_options(storage_options);
        }
        builder.load().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetOpen,
                format!("dataset open '{table_uri}': {err}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    Ok((DatasetHandle::new(Arc::new(dataset)), table_uri))
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_dir_namespace(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    out_table_uri: *mut *const c_char,
) -> *mut c_void {
    if !out_table_uri.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_table_uri, ptr::null());
        }
    }

    match open_dataset_in_dir_namespace_inner(
        root,
        table_name,
        option_keys,
        option_values,
        options_len,
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

fn dir_namespace_drop_table_inner(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> FfiResult<()> {
    let root = unsafe { cstr_to_str(root, "root")? };
    let table_name = unsafe { cstr_to_str(table_name, "table_name")? };
    let storage_options = parse_storage_options(option_keys, option_values, options_len)?;

    runtime::block_on(async move {
        let mut builder = DirectoryNamespaceBuilder::new(root).manifest_enabled(false);
        if !storage_options.is_empty() {
            builder = builder.storage_options(storage_options);
        }
        let namespace = builder.build().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DirNamespaceDropTable,
                format!("dir namespace build '{root}': {err}"),
            )
        })?;

        let mut req = DropTableRequest::new();
        req.id = Some(vec![table_name.to_string()]);

        match namespace.drop_table(req).await {
            Ok(_) => Ok(()),
            Err(LanceError::NotFound { .. }) => Ok(()),
            Err(err) => Err(FfiError::new(
                ErrorCode::DirNamespaceDropTable,
                format!("dir namespace drop_table '{root}/{table_name}': {err}"),
            )),
        }
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?
}

#[no_mangle]
pub unsafe extern "C" fn lance_dir_namespace_drop_table(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> i32 {
    match dir_namespace_drop_table_inner(root, table_name, option_keys, option_values, options_len)
    {
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
