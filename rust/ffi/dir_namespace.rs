use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;
use std::sync::{Arc, LazyLock};

use lance::dataset::builder::DatasetBuilder;
use lance_core::Error as LanceError;
use lance_io::object_store::{uri_to_url, ObjectStoreRegistry};
use lance_namespace::models::{DropTableRequest, ListTablesRequest};
use lance_namespace::LanceNamespace;
use lance_namespace_impls::DirectoryNamespaceBuilder;

use crate::error::{clear_last_error, namespace_error_code, set_last_error, ErrorCode};
use crate::runtime;

use super::session::record_dataset_open;
use super::source_resolver::activate_source_resolution_for_session;
use super::types::DatasetHandle;
use super::util::{
    cstr_to_str, export_string_list, optional_session_handle, slice_from_ptr, to_c_string,
    FfiError, FfiResult, LanceStringList,
};

// Use the same locator for statement identity and the subsequent namespace open.
// Lance resolves local dot components before the filesystem follows symlinks.
fn directory_snapshot_root(root: &str, resolve_links: bool) -> String {
    let trimmed = root.trim_end_matches('/');
    if trimmed.is_empty() {
        return if root.is_empty() { "" } else { "/." }.to_owned();
    }
    let path = uri_to_url(trimmed)
        .ok()
        .filter(|url| url.scheme() == "file")
        .and_then(|url| url.to_file_path().ok())
        .map(|path| {
            if resolve_links {
                path.canonicalize().unwrap_or(path)
            } else {
                path.components().collect()
            }
        });
    match path.as_deref().and_then(std::path::Path::to_str) {
        // The builder trims trailing '/' even for a filesystem root.
        Some("/") => "/.".to_owned(),
        Some(path) => path.to_owned(),
        // Let the provider resolve its own keys and report missing paths.
        None => trimmed.to_owned(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_directory_snapshot_root(
    root: *const c_char,
    resolve_links: bool,
) -> *const c_char {
    match unsafe { cstr_to_str(root, "root") } {
        Ok(root) => {
            clear_last_error();
            to_c_string(directory_snapshot_root(root, resolve_links)).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

// The namespace appends the table name before Lance parses the URL. Keep the
// locator used to open it separate from the key used to pin it in a statement.
fn directory_snapshot_identity(
    root: &str,
    name: &str,
    options: &HashMap<String, String>,
) -> String {
    let Ok(url) = url::Url::parse(&format!("{root}/{name}.lance")) else {
        return root.to_owned();
    };
    static REGISTRY: LazyLock<ObjectStoreRegistry> = LazyLock::new(ObjectStoreRegistry::default);
    let Some(provider) = REGISTRY.get_provider(url.scheme()) else {
        return url.to_string();
    };
    let (Ok(prefix), Ok(path)) = (
        provider.calculate_object_store_prefix(&url, Some(options)),
        provider.extract_path(&url),
    ) else {
        // Leave errors to the namespace open. An invalid spelling cannot hit
        // the key of a successfully resolved provider path.
        return url.to_string();
    };
    let route = match url.scheme() {
        "s3" | "s3+ddb" | "oss" | "memory" | "shared-memory" => "",
        // These providers set the OpenDAL operator root to the raw URL path.
        "gs" | "az" | "abfss"
            if !options
                .get("use_opendal")
                .is_some_and(|v| lance_core::utils::parse::str_is_truthy(v)) =>
        {
            ""
        }
        // The provider prefix omits the revision. Retain its raw owner/repo
        // routing components while using the extracted path inside the repo.
        "hf" => {
            let path = url.path().trim_start_matches('/');
            path.match_indices('/')
                .nth(1)
                .map_or(path, |(end, _)| &path[..end])
        }
        _ => url.path(),
    };
    serde_json::json!([
        url.scheme(),
        url.authority(),
        prefix,
        route,
        path.as_ref(),
        url.query(),
        url.fragment()
    ])
    .to_string()
}

#[no_mangle]
pub unsafe extern "C" fn lance_directory_snapshot_identity(
    root: *const c_char,
    name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
) -> *const c_char {
    let identity = || -> FfiResult<String> {
        let root = unsafe { cstr_to_str(root, "root")? };
        let name = unsafe { cstr_to_str(name, "name")? };
        let options = parse_storage_options(option_keys, option_values, options_len)?;
        Ok(directory_snapshot_identity(root, name, &options))
    };
    match identity() {
        Ok(identity) => {
            clear_last_error();
            to_c_string(identity).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null()
        }
    }
}

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
    out: *mut LanceStringList,
) -> i32 {
    unsafe {
        export_string_list(
            dir_namespace_list_tables_inner(root, option_keys, option_values, options_len),
            out,
        )
    }
}

fn open_dataset_in_dir_namespace_inner(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    version: u64,
) -> FfiResult<(DatasetHandle, String)> {
    let session_ptr = session;
    let root = unsafe { cstr_to_str(root, "root")? }
        .trim_end_matches('/')
        .to_string();
    let table_name = unsafe { cstr_to_str(table_name, "table_name")? };
    let storage_options = parse_storage_options(option_keys, option_values, options_len)?;
    let session = unsafe { optional_session_handle(session)? };

    let dataset = runtime::block_on(async {
        let mut ns_builder = DirectoryNamespaceBuilder::new(&root).manifest_enabled(false);
        if !storage_options.is_empty() {
            ns_builder = ns_builder.storage_options(storage_options.clone());
        }
        let namespace = ns_builder.build().await.map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetOpen,
                format!("dir namespace build '{root}': {err}"),
            )
        })?;
        let mut builder =
            DatasetBuilder::from_namespace(Arc::new(namespace), vec![table_name.to_string()])
                .await
                .map_err(|err| {
                    FfiError::new(
                        namespace_error_code(&err, ErrorCode::DatasetOpen),
                        format!("dir namespace describe '{root}/{table_name}': {err}"),
                    )
                })?;
        // Forward the caller's storage_options to the dataset open path.
        //
        // `DirectoryNamespace::describe_table` returns `storage_options: None`
        // when no credential vendor is configured, by design — to avoid
        // leaking the namespace's own static credentials to clients (see
        // lance-namespace-impls/src/dir.rs::get_storage_options_for_table).
        // Without this re-injection, `builder.load()` builds an ObjectStore
        // with empty options. For non-AWS S3-compatible endpoints (TOS,
        // MinIO, R2, ...) this falls into AWS bucket-region resolution
        // (lance-io/src/object_store/providers/aws.rs:217) and fails with
        // "Bucket '<name>' not found".
        //
        // `with_storage_options` merges into any accessor already attached
        // by `from_namespace`, preserving a `StorageOptionsProvider` if one
        // was returned by a credential vendor.
        if !storage_options.is_empty() {
            builder = builder.with_storage_options(storage_options);
        }
        if let Some(session) = session {
            builder = builder.with_session(session);
        }
        if version != 0 {
            builder = builder.with_version(version);
        }
        builder.load().await.map_err(|err| {
            FfiError::new(
                namespace_error_code(&err, ErrorCode::DatasetOpen),
                format!("dir namespace dataset open: {err}"),
            )
        })
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))??;

    let table_uri = dataset.uri().to_string();
    unsafe {
        activate_source_resolution_for_session(session_ptr, &dataset);
    }
    record_dataset_open();
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
    unsafe {
        lance_open_dataset_in_dir_namespace_with_session(
            root,
            table_name,
            option_keys,
            option_values,
            options_len,
            ptr::null_mut(),
            0,
            out_table_uri,
        )
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset_in_dir_namespace_with_session(
    root: *const c_char,
    table_name: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
    version: u64,
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
        session,
        version,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_identity_uses_provider_paths_and_keeps_routing_information() {
        let options = HashMap::new();
        let key = |root: &str| directory_snapshot_identity(root, "t", &options);
        assert_eq!(
            key("shared-memory://a/root"),
            key("shared-memory://a/x/../root/.")
        );
        assert_eq!(
            key("shared-memory://a/x//root"),
            key("shared-memory://a/x/root")
        );
        assert_ne!(key("shared-memory://a/root"), key("shared-memory://b/root"));
        assert_ne!(key("shared-memory://a/x%2Fy"), key("shared-memory://a/x/y"));
        assert_eq!(key("s3://bucket/x%2Fy"), key("s3://bucket/x/y"));
        assert_ne!(key("s3://bucket/x%252Fy"), key("s3://bucket/x/y"));
        assert_ne!(key("s3://a/root"), key("s3://b/root"));
        assert_ne!(
            key("s3+ddb://a/root?ddbTableName=x"),
            key("s3+ddb://a/root?ddbTableName=y")
        );
        assert_ne!(
            key("hf://datasets/a/repo@main/root"),
            key("hf://datasets/a/repo@dev/root")
        );
        let opendal = HashMap::from([("use_opendal".to_owned(), "true".to_owned())]);
        for bucket in ["gs://bucket", "abfss://bucket@account.dfs.core.windows.net"] {
            let encoded = format!("{bucket}/x%2Fy");
            let nested = format!("{bucket}/x/y");
            assert_eq!(key(&encoded), key(&nested));
            assert_ne!(
                directory_snapshot_identity(&encoded, "t", &opendal),
                directory_snapshot_identity(&nested, "t", &opendal)
            );
        }
    }

    #[test]
    fn snapshot_roots_follow_native_local_path_semantics() {
        let root = std::env::current_dir().unwrap();
        let uri = url::Url::from_directory_path(&root).unwrap();
        for resolve_links in [false, true] {
            let normalize = |root| directory_snapshot_root(root, resolve_links);
            let expected = normalize(root.to_str().unwrap());
            assert_eq!(normalize("./test/../"), expected);
            assert_eq!(normalize(uri.as_str()), expected);
            assert_eq!(normalize("///"), "/.");
            assert_eq!(normalize(""), "");
            assert_eq!(
                normalize("shared-memory://s/a%2Fb///"),
                "shared-memory://s/a%2Fb"
            );
        }
    }
}
