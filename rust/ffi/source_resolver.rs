// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-dataset source storage resolution through DuckDB secrets.
//!
//! A dataset opened by the extension receives a derived Lance session. It shares
//! the database-wide index and metadata cache backends, but has an isolated
//! object-store registry. The registry wraps Lance's public providers so a Blob
//! external URI can resolve its own DuckDB secret only after the dataset's
//! primary store has opened. This covers absolute references stored with
//! `base_id=0` as well as registered external bases.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use lance::dataset::Dataset;
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance_core::Error;
use lance_io::object_store::ObjectStoreProvider;
use url::Url;

use super::types::SessionHandle;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SourceResolverCallbacks {
    pub resolver: *mut c_void,
    pub resolve_source: Option<
        unsafe extern "C" fn(
            resolver: *mut c_void,
            uri: *const c_char,
            out_keys: *mut *const *const c_char,
            out_values: *mut *const *const c_char,
            out_count: *mut usize,
            out_error: *mut *const c_char,
        ) -> i32,
    >,
    pub release_source: Option<unsafe extern "C" fn(resolver: *mut c_void)>,
    pub destroy: Option<unsafe extern "C" fn(resolver: *mut c_void)>,
}

#[derive(Debug)]
struct SourceResolver {
    callbacks: SourceResolverCallbacks,
}

impl SourceResolver {
    fn new(callbacks: SourceResolverCallbacks) -> Result<Self, String> {
        if callbacks.resolver.is_null()
            || callbacks.resolve_source.is_none()
            || callbacks.release_source.is_none()
            || callbacks.destroy.is_none()
        {
            return Err("DuckDB source resolver callback is null".to_string());
        }
        Ok(Self { callbacks })
    }

    fn resolve(&self, uri: &str) -> Result<HashMap<String, String>, Error> {
        let uri = std::ffi::CString::new(uri)
            .map_err(|_| Error::invalid_input("source URI contains a NUL byte"))?;
        let mut keys = std::ptr::null();
        let mut values = std::ptr::null();
        let mut count = 0usize;
        let mut error = std::ptr::null();
        let callback = self
            .callbacks
            .resolve_source
            .expect("callback checked during construction");
        let status = unsafe {
            callback(
                self.callbacks.resolver,
                uri.as_ptr(),
                &mut keys,
                &mut values,
                &mut count,
                &mut error,
            )
        };
        let result = (|| {
            if status != 0 {
                let message = if error.is_null() {
                    "DuckDB source resolution failed".to_string()
                } else {
                    unsafe { CStr::from_ptr(error) }
                        .to_string_lossy()
                        .into_owned()
                };
                return Err(Error::io(message));
            }
            if count == 0 {
                return Ok(HashMap::new());
            }
            if keys.is_null() || values.is_null() {
                return Err(Error::io(
                    "DuckDB source resolver returned null option arrays",
                ));
            }
            let keys = unsafe { std::slice::from_raw_parts(keys, count) };
            let values = unsafe { std::slice::from_raw_parts(values, count) };
            let mut options = HashMap::with_capacity(count);
            for idx in 0..count {
                if keys[idx].is_null() || values[idx].is_null() {
                    return Err(Error::io(format!(
                        "DuckDB source resolver returned null option at index {idx}"
                    )));
                }
                let key = unsafe { CStr::from_ptr(keys[idx]) }
                    .to_string_lossy()
                    .into_owned();
                let value = unsafe { CStr::from_ptr(values[idx]) }
                    .to_string_lossy()
                    .into_owned();
                options.insert(key, value);
            }
            Ok(options)
        })();
        unsafe {
            self.callbacks
                .release_source
                .expect("callback checked during construction")(self.callbacks.resolver)
        };
        result
    }
}

impl Drop for SourceResolver {
    fn drop(&mut self) {
        if let Some(destroy) = self.callbacks.destroy {
            unsafe { destroy(self.callbacks.resolver) };
        }
    }
}

// C++ holds only a weak ClientContext and uses DuckDB's native, internally
// locked SecretManager lookup plus per-thread callback storage.
unsafe impl Send for SourceResolver {}
unsafe impl Sync for SourceResolver {}

fn options_fingerprint(options: &HashMap<String, String>) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut entries = options.iter().collect::<Vec<_>>();
    entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
    for (key, value) in entries {
        key.hash(&mut hasher);
        value.hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Debug)]
pub(crate) struct SourceResolutionState {
    resolver: SourceResolver,
    target_uri: RwLock<Option<String>>,
    target_storage_options: RwLock<HashMap<String, String>>,
}

impl SourceResolutionState {
    fn new(resolver: SourceResolver) -> Self {
        Self {
            resolver,
            target_uri: RwLock::new(None),
            target_storage_options: RwLock::new(HashMap::new()),
        }
    }

    pub(crate) fn activate(&self, dataset: &Dataset) {
        *self.target_uri.write().unwrap() = Some(normalize_target_uri(dataset.uri()));
        *self.target_storage_options.write().unwrap() = dataset
            .store_params()
            .and_then(ObjectStoreParams::storage_options)
            .cloned()
            .unwrap_or_default();
    }

    fn should_resolve_source(
        &self,
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> bool {
        let target_uri = self.target_uri.read().unwrap();
        let Some(target_uri) = target_uri.as_deref() else {
            return false;
        };
        if is_under_target(uri, target_uri) {
            return false;
        }
        // Exact target options identify the dataset-level fallback used for
        // absolute `base_id=0` references. Registered bases with explicit
        // runtime params keep their own store configuration.
        let target_storage_options = self.target_storage_options.read().unwrap();
        storage_options
            .map(|options| options == &*target_storage_options)
            .unwrap_or_else(|| target_storage_options.is_empty())
    }

    fn source_options(
        &self,
        uri: &str,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Result<Option<HashMap<String, String>>, Error> {
        if !self.should_resolve_source(uri, storage_options) {
            return Ok(None);
        }
        self.resolver.resolve(uri).map(Some)
    }
}

fn normalize_target_uri(uri: &str) -> String {
    uri.trim_end_matches('/').to_string()
}

fn is_under_target(uri: &str, target_uri: &str) -> bool {
    let uri = uri.trim_end_matches('/');
    uri == target_uri
        || uri
            .strip_prefix(target_uri)
            .is_some_and(|suffix| suffix.starts_with('/'))
}

#[derive(Debug)]
struct SourceResolvingProvider {
    inner: Arc<dyn ObjectStoreProvider>,
    state: Arc<SourceResolutionState>,
}

fn apply_source_options(
    params: &ObjectStoreParams,
    options: HashMap<String, String>,
) -> ObjectStoreParams {
    let mut effective = params.clone();
    effective.storage_options_accessor = Some(Arc::new(
        StorageOptionsAccessor::with_static_options(options),
    ));
    effective
}

#[async_trait]
impl ObjectStoreProvider for SourceResolvingProvider {
    async fn new_store(
        &self,
        base_path: Url,
        params: &ObjectStoreParams,
    ) -> lance_core::Result<ObjectStore> {
        let uri = base_path.as_str();
        let Some(source_options) = self.state.source_options(uri, params.storage_options())? else {
            return self.inner.new_store(base_path, params).await;
        };
        let effective = apply_source_options(params, source_options);
        self.inner.new_store(base_path, &effective).await
    }

    fn extract_path(&self, url: &Url) -> lance_core::Result<object_store::path::Path> {
        self.inner.extract_path(url)
    }

    fn calculate_object_store_prefix(
        &self,
        url: &Url,
        storage_options: Option<&HashMap<String, String>>,
    ) -> lance_core::Result<String> {
        let uri = url.as_str();
        let Some(source_options) = self.state.source_options(uri, storage_options)? else {
            return self
                .inner
                .calculate_object_store_prefix(url, storage_options);
        };
        let prefix = self
            .inner
            .calculate_object_store_prefix(url, Some(&source_options))?;
        Ok(format!(
            "{prefix}$duckdb-source-{:016x}",
            options_fingerprint(&source_options)
        ))
    }
}

const PROVIDER_SCHEMES: &[&str] = &[
    "memory",
    "shared-memory",
    "file",
    "file-object-store",
    "file+uring",
    "s3",
    "s3+ddb",
    "az",
    "abfss",
    "gs",
    "goosefs",
    "oss",
    "cos",
    "tos",
    "hf",
];

fn build_source_registry(state: Arc<SourceResolutionState>) -> Arc<ObjectStoreRegistry> {
    let registry = Arc::new(ObjectStoreRegistry::default());
    for scheme in PROVIDER_SCHEMES {
        if let Some(inner) = registry.get_provider(scheme) {
            registry.insert(
                scheme,
                Arc::new(SourceResolvingProvider {
                    inner,
                    state: state.clone(),
                }),
            );
        }
    }
    registry
}

pub(crate) fn create_source_session(
    parent: &SessionHandle,
    callbacks: SourceResolverCallbacks,
) -> Result<SessionHandle, String> {
    let resolver = SourceResolver::new(callbacks)?;
    let state = Arc::new(SourceResolutionState::new(resolver));
    let registry = build_source_registry(state.clone());
    let session = Arc::new(lance::session::Session::with_cache_backends(
        lance::session::CacheSpec::Backend(parent.index_cache.clone()),
        lance::session::CacheSpec::Backend(parent.metadata_cache.clone()),
        registry,
    ));
    Ok(SessionHandle {
        session,
        index_cache: parent.index_cache.clone(),
        metadata_cache: parent.metadata_cache.clone(),
        source_resolution: Some(state),
        clear_caches_on_close: false,
    })
}

pub(crate) fn activate_source_resolution(session: &SessionHandle, dataset: &Dataset) {
    if let Some(state) = &session.source_resolution {
        state.activate(dataset);
    }
}

pub(crate) unsafe fn activate_source_resolution_for_session(
    session: *mut c_void,
    dataset: &Dataset,
) {
    if !session.is_null() {
        let handle = unsafe { &*(session as *const SessionHandle) };
        activate_source_resolution(handle, dataset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_roots_use_path_boundaries() {
        assert!(is_under_target(
            "s3://bucket/table.lance/data/file",
            "s3://bucket/table.lance"
        ));
        assert!(!is_under_target(
            "s3://bucket/table.lance-other/file",
            "s3://bucket/table.lance"
        ));
        assert!(!is_under_target(
            "s3://bucket/other/file",
            "s3://bucket/table.lance"
        ));
    }
}
