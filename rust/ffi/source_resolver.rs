// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Per-dataset source storage resolution through DuckDB secrets.
//!
//! A dataset opened by the extension receives a derived Lance session. It shares
//! the database-wide index and metadata cache backends, but has an isolated
//! object-store registry. The registry wraps Lance's public providers so a Blob
//! external URI can resolve its own DuckDB secret only after the dataset's
//! primary store has opened. This covers absolute references stored with
//! `base_id=0`. Registered bases with explicit runtime options keep their own
//! store configuration.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use lance::dataset::Dataset;
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance_core::Error;
use lance_io::object_store::{ObjectStoreProvider, StorageOptionsProvider};
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

#[derive(Debug, Default)]
struct SourceOptionScopeIds {
    ids: RwLock<HashMap<[u8; 32], u64>>,
    next_id: AtomicU64,
}

impl SourceOptionScopeIds {
    fn id_for(&self, options: &HashMap<String, String>) -> u64 {
        let mut hasher = blake3::Hasher::new();
        let mut entries = options.iter().collect::<Vec<_>>();
        entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
        for (key, value) in entries {
            hasher.update(&(key.len() as u64).to_le_bytes());
            hasher.update(key.as_bytes());
            hasher.update(&(value.len() as u64).to_le_bytes());
            hasher.update(value.as_bytes());
        }
        let digest = *hasher.finalize().as_bytes();
        if let Some(id) = self.ids.read().unwrap().get(&digest) {
            return *id;
        }
        let mut ids = self.ids.write().unwrap();
        if let Some(id) = ids.get(&digest) {
            return *id;
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        ids.insert(digest, id);
        id
    }
}

static NEXT_SOURCE_STATE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct SourceStorageOptionsProvider {
    id: String,
    options: HashMap<String, String>,
}

#[async_trait]
impl StorageOptionsProvider for SourceStorageOptionsProvider {
    async fn fetch_storage_options(&self) -> lance_core::Result<Option<HashMap<String, String>>> {
        Ok(Some(self.options.clone()))
    }

    fn provider_id(&self) -> String {
        self.id.clone()
    }
}

fn apply_source_options(
    params: &ObjectStoreParams,
    options: HashMap<String, String>,
    source_scope_id: String,
) -> ObjectStoreParams {
    let provider = Arc::new(SourceStorageOptionsProvider {
        id: source_scope_id,
        options: options.clone(),
    });
    let accessor = StorageOptionsAccessor::with_initial_and_provider(options, provider);
    let mut effective = params.clone();
    effective.storage_options_accessor = Some(Arc::new(accessor));
    effective
}

fn is_target_root(uri: &str, target_uri: &str) -> bool {
    uri.trim_end_matches('/') == target_uri
}

fn normalize_target_uri(uri: &str) -> String {
    uri.trim_end_matches('/').to_string()
}

fn resolved_fallback_options(params: &ObjectStoreParams) -> HashMap<String, String> {
    params
        .scoped_to_base(None)
        .storage_options()
        .cloned()
        .unwrap_or_default()
}

fn is_dataset_fallback(
    target_uri: &str,
    target_storage_options: &HashMap<String, String>,
    uri: &str,
    storage_options: Option<&HashMap<String, String>>,
) -> bool {
    !is_target_root(uri, target_uri)
        && storage_options
            .map(|options| options == target_storage_options)
            .unwrap_or_else(|| target_storage_options.is_empty())
}

#[derive(Debug)]
pub(crate) struct SourceResolutionState {
    resolver: SourceResolver,
    state_id: u64,
    // Scope identities follow the derived session lifetime; resolved credentials
    // never accumulate in a process-global map.
    source_option_ids: SourceOptionScopeIds,
    target_uri: RwLock<Option<String>>,
    target_storage_options: RwLock<HashMap<String, String>>,
}

impl SourceResolutionState {
    fn new(resolver: SourceResolver) -> Self {
        Self {
            resolver,
            state_id: NEXT_SOURCE_STATE_ID.fetch_add(1, Ordering::Relaxed),
            source_option_ids: SourceOptionScopeIds::default(),
            target_uri: RwLock::new(None),
            target_storage_options: RwLock::new(HashMap::new()),
        }
    }

    pub(crate) fn activate(&self, dataset: &Dataset) {
        *self.target_uri.write().unwrap() = Some(normalize_target_uri(dataset.uri()));
        *self.target_storage_options.write().unwrap() = dataset
            .store_params()
            .map(resolved_fallback_options)
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
        // Exact target options identify the dataset-level fallback used for
        // absolute `base_id=0` references. Registered bases with explicit
        // runtime params keep their own store configuration.
        let target_storage_options = self.target_storage_options.read().unwrap();
        is_dataset_fallback(target_uri, &target_storage_options, uri, storage_options)
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

    fn source_params(
        &self,
        params: &ObjectStoreParams,
        options: HashMap<String, String>,
    ) -> ObjectStoreParams {
        let source_scope_id = self.source_scope_id(&options);
        apply_source_options(params, options, source_scope_id)
    }

    fn source_scope_id(&self, options: &HashMap<String, String>) -> String {
        let option_id = self.source_option_ids.id_for(options);
        format!("duckdb-source-{:016x}-{option_id:016x}", self.state_id)
    }
}

#[derive(Debug)]
struct SourceResolvingProvider {
    inner: Arc<dyn ObjectStoreProvider>,
    state: Arc<SourceResolutionState>,
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
        let effective = self.state.source_params(params, source_options);
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
            "{prefix}${}",
            self.state.source_scope_id(&source_options)
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
    fn only_the_exact_target_root_is_primary() {
        assert!(is_target_root(
            "s3://bucket/table.lance/",
            "s3://bucket/table.lance"
        ));
        assert!(!is_target_root(
            "s3://bucket/table.lance/data/file",
            "s3://bucket/table.lance"
        ));
        assert!(!is_target_root(
            "s3://bucket/table.lance-other/file",
            "s3://bucket/table.lance"
        ));
    }

    #[test]
    fn only_dataset_fallback_options_resolve_source_secrets() {
        let target_options =
            HashMap::from([("endpoint".to_string(), "http://target.example".to_string())]);
        assert!(is_dataset_fallback(
            "s3://bucket/table.lance",
            &target_options,
            "s3://other/source.bin",
            Some(&target_options),
        ));
        assert!(is_dataset_fallback(
            "s3://bucket/table.lance",
            &HashMap::new(),
            "s3://other/source.bin",
            None,
        ));
        assert!(!is_dataset_fallback(
            "s3://bucket/table.lance",
            &target_options,
            "s3://other/source.bin",
            Some(&HashMap::from([(
                "base_1.endpoint".to_string(),
                "http://source.example".to_string(),
            )])),
        ));
        assert!(!is_dataset_fallback(
            "s3://bucket/table.lance",
            &target_options,
            "s3://bucket/table.lance/",
            Some(&target_options),
        ));
        assert!(is_dataset_fallback(
            "s3://bucket/table.lance",
            &target_options,
            "s3://bucket/table.lance/assets/payload.bin",
            Some(&target_options),
        ));
    }

    #[test]
    fn unrelated_base_scope_does_not_disable_absolute_source_resolution() {
        let raw = HashMap::from([("base_1.download_retry_count".to_string(), "3".to_string())]);
        let params = ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                raw,
            ))),
            ..Default::default()
        };
        let resolved = params.scoped_to_base(None);
        assert!(is_dataset_fallback(
            "s3://bucket/table.lance",
            &resolved_fallback_options(&params),
            "s3://other/source.bin",
            resolved.storage_options(),
        ));
    }

    #[test]
    fn source_option_ids_are_stable_across_states_and_never_expose_values() {
        let options = HashMap::from([
            ("access_key_id".to_string(), "reader".to_string()),
            ("secret_access_key".to_string(), "super-secret".to_string()),
        ]);
        let first = SourceOptionScopeIds::default();
        let second = SourceOptionScopeIds::default();

        let first_id = first.id_for(&options);
        assert_eq!(first_id, first.id_for(&options));
        assert_eq!(first_id, second.id_for(&options));
        assert_ne!(
            first_id,
            first.id_for(&HashMap::from([(
                "secret_access_key".to_string(),
                "rotated-secret".to_string(),
            )]))
        );

        let params = apply_source_options(
            &ObjectStoreParams::default(),
            options,
            format!("duckdb-source-0000000000000007-{first_id:016x}"),
        );
        assert_eq!(
            params.storage_options(),
            Some(&HashMap::from([
                ("access_key_id".to_string(), "reader".to_string()),
                ("secret_access_key".to_string(), "super-secret".to_string()),
            ]))
        );
        let accessor_id = params
            .storage_options_accessor
            .as_ref()
            .expect("source accessor")
            .accessor_id();
        assert_eq!(
            accessor_id,
            "duckdb-source-0000000000000007-0000000000000000"
        );
        assert!(!accessor_id.contains("reader"));
        assert!(!accessor_id.contains("super-secret"));
    }
}
