use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Future;
use lance::session::{CacheSpec, Session};
use lance_core::cache::{
    CacheBackend, CacheCodec, CacheEntry, CacheKeySchema, CacheNamespace, InternalCacheKey,
    KeyBuilder,
};

use super::types::SessionHandle;

/// Lance 14 cache keys omit the registered manifest path. Put a bind's keys
/// in their own namespace while charging entries to the existing SQL caches.
#[derive(Debug)]
struct SnapshotCache {
    backend: Arc<dyn CacheBackend>,
    namespace: CacheNamespace,
}

impl SnapshotCache {
    fn key(&self, key: &InternalCacheKey) -> InternalCacheKey {
        let mut builder = KeyBuilder::new(
            self.namespace,
            "lance_duckdb.snapshot",
            CacheKeySchema::new("cache-entry", 1),
        );
        builder.write_fixed_bytes(key.as_bytes());
        builder.finish()
    }
}

#[async_trait]
impl CacheBackend for SnapshotCache {
    async fn get_resident(&self, key: &InternalCacheKey) -> Option<CacheEntry> {
        self.backend.get_resident(&self.key(key)).await
    }

    async fn peek_resident(&self, key: &InternalCacheKey) -> bool {
        self.backend.peek_resident(&self.key(key)).await
    }

    fn plane_admission_gated(&self) -> bool {
        self.backend.plane_admission_gated()
    }

    async fn get(&self, key: &InternalCacheKey, codec: Option<CacheCodec>) -> Option<CacheEntry> {
        self.backend.get(&self.key(key), codec).await
    }

    async fn insert(
        &self,
        key: &InternalCacheKey,
        entry: CacheEntry,
        size: usize,
        codec: Option<CacheCodec>,
    ) {
        self.backend
            .insert(&self.key(key), entry, size, codec)
            .await
    }

    async fn get_or_insert<'a>(
        &self,
        key: &InternalCacheKey,
        loader: Pin<Box<dyn Future<Output = lance_core::Result<(CacheEntry, usize)>> + Send + 'a>>,
        codec: Option<CacheCodec>,
    ) -> lance_core::Result<(CacheEntry, bool)> {
        self.backend
            .get_or_insert(&self.key(key), loader, codec)
            .await
    }

    async fn clear(&self) {
        self.backend.clear().await
    }

    async fn num_entries(&self) -> usize {
        self.backend.num_entries().await
    }

    async fn size_bytes(&self) -> usize {
        self.backend.size_bytes().await
    }

    fn capacity_bytes(&self) -> Option<usize> {
        self.backend.capacity_bytes()
    }

    fn approx_num_entries(&self) -> usize {
        self.backend.approx_num_entries()
    }

    fn approx_size_bytes(&self) -> usize {
        self.backend.approx_size_bytes()
    }
}

pub(super) fn snapshot_session(parent: &SessionHandle) -> Arc<Session> {
    let namespace = CacheNamespace::root().child(&rand::random::<u128>().to_string());
    let scope = |backend: &Arc<dyn CacheBackend>| {
        CacheSpec::Backend(Arc::new(SnapshotCache {
            backend: backend.clone(),
            namespace,
        }))
    };
    Arc::new(Session::with_cache_backends(
        scope(&parent.index_cache),
        scope(&parent.metadata_cache),
        parent.session.store_registry(),
    ))
}
