use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use lance::session::Session;
use lance::Dataset;
use lance_core::cache::CacheBackend;

use crate::datafusion_stream::DataFusionStream;
use crate::scanner::{LanceStream, LanceTakeStream};

use super::projection;
use super::source_resolver::SourceResolutionState;

pub(crate) type SchemaHandle = Arc<Schema>;

pub(crate) struct SessionHandle {
    pub(crate) session: Arc<Session>,
    // Lance does not expose clearing its index cache through Session.
    pub(crate) index_cache: Arc<dyn CacheBackend>,
    // Derived source sessions share these backends but clear neither on close.
    pub(crate) metadata_cache: Arc<dyn CacheBackend>,
    pub(crate) source_resolution: Option<Arc<SourceResolutionState>>,
    pub(crate) clear_caches_on_close: bool,
}

pub(crate) struct DatasetHandle {
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) read_schema: SchemaHandle,
    pub(crate) base_projection: Arc<[String]>,
    pub(crate) fts_projection: Arc<[String]>,
}

impl DatasetHandle {
    pub(crate) fn new(dataset: Arc<Dataset>) -> Self {
        let read_schema: Schema = (&projection::materialized_schema(dataset.schema())).into();
        let read_schema = Arc::new(read_schema);
        let base_projection = projection::build_base_projection(&read_schema);
        let fts_projection = projection::build_fts_projection(&base_projection);
        Self {
            dataset,
            read_schema,
            base_projection,
            fts_projection,
        }
    }
}

pub(crate) enum StreamHandle {
    Lance(LanceStream),
    Take(LanceTakeStream),
    DataFusion(DataFusionStream),
    Batches(std::vec::IntoIter<RecordBatch>),
}

impl StreamHandle {
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>, anyhow::Error> {
        match self {
            StreamHandle::Lance(stream) => stream.next().map_err(anyhow::Error::new),
            StreamHandle::Take(stream) => stream.next().map_err(anyhow::Error::new),
            StreamHandle::DataFusion(stream) => stream.next().map_err(anyhow::Error::new),
            StreamHandle::Batches(iter) => Ok(iter.next()),
        }
    }
}
