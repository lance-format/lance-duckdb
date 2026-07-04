use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use lance::session::Session;
use lance::Dataset;

use crate::datafusion_stream::DataFusionStream;
use crate::scanner::{LanceStream, LanceTakeStream};

use super::projection;

pub(crate) type SchemaHandle = Arc<Schema>;

pub(crate) struct SessionHandle {
    pub(crate) session: Arc<Session>,
}

pub(crate) struct DatasetHandle {
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) arrow_schema: SchemaHandle,
    pub(crate) base_projection: Arc<[String]>,
    pub(crate) fts_projection: Arc<[String]>,
    /// Storage options the namespace vended when this handle was opened
    /// through a REST namespace (`None` for plain/path opens). Revalidation
    /// compares them against a fresh describe so rotated credentials or other
    /// option changes trigger a reopen even when the location is unchanged.
    pub(crate) namespace_storage_options: Option<std::collections::HashMap<String, String>>,
}

impl DatasetHandle {
    pub(crate) fn new(dataset: Arc<Dataset>) -> Self {
        let arrow_schema: Schema = dataset.schema().into();
        let arrow_schema = Arc::new(arrow_schema);
        let base_projection = projection::build_base_projection(&arrow_schema);
        let fts_projection = projection::build_fts_projection(&base_projection);
        Self {
            dataset,
            arrow_schema,
            base_projection,
            fts_projection,
            namespace_storage_options: None,
        }
    }

    pub(crate) fn with_namespace_storage_options(
        mut self,
        options: Option<std::collections::HashMap<String, String>>,
    ) -> Self {
        self.namespace_storage_options = options;
        self
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
