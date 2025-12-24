use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use lance::Dataset;

use crate::scanner::LanceStream;

use super::projection;

pub(crate) type SchemaHandle = Arc<Schema>;

pub(crate) struct DatasetHandle {
    pub(crate) dataset: Arc<Dataset>,
    pub(crate) arrow_schema: SchemaHandle,
    pub(crate) base_projection: Arc<[String]>,
    pub(crate) fts_projection: Arc<[String]>,
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
        }
    }
}

pub(crate) enum StreamHandle {
    Lance(LanceStream),
    Batches(std::vec::IntoIter<RecordBatch>),
}

impl StreamHandle {
    pub(crate) fn next_batch(&mut self) -> Result<Option<RecordBatch>, anyhow::Error> {
        match self {
            StreamHandle::Lance(stream) => stream.next().map_err(anyhow::Error::new),
            StreamHandle::Batches(iter) => Ok(iter.next()),
        }
    }
}
