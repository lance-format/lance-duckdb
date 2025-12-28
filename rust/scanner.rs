use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use lance::dataset::{Dataset, ProjectionRequest};
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance::io::RecordBatchStream;
use tokio::runtime::Handle;

/// A stream wrapper that holds the Lance RecordBatchStream
pub struct LanceStream {
    handle: Handle,
    stream: Pin<Box<DatasetRecordBatchStream>>,
}

impl LanceStream {
    /// Create a new stream from a Lance scanner
    pub fn from_scanner(scanner: Scanner) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        let stream = handle.block_on(async { scanner.try_into_stream().await })?;

        Ok(Self {
            handle,
            stream: Box::pin(stream),
        })
    }

    pub fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }

    /// Get the next batch from the stream
    pub fn next(&mut self) -> Result<Option<RecordBatch>, lance::Error> {
        use futures::StreamExt;

        self.handle.block_on(async {
            match self.stream.next().await {
                Some(Ok(batch)) => Ok(Some(batch)),
                Some(Err(err)) => Err(err),
                None => Ok(None),
            }
        })
    }
}

pub struct LanceTakeStream {
    handle: Handle,
    dataset: Arc<Dataset>,
    projection: ProjectionRequest,
    row_indices: Vec<u64>,
    pos: usize,
    batch_size: usize,
}

impl LanceTakeStream {
    pub fn try_new(
        dataset: Arc<Dataset>,
        projection: ProjectionRequest,
        row_indices: Vec<u64>,
        batch_size: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        Ok(Self {
            handle,
            dataset,
            projection,
            row_indices,
            pos: 0,
            batch_size: std::cmp::max(1, batch_size),
        })
    }

    pub fn next(&mut self) -> Result<Option<RecordBatch>, lance::Error> {
        if self.pos >= self.row_indices.len() {
            return Ok(None);
        }
        let end = std::cmp::min(self.pos + self.batch_size, self.row_indices.len());
        let start = self.pos;
        self.pos = end;

        self.handle.block_on(async {
            let batch = self
                .dataset
                .take(&self.row_indices[start..end], self.projection.clone())
                .await?;
            Ok(Some(batch))
        })
    }
}
