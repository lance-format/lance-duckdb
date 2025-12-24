use std::pin::Pin;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
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
