use std::pin::Pin;

use arrow::array::RecordBatch;
use futures::stream::Stream;
use lance::Dataset;
use tokio::runtime::Handle;

/// A stream wrapper that holds the Lance RecordBatchStream
pub struct LanceStream {
    handle: Handle,
    stream: Pin<Box<dyn Stream<Item = Result<RecordBatch, lance::Error>> + Send>>,
}

impl LanceStream {
    /// Create a new stream from a dataset path
    pub fn new(dataset: &Dataset) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        let scanner = dataset.scan();

        let stream = handle.block_on(async { scanner.try_into_stream().await })?;

        Ok(Self {
            handle,
            stream: Box::pin(stream),
        })
    }

    /// Get the next batch from the stream
    pub fn next(&mut self) -> Option<RecordBatch> {
        use futures::StreamExt;

        self.handle.block_on(async {
            match self.stream.next().await {
                Some(Ok(batch)) => Some(batch),
                _ => None,
            }
        })
    }
}
