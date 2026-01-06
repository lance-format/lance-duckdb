use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use tokio::runtime::Handle;

pub struct DataFusionStream {
    handle: Handle,
    stream: SendableRecordBatchStream,
}

impl DataFusionStream {
    pub fn try_new(stream: SendableRecordBatchStream) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        Ok(Self { handle, stream })
    }

    pub fn schema(&self) -> SchemaRef {
        self.stream.schema()
    }

    pub fn next(&mut self) -> Result<Option<RecordBatch>, datafusion_common::DataFusionError> {
        self.handle.block_on(async {
            let next = self.stream.next().await;
            match next {
                Some(Ok(batch)) => Ok(Some(batch)),
                Some(Err(err)) => Err(err),
                None => Ok(None),
            }
        })
    }
}
