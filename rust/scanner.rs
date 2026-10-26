use std::pin::Pin;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use lance::io::RecordBatchStream;
use lance_core::datatypes::BlobHandling;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::OffsetMapper;
use lance_datafusion::projection::ProjectionPlan;
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
    projection: Arc<ProjectionPlan>,
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
        let mut projection = projection
            .into_projection_plan(dataset.clone())
            .map_err(|err| err.to_string())?;
        projection.physical_projection = projection
            .physical_projection
            .clone()
            .with_blob_handling(BlobHandling::AllBinary);
        Ok(Self {
            handle,
            dataset,
            projection: Arc::new(projection),
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
            let row_addrs =
                row_offsets_to_addresses(&self.dataset, &self.row_indices[start..end]).await?;
            let batch = TakeBuilder::try_new_from_addresses(
                self.dataset.clone(),
                row_addrs,
                self.projection.clone(),
            )?
            .execute()
            .await?;
            Ok(Some(batch))
        })
    }
}

/// Map logical row offsets to physical row addresses in input order.
///
/// This mirrors the public behavior of Lance's internal offset mapping:
/// offsets address live rows (deletions are skipped) and each fragment's
/// mapping is independent. Sample paths compute offsets up front, so they
/// must produce addresses at take time instead of re-reading descriptors.
async fn row_offsets_to_addresses(
    dataset: &Dataset,
    row_indices: &[u64],
) -> lance::Result<Vec<u64>> {
    let fragments = dataset.get_fragments();
    let mut sorted: Vec<(u64, usize)> = row_indices
        .iter()
        .enumerate()
        .map(|(orig_idx, &offset)| (offset, orig_idx))
        .collect();
    sorted.sort_by_key(|&(offset, _)| offset);

    let mut out = vec![u64::from(RowAddress::TOMBSTONE_ROW); row_indices.len()];
    let mut fragment_iter = fragments.iter();
    let mut fragment = fragment_iter.next();
    let mut fragment_logical_rows: u64 = match fragment {
        Some(fragment) => fragment.count_rows(None).await? as u64,
        None => 0,
    };
    let mut fragment_offset = 0u64;
    let mut offset_mapper = match fragment {
        Some(fragment) => fragment
            .get_deletion_vector()
            .await?
            .as_ref()
            .map(|dv| OffsetMapper::new(dv.clone())),
        None => None,
    };

    for (offset, orig_idx) in sorted {
        while fragment.is_some() && offset >= fragment_offset + fragment_logical_rows {
            fragment_offset += fragment_logical_rows;
            fragment = fragment_iter.next();
            fragment_logical_rows = match fragment {
                Some(fragment) => fragment.count_rows(None).await? as u64,
                None => 0,
            };
            offset_mapper = match fragment {
                Some(fragment) => fragment
                    .get_deletion_vector()
                    .await?
                    .as_ref()
                    .map(|dv| OffsetMapper::new(dv.clone())),
                None => None,
            };
        }
        let Some(fragment) = fragment else {
            continue;
        };
        let logical_offset = (offset - fragment_offset) as u32;
        let physical_offset = offset_mapper
            .as_mut()
            .map(|mapper| mapper.map_offset(logical_offset))
            .unwrap_or(logical_offset);
        out[orig_idx] = u64::from(RowAddress::new_from_parts(
            fragment.id() as u32,
            physical_offset,
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, Int64Array, RecordBatch, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use lance::dataset::WriteParams;
    use lance::{blob_field, BlobArrayBuilder};
    use lance_core::utils::tempfile::TempStrDir;

    fn fixture_batches() -> (Arc<ArrowSchema>, Vec<RecordBatch>) {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            blob_field("payload", true),
        ]));
        let mut builder = BlobArrayBuilder::new(5);
        for (id, length) in [1u8, 2, 3, 4, 5].into_iter().enumerate() {
            if id == 3 {
                builder.push_null().unwrap();
            } else {
                builder.push_bytes(vec![b'x'; length as usize]).unwrap();
            }
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..5)),
                builder.finish().unwrap(),
            ],
        )
        .unwrap();
        (schema, vec![batch])
    }

    async fn write_fixture(stable: bool) -> (TempStrDir, Arc<Dataset>) {
        let dir = TempStrDir::default();
        let (schema, batches) = fixture_batches();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            dir.as_ref(),
            Some(WriteParams {
                enable_stable_row_ids: stable,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dir, Arc::new(dataset))
    }

    #[tokio::test]
    async fn take_stream_preserves_order_duplicates_and_skips_deleted_rows() {
        let (_dir, dataset) = write_fixture(false).await;
        let mut dataset = (*dataset).clone();
        dataset.delete("id = 2").await.unwrap();

        let projection = ProjectionRequest::from_columns(["id", "payload"], dataset.schema());
        let mut stream =
            LanceTakeStream::try_new(Arc::new(dataset), projection, vec![2, 0, 0, 3], 4).unwrap();
        let batch = stream.next().unwrap().unwrap();
        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[3, 0, 0, 4]);
        let payloads = batch.column(1).as_binary::<i64>();
        assert_eq!(payloads.value(0), &[b'x'; 4]);
        assert_eq!(payloads.value(1), &[b'x'; 1]);
        assert_eq!(payloads.value(2), &[b'x'; 1]);
        assert_eq!(payloads.value(3), &[b'x'; 5]);
    }

    #[tokio::test]
    async fn offset_mapping_flags_out_of_range_offsets() {
        let (_dir, dataset) = write_fixture(false).await;
        let addresses = row_offsets_to_addresses(&dataset, &[99]).await.unwrap();
        assert_eq!(addresses[0], u64::from(RowAddress::TOMBSTONE_ROW));
    }

    #[tokio::test]
    async fn stable_row_ids_resolve_to_binary_take_rows() {
        let (_dir, dataset) = write_fixture(true).await;
        let mut dataset = (*dataset).clone();
        dataset.delete("id = 2").await.unwrap();

        let mut scan = dataset.scan();
        scan.with_row_id()
            .project(&["id"])
            .unwrap()
            .scan_in_order(false);
        let batch = scan.try_into_batch().await.unwrap();
        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        let row_ids = batch
            .column(1)
            .as_primitive::<arrow::datatypes::UInt64Type>();
        let mut pairs = ids
            .values()
            .iter()
            .zip(row_ids.values().iter())
            .collect::<Vec<_>>();
        pairs.sort_by_key(|(id, _)| **id);

        let row_ids = pairs.iter().map(|(_, row_id)| **row_id).collect::<Vec<_>>();
        let addresses = crate::ffi::projection::row_ids_to_addresses(&dataset, &row_ids)
            .await
            .unwrap();
        let plan = crate::ffi::projection::binary_projection_plan(
            Arc::new(dataset.clone()),
            &["id".to_string(), "payload".to_string()],
        )
        .unwrap();
        let taken = lance::dataset::TakeBuilder::try_new_from_addresses(
            Arc::new(dataset.clone()),
            addresses,
            Arc::new(plan),
        )
        .unwrap()
        .execute()
        .await
        .unwrap();
        let taken_ids = taken
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(taken_ids.values(), &[0, 1, 3, 4]);
        let payloads = taken.column(1).as_binary::<i64>();
        assert_eq!(payloads.value(2), &[b'x'; 4]);
    }
}
