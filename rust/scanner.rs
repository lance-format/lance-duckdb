use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use arrow::array::{new_null_array, Array, AsArray, RecordBatch, RecordBatchOptions, UInt64Array};
use arrow::compute::take as arrow_take;
use arrow::datatypes::Schema as ArrowSchema;
use arrow::datatypes::SchemaRef;
use arrow::datatypes::{Field, UInt64Type};
use lance::dataset::scanner::{DatasetRecordBatchStream, Scanner};
use lance::dataset::scanner::{RowAddrMask, RowAddrTreeMap};
use lance::dataset::{Dataset, ProjectionRequest, TakeBuilder};
use lance_core::datatypes::BlobHandling;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::OffsetMapper;
use lance_core::ROW_ID;
use lance_datafusion::projection::ProjectionPlan;
use tokio::runtime::Handle;

pub(crate) fn nullable_output_schema(schema: SchemaRef) -> SchemaRef {
    Arc::new(ArrowSchema::new_with_metadata(
        schema
            .fields()
            .iter()
            .map(|field| field.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>(),
        schema.metadata().clone(),
    ))
}

/// Pad a projected batch back to the bound Arrow column layout.
///
/// Missing columns are null placeholders only: they preserve column indexes
/// for the caller's bound schema and must never be exposed as SQL values.
/// Present columns take the batch's field types so nested runtime metadata
/// matches the arrays before constructing the padded batch.
pub(crate) fn fill_missing_columns(
    batch: RecordBatch,
    full_schema: &SchemaRef,
) -> Result<RecordBatch, arrow::error::ArrowError> {
    let row_count = batch.num_rows();
    let batch_schema = batch.schema();
    let mut fields = Vec::with_capacity(full_schema.fields().len());
    let mut arrays = Vec::with_capacity(full_schema.fields().len());
    for field in full_schema.fields() {
        let batch_index = batch_schema
            .fields()
            .iter()
            .position(|candidate| candidate.name() == field.name());
        let (field, array) = if let Some(index) = batch_index {
            (
                batch_schema.field(index).clone().with_nullable(true),
                batch.column(index).clone(),
            )
        } else {
            (
                field.as_ref().clone().with_nullable(true),
                new_null_array(field.data_type(), row_count),
            )
        };
        fields.push(field);
        arrays.push(array);
    }
    let schema = Arc::new(ArrowSchema::new_with_metadata(
        fields,
        full_schema.metadata().clone(),
    ));
    RecordBatch::try_new(schema, arrays)
}

/// A stream wrapper that holds the Lance RecordBatchStream
pub struct LanceStream {
    handle: Handle,
    stream: Pin<Box<DatasetRecordBatchStream>>,
    pad_schema: Option<SchemaRef>,
}

impl LanceStream {
    /// Create a new stream from a Lance scanner
    pub fn from_scanner(scanner: Scanner) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        let stream = handle.block_on(async { scanner.try_into_stream().await })?;

        Ok(Self {
            handle,
            stream: Box::pin(stream),
            pad_schema: None,
        })
    }

    pub fn from_scanner_with_schema(
        scanner: Scanner,
        fill_schema: SchemaRef,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let handle = crate::runtime::handle()?;
        let stream = handle.block_on(async { scanner.try_into_stream().await })?;
        let schema = nullable_output_schema(fill_schema);
        Ok(Self {
            handle,
            stream: Box::pin(stream),
            pad_schema: Some(schema),
        })
    }

    /// Get the next batch from the stream
    pub fn next(&mut self) -> Result<Option<RecordBatch>, lance::Error> {
        use futures::StreamExt;

        self.handle.block_on(async {
            match self.stream.next().await {
                Some(Ok(batch)) => {
                    let batch = if let Some(full_schema) = &self.pad_schema {
                        fill_missing_columns(batch, full_schema).map_err(|err| {
                            lance::Error::internal(format!("fill projected columns: {err}"))
                        })?
                    } else {
                        batch
                    };
                    Ok(Some(batch))
                }
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
    projection_names: Vec<String>,
    use_blob_v2_scanner: bool,
    row_indices: Vec<u64>,
    pos: usize,
    batch_size: usize,
}

/// Take exact logical row IDs through the native scanner so blob columns go
/// through Lance's materialization path instead of a bare fragment take.
///
/// `row_ids` are in the dataset's `_rowid` space (stable row IDs when enabled,
/// physical row addresses otherwise), matching `with_row_id_prefilter`.
/// Output rows follow the input `columns` order and repeat/omit rows to match
/// the input `row_ids` exactly.
pub async fn take_rows_by_row_ids(
    dataset: Arc<Dataset>,
    row_ids: &[u64],
    columns: &[String],
) -> Result<RecordBatch, lance::Error> {
    let has_row_id = columns.iter().any(|name| name == ROW_ID);
    let mut scanner = dataset.scan();
    if columns.is_empty() {
        scanner.empty_project()?;
    } else {
        scanner.project(columns)?;
    }
    scanner.blob_handling(BlobHandling::AllBinary);
    if !has_row_id {
        scanner.with_row_id();
    }
    let mask = if row_ids.is_empty() {
        RowAddrMask::allow_nothing()
    } else {
        RowAddrMask::from_allowed(RowAddrTreeMap::from_iter(row_ids.iter().copied()))
    };
    scanner.with_row_id_prefilter(mask);

    let scanned = scanner.try_into_batch().await?;
    let rowid_pos = scanned
        .schema()
        .fields
        .iter()
        .position(|field| field.name() == ROW_ID)
        .ok_or_else(|| {
            lance::Error::internal(format!(
                "rowid prefilter scan did not return the {ROW_ID} column"
            ))
        })?;
    let scanned_row_ids = scanned.column(rowid_pos).as_primitive::<UInt64Type>();
    let mut position_by_row_id: HashMap<u64, usize> = HashMap::with_capacity(scanned.num_rows());
    for (pos, row_id) in scanned_row_ids.values().iter().enumerate() {
        position_by_row_id.entry(*row_id).or_insert(pos);
    }

    let indices: Vec<u64> = row_ids
        .iter()
        .filter_map(|row_id| {
            position_by_row_id
                .get(row_id)
                .copied()
                .map(|pos| pos as u64)
        })
        .collect();
    let indices = UInt64Array::from(indices);

    if columns.is_empty() {
        return RecordBatch::try_new_with_options(
            Arc::new(ArrowSchema::empty()),
            Vec::new(),
            &RecordBatchOptions::new().with_row_count(Some(indices.len())),
        )
        .map_err(|err| lance::Error::internal(format!("empty take projection: {err}")));
    }

    let schema = scanned.schema();
    let mut out_fields: Vec<Field> = Vec::with_capacity(columns.len());
    let mut out_arrays: Vec<std::sync::Arc<dyn Array>> = Vec::with_capacity(columns.len());
    for name in columns {
        let pos = schema
            .fields
            .iter()
            .position(|field| field.name() == name)
            .ok_or_else(|| {
                lance::Error::invalid_input(format!(
                    "take projection column {name} not found in scan output"
                ))
            })?;
        out_fields.push(schema.field(pos).clone());
        out_arrays.push(arrow_take(scanned.column(pos).as_ref(), &indices, None)?);
    }
    let out_schema = Arc::new(ArrowSchema::new(out_fields));
    RecordBatch::try_new(out_schema, out_arrays).map_err(Into::into)
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
        let projection_names: Vec<String> = if projection.requested_output_expr.is_empty() {
            dataset
                .schema()
                .fields
                .iter()
                .map(|field| field.name.clone())
                .collect()
        } else {
            projection
                .requested_output_expr
                .iter()
                .map(|column| column.name.clone())
                .collect()
        };
        let use_blob_v2_scanner =
            crate::ffi::projection::projection_contains_blob_v2(&dataset, &projection_names)
                .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        Ok(Self {
            handle,
            dataset,
            projection: Arc::new(projection),
            projection_names,
            use_blob_v2_scanner,
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
            let batch = if self.use_blob_v2_scanner {
                // The native take returns logical `_rowid` values, which are
                // stable row IDs when enabled and row addresses otherwise.
                let row_ids = addresses_to_row_ids(&self.dataset, &row_addrs).await?;
                take_rows_by_row_ids(self.dataset.clone(), &row_ids, &self.projection_names).await?
            } else {
                TakeBuilder::try_new_from_addresses(
                    self.dataset.clone(),
                    row_addrs,
                    self.projection.clone(),
                )?
                .execute()
                .await?
            };
            Ok(Some(batch))
        })
    }
}

async fn addresses_to_row_ids(
    dataset: &Arc<Dataset>,
    addresses: &[u64],
) -> lance::Result<Vec<u64>> {
    if addresses.is_empty() {
        return Ok(Vec::new());
    }
    let request = ProjectionRequest::from_columns([ROW_ID], dataset.schema());
    let plan = request.into_projection_plan(dataset.clone())?;
    let batch =
        TakeBuilder::try_new_from_addresses(dataset.clone(), addresses.to_vec(), Arc::new(plan))?
            .execute()
            .await?;
    let rowid_pos = batch
        .schema()
        .fields
        .iter()
        .position(|field| field.name() == ROW_ID)
        .ok_or_else(|| {
            lance::Error::internal(format!(
                "rowid-only take did not return the {ROW_ID} column"
            ))
        })?;
    Ok(batch
        .column(rowid_pos)
        .as_primitive::<UInt64Type>()
        .values()
        .to_vec())
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

    let mut out = vec![RowAddress::TOMBSTONE_ROW; row_indices.len()];
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
    use lance_file::version::LanceFileVersion;

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

    async fn write_multi_fragment_stable_fixture() -> (TempStrDir, Arc<Dataset>) {
        let dir = TempStrDir::default();
        let (schema, batches) = fixture_batches();
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
        let dataset = Dataset::write(
            reader,
            dir.as_ref(),
            Some(WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 2,
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        (dir, Arc::new(dataset))
    }

    async fn write_legacy_plain_fixture() -> (TempStrDir, Arc<Dataset>) {
        let dir = TempStrDir::default();
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", DataType::LargeBinary, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..3)),
                Arc::new(arrow::array::LargeBinaryArray::from(vec![
                    b"a".as_slice(),
                    b"bb".as_slice(),
                    b"ccc".as_slice(),
                ])),
            ],
        )
        .unwrap();
        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let dataset = Dataset::write(
            reader,
            dir.as_ref(),
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::Legacy),
                max_rows_per_file: 2,
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
        let batch = tokio::task::spawn_blocking(move || stream.next().unwrap().unwrap())
            .await
            .unwrap();
        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[3, 0, 0, 4]);
        let payloads = batch.column(1).as_binary::<i64>();
        assert!(payloads.is_null(0));
        assert_eq!(payloads.value(1), &[b'x'; 1]);
        assert_eq!(payloads.value(2), &[b'x'; 1]);
        assert_eq!(payloads.value(3), &[b'x'; 5]);
    }

    #[tokio::test]
    async fn stable_sample_maps_physical_addresses_back_to_row_ids() {
        let (_dir, dataset) = write_multi_fragment_stable_fixture().await;
        let mut dataset = (*dataset).clone();
        dataset.delete("id = 2").await.unwrap();

        let projection = ProjectionRequest::from_columns(["id", "payload"], dataset.schema());
        let mut stream =
            LanceTakeStream::try_new(Arc::new(dataset), projection, vec![0, 1, 2, 3], 4).unwrap();
        let batch = tokio::task::spawn_blocking(move || stream.next().unwrap().unwrap())
            .await
            .unwrap();
        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[0, 1, 3, 4]);
        let payloads = batch.column(1).as_binary::<i64>();
        assert_eq!(payloads.value(0), &[b'x'; 1]);
        assert_eq!(payloads.value(1), &[b'x'; 2]);
        assert!(payloads.is_null(2));
        assert_eq!(payloads.value(3), &[b'x'; 5]);
    }

    #[tokio::test]
    async fn legacy_plain_sample_keeps_native_take_path() {
        let (_dir, dataset) = write_legacy_plain_fixture().await;
        let projection = ProjectionRequest::from_columns(["id", "payload"], dataset.schema());
        let mut stream = LanceTakeStream::try_new(dataset, projection, vec![0, 2], 2).unwrap();
        let batch = tokio::task::spawn_blocking(move || stream.next().unwrap().unwrap())
            .await
            .unwrap();
        let ids = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(ids.values(), &[0, 2]);
        let payloads = batch.column(1).as_binary::<i64>();
        assert_eq!(payloads.value(0), b"a");
        assert_eq!(payloads.value(1), b"ccc");
    }

    #[tokio::test]
    async fn offset_mapping_flags_out_of_range_offsets() {
        let (_dir, dataset) = write_fixture(false).await;
        let addresses = row_offsets_to_addresses(&dataset, &[99]).await.unwrap();
        assert_eq!(addresses[0], RowAddress::TOMBSTONE_ROW);
    }

    #[tokio::test]
    async fn stable_row_ids_take_materializes_blobs() {
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
        let taken = take_rows_by_row_ids(
            Arc::new(dataset.clone()),
            &row_ids,
            &["id".to_string(), "payload".to_string()],
        )
        .await
        .unwrap();
        let taken_ids = taken
            .column(0)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(taken_ids.values(), &[0, 1, 3, 4]);
        let payloads = taken.column(1).as_binary::<i64>();
        assert_eq!(payloads.value(0), &[b'x'; 1]);
        assert_eq!(payloads.value(1), &[b'x'; 2]);
        assert!(payloads.is_null(2));
        assert_eq!(payloads.value(3), &[b'x'; 5]);
    }
}
