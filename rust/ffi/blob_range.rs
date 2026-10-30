use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::Arc;

use lance::dataset::{BlobRangeRequest, Dataset, ReadBlobRange};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, dataset_handle, slice_from_ptr, FfiError, FfiResult};

/// Selector kind values shared with the C++ caller.
pub(crate) const BLOB_SELECTOR_ROW_ADDRESS: u8 = 0;
pub(crate) const BLOB_SELECTOR_ROW_ID: u8 = 1;

/// One blob range request forwarded from the SQL request relation.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LanceBlobRangeRequest {
    pub row: u64,
    pub offset: u64,
    pub length: u64,
}

/// One materialized blob range returned to the C++ caller.
///
/// `data_ptr` is null when `is_null` is set or when the requested range is
/// empty. Non-empty payloads are owned by this array and released by
/// [`lance_free_blob_range_results`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LanceBlobRangeResult {
    pub request_index: u64,
    pub row_address: u64,
    pub is_null: u8,
    pub data_len: usize,
    pub data_ptr: *const u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlobRowSelector {
    RowAddress,
    RowId,
}

impl BlobRowSelector {
    pub(crate) fn from_ffi(value: u8) -> FfiResult<Self> {
        match value {
            BLOB_SELECTOR_ROW_ADDRESS => Ok(Self::RowAddress),
            BLOB_SELECTOR_ROW_ID => Ok(Self::RowId),
            other => Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("unknown blob row selector kind: {other}"),
            )),
        }
    }
}

fn validate_row_addresses(
    dataset: &Arc<Dataset>,
    requests: &[BlobRangeRequest],
) -> lance::Result<()> {
    let fragments = &dataset.manifest.fragments;
    for (index, request) in requests.iter().enumerate() {
        let fragment_id = request.row >> 32;
        let row_offset = request.row as u32 as u64;
        if let Ok(fragment_index) =
            fragments.binary_search_by_key(&fragment_id, |fragment| fragment.id)
        {
            if let Some(physical_rows) = fragments[fragment_index].physical_rows {
                if row_offset >= physical_rows as u64 {
                    return Err(lance::Error::invalid_input(format!(
                        "blob range request {index}: row address {} is out of range for fragment {fragment_id} with {physical_rows} rows",
                        request.row
                    )));
                }
            }
        } else {
            return Err(lance::Error::invalid_input(format!(
                "blob range request {index}: row address {} references missing fragment {fragment_id}",
                request.row
            )));
        }
    }
    Ok(())
}

/// Read one bounded window of blob ranges from an already-pinned dataset
/// handle. `selector` decides how `row` is interpreted; rows carry one
/// selector kind per call so stable row ids and physical row addresses never
/// mix inside a single Lance request batch.
pub(crate) async fn read_blob_ranges_core(
    dataset: &Arc<Dataset>,
    column: &str,
    selector: BlobRowSelector,
    requests: &[BlobRangeRequest],
) -> lance::Result<Vec<ReadBlobRange>> {
    if selector == BlobRowSelector::RowAddress {
        validate_row_addresses(dataset, requests)?;
    }
    let builder = dataset.read_blob_ranges(column)?;
    let builder = match selector {
        BlobRowSelector::RowAddress => builder.with_row_addresses(requests.iter().copied()),
        BlobRowSelector::RowId => builder.with_row_ids(requests.iter().copied()),
    };
    builder.execute().await
}

fn into_c_results(results: Vec<ReadBlobRange>) -> Result<Vec<LanceBlobRangeResult>, FfiError> {
    let mut converted = Vec::with_capacity(results.len());
    for result in results {
        let (is_null, data_len, data_ptr) = match result.data {
            None => (1_u8, 0_usize, ptr::null()),
            Some(data) if data.is_empty() => (0_u8, 0_usize, ptr::null()),
            Some(data) => {
                let buffer: Box<[u8]> = data.to_vec().into_boxed_slice();
                let data_len = buffer.len();
                (0_u8, data_len, Box::into_raw(buffer) as *const u8)
            }
        };
        converted.push(LanceBlobRangeResult {
            request_index: u64::try_from(result.request_index).map_err(|_| {
                FfiError::new(
                    ErrorCode::BlobRangeRead,
                    "blob range request index does not fit in u64",
                )
            })?,
            row_address: result.row_address,
            is_null,
            data_len,
            data_ptr,
        });
    }
    Ok(converted)
}

fn read_blob_ranges_inner(
    dataset: *mut c_void,
    column: *const c_char,
    selector: u8,
    requests: *const LanceBlobRangeRequest,
    requests_len: usize,
) -> FfiResult<Vec<LanceBlobRangeResult>> {
    if requests_len > 0 && requests.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "blob range requests pointer is null with non-zero length",
        ));
    }
    let handle = unsafe { dataset_handle(dataset)? };
    let column = unsafe { cstr_to_str(column, "column")? };
    let selector = BlobRowSelector::from_ffi(selector)?;
    let raw_requests = if requests_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(requests, requests_len, "requests")? }
    };
    let requests = raw_requests
        .iter()
        .map(|request| BlobRangeRequest::new(request.row, request.offset, request.length))
        .collect::<Vec<_>>();

    let results = match runtime::block_on(read_blob_ranges_core(
        &handle.dataset,
        column,
        selector,
        &requests,
    )) {
        Ok(Ok(results)) => results,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::BlobRangeRead,
                format!("dataset blob range read: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };
    into_c_results(results)
}

#[no_mangle]
pub unsafe extern "C" fn lance_read_blob_ranges(
    dataset: *mut c_void,
    column: *const c_char,
    selector: u8,
    requests: *const LanceBlobRangeRequest,
    requests_len: usize,
    out_results: *mut *mut LanceBlobRangeResult,
    out_results_len: *mut usize,
) -> i32 {
    if out_results.is_null() || out_results_len.is_null() {
        set_last_error(
            ErrorCode::InvalidArgument,
            "out_results/out_results_len must not be null",
        );
        return -1;
    }
    match read_blob_ranges_inner(dataset, column, selector, requests, requests_len) {
        Ok(results) if results.is_empty() => {
            clear_last_error();
            unsafe {
                *out_results = ptr::null_mut();
                *out_results_len = 0;
            }
            0
        }
        Ok(results) => {
            clear_last_error();
            let results_len = results.len();
            let boxed: Box<[LanceBlobRangeResult]> = results.into_boxed_slice();
            unsafe {
                *out_results = Box::into_raw(boxed) as *mut LanceBlobRangeResult;
                *out_results_len = results_len;
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

/// Release a result array returned by [`lance_read_blob_ranges`].
#[no_mangle]
pub unsafe extern "C" fn lance_free_blob_range_results(
    results: *mut LanceBlobRangeResult,
    results_len: usize,
) {
    if results.is_null() {
        return;
    }
    let slice = unsafe { std::slice::from_raw_parts_mut(results, results_len) };
    for result in slice.iter() {
        if result.data_len > 0 && !result.data_ptr.is_null() {
            let buffer = ptr::slice_from_raw_parts_mut(result.data_ptr as *mut u8, result.data_len);
            drop(unsafe { Box::from_raw(buffer) });
        }
    }
    let array = ptr::slice_from_raw_parts_mut(results, results_len);
    drop(unsafe { Box::from_raw(array) });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::types::UInt64Type;
    use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, UInt64Array};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use lance::blob::{blob_field_with_options, BlobArrayBuilder, BlobFieldOptions};
    use lance::dataset::{Dataset, ProjectionRequest, WriteParams};
    use lance_file::version::LanceFileVersion;
    use lance_io::object_store::WrappingObjectStore;
    use lance_io::utils::tracking_store::IOTracker;

    use super::*;

    const INLINE_THRESHOLD: usize = 1024;
    const DEDICATED_THRESHOLD: usize = 4096;
    const DEDICATED_VALUE_SIZE: usize = 256 * 1024;

    fn blob_fixture_schema() -> Arc<ArrowSchema> {
        let blob_column = blob_field_with_options(
            "payload",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(INLINE_THRESHOLD)
                .with_dedicated_size_threshold(
                    std::num::NonZeroUsize::new(DEDICATED_THRESHOLD).unwrap(),
                ),
        );
        Arc::new(ArrowSchema::new(vec![
            blob_column,
            ArrowField::new("id", arrow_schema::DataType::UInt64, false),
        ]))
    }

    fn write_blob_fixture(uri: &str, stable_row_ids: bool) -> Arc<Dataset> {
        let schema = blob_fixture_schema();
        let mut builder = BlobArrayBuilder::new(5);
        builder.push_bytes(b"hello world").unwrap();
        builder
            .push_bytes(vec![7_u8; 2 * INLINE_THRESHOLD])
            .unwrap();
        builder
            .push_bytes(vec![0xA5_u8; DEDICATED_VALUE_SIZE])
            .unwrap();
        builder.push_null().unwrap();
        builder.push_empty().unwrap();
        let blob_array: ArrayRef = builder.finish().unwrap();
        let ids = Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4])) as ArrayRef;
        let batch = RecordBatch::try_new(schema.clone(), vec![blob_array, ids]).unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let dataset = runtime::block_on(Dataset::write(
            reader,
            uri,
            Some(WriteParams {
                data_storage_version: Some(LanceFileVersion::V2_2),
                enable_stable_row_ids: stable_row_ids,
                ..Default::default()
            }),
        ))
        .unwrap()
        .unwrap();
        Arc::new(dataset)
    }

    fn row_addresses(dataset: &Arc<Dataset>, row_indices: &[u64]) -> Vec<u64> {
        let projection = ProjectionRequest::from_columns(["id"], dataset.schema());
        let batch = runtime::block_on(
            dataset
                .take_builder(row_indices, projection)
                .unwrap()
                .with_row_address(true)
                .execute(),
        )
        .unwrap()
        .unwrap();
        batch
            .column(1)
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec()
    }

    #[test]
    fn range_read_by_row_address_preserves_requests_and_null_empty_semantics() {
        let dataset = write_blob_fixture("memory://blob-range-address", false);
        let addresses = row_addresses(&dataset, &[0, 1, 2, 3, 4]);
        let requests = [
            BlobRangeRequest::new(addresses[2], 1024, 4),
            BlobRangeRequest::new(addresses[0], 0, 5),
            BlobRangeRequest::new(addresses[2], 1024, 4),
            BlobRangeRequest::new(addresses[3], 0, 4),
            BlobRangeRequest::new(addresses[4], 0, 0),
            BlobRangeRequest::new(addresses[1], 0, 3),
        ];
        let results = runtime::block_on(read_blob_ranges_core(
            &dataset,
            "payload",
            BlobRowSelector::RowAddress,
            &requests,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(results.len(), requests.len());
        assert_eq!(results[0].data.as_deref(), Some(&[0xA5_u8; 4][..]));
        assert_eq!(results[1].data.as_deref(), Some(&b"hello"[..]));
        assert_eq!(results[2].data.as_deref(), Some(&[0xA5_u8; 4][..]));
        assert_eq!(results[3].data, None);
        assert_eq!(results[4].data.as_deref(), Some(&[][..]));
        assert_eq!(results[5].data.as_deref(), Some(&[7_u8; 3][..]));
        for (index, result) in results.iter().enumerate() {
            assert_eq!(result.request_index, index);
            assert_eq!(result.row_address, requests[index].row);
        }
    }

    #[test]
    fn range_read_by_stable_row_id_pins_results_to_stable_identity() {
        let dataset = write_blob_fixture("memory://blob-range-row-id", true);
        assert!(dataset.manifest.uses_stable_row_ids());
        // Stable row ids are assigned in write order for a fresh dataset.
        let row_ids = [4_u64, 0, 1];
        let requests = [
            BlobRangeRequest::new(row_ids[0], 0, 0),
            BlobRangeRequest::new(row_ids[1], 6, 5),
            BlobRangeRequest::new(row_ids[2], 0, 2),
        ];
        let results = runtime::block_on(read_blob_ranges_core(
            &dataset,
            "payload",
            BlobRowSelector::RowId,
            &requests,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].data.as_deref(), Some(&[][..]));
        assert_eq!(results[1].data.as_deref(), Some(&b"world"[..]));
        assert_eq!(results[2].data.as_deref(), Some(&[7_u8; 2][..]));
    }

    #[test]
    fn range_read_rejects_missing_rows_overflow_and_out_of_bounds() {
        let dataset = write_blob_fixture("memory://blob-range-errors", false);
        let addresses = row_addresses(&dataset, &[0]);

        let overflow = runtime::block_on(read_blob_ranges_core(
            &dataset,
            "payload",
            BlobRowSelector::RowAddress,
            &[BlobRangeRequest::new(addresses[0], u64::MAX, 1)],
        ));
        let err = overflow.unwrap().unwrap_err();
        assert!(err.to_string().contains("overflowed"), "{err}");

        let out_of_bounds = runtime::block_on(read_blob_ranges_core(
            &dataset,
            "payload",
            BlobRowSelector::RowAddress,
            &[BlobRangeRequest::new(addresses[0], 0, 5000)],
        ));
        let err = out_of_bounds.unwrap().unwrap_err();
        assert!(
            err.to_string().contains("exceeds blob size")
                || err.to_string().contains("out of bounds")
                || err.to_string().contains("invalid"),
            "{err}"
        );

        let missing = runtime::block_on(read_blob_ranges_core(
            &dataset,
            "payload",
            BlobRowSelector::RowAddress,
            &[BlobRangeRequest::new(999_u64 << 32 | 3, 0, 4)],
        ));
        let err = missing.unwrap().unwrap_err();
        assert!(err.to_string().contains("missing fragment"), "{err}");
    }

    #[test]
    fn range_read_does_not_materialize_whole_dedicated_values() {
        let dataset = write_blob_fixture("memory://blob-range-io", false);
        let tracker = IOTracker::default();
        let wrapper: Arc<dyn WrappingObjectStore> = Arc::new(tracker.clone());
        let wrapped = Arc::new(dataset.with_object_store_wrappers([wrapper]));
        let addresses = row_addresses(&wrapped, &[2]);
        let _ = tracker.incremental_stats();

        let results = runtime::block_on(read_blob_ranges_core(
            &wrapped,
            "payload",
            BlobRowSelector::RowAddress,
            &[BlobRangeRequest::new(addresses[0], 64 * 1024, 4096)],
        ))
        .unwrap()
        .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].data.as_deref(), Some(&[0xA5_u8; 4096][..]));
        let stats = tracker.incremental_stats();
        eprintln!(
            "dedicated {} byte blob, 4096 byte window -> read_bytes={} requests={:?}",
            DEDICATED_VALUE_SIZE, stats.read_bytes, stats.requests
        );
        assert!(
            stats.read_bytes < DEDICATED_VALUE_SIZE as u64,
            "read {} bytes for a {} byte dedicated value",
            stats.read_bytes,
            DEDICATED_VALUE_SIZE
        );
        for request in &stats.requests {
            if let Some(range) = &request.range {
                assert!(
                    range.end - range.start < DEDICATED_VALUE_SIZE as u64,
                    "whole-value read requested: {request:?}"
                );
            }
        }
    }
}
