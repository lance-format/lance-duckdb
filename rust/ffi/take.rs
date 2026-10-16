use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::Arc;

use futures::{stream, StreamExt, TryStreamExt};
use lance::dataset::{fragment::FileFragment, Dataset, ProjectionRequest};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::StreamHandle;
use super::util::{optional_cstr_array, slice_from_ptr, FfiError, FfiResult};

async fn fragment_physical_row_counts(
    dataset: &Arc<Dataset>,
    row_ids: &[u64],
) -> lance::Result<Vec<(u64, usize)>> {
    let mut fragment_ids = row_ids
        .iter()
        .map(|row_id| row_id >> 32)
        .collect::<Vec<_>>();
    fragment_ids.sort_unstable();
    fragment_ids.dedup();

    let fragments = fragment_ids
        .into_iter()
        .filter_map(|fragment_id| {
            let fragment_idx = dataset
                .manifest
                .fragments
                .binary_search_by_key(&fragment_id, |fragment| fragment.id)
                .ok()?;
            Some((
                fragment_id,
                FileFragment::new(
                    dataset.clone(),
                    dataset.manifest.fragments[fragment_idx].clone(),
                ),
            ))
        })
        .collect::<Vec<_>>();
    let io_parallelism = dataset.object_store(None).await?.io_parallelism();
    stream::iter(fragments)
        .map(|(fragment_id, fragment)| async move {
            fragment
                .physical_rows()
                .await
                .map(|row_count| (fragment_id, row_count))
        })
        .buffered(io_parallelism)
        .try_collect()
        .await
}

fn row_address_is_in_range(fragment_row_counts: &[(u64, usize)], row_id: u64) -> bool {
    let fragment_id = row_id >> 32;
    let row_offset = row_id as u32 as usize;
    fragment_row_counts
        .binary_search_by_key(&fragment_id, |(fragment_id, _)| *fragment_id)
        .is_ok_and(|idx| row_offset < fragment_row_counts[idx].1)
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_take_stream(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
) -> *mut c_void {
    match create_dataset_take_stream_inner(
        dataset,
        row_ids,
        row_ids_len,
        columns,
        columns_len,
        true,
    ) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_take_stream_unfiltered(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
) -> *mut c_void {
    match create_dataset_take_stream_inner(
        dataset,
        row_ids,
        row_ids_len,
        columns,
        columns_len,
        false,
    ) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn create_dataset_take_stream_inner(
    dataset: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
    columns: *const *const c_char,
    columns_len: usize,
    filter_out_of_range: bool,
) -> FfiResult<StreamHandle> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };

    let row_ids = if row_ids_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? }
    };
    let row_ids_filtered;
    let row_ids = if !filter_out_of_range || row_ids.is_empty() {
        row_ids
    } else if handle.dataset.manifest.uses_stable_row_ids() {
        let max_row_id = handle.dataset.manifest.next_row_id;
        if row_ids.iter().all(|id| *id < max_row_id) {
            row_ids
        } else {
            row_ids_filtered = row_ids
                .iter()
                .copied()
                .filter(|id| *id < max_row_id)
                .collect::<Vec<_>>();
            row_ids_filtered.as_slice()
        }
    } else {
        let fragment_row_counts =
            match runtime::block_on(fragment_physical_row_counts(&handle.dataset, row_ids)) {
                Ok(Ok(row_counts)) => row_counts,
                Ok(Err(err)) => {
                    return Err(FfiError::new(
                        ErrorCode::DatasetTake,
                        format!("dataset fragment physical rows: {err}"),
                    ))
                }
                Err(err) => {
                    return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))
                }
            };
        if row_ids
            .iter()
            .all(|id| row_address_is_in_range(&fragment_row_counts, *id))
        {
            row_ids
        } else {
            row_ids_filtered = row_ids
                .iter()
                .copied()
                .filter(|id| row_address_is_in_range(&fragment_row_counts, *id))
                .collect::<Vec<_>>();
            row_ids_filtered.as_slice()
        }
    };

    let projection_cols = unsafe { optional_cstr_array(columns, columns_len, "columns")? };
    let projection = if projection_cols.is_empty() {
        ProjectionRequest::from_schema(handle.dataset.schema().clone())
    } else {
        ProjectionRequest::from_columns(
            projection_cols.iter().map(|s| s.as_str()),
            handle.dataset.schema(),
        )
    };

    let batch = match runtime::block_on(handle.dataset.take_rows(row_ids, projection)) {
        Ok(Ok(batch)) => batch,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetTake,
                format!("dataset take_rows: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    Ok(StreamHandle::Batches(vec![batch].into_iter()))
}

#[cfg(test)]
mod tests {
    use arrow::array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow::datatypes::{DataType, Field, Schema};
    use lance::dataset::WriteParams;

    use super::*;
    use crate::ffi::types::DatasetHandle;

    #[test]
    fn legacy_unknown_physical_rows_remain_addressable() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![10_i64, 20]))],
        )
        .unwrap();
        let batches = RecordBatchIterator::new([Ok(batch)], schema);
        let params = WriteParams {
            max_rows_per_file: 1,
            ..Default::default()
        };
        let mut dataset = runtime::block_on(Dataset::write(
            batches,
            "memory://take-legacy-unknown-rows",
            Some(params),
        ))
        .unwrap()
        .unwrap();

        let manifest = Arc::make_mut(&mut dataset.manifest);
        manifest.writer_version = None;
        for fragment in Arc::make_mut(&mut manifest.fragments) {
            fragment.physical_rows = None;
        }

        let mut handle = Box::new(DatasetHandle::new(Arc::new(dataset)));
        let handle_ptr = handle.as_mut() as *mut DatasetHandle as *mut c_void;
        let row_ids = [1_u64 << 32];
        let stream = create_dataset_take_stream_inner(
            handle_ptr,
            row_ids.as_ptr(),
            row_ids.len(),
            ptr::null(),
            0,
            true,
        )
        .unwrap();
        let StreamHandle::Batches(mut batches) = stream else {
            panic!("expected batches stream");
        };
        let result = batches.next().unwrap();
        assert_eq!(result.num_rows(), 1, "valid legacy row address was dropped");
    }
}
