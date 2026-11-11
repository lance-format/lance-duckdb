use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::Arc;

use lance::dataset::scanner::RowAddrTreeMap;
use lance::dataset::transaction::{Operation, Transaction};
use lance::Dataset;
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::update::{apply_deletions, build_row_id_index, lookup_row_address};
use super::util::{dataset_handle, slice_from_ptr, FfiError, FfiResult};
use super::write::UncommittedTransaction;

// DuckDB evaluates the DELETE's WHERE clause (subqueries and joins included)
// and streams the `_rowid` of every selected row here. The transaction is
// built against the dataset handle pinned for the statement, the one native
// scans of the table read, so their row ids and Lance's conflict check refer
// to the same version.

struct DeleteHandle {
    dataset: Arc<Dataset>,
    row_ids: RoaringTreemap,
}

#[no_mangle]
pub unsafe extern "C" fn lance_delete_begin(
    dataset: *mut c_void,
    out_delete_handle: *mut *mut c_void,
) -> i32 {
    match delete_begin_inner(dataset, out_delete_handle) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn delete_begin_inner(dataset: *mut c_void, out_delete_handle: *mut *mut c_void) -> FfiResult<()> {
    if out_delete_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_delete_handle is null",
        ));
    }
    let dataset = unsafe { dataset_handle(dataset)? }.dataset.clone();
    let handle = Box::new(DeleteHandle {
        dataset,
        row_ids: RoaringTreemap::new(),
    });
    unsafe {
        *out_delete_handle = Box::into_raw(handle) as *mut c_void;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_delete_add_rowids(
    delete_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> i32 {
    match delete_add_rowids_inner(delete_handle, row_ids, row_ids_len) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn delete_add_rowids_inner(
    delete_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> FfiResult<()> {
    if delete_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "delete_handle is null",
        ));
    }
    if row_ids_len > 0 && row_ids.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "row_ids is null with non-zero length",
        ));
    }
    let handle = unsafe { &mut *(delete_handle as *mut DeleteHandle) };
    if row_ids_len > 0 {
        let ids = unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? };
        // Row ids arrive in no particular order, and a join (`USING`) can
        // repeat one; the set counts every row once.
        handle.row_ids.extend(ids.iter().copied());
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_delete_finish_uncommitted(
    delete_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> i32 {
    match delete_finish_uncommitted_inner(delete_handle, out_transaction, out_deleted_rows) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn delete_finish_uncommitted_inner(
    delete_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> FfiResult<()> {
    if delete_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "delete_handle is null",
        ));
    }
    // Takes ownership: the handle is released on every path.
    let handle = unsafe { Box::from_raw(delete_handle as *mut DeleteHandle) };
    if out_transaction.is_null() || out_deleted_rows.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction/out_deleted_rows is null",
        ));
    }

    let (maybe_txn, deleted_rows) =
        match runtime::block_on(row_id_delete_transaction(handle.dataset, handle.row_ids)) {
            Ok(Ok(v)) => v,
            Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetDelete, message)),
            Err(err) => {
                return Err(FfiError::new(
                    ErrorCode::DatasetDelete,
                    format!("runtime: {err}"),
                ))
            }
        };
    unsafe { write_delete_result(out_transaction, out_deleted_rows, maybe_txn, deleted_rows) };
    Ok(())
}

async fn row_id_delete_transaction(
    dataset: Arc<Dataset>,
    row_ids: RoaringTreemap,
) -> Result<(Option<UncommittedTransaction>, i64), String> {
    if row_ids.is_empty() {
        return Ok((None, 0));
    }

    let row_addrs = if dataset.manifest.uses_stable_row_ids() {
        let row_id_index = build_row_id_index(dataset.as_ref()).await?;
        let mut addrs = RoaringTreemap::new();
        for row_id in row_ids.iter() {
            addrs.insert(lookup_row_address(&row_id_index, row_id)?);
        }
        addrs
    } else {
        row_ids
    };

    // `apply_deletions` skips fragments it does not find, which would report
    // rows as deleted that are still there.
    let fragment_ids: HashSet<u64> = dataset.manifest.fragments.iter().map(|f| f.id).collect();
    if let Some((fragment_id, _)) = row_addrs
        .bitmaps()
        .find(|(id, _)| !fragment_ids.contains(&u64::from(*id)))
    {
        return Err(format!(
            "row id refers to fragment {fragment_id}, which is not in dataset version {}",
            dataset.manifest.version
        ));
    }

    let deleted_rows =
        i64::try_from(row_addrs.len()).map_err(|_| "deleted row count overflow".to_string())?;
    let (updated_fragments, deleted_fragment_ids) =
        apply_deletions(dataset.as_ref(), &row_addrs).await?;
    let operation = Operation::Delete {
        updated_fragments,
        deleted_fragment_ids,
        // DuckDB selected the rows; there is no Lance predicate to record.
        predicate: String::new(),
    };
    let txn = Transaction::new(dataset.manifest.version, operation, None);
    // Lance rebases the commit over concurrent deletes of other rows only
    // when it knows the rows this delete removes.
    let mut txn = UncommittedTransaction::new(txn);
    txn.affected_rows = Some(RowAddrTreeMap::from(row_addrs));
    Ok((Some(txn), deleted_rows))
}

/// A plain `DELETE FROM t`: remove every fragment of the dataset version
/// pinned for the statement, without streaming its row ids.
#[no_mangle]
pub unsafe extern "C" fn lance_delete_all_uncommitted(
    dataset: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> i32 {
    match delete_all_uncommitted_inner(dataset, out_transaction, out_deleted_rows) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn delete_all_uncommitted_inner(
    dataset: *mut c_void,
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
) -> FfiResult<()> {
    if out_transaction.is_null() || out_deleted_rows.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction/out_deleted_rows is null",
        ));
    }
    let dataset = unsafe { dataset_handle(dataset)? }.dataset.clone();
    let deleted_fragment_ids: Vec<u64> = dataset.manifest.fragments.iter().map(|f| f.id).collect();
    if deleted_fragment_ids.is_empty() {
        unsafe { write_delete_result(out_transaction, out_deleted_rows, None, 0) };
        return Ok(());
    }

    let deleted_rows = match runtime::block_on(dataset.count_rows(None)) {
        Ok(Ok(rows)) => rows,
        Ok(Err(err)) => return Err(FfiError::new(ErrorCode::DatasetDelete, err.to_string())),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDelete,
                format!("runtime: {err}"),
            ))
        }
    };
    let deleted_rows = i64::try_from(deleted_rows)
        .map_err(|_| FfiError::new(ErrorCode::DatasetDelete, "deleted row count overflow"))?;
    let operation = Operation::Delete {
        updated_fragments: Vec::new(),
        deleted_fragment_ids,
        predicate: "true".to_string(),
    };
    // Removing whole fragments has no row addresses to rebase with, as for
    // Lance's own delete of everything.
    let txn = Transaction::new(dataset.manifest.version, operation, None);
    unsafe {
        write_delete_result(
            out_transaction,
            out_deleted_rows,
            Some(UncommittedTransaction::new(txn)),
            deleted_rows,
        )
    };
    Ok(())
}

unsafe fn write_delete_result(
    out_transaction: *mut *mut c_void,
    out_deleted_rows: *mut i64,
    txn: Option<UncommittedTransaction>,
    deleted_rows: i64,
) {
    unsafe {
        *out_deleted_rows = deleted_rows;
        *out_transaction = match txn {
            Some(txn) => txn.into_raw(),
            None => std::ptr::null_mut(),
        };
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_delete_abort(delete_handle: *mut c_void) {
    if delete_handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(delete_handle as *mut DeleteHandle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Int64Type;
    use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
    use futures::TryStreamExt;
    use lance::dataset::{WriteMode, WriteParams};

    use crate::ffi::types::DatasetHandle;

    /// Rows `id` 0..8 in two fragments of four. With stable row ids, `_rowid`
    /// equals `id`, so in the second fragment it differs from the address.
    fn write_dataset(tag: &str, stable_row_ids: bool) -> (String, Arc<Dataset>) {
        let dir =
            std::env::temp_dir().join(format!("lance-delete-{tag}-{}", rand::random::<u64>()));
        let uri = dir.to_string_lossy().to_string();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from_iter_values(0..8))],
        )
        .unwrap();
        let params = WriteParams {
            mode: WriteMode::Create,
            max_rows_per_file: 4,
            enable_stable_row_ids: stable_row_ids,
            ..Default::default()
        };
        let dataset = runtime::block_on(Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
            &uri,
            Some(params),
        ))
        .unwrap()
        .unwrap();
        assert_eq!(dataset.get_fragments().len(), 2);
        (uri, Arc::new(dataset))
    }

    fn remaining_ids(uri: &str) -> Vec<i64> {
        let dataset = runtime::block_on(Dataset::open(uri)).unwrap().unwrap();
        let batches: Vec<RecordBatch> = runtime::block_on(async move {
            dataset.scan().try_into_stream().await?.try_collect().await
        })
        .unwrap()
        .unwrap();
        let mut ids: Vec<i64> = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Stages a DELETE of `row_ids` on `dataset` through the FFI.
    fn stage_delete(dataset: &DatasetHandle, row_ids: &[u64]) -> (*mut c_void, i64) {
        let mut delete = std::ptr::null_mut();
        let mut txn = std::ptr::null_mut();
        let mut deleted_rows = 0;
        unsafe {
            assert_eq!(
                lance_delete_begin(dataset as *const DatasetHandle as *mut c_void, &mut delete),
                0
            );
            assert_eq!(
                lance_delete_add_rowids(delete, row_ids.as_ptr(), row_ids.len()),
                0
            );
            assert_eq!(
                lance_delete_finish_uncommitted(delete, &mut txn, &mut deleted_rows),
                0
            );
        }
        (txn, deleted_rows)
    }

    /// Commits a staged transaction to `dataset`, as DuckDB's COMMIT does.
    fn commit_to(dataset: &DatasetHandle, txn: *mut c_void) -> i32 {
        unsafe {
            crate::ffi::write::lance_commit_transaction_to_dataset(
                dataset as *const DatasetHandle as *mut c_void,
                txn,
            )
        }
    }

    #[test]
    fn delete_maps_stable_row_ids_to_addresses() {
        let (uri, dataset) = write_dataset("stable", true);
        let dataset = DatasetHandle::new(dataset);
        let mut delete = std::ptr::null_mut();
        let mut txn = std::ptr::null_mut();
        let mut deleted_rows = 0;
        unsafe {
            assert_eq!(
                lance_delete_begin(&dataset as *const DatasetHandle as *mut c_void, &mut delete),
                0
            );
            // Unsorted, with a duplicate, across two chunks.
            for chunk in [&[6_u64, 1, 6][..], &[5][..]] {
                assert_eq!(
                    lance_delete_add_rowids(delete, chunk.as_ptr(), chunk.len()),
                    0
                );
            }
            assert_eq!(
                lance_delete_finish_uncommitted(delete, &mut txn, &mut deleted_rows),
                0
            );
        }
        assert_eq!(deleted_rows, 3);

        assert_eq!(commit_to(&dataset, txn), 0);
        assert_eq!(remaining_ids(&uri), vec![0, 2, 3, 4, 7]);
    }

    #[test]
    fn delete_rejects_row_ids_outside_the_dataset_version() {
        let (_uri, dataset) = write_dataset("unknown-fragment", false);
        let dataset = DatasetHandle::new(dataset);
        let row_ids = [7_u64 << 32];
        let mut delete = std::ptr::null_mut();
        let mut txn = std::ptr::null_mut();
        let mut deleted_rows = 0;
        unsafe {
            assert_eq!(
                lance_delete_begin(&dataset as *const DatasetHandle as *mut c_void, &mut delete),
                0
            );
            assert_eq!(
                lance_delete_add_rowids(delete, row_ids.as_ptr(), row_ids.len()),
                0
            );
            assert_eq!(
                lance_delete_finish_uncommitted(delete, &mut txn, &mut deleted_rows),
                -1
            );
        }
        assert!(txn.is_null());
    }

    #[test]
    fn delete_reads_the_pinned_dataset_version() {
        let (uri, dataset) = write_dataset("pinned", false);
        let pinned_version = dataset.version().version;

        // Another writer commits after the statement pinned its version.
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![100, 101]))],
        )
        .unwrap();
        let params = WriteParams {
            mode: WriteMode::Append,
            ..Default::default()
        };
        runtime::block_on(Dataset::write(
            RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
            &uri,
            Some(params),
        ))
        .unwrap()
        .unwrap();

        let dataset = DatasetHandle::new(dataset);
        let (txn, deleted_rows) = stage_delete(&dataset, &[1]);
        assert_eq!(deleted_rows, 1);
        let read_version = unsafe { &*(txn as *const UncommittedTransaction) }
            .transaction
            .read_version;
        assert_eq!(read_version, pinned_version);
        // Lance rebases the delete over the concurrent append.
        assert_eq!(commit_to(&dataset, txn), 0);
        assert_eq!(remaining_ids(&uri), vec![0, 2, 3, 4, 5, 6, 7, 100, 101]);
    }

    #[test]
    fn delete_rebases_over_a_concurrent_delete_of_other_rows() {
        for stable_row_ids in [false, true] {
            let (uri, dataset) = write_dataset("concurrent", stable_row_ids);
            let dataset = DatasetHandle::new(dataset);
            let row_ids = |ids: &[u64]| {
                ids.iter()
                    .map(|id| row_id_of(*id, stable_row_ids))
                    .collect::<Vec<_>>()
            };
            // All three stage on the same version, as DELETEs in one
            // transaction or in concurrent connections do, and all touch
            // fragment 0.
            let (first, _) = stage_delete(&dataset, &row_ids(&[1]));
            let (second, _) = stage_delete(&dataset, &row_ids(&[2, 5]));
            let (overlapping, _) = stage_delete(&dataset, &row_ids(&[2, 3]));
            assert_eq!(commit_to(&dataset, first), 0);
            assert_eq!(commit_to(&dataset, second), 0);
            assert_eq!(remaining_ids(&uri), vec![0, 3, 4, 6, 7]);

            // Row 2 was deleted by a transaction committed meanwhile.
            assert_eq!(commit_to(&dataset, overlapping), -1);
            let message = last_error_message();
            assert!(message.contains("Retryable commit conflict"), "{message}");
            assert_eq!(remaining_ids(&uri), vec![0, 3, 4, 6, 7]);
        }
    }

    #[test]
    fn rebased_deletes_that_empty_a_fragment_remove_it() {
        let (uri, dataset) = write_dataset("empty-fragment", false);
        let dataset = DatasetHandle::new(dataset);
        let (first, _) = stage_delete(&dataset, &[0, 1]);
        let (second, _) = stage_delete(&dataset, &[2, 3]);
        assert_eq!(commit_to(&dataset, first), 0);
        assert_eq!(commit_to(&dataset, second), 0);
        assert_eq!(remaining_ids(&uri), vec![4, 5, 6, 7]);
        let latest = runtime::block_on(Dataset::open(&uri)).unwrap().unwrap();
        let fragment_ids: Vec<u64> = latest.manifest.fragments.iter().map(|f| f.id).collect();
        assert_eq!(fragment_ids, vec![1]);
    }

    /// The `_rowid` DuckDB's scan returns for `id` in `write_dataset`: the
    /// stable row id when the dataset has them, otherwise the row address.
    fn row_id_of(id: u64, stable_row_ids: bool) -> u64 {
        if stable_row_ids {
            id
        } else {
            ((id / 4) << 32) | (id % 4)
        }
    }

    fn last_error_message() -> String {
        unsafe {
            let ptr = crate::error::lance_last_error_message();
            let text = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
            crate::error::lance_free_string(ptr);
            text
        }
    }
}
