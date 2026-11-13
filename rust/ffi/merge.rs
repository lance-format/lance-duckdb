use std::ffi::{c_char, c_void};
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::Arc;
use std::thread::JoinHandle;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance::dataset::{ExternalBlobMode, InsertBuilder, WriteMode, WriteParams};
use lance::io::ObjectStoreParams;
use lance_table::format::Fragment;
use lance_table::transaction::UpdatedFragmentOffsets;
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::blob::BlobWriteContext;
use super::stored_type::cast_batch_to_stored;
use super::update::{
    apply_deletions, build_row_id_index, convert_blob_batch, lookup_row_address,
    ColumnUpdateWriter, UpdateValueBatch,
};
use super::util::{dataset_handle, optional_cstr_to_str, slice_from_ptr, FfiError, FfiResult};
use super::write::{ReceiverRecordBatchReader, UncommittedTransaction};

struct MergeHandle {
    // The dataset version the statement's scans read. The transaction is built
    // on it, so Lance detects any commit made since.
    dataset: Arc<lance::Dataset>,
    existing_schema: arrow_schema::Schema,
    blob_context: BlobWriteContext,
    write_params: WriteParams,
    delete_row_ids: RoaringTreemap,
    updates: ColumnUpdateWriter,
    inserts: Option<InsertStream>,
}

/// MERGE inserts streamed to Lance's writer on a background thread through a
/// bounded channel, as INSERT does, instead of buffered until the statement
/// finishes.
struct InsertStream {
    sender: Option<SyncSender<RecordBatch>>,
    join: Option<JoinHandle<Result<Vec<Fragment>, String>>>,
}

impl InsertStream {
    fn start(dataset: Arc<lance::Dataset>, params: WriteParams, schema: SchemaRef) -> Self {
        let (sender, receiver) = sync_channel::<RecordBatch>(2);
        let join = std::thread::spawn(move || -> Result<Vec<Fragment>, String> {
            let source: Box<dyn RecordBatchReader + Send> =
                Box::new(ReceiverRecordBatchReader::new(schema, receiver));
            let builder = InsertBuilder::new(dataset).with_params(&params);
            let txn = runtime::block_on(builder.execute_uncommitted_stream(source))
                .map_err(|err| format!("runtime: {err}"))?
                .map_err(|err| err.to_string())?;
            let Operation::Append { fragments } = txn.operation else {
                return Err("unexpected transaction operation for merge insert write".to_string());
            };
            Ok(fragments)
        });
        Self {
            sender: Some(sender),
            join: Some(join),
        }
    }

    fn send(&mut self, batch: RecordBatch) -> Result<(), String> {
        if let Some(sender) = self.sender.as_ref() {
            if sender.send(batch).is_ok() {
                return Ok(());
            }
        }
        // The writer stopped taking input; report why.
        match self.finish() {
            Err(err) => Err(err),
            Ok(_) => Err("merge insert writer exited".to_string()),
        }
    }

    fn finish(&mut self) -> Result<Vec<Fragment>, String> {
        drop(self.sender.take());
        let join = self
            .join
            .take()
            .ok_or("merge insert writer is already finished")?;
        join.join()
            .map_err(|_| "merge insert writer panicked".to_string())?
    }
}

impl Drop for InsertStream {
    fn drop(&mut self) {
        // An aborted statement drops the uncommitted append.
        if self.join.is_some() {
            let _ = self.finish();
        }
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn lance_merge_begin(
    dataset: *mut c_void,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    ingest_batch_budget: u64,
    spill_directory: *const c_char,
    spill_limit: u64,
    out_merge_handle: *mut *mut c_void,
) -> i32 {
    match merge_begin_inner(
        dataset,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        ingest_batch_budget,
        spill_directory,
        spill_limit,
        out_merge_handle,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn merge_begin_inner(
    dataset: *mut c_void,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    ingest_batch_budget: u64,
    spill_directory: *const c_char,
    spill_limit: u64,
    out_merge_handle: *mut *mut c_void,
) -> FfiResult<()> {
    if out_merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_merge_handle is null",
        ));
    }
    let dataset = unsafe { dataset_handle(dataset)? }.dataset.clone();

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;
    let ingest_batch_budget = if ingest_batch_budget == 0 {
        super::blob::DEFAULT_INGEST_BATCH_BUDGET
    } else {
        usize::try_from(ingest_batch_budget).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("invalid ingest memory budget: {err}"),
            )
        })?
    };

    let spill_root =
        unsafe { optional_cstr_to_str(spill_directory, "spill_directory")? }.map(PathBuf::from);

    // The writer resolves external blob sources with the dataset's own
    // storage options.
    let store_params = ObjectStoreParams {
        storage_options_accessor: dataset.storage_options_accessor(),
        ..Default::default()
    };
    let write_params = WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        session: Some(dataset.session()),
        store_params: Some(store_params),
        // SQL reference values keep Lance ownership semantics: matching
        // registered bases reuse them, other absolute URIs are stored as-is.
        // Ingest values are materialized before they reach the writer.
        allow_external_blob_outside_bases: true,
        external_blob_mode: ExternalBlobMode::Reference,
        ..Default::default()
    };
    let handle = Box::new(MergeHandle {
        existing_schema: dataset.schema().into(),
        blob_context: BlobWriteContext {
            registry: dataset.session().store_registry(),
            ingest_batch_budget,
        },
        write_params,
        delete_row_ids: RoaringTreemap::new(),
        updates: ColumnUpdateWriter::new(
            dataset.clone(),
            ingest_batch_budget,
            spill_root,
            spill_limit,
        ),
        inserts: None,
        dataset,
    });

    unsafe {
        *out_merge_handle = Box::into_raw(handle) as *mut c_void;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_delete_rowids(
    merge_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> i32 {
    match merge_add_delete_rowids_inner(merge_handle, row_ids, row_ids_len) {
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

fn merge_add_delete_rowids_inner(
    merge_handle: *mut c_void,
    row_ids: *const u64,
    row_ids_len: usize,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    if row_ids_len > 0 && row_ids.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "row_ids is null with non-zero length",
        ));
    }

    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let ids = if row_ids_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(row_ids, row_ids_len, "row_ids")? }
    };

    for row_id in ids {
        handle.delete_row_ids.insert(*row_id);
    }

    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_update_batch(
    merge_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> i32 {
    match merge_add_update_batch_inner(
        merge_handle,
        schema,
        array,
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn merge_add_update_batch_inner(
    merge_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let batch = super::update::import_struct_batch(array, schema)?;
    let columns = super::update::take_update_batch_fields(&batch)?;
    let source_options = super::write::parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    )?;
    runtime::block_on(handle.updates.add(UpdateValueBatch {
        columns,
        batch,
        source_options,
    }))
    .map_err(|err| FfiError::new(ErrorCode::DatasetMerge, format!("runtime: {err}")))?
    .map_err(|message| FfiError::new(ErrorCode::DatasetMerge, message))
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_add_insert_batch(
    merge_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> i32 {
    match merge_add_insert_batch_inner(
        merge_handle,
        schema,
        array,
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    ) {
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

#[allow(clippy::too_many_arguments)]
fn merge_add_insert_batch_inner(
    merge_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };
    let batch = super::update::import_struct_batch(array, schema)?;
    let source_options = super::write::parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    )?;
    let batch = convert_blob_batch(
        &batch,
        &handle.existing_schema,
        &handle.blob_context,
        &source_options,
    )
    .and_then(|batch| cast_batch_to_stored(&batch, &handle.existing_schema))
    .map_err(|message| FfiError::new(ErrorCode::DatasetMerge, message))?;
    if handle.inserts.is_none() {
        handle.inserts = Some(InsertStream::start(
            handle.dataset.clone(),
            handle.write_params.clone(),
            batch.schema(),
        ));
    }
    if let Some(inserts) = handle.inserts.as_mut() {
        inserts
            .send(batch)
            .map_err(|message| FfiError::new(ErrorCode::DatasetMerge, message))?;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_finish_uncommitted(
    merge_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> i32 {
    match merge_finish_uncommitted_inner(merge_handle, out_transaction) {
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

fn merge_finish_uncommitted_inner(
    merge_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }

    let MergeHandle {
        dataset,
        delete_row_ids,
        updates,
        inserts,
        ..
    } = *unsafe { Box::from_raw(merge_handle as *mut MergeHandle) };

    let new_fragments = match inserts {
        Some(mut inserts) => inserts
            .finish()
            .map_err(|message| FfiError::new(ErrorCode::DatasetMerge, message))?,
        None => Vec::new(),
    };

    let maybe_txn = match runtime::block_on(async move {
        let mut row_addrs = RoaringTreemap::new();
        if !delete_row_ids.is_empty() {
            if dataset.manifest.uses_stable_row_ids() {
                let row_id_index = build_row_id_index(dataset.as_ref())
                    .await
                    .map_err(|e| e.to_string())?;
                for row_id in delete_row_ids.iter() {
                    row_addrs.insert(lookup_row_address(&row_id_index, row_id)?);
                }
            } else {
                row_addrs = delete_row_ids;
            }
        }

        let (mut updated_fragments, fields_modified, fragment_offsets) =
            updates.finish(&row_addrs).await?;

        let (deleted_fragments, removed_fragment_ids) = if row_addrs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            apply_deletions(dataset.as_ref(), &row_addrs)
                .await
                .map_err(|e| e.to_string())?
        };
        for deleted in deleted_fragments {
            if let Some(existing) = updated_fragments.iter_mut().find(|f| f.id == deleted.id) {
                existing.deletion_file = deleted.deletion_file.clone();
            } else {
                updated_fragments.push(deleted);
            }
        }

        if new_fragments.is_empty()
            && updated_fragments.is_empty()
            && removed_fragment_ids.is_empty()
        {
            return Ok::<_, String>(None);
        }

        let updated_fragment_offsets = if fragment_offsets.is_empty() {
            None
        } else {
            Some(UpdatedFragmentOffsets(fragment_offsets))
        };
        let operation = Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets,
        };
        let txn = Transaction::new(dataset.manifest.version, operation, None);

        Ok::<_, String>(Some(txn))
    }) {
        Ok(Ok(txn)) => txn,
        Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetMerge, message)),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("runtime: {err}"),
            ))
        }
    };

    unsafe {
        *out_transaction = std::ptr::null_mut();
        if let Some(txn) = maybe_txn {
            // No affected rows: an update rewrites columns, and Lance would
            // rebase affected rows as deletions.
            *out_transaction = UncommittedTransaction::new(txn).into_raw();
        }
    }

    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_abort(merge_handle: *mut c_void) {
    if merge_handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(merge_handle as *mut MergeHandle);
    }
}
