use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance::dataset::{ExternalBlobMode, InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance_table::transaction::UpdatedFragmentOffsets;
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::blob::{BlobWriteContext, SourceOptionsMap};
use super::update::{
    apply_deletions, build_row_id_index, compose_column_updates, convert_blob_batch,
    lookup_row_address, UpdateValueBatch,
};
use super::util::{cstr_to_str, optional_session_handle, slice_from_ptr, FfiError, FfiResult};

struct MergeHandle {
    input_schema: Arc<arrow_schema::Schema>,
    path: String,
    session: Option<Arc<lance::session::Session>>,
    storage_options: HashMap<String, String>,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    max_bytes_per_file: usize,
    ingest_batch_budget: usize,
    delete_row_ids: RoaringTreemap,
    update_batches: Vec<UpdateValueBatch>,
    insert_batches: Vec<(RecordBatch, SourceOptionsMap)>,
}

#[no_mangle]
pub unsafe extern "C" fn lance_merge_begin_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_merge_handle: *mut *mut c_void,
) -> i32 {
    match merge_begin_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        ingest_batch_budget,
        session,
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
fn merge_begin_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_merge_handle: *mut *mut c_void,
) -> FfiResult<()> {
    if out_merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_merge_handle is null",
        ));
    }

    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    if options_len > 0 && (option_keys.is_null() || option_values.is_null()) {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "option_keys/option_values is null with non-zero length",
        ));
    }

    let keys = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_keys, options_len, "option_keys")? }
    };
    let values = if options_len == 0 {
        &[][..]
    } else {
        unsafe { slice_from_ptr(option_values, options_len, "option_values")? }
    };

    let mut storage_options = HashMap::<String, String>::new();
    for (idx, (&key_ptr, &val_ptr)) in keys.iter().zip(values.iter()).enumerate() {
        if key_ptr.is_null() || val_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("option key/value is null at index {idx}"),
            ));
        }
        let key = unsafe { CStr::from_ptr(key_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_keys[{idx}] utf8: {err}"))
        })?;
        let value = unsafe { CStr::from_ptr(val_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("option_values[{idx}] utf8: {err}"))
        })?;
        storage_options.insert(key.to_string(), value.to_string());
    }

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
    let session = unsafe { optional_session_handle(session)? };
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

    let input_schema: Arc<arrow_schema::Schema> = match runtime::block_on(async {
        let mut builder =
            DatasetBuilder::from_uri(path.as_str()).with_storage_options(storage_options.clone());
        if let Some(session) = session.clone() {
            builder = builder.with_session(session);
        }
        builder.load().await.map_err(|e| e.to_string())
    }) {
        Ok(Ok(dataset)) => Arc::new(dataset.schema().into()),
        Ok(Err(message)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("open dataset: {message}"),
            ))
        }
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetMerge,
                format!("runtime: {err}"),
            ))
        }
    };

    let handle = Box::new(MergeHandle {
        input_schema,
        path,
        session,
        storage_options,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        ingest_batch_budget,
        delete_row_ids: RoaringTreemap::new(),
        update_batches: Vec::new(),
        insert_batches: Vec::new(),
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
    handle.update_batches.push(UpdateValueBatch {
        columns,
        batch,
        source_options,
    });
    Ok(())
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
    handle.insert_batches.push((batch, source_options));
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

    let handle = unsafe { Box::from_raw(merge_handle as *mut MergeHandle) };

    let maybe_txn = match runtime::block_on(async move {
        let mut builder = DatasetBuilder::from_uri(handle.path.as_str())
            .with_storage_options(handle.storage_options.clone());
        if let Some(session) = handle.session.clone() {
            builder = builder.with_session(session);
        }
        let dataset = builder.load().await.map_err(|e| e.to_string())?;
        let dataset = Arc::new(dataset);

        let existing_schema: arrow_schema::Schema = dataset.schema().into();
        let blob_context = BlobWriteContext {
            registry: handle
                .session
                .as_ref()
                .map(|session| session.store_registry())
                .unwrap_or_default(),
            ingest_batch_budget: handle.ingest_batch_budget,
        };

        let mut new_fragments = Vec::new();
        if !handle.insert_batches.is_empty() {
            let mut insert_batches = Vec::with_capacity(handle.insert_batches.len());
            for (batch, source_options) in handle.insert_batches.iter() {
                insert_batches.push(convert_blob_batch(
                    batch,
                    &existing_schema,
                    &blob_context,
                    source_options,
                )?);
            }

            let mut store_params = ObjectStoreParams::default();
            if !handle.storage_options.is_empty() {
                store_params.storage_options_accessor = Some(Arc::new(
                    StorageOptionsAccessor::with_static_options(handle.storage_options.clone()),
                ));
            }

            let write_params = WriteParams {
                mode: WriteMode::Append,
                max_rows_per_file: handle.max_rows_per_file,
                max_rows_per_group: handle.max_rows_per_group,
                max_bytes_per_file: handle.max_bytes_per_file,
                session: handle.session.clone(),
                store_params: Some(store_params),
                // SQL reference values keep Lance ownership semantics: matching
                // registered bases reuse them, other absolute URIs are stored
                // as-is. Ingest values were materialized above.
                allow_external_blob_outside_bases: true,
                external_blob_mode: ExternalBlobMode::Reference,
                ..Default::default()
            };

            let insert_schema = insert_batches
                .first()
                .map(|batch| batch.schema())
                .unwrap_or_else(|| handle.input_schema.clone());
            let stream = futures::stream::iter(insert_batches.into_iter().map(Ok)).boxed();
            let stream: SendableRecordBatchStream =
                Box::pin(RecordBatchStreamAdapter::new(insert_schema, stream));

            let append_txn = InsertBuilder::new(dataset.clone())
                .with_params(&write_params)
                .execute_uncommitted_stream(stream)
                .await
                .map_err(|e| e.to_string())?;

            let Operation::Append { fragments } = append_txn.operation else {
                return Err("unexpected transaction operation for merge insert write".to_string());
            };
            new_fragments = fragments;
        }

        let mut row_addrs = RoaringTreemap::new();
        if !handle.delete_row_ids.is_empty() {
            if dataset.manifest.uses_stable_row_ids() {
                let row_id_index = build_row_id_index(dataset.as_ref())
                    .await
                    .map_err(|e| e.to_string())?;
                for row_id in handle.delete_row_ids.iter() {
                    row_addrs.insert(lookup_row_address(&row_id_index, row_id)?);
                }
            } else {
                row_addrs = handle.delete_row_ids;
            }
        }

        let mut update_batches = Vec::with_capacity(handle.update_batches.len());
        for mut item in handle.update_batches {
            item.batch = convert_blob_batch(
                &item.batch,
                &existing_schema,
                &blob_context,
                &item.source_options,
            )?;
            update_batches.push(item);
        }

        let compose_result = compose_column_updates(
            &dataset,
            &update_batches,
            &row_addrs,
            handle.ingest_batch_budget,
        )
        .await?;
        let mut updated_fragments = compose_result.0;
        let fields_modified = compose_result.1;
        let fragment_offsets = compose_result.2;

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
            let boxed = Box::new(txn);
            *out_transaction = Box::into_raw(boxed) as *mut c_void;
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
