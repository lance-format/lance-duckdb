use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::datatypes::DataType;
use arrow_array::{make_array, RecordBatch, StructArray};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::update::{apply_deletions, build_row_id_index};
use super::util::{cstr_to_str, slice_from_ptr, FfiError, FfiResult};

struct MergeHandle {
    input_schema: Arc<arrow_schema::Schema>,
    data_type: DataType,
    path: String,
    storage_options: HashMap<String, String>,
    max_rows_per_file: usize,
    max_rows_per_group: usize,
    max_bytes_per_file: usize,
    delete_row_ids: RoaringTreemap,
    insert_batches: Vec<RecordBatch>,
}

#[repr(C)]
struct RawArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut RawArrowArray,
    dictionary: *mut RawArrowArray,
    release: Option<unsafe extern "C" fn(arg1: *mut RawArrowArray)>,
    private_data: *mut c_void,
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

    let input_schema: Arc<arrow_schema::Schema> = match runtime::block_on(async {
        DatasetBuilder::from_uri(path.as_str())
            .with_storage_options(storage_options.clone())
            .load()
            .await
            .map_err(|e| e.to_string())
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

    let data_type = DataType::Struct(input_schema.fields().clone());
    let handle = Box::new(MergeHandle {
        input_schema,
        data_type,
        path,
        storage_options,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        delete_row_ids: RoaringTreemap::new(),
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
pub unsafe extern "C" fn lance_merge_add_insert_batch(
    merge_handle: *mut c_void,
    array: *mut c_void,
) -> i32 {
    match merge_add_insert_batch_inner(merge_handle, array) {
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

fn merge_add_insert_batch_inner(merge_handle: *mut c_void, array: *mut c_void) -> FfiResult<()> {
    if merge_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "merge_handle is null",
        ));
    }
    if array.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "array is null"));
    }

    let handle = unsafe { &mut *(merge_handle as *mut MergeHandle) };

    let raw_array = unsafe { ptr::read(array as *mut RawArrowArray) };
    unsafe {
        (*(array as *mut RawArrowArray)).release = None;
    }
    let ffi_array: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(raw_array) };

    let array_data =
        unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, handle.data_type.clone()) }
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetMerge, format!("array import: {err}"))
            })?;

    let array = make_array(array_data);
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetMerge, "array is not a struct"))?;

    let batch = RecordBatch::try_new(handle.input_schema.clone(), struct_array.columns().to_vec())
        .map_err(|err| FfiError::new(ErrorCode::DatasetMerge, format!("record batch: {err}")))?;

    handle.insert_batches.push(batch);
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
        let dataset = DatasetBuilder::from_uri(handle.path.as_str())
            .with_storage_options(handle.storage_options.clone())
            .load()
            .await
            .map_err(|e| e.to_string())?;
        let dataset = Arc::new(dataset);

        let mut new_fragments = Vec::new();
        if !handle.insert_batches.is_empty() {
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
                store_params: Some(store_params),
                ..Default::default()
            };

            let stream = futures::stream::iter(handle.insert_batches.into_iter().map(Ok)).boxed();
            let stream: SendableRecordBatchStream = Box::pin(RecordBatchStreamAdapter::new(
                handle.input_schema.clone(),
                stream,
            ));

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
                    let addr = row_id_index
                        .get(row_id)
                        .ok_or_else(|| format!("row id missing from row id index: {row_id}"))?;
                    row_addrs.insert(u64::from(addr));
                }
            } else {
                row_addrs = handle.delete_row_ids;
            }
        }

        let (updated_fragments, removed_fragment_ids) = if row_addrs.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            apply_deletions(dataset.as_ref(), &row_addrs)
                .await
                .map_err(|e| e.to_string())?
        };

        if new_fragments.is_empty()
            && updated_fragments.is_empty()
            && removed_fragment_ids.is_empty()
        {
            return Ok::<_, String>(None);
        }

        let operation = Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified: vec![],
            merged_generations: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteRows),
            inserted_rows_filter: None,
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
