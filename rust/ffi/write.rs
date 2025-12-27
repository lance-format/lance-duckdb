use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Mutex;
use std::sync::RwLock;
use std::thread::JoinHandle;

use arrow_array::{make_array, RecordBatch, RecordBatchReader, StructArray};
use arrow_schema::{ArrowError, DataType, Schema, SchemaRef};
use lance::dataset::{CommitBuilder, Dataset, InsertBuilder, WriteMode, WriteParams};
use lance::io::ObjectStoreParams;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, slice_from_ptr, FfiError, FfiResult};

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

struct ReceiverRecordBatchReader {
    schema: SchemaRef,
    receiver: Receiver<RecordBatch>,
}

impl ReceiverRecordBatchReader {
    fn new(schema: SchemaRef, receiver: Receiver<RecordBatch>) -> Self {
        Self { schema, receiver }
    }
}

impl Iterator for ReceiverRecordBatchReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(batch) => Some(Ok(batch)),
            Err(_) => None,
        }
    }
}

impl RecordBatchReader for ReceiverRecordBatchReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

struct WriterHandle {
    schema: SchemaRef,
    data_type: DataType,
    sender: RwLock<Option<SyncSender<RecordBatch>>>,
    join: Mutex<Option<JoinHandle<Result<WriterResult, String>>>>,
    batches_sent: AtomicU64,
}

enum WriterResult {
    Committed,
    Uncommitted(lance::dataset::transaction::Transaction),
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        let sender = self
            .sender
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        drop(sender);
        let mut guard = self
            .join
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(join) = guard.take() {
            let _ = join.join();
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_writer_with_storage_options(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    schema: *const c_void,
) -> *mut c_void {
    match open_writer_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        schema,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_uncommitted_writer_with_storage_options(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    schema: *const c_void,
) -> *mut c_void {
    match open_uncommitted_writer_inner(
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        schema,
    ) {
        Ok(handle) => {
            clear_last_error();
            Box::into_raw(Box::new(handle)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn open_uncommitted_writer_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    schema: *const c_void,
) -> FfiResult<WriterHandle> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? };

    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }

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

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let data_type = DataType::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let DataType::Struct(fields) = &data_type else {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "schema must be a struct",
        ));
    };
    let schema: SchemaRef = std::sync::Arc::new(Schema::new(fields.clone()));

    let write_mode = WriteMode::try_from(mode).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid write mode '{mode}': {err}"),
        )
    })?;

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;

    let (sender, receiver) = sync_channel::<RecordBatch>(2);

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options = Some(storage_options);
    }

    let params = WriteParams {
        mode: write_mode,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        store_params: Some(store_params),
        ..Default::default()
    };

    let schema_for_thread = schema.clone();
    let join = std::thread::spawn(move || -> Result<WriterResult, String> {
        let reader = ReceiverRecordBatchReader::new(schema_for_thread, receiver);
        let source: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        let builder = InsertBuilder::new(path.as_str()).with_params(&params);
        let fut = builder.execute_uncommitted_stream(source);
        match runtime::block_on(fut) {
            Ok(Ok(txn)) => Ok(WriterResult::Uncommitted(txn)),
            Ok(Err(err)) => Err(err.to_string()),
            Err(err) => Err(format!("runtime: {err}")),
        }
    });

    Ok(WriterHandle {
        schema,
        data_type,
        sender: RwLock::new(Some(sender)),
        join: Mutex::new(Some(join)),
        batches_sent: AtomicU64::new(0),
    })
}

fn open_writer_inner(
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    schema: *const c_void,
) -> FfiResult<WriterHandle> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let mode = unsafe { cstr_to_str(mode, "mode")? };

    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }

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

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let data_type = DataType::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::DatasetWriteOpen, format!("schema import: {err}"))
    })?;
    let DataType::Struct(fields) = &data_type else {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "schema must be a struct",
        ));
    };
    let schema: SchemaRef = std::sync::Arc::new(Schema::new(fields.clone()));

    let write_mode = WriteMode::try_from(mode).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid write mode '{mode}': {err}"),
        )
    })?;

    let max_rows_per_file = usize::try_from(max_rows_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_file: {err}"),
        )
    })?;
    let max_rows_per_group = usize::try_from(max_rows_per_group).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_rows_per_group: {err}"),
        )
    })?;
    let max_bytes_per_file = usize::try_from(max_bytes_per_file).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid max_bytes_per_file: {err}"),
        )
    })?;

    let (sender, receiver) = sync_channel::<RecordBatch>(2);

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options = Some(storage_options);
    }

    let params = WriteParams {
        mode: write_mode,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        store_params: Some(store_params),
        ..Default::default()
    };

    let schema_for_thread = schema.clone();
    let join = std::thread::spawn(move || -> Result<WriterResult, String> {
        let reader = ReceiverRecordBatchReader::new(schema_for_thread, receiver);
        let fut = Dataset::write(reader, &path, Some(params));
        match runtime::block_on(fut) {
            Ok(Ok(_)) => Ok(WriterResult::Committed),
            Ok(Err(err)) => Err(err.to_string()),
            Err(err) => Err(format!("runtime: {err}")),
        }
    });

    Ok(WriterHandle {
        schema,
        data_type,
        sender: RwLock::new(Some(sender)),
        join: Mutex::new(Some(join)),
        batches_sent: AtomicU64::new(0),
    })
}

#[no_mangle]
pub unsafe extern "C" fn lance_writer_write_batch(writer: *mut c_void, array: *mut c_void) -> i32 {
    match writer_write_batch_inner(writer, array) {
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

fn writer_write_batch_inner(writer: *mut c_void, array: *mut c_void) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }
    if array.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "array is null"));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };
    let sender = {
        let guard = handle
            .sender
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .as_ref()
            .ok_or_else(|| {
                FfiError::new(ErrorCode::DatasetWriteBatch, "writer is already finished")
            })?
            .clone()
    };

    let raw_array = unsafe { ptr::read(array as *mut RawArrowArray) };
    unsafe {
        (*(array as *mut RawArrowArray)).release = None;
    }

    let ffi_array: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(raw_array) };

    let array_data =
        unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, handle.data_type.clone()) }
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetWriteBatch, format!("array import: {err}"))
            })?;
    let array = make_array(array_data);
    let struct_array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| FfiError::new(ErrorCode::DatasetWriteBatch, "array is not a struct"))?;

    let batch = RecordBatch::try_new(handle.schema.clone(), struct_array.columns().to_vec())
        .map_err(|err| {
            FfiError::new(ErrorCode::DatasetWriteBatch, format!("record batch: {err}"))
        })?;

    sender.send(batch).map_err(|_| {
        FfiError::new(
            ErrorCode::DatasetWriteBatch,
            "writer background task exited",
        )
    })?;

    handle.batches_sent.fetch_add(1, Ordering::Relaxed);

    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_writer_finish(writer: *mut c_void) -> i32 {
    match writer_finish_inner(writer) {
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

fn writer_finish_inner(writer: *mut c_void) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };

    if handle.batches_sent.load(Ordering::Acquire) == 0 {
        let sender = {
            let guard = handle
                .sender
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.clone()
        };
        if let Some(sender) = sender {
            let empty = RecordBatch::new_empty(handle.schema.clone());
            sender.send(empty).map_err(|_| {
                FfiError::new(
                    ErrorCode::DatasetWriteFinish,
                    "writer background task exited",
                )
            })?;
        }
    }

    let sender = handle
        .sender
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    drop(sender);

    let join = {
        let mut guard = handle
            .join
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take().ok_or_else(|| {
            FfiError::new(ErrorCode::DatasetWriteFinish, "writer is already finished")
        })?
    };

    match join.join() {
        Ok(Ok(WriterResult::Committed)) => Ok(()),
        Ok(Ok(WriterResult::Uncommitted(_))) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            "writer returned an uncommitted transaction",
        )),
        Ok(Err(message)) => Err(FfiError::new(ErrorCode::DatasetWriteFinish, message)),
        Err(_) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            "writer thread panicked",
        )),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_writer_finish_uncommitted(
    writer: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> i32 {
    match writer_finish_uncommitted_inner(writer, out_transaction) {
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

fn writer_finish_uncommitted_inner(
    writer: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> FfiResult<()> {
    if writer.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "writer is null"));
    }
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }

    let handle = unsafe { &*(writer as *const WriterHandle) };

    if handle.batches_sent.load(Ordering::Acquire) == 0 {
        let sender = {
            let guard = handle
                .sender
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.clone()
        };
        if let Some(sender) = sender {
            let empty = RecordBatch::new_empty(handle.schema.clone());
            sender.send(empty).map_err(|_| {
                FfiError::new(
                    ErrorCode::DatasetWriteFinishUncommitted,
                    "writer background task exited",
                )
            })?;
        }
    }

    let sender = handle
        .sender
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    drop(sender);

    let join = {
        let mut guard = handle
            .join
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.take().ok_or_else(|| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer is already finished",
            )
        })?
    };

    let txn = match join.join() {
        Ok(Ok(WriterResult::Uncommitted(txn))) => txn,
        Ok(Ok(WriterResult::Committed)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer did not return an uncommitted transaction",
            ))
        }
        Ok(Err(message)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                message,
            ))
        }
        Err(_) => {
            return Err(FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer thread panicked",
            ))
        }
    };

    let boxed = Box::new(txn);
    unsafe {
        *out_transaction = Box::into_raw(boxed) as *mut c_void;
    }

    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_writer(writer: *mut c_void) {
    if writer.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(writer as *mut WriterHandle);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_commit_transaction_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    transaction: *mut c_void,
) -> i32 {
    match commit_transaction_inner(path, option_keys, option_values, options_len, transaction) {
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

fn commit_transaction_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    transaction: *mut c_void,
) -> FfiResult<()> {
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }

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

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options = Some(storage_options);
    }

    let txn =
        unsafe { Box::from_raw(transaction as *mut lance::dataset::transaction::Transaction) };
    let fut = CommitBuilder::new(path.as_str())
        .with_store_params(store_params)
        .execute(*txn);
    match runtime::block_on(fut) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            err.to_string(),
        )),
        Err(err) => Err(FfiError::new(
            ErrorCode::DatasetCommitTransaction,
            format!("runtime: {err}"),
        )),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_transaction(transaction: *mut c_void) {
    if transaction.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(transaction as *mut lance::dataset::transaction::Transaction);
    }
}
