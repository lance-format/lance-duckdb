use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;

use arrow_array::builder::{FixedSizeListBuilder, Float32Builder, Float64Builder};
use arrow_array::{
    make_array, Array, FixedSizeListArray, Float32Array, Float64Array, LargeListArray, ListArray,
    RecordBatch, RecordBatchReader, StructArray,
};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
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
    input_schema: SchemaRef,
    data_type: DataType,
    state: Mutex<WriterState>,
    batches_sent: AtomicU64,
}

enum WriterResult {
    Committed,
    Uncommitted(lance::dataset::transaction::Transaction),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterKind {
    Committed,
    Uncommitted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorListKind {
    List,
    LargeList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorElementType {
    Float32,
    Float64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VectorConversion {
    col_idx: usize,
    dim: usize,
    list_kind: VectorListKind,
    element_type: VectorElementType,
}

struct WriterState {
    kind: WriterKind,
    path: String,
    params: WriteParams,

    vector_candidates: Vec<VectorConversion>,
    buffered_batches: Vec<RecordBatch>,

    output_schema: Option<SchemaRef>,
    output_sender: Option<SyncSender<RecordBatch>>,
    output_join: Option<JoinHandle<Result<WriterResult, String>>>,
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        let (sender, join) = {
            let mut guard = self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            (guard.output_sender.take(), guard.output_join.take())
        };
        drop(sender);
        if let Some(join) = join {
            let _ = join.join();
        }
    }
}

const MAX_VECTOR_DIM_INFERENCE_BATCHES: usize = 4;

fn is_variable_list_vector_type(dt: &DataType) -> Option<(VectorListKind, VectorElementType)> {
    match dt {
        DataType::List(field) => match field.data_type() {
            DataType::Float32 => Some((VectorListKind::List, VectorElementType::Float32)),
            DataType::Float64 => Some((VectorListKind::List, VectorElementType::Float64)),
            _ => None,
        },
        DataType::LargeList(field) => match field.data_type() {
            DataType::Float32 => Some((VectorListKind::LargeList, VectorElementType::Float32)),
            DataType::Float64 => Some((VectorListKind::LargeList, VectorElementType::Float64)),
            _ => None,
        },
        _ => None,
    }
}

fn infer_vector_dim_from_array(
    array: &dyn Array,
    list_kind: VectorListKind,
) -> Option<Result<usize, String>> {
    match list_kind {
        VectorListKind::List => {
            let list = array.as_any().downcast_ref::<ListArray>()?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim == 0 {
                    return Some(Err("vector dim must be non-zero".to_string()));
                }
                return Some(Ok(dim));
            }
            None
        }
        VectorListKind::LargeList => {
            let list = array.as_any().downcast_ref::<LargeListArray>()?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim == 0 {
                    return Some(Err("vector dim must be non-zero".to_string()));
                }
                return Some(Ok(dim));
            }
            None
        }
    }
}

fn validate_list_vector_dim(
    array: &dyn Array,
    list_kind: VectorListKind,
    expected_dim: usize,
) -> Result<(), String> {
    match list_kind {
        VectorListKind::List => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim != expected_dim {
                    return Err(format!(
                        "vector dim mismatch: expected {expected_dim} got {dim}"
                    ));
                }
            }
            Ok(())
        }
        VectorListKind::LargeList => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            for i in 0..list.len() {
                if list.is_null(i) {
                    continue;
                }
                let dim = list.value_length(i) as usize;
                if dim != expected_dim {
                    return Err(format!(
                        "vector dim mismatch: expected {expected_dim} got {dim}"
                    ));
                }
            }
            Ok(())
        }
    }
}

fn convert_list_array_to_fixed_size(
    array: &dyn Array,
    list_kind: VectorListKind,
    element_type: VectorElementType,
    dim: usize,
) -> Result<FixedSizeListArray, String> {
    let dim_i32 = i32::try_from(dim).map_err(|_| "vector dim is too large".to_string())?;

    match (list_kind, element_type) {
        (VectorListKind::List, VectorElementType::Float32) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector values are not Float32".to_string())?;
            let field = match list.data_type() {
                DataType::List(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float32Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for i in 0..list.len() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!(
                        "vector dim mismatch: expected {dim} got {len}"
                    ));
                }
                let start = offsets[i] as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::List, VectorElementType::Float64) => {
            let list = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| "vector column is not ListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "vector values are not Float64".to_string())?;
            let field = match list.data_type() {
                DataType::List(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float64Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for i in 0..list.len() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!(
                        "vector dim mismatch: expected {dim} got {len}"
                    ));
                }
                let start = offsets[i] as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::LargeList, VectorElementType::Float32) => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| "vector values are not Float32".to_string())?;
            let field = match list.data_type() {
                DataType::LargeList(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float32Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for i in 0..list.len() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!(
                        "vector dim mismatch: expected {dim} got {len}"
                    ));
                }
                let start = offsets[i] as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
        (VectorListKind::LargeList, VectorElementType::Float64) => {
            let list = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| "vector column is not LargeListArray".to_string())?;
            let values = list
                .values()
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| "vector values are not Float64".to_string())?;
            let field = match list.data_type() {
                DataType::LargeList(field) => field.clone(),
                _ => return Err("vector column has unexpected data type".to_string()),
            };

            let mut builder =
                FixedSizeListBuilder::with_capacity(Float64Builder::new(), dim_i32, list.len())
                    .with_field(field);
            let offsets = list.value_offsets();
            for i in 0..list.len() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!(
                        "vector dim mismatch: expected {dim} got {len}"
                    ));
                }
                let start = offsets[i] as usize;
                for j in 0..dim {
                    let idx = start + j;
                    if idx >= values.len() {
                        return Err("vector offsets are out of bounds".to_string());
                    }
                    if values.is_null(idx) {
                        builder.values().append_null();
                    } else {
                        builder.values().append_value(values.value(idx));
                    }
                }
                builder.append(true);
            }
            Ok(builder.finish())
        }
    }
}

fn build_output_schema(
    input_schema: &SchemaRef,
    conversions: &[VectorConversion],
) -> Result<SchemaRef, String> {
    if conversions.is_empty() {
        return Ok(input_schema.clone());
    }
    let mut fields = input_schema.fields().as_ref().to_vec();
    for conv in conversions {
        let idx = conv.col_idx;
        if idx >= fields.len() {
            return Err("vector column index is out of bounds".to_string());
        }
        let original = fields[idx].as_ref();
        let (list_kind, element_type) = is_variable_list_vector_type(original.data_type())
            .ok_or_else(|| "vector column has unexpected data type".to_string())?;
        if list_kind != conv.list_kind || element_type != conv.element_type {
            return Err("vector column has unexpected data type".to_string());
        }
        let child_field = match original.data_type() {
            DataType::List(field) | DataType::LargeList(field) => field.clone(),
            _ => return Err("vector column has unexpected data type".to_string()),
        };
        let dim_i32 =
            i32::try_from(conv.dim).map_err(|_| "vector dim is too large".to_string())?;
        fields[idx] = Arc::new(Field::new(
            original.name(),
            DataType::FixedSizeList(child_field, dim_i32),
            original.is_nullable(),
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn convert_record_batch(
    input_batch: &RecordBatch,
    output_schema: &SchemaRef,
    conversions: &[VectorConversion],
) -> Result<RecordBatch, String> {
    if conversions.is_empty() {
        return Ok(RecordBatch::try_new(output_schema.clone(), input_batch.columns().to_vec())
            .map_err(|e| e.to_string())?);
    }
    let mut cols = input_batch.columns().to_vec();
    for conv in conversions {
        let arr = cols
            .get(conv.col_idx)
            .ok_or_else(|| "vector column index is out of bounds".to_string())?
            .as_ref();
        validate_list_vector_dim(arr, conv.list_kind, conv.dim)?;
        let fixed = convert_list_array_to_fixed_size(arr, conv.list_kind, conv.element_type, conv.dim)?;
        cols[conv.col_idx] = Arc::new(fixed);
    }
    RecordBatch::try_new(output_schema.clone(), cols).map_err(|e| e.to_string())
}

fn spawn_writer_thread(
    kind: WriterKind,
    path: String,
    params: WriteParams,
    schema: SchemaRef,
    receiver: Receiver<RecordBatch>,
) -> JoinHandle<Result<WriterResult, String>> {
    std::thread::spawn(move || -> Result<WriterResult, String> {
        let reader = ReceiverRecordBatchReader::new(schema, receiver);
        match kind {
            WriterKind::Committed => {
                let fut = Dataset::write(reader, &path, Some(params));
                match runtime::block_on(fut) {
                    Ok(Ok(_)) => Ok(WriterResult::Committed),
                    Ok(Err(err)) => Err(err.to_string()),
                    Err(err) => Err(format!("runtime: {err}")),
                }
            }
            WriterKind::Uncommitted => {
                let source: Box<dyn RecordBatchReader + Send> = Box::new(reader);
                let builder = InsertBuilder::new(path.as_str()).with_params(&params);
                let fut = builder.execute_uncommitted_stream(source);
                match runtime::block_on(fut) {
                    Ok(Ok(txn)) => Ok(WriterResult::Uncommitted(txn)),
                    Ok(Err(err)) => Err(err.to_string()),
                    Err(err) => Err(format!("runtime: {err}")),
                }
            }
        }
    })
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
    let mut vector_candidates = Vec::<VectorConversion>::new();
    for (idx, field) in schema.fields().iter().enumerate() {
        if let Some((list_kind, element_type)) = is_variable_list_vector_type(field.data_type()) {
            vector_candidates.push(VectorConversion {
                col_idx: idx,
                dim: 0,
                list_kind,
                element_type,
            });
        }
    }

    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Uncommitted,
            path,
            params,
            vector_candidates,
            buffered_batches: Vec::new(),
            output_schema: None,
            output_sender: None,
            output_join: None,
        }),
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

    let mut vector_candidates = Vec::<VectorConversion>::new();
    for (idx, field) in schema.fields().iter().enumerate() {
        if let Some((list_kind, element_type)) = is_variable_list_vector_type(field.data_type()) {
            vector_candidates.push(VectorConversion {
                col_idx: idx,
                dim: 0,
                list_kind,
                element_type,
            });
        }
    }

    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind: WriterKind::Committed,
            path,
            params,
            vector_candidates,
            buffered_batches: Vec::new(),
            output_schema: None,
            output_sender: None,
            output_join: None,
        }),
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

    let input_batch = RecordBatch::try_new(handle.input_schema.clone(), struct_array.columns().to_vec())
        .map_err(|err| {
            FfiError::new(ErrorCode::DatasetWriteBatch, format!("record batch: {err}"))
        })?;

    let (sender, to_send) = {
        let mut guard = handle.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        if guard.output_sender.is_none() {
            guard.buffered_batches.push(input_batch);

            if !guard.vector_candidates.is_empty() {
                let batches = guard.buffered_batches.clone();
                for cand in guard.vector_candidates.iter_mut() {
                    if cand.dim != 0 {
                        continue;
                    }
                    for batch in batches.iter() {
                        let arr = batch
                            .column(cand.col_idx)
                            .as_ref();
                        match infer_vector_dim_from_array(arr, cand.list_kind) {
                            Some(Ok(dim)) => {
                                cand.dim = dim;
                                break;
                            }
                            Some(Err(e)) => {
                                return Err(FfiError::new(ErrorCode::DatasetWriteBatch, e));
                            }
                            None => {}
                        }
                    }
                }

                let batches = guard.buffered_batches.clone();
                for cand in guard.vector_candidates.iter() {
                    if cand.dim == 0 {
                        continue;
                    }
                    for batch in batches.iter() {
                        let arr = batch.column(cand.col_idx).as_ref();
                        if let Err(e) = validate_list_vector_dim(arr, cand.list_kind, cand.dim) {
                            return Err(FfiError::new(ErrorCode::DatasetWriteBatch, e));
                        }
                    }
                }
            }

            let can_start = guard.vector_candidates.iter().all(|c| c.dim != 0)
                || guard.buffered_batches.len() >= MAX_VECTOR_DIM_INFERENCE_BATCHES;
            if can_start {
                let conversions: Vec<VectorConversion> = guard
                    .vector_candidates
                    .iter()
                    .filter(|c| c.dim != 0)
                    .cloned()
                    .collect();

                let output_schema = build_output_schema(&handle.input_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                let (sender, receiver) = sync_channel::<RecordBatch>(2);
                let join = spawn_writer_thread(
                    guard.kind,
                    guard.path.clone(),
                    guard.params.clone(),
                    output_schema.clone(),
                    receiver,
                );

                let buffered = std::mem::take(&mut guard.buffered_batches);
                let mut out_batches = Vec::with_capacity(buffered.len());
                for b in buffered.iter() {
                    let out = convert_record_batch(b, &output_schema, &conversions)
                        .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                    out_batches.push(out);
                }

                guard.output_schema = Some(output_schema);
                guard.output_sender = Some(sender.clone());
                guard.output_join = Some(join);
                (Some(sender), out_batches)
            } else {
                (None, Vec::new())
            }
        } else {
            let sender = guard.output_sender.as_ref().cloned();
            let schema = guard
                .output_schema
                .as_ref()
                .ok_or_else(|| FfiError::new(ErrorCode::DatasetWriteBatch, "writer is not initialized"))?
                .clone();
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let out = convert_record_batch(&input_batch, &schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
            (sender, vec![out])
        }
    };

    if let Some(sender) = sender {
        for batch in to_send {
            sender.send(batch).map_err(|_| {
                FfiError::new(
                    ErrorCode::DatasetWriteBatch,
                    "writer background task exited",
                )
            })?;
        }
    }

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
    let (sender, join, to_send) = {
        let mut guard = handle.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
            );
            let buffered = std::mem::take(&mut guard.buffered_batches);
            let mut out_batches = Vec::with_capacity(buffered.len() + 1);
            for b in buffered.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_batches = out_batches;
        }

        let sender = guard
            .output_sender
            .as_ref()
            .cloned()
            .ok_or_else(|| FfiError::new(ErrorCode::DatasetWriteFinish, "writer is not initialized"))?;
        let join = guard
            .output_join
            .take()
            .ok_or_else(|| FfiError::new(ErrorCode::DatasetWriteFinish, "writer is already finished"))?;
        let to_send = std::mem::take(&mut guard.buffered_batches);
        guard.output_sender = None;
        (sender, join, to_send)
    };

    for b in to_send {
        sender.send(b).map_err(|_| {
            FfiError::new(
                ErrorCode::DatasetWriteFinish,
                "writer background task exited",
            )
        })?;
    }
    drop(sender);

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
    let (sender, join, to_send) = {
        let mut guard = handle.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.output_sender.is_none() {
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let output_schema = build_output_schema(&handle.input_schema, &conversions)
                .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
            let (sender, receiver) = sync_channel::<RecordBatch>(2);
            let join = spawn_writer_thread(
                guard.kind,
                guard.path.clone(),
                guard.params.clone(),
                output_schema.clone(),
                receiver,
            );
            let buffered = std::mem::take(&mut guard.buffered_batches);
            let mut out_batches = Vec::with_capacity(buffered.len() + 1);
            for b in buffered.iter() {
                let out = convert_record_batch(b, &output_schema, &conversions)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?;
                out_batches.push(out);
            }
            if handle.batches_sent.load(Ordering::Acquire) == 0 {
                out_batches.push(RecordBatch::new_empty(output_schema.clone()));
            }
            guard.output_schema = Some(output_schema);
            guard.output_sender = Some(sender.clone());
            guard.output_join = Some(join);
            guard.buffered_batches = out_batches;
        }

        let sender = guard
            .output_sender
            .as_ref()
            .cloned()
            .ok_or_else(|| {
                FfiError::new(
                    ErrorCode::DatasetWriteFinishUncommitted,
                    "writer is not initialized",
                )
            })?;
        let join = guard.output_join.take().ok_or_else(|| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer is already finished",
            )
        })?;
        let to_send = std::mem::take(&mut guard.buffered_batches);
        guard.output_sender = None;
        (sender, join, to_send)
    };

    for b in to_send {
        sender.send(b).map_err(|_| {
            FfiError::new(
                ErrorCode::DatasetWriteFinishUncommitted,
                "writer background task exited",
            )
        })?;
    }
    drop(sender);

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
