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
use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, Schema, SchemaRef};
use lance::dataset::write::ExternalBlobMode;
use lance::dataset::{CommitBuilder, InsertBuilder, WriteMode, WriteParams};
use lance::io::{ObjectStoreParams, StorageOptionsAccessor};
use lance_arrow::json::is_arrow_json_field;
use lance_arrow::BLOB_V2_EXT_NAME;
use lance_core::datatypes::Field as LanceField;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::blob::{
    apply_blob_output_fields, convert_blob_column, plan_blob_conversions, BlobColumnConversion,
    BlobWriteContext, SourceOptionsMap,
};
use super::session::record_commit;
use super::util::{
    cstr_to_str, dataset_handle, optional_session_handle, slice_from_ptr, FfiError, FfiResult,
};

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
    /// The background writer's failure, recorded so a failed channel send can
    /// report the real cause instead of only "writer background task exited".
    writer_error: Arc<Mutex<Option<String>>>,
}

struct WriterResult(Box<lance::dataset::transaction::Transaction>);

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
    #[allow(dead_code)]
    kind: WriterKind,
    path: String,
    params: WriteParams,
    /// The destination's fields when appending to an existing dataset.
    append_reference: Option<Fields>,

    vector_candidates: Vec<VectorConversion>,
    blob_conversions: Vec<BlobColumnConversion>,
    blob_context: BlobWriteContext,
    buffered_batches: Vec<(RecordBatch, SourceOptionsMap)>,
    writer_error: Arc<Mutex<Option<String>>>,

    output_schema: Option<SchemaRef>,
    output_sender: Option<SyncSender<RecordBatch>>,
    output_join: Option<JoinHandle<Result<WriterResult, String>>>,
}

impl WriterState {
    fn start(
        &mut self,
        input_schema: &SchemaRef,
        write_empty: bool,
    ) -> Result<Vec<RecordBatch>, String> {
        let conversions: Vec<VectorConversion> = self
            .vector_candidates
            .iter()
            .filter(|c| c.dim != 0)
            .cloned()
            .collect();
        let output_schema = build_output_schema(
            input_schema,
            &conversions,
            self.append_reference.as_ref(),
            &self.blob_conversions,
        )?;
        // Validate buffered input before starting a task that can publish a dataset.
        let mut batches = self
            .buffered_batches
            .iter()
            .map(|(batch, source_options)| {
                convert_record_batch(
                    batch,
                    &output_schema,
                    &conversions,
                    &self.blob_conversions,
                    &self.blob_context,
                    source_options,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        if write_empty {
            batches.push(RecordBatch::new_empty(output_schema.clone()));
        }
        let (sender, receiver) = sync_channel::<RecordBatch>(2);
        let join = spawn_writer_thread(
            self.path.clone(),
            self.params.clone(),
            output_schema.clone(),
            receiver,
            self.writer_error.clone(),
        );
        self.buffered_batches.clear();
        self.output_schema = Some(output_schema);
        self.output_sender = Some(sender);
        self.output_join = Some(join);
        Ok(batches)
    }
}

impl WriterHandle {
    fn finish(&self) -> Result<WriterResult, String> {
        let (sender, join, batches) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let batches = if state.output_sender.is_none() {
                state.start(
                    &self.input_schema,
                    self.batches_sent.load(Ordering::Acquire) == 0,
                )?
            } else {
                Vec::new()
            };
            let sender = state
                .output_sender
                .take()
                .ok_or("writer is not initialized")?;
            let join = state
                .output_join
                .take()
                .ok_or("writer is already finished")?;
            (sender, join, batches)
        };
        let sent = batches.into_iter().try_for_each(|batch| {
            sender.send(batch).map_err(|_| {
                let detail = self
                    .writer_error
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                match detail {
                    Some(detail) => format!("writer background task failed: {detail}"),
                    None => "writer background task exited".to_string(),
                }
            })
        });
        drop(sender);
        // Always join, including when the worker stopped accepting buffered input.
        let result = join
            .join()
            .map_err(|_| "writer thread panicked".to_string())?;
        sent?;
        result
    }
}

impl Drop for WriterHandle {
    fn drop(&mut self) {
        let (sender, join) = {
            let mut guard = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
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
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
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
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
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
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
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
            for (i, start) in offsets.iter().take(list.len()).enumerate() {
                if list.is_null(i) {
                    for _ in 0..dim {
                        builder.values().append_null();
                    }
                    builder.append(false);
                    continue;
                }
                let len = list.value_length(i) as usize;
                if len != dim {
                    return Err(format!("vector dim mismatch: expected {dim} got {len}"));
                }
                let start = *start as usize;
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

/// Best-effort read of the destination dataset's schema and file version;
/// `None` when the dataset does not exist yet, in which case an append behaves
/// like a create.
fn existing_dataset_context(
    path: &str,
    storage_options: &HashMap<String, String>,
) -> Option<(Schema, String)> {
    let builder = lance::dataset::builder::DatasetBuilder::from_uri(path)
        .with_storage_options(storage_options.clone());
    let dataset = runtime::block_on(builder.load()).ok()?.ok()?;
    let version = dataset
        .manifest
        .data_storage_format
        .lance_file_format()
        .to_manifest_string()
        .to_string();
    Some((Schema::from(dataset.schema()), version))
}

/// Rewrites list child field names in `schema` to match `reference`.
///
/// DuckDB's Arrow exporter names list children `l`; the C++ write paths rewrite
/// them to Arrow's `item` so DataFusion leaves list columns uncoerced and Lance
/// can answer `array_has_*` from a LABEL_LIST index. Datasets written before
/// that change still carry `l`, and Lance rejects an append whose schema
/// disagrees, so an append has to speak whatever the destination already uses.
fn align_list_field_names(fields: &Fields, reference: &Fields) -> Option<Fields> {
    let mut aligned = fields.as_ref().to_vec();
    let mut changed = false;
    for slot in &mut aligned {
        let Some(reference_field) = reference.iter().find(|f| f.name() == slot.name()) else {
            continue;
        };
        let Some(field) = align_list_field(slot, reference_field) else {
            continue;
        };
        *slot = field;
        changed = true;
    }
    changed.then(|| Fields::from(aligned))
}

fn align_list_field(field: &FieldRef, reference: &FieldRef) -> Option<FieldRef> {
    let data_type = match (field.data_type(), reference.data_type()) {
        (DataType::List(child), DataType::List(reference_child)) => {
            let child = align_list_child(child, reference_child)?;
            DataType::List(child)
        }
        (DataType::LargeList(child), DataType::LargeList(reference_child)) => {
            let child = align_list_child(child, reference_child)?;
            DataType::LargeList(child)
        }
        (DataType::Struct(children), DataType::Struct(reference_children)) => {
            DataType::Struct(align_list_field_names(children, reference_children)?)
        }
        _ => return None,
    };
    Some(Arc::new(
        Field::new(field.name(), data_type, field.is_nullable())
            .with_metadata(field.metadata().clone()),
    ))
}

fn align_list_child(child: &FieldRef, reference: &FieldRef) -> Option<FieldRef> {
    let nested = match (child.data_type(), reference.data_type()) {
        (DataType::Struct(children), DataType::Struct(reference_children)) => {
            align_list_field_names(children, reference_children).map(DataType::Struct)
        }
        _ => None,
    };
    if child.name() == reference.name() && nested.is_none() {
        return None;
    }
    Some(Arc::new(
        Field::new(
            reference.name(),
            nested.unwrap_or_else(|| child.data_type().clone()),
            child.is_nullable(),
        )
        .with_metadata(child.metadata().clone()),
    ))
}

/// Restates `fields` the way the destination dataset already records them:
/// list child names, then Arrow extension types.
fn align_append_fields(fields: &Fields, reference: &Fields) -> Option<Fields> {
    let renamed = align_list_field_names(fields, reference);
    align_extension_types(renamed.as_ref().unwrap_or(fields), reference).or(renamed)
}

/// Gives incoming fields the Arrow extension types the destination records.
///
/// Lance checks an append against the destination and writes it under the
/// destination's fields; its check ignores most extensions, so Lance 12
/// recorded the destination's extensions whatever a batch carried. Lance 13's
/// dataset writer still leaves each batch its own extensions, but its file
/// writers reject those that differ from the destination's
/// (lance-format/lance#9478). DuckDB drops extensions it does not model, such
/// as `arrow.fixed_shape_tensor`, adds its own under
/// `arrow_lossless_conversion`, such as `arrow.opaque` on `BIT`, and spells
/// extension metadata its own way, such as `{}` on a `geoarrow.wkb` without a
/// CRS. So wherever Lance gives both sides the same type, a field takes the
/// destination's extension (JSON text only its metadata, as
/// `restated_extension_metadata` explains), which is what Lance 12 recorded;
/// as with Lance 12, the values are not checked against it. Other fields, and
/// those `align_extension_type` sets aside, are left to Lance.
fn align_extension_types(fields: &Fields, reference: &Fields) -> Option<Fields> {
    let mut aligned = fields.as_ref().to_vec();
    let mut changed = false;
    for slot in &mut aligned {
        let Some(reference_field) = reference.iter().find(|f| f.name() == slot.name()) else {
            continue;
        };
        let Some(field) = align_extension_type(slot, reference_field) else {
            continue;
        };
        *slot = field;
        changed = true;
    }
    changed.then(|| Fields::from(aligned))
}

/// Walks `field` against `reference` the way Lance's check pairs them: struct
/// children by name, and list, map and fixed-size list items in place.
fn align_extension_type(field: &FieldRef, reference: &FieldRef) -> Option<FieldRef> {
    // Lance's blob handling reads Blob v2 input by its extension, so it is left
    // as is.
    let blob_v2 = |field: &Field| field.extension_type_name() == Some(BLOB_V2_EXT_NAME);
    if blob_v2(field) || blob_v2(reference) || !same_lance_type(field, reference) {
        return None;
    }
    let data_type = match (field.data_type(), reference.data_type()) {
        (DataType::Struct(children), DataType::Struct(reference_children)) => {
            align_extension_types(children, reference_children).map(DataType::Struct)
        }
        (DataType::List(item), DataType::List(reference_item)) => {
            align_extension_type(item, reference_item).map(DataType::List)
        }
        (DataType::LargeList(item), DataType::LargeList(reference_item)) => {
            align_extension_type(item, reference_item).map(DataType::LargeList)
        }
        // Lance's check compares the item of a fixed-size list of structs
        // whole, extensions included, so that item is left as it is.
        (DataType::FixedSizeList(item, size), DataType::FixedSizeList(reference_item, _))
            if !matches!(item.data_type(), DataType::Struct(_)) =>
        {
            align_extension_type(item, reference_item)
                .map(|item| DataType::FixedSizeList(item, *size))
        }
        (DataType::Map(entries, sorted), DataType::Map(reference_entries, _)) => {
            align_extension_type(entries, reference_entries)
                .map(|entries| DataType::Map(entries, *sorted))
        }
        _ => None,
    };
    let metadata = restated_extension_metadata(field, reference);
    if data_type.is_none() && metadata.is_none() {
        return None;
    }
    Some(Arc::new(
        Field::new(
            field.name(),
            data_type.unwrap_or_else(|| field.data_type().clone()),
            field.is_nullable(),
        )
        .with_metadata(metadata.unwrap_or_else(|| field.metadata().clone())),
    ))
}

/// The metadata `field` needs to carry the destination's extension, or `None`
/// when it already does.
///
/// JSON text keeps its `arrow.json` name, which is what makes Lance convert it
/// to JSONB; the conversion keeps the text's extension metadata, so only that
/// follows the destination.
fn restated_extension_metadata(
    field: &Field,
    reference: &Field,
) -> Option<HashMap<String, String>> {
    let name = if is_arrow_json_field(field) {
        field.extension_type_name()
    } else {
        reference.extension_type_name()
    };
    let metadata = reference.extension_type_metadata();
    if (field.extension_type_name(), field.extension_type_metadata()) == (name, metadata) {
        return None;
    }
    let mut restated = field.metadata().clone();
    restated.remove(EXTENSION_TYPE_NAME_KEY);
    restated.remove(EXTENSION_TYPE_METADATA_KEY);
    if let Some(name) = name {
        restated.insert(EXTENSION_TYPE_NAME_KEY.to_string(), name.to_string());
    }
    if let Some(metadata) = metadata {
        restated.insert(
            EXTENSION_TYPE_METADATA_KEY.to_string(),
            metadata.to_string(),
        );
    }
    Some(restated)
}

/// Whether Lance gives both fields the same logical type, which for JSON and
/// bfloat16 vectors depends on their extensions. `align_extension_type` makes
/// this check again at each child it descends into.
fn same_lance_type(field: &Field, reference: &Field) -> bool {
    matches!(
        (LanceField::try_from(field), LanceField::try_from(reference)),
        (Ok(incoming), Ok(stored)) if incoming.logical_type == stored.logical_type
    )
}

/// `reference` holds the destination's fields when appending; a list column
/// converted to a fixed-size vector takes the extension the destination
/// records for it, which `align_append_fields` could not match on the list.
fn build_output_schema(
    input_schema: &SchemaRef,
    conversions: &[VectorConversion],
    reference: Option<&Fields>,
    blob_conversions: &[BlobColumnConversion],
) -> Result<SchemaRef, String> {
    let blob_schema = apply_blob_output_fields(input_schema, blob_conversions);
    let mut fields = blob_schema.fields().as_ref().to_vec();
    if conversions.is_empty() {
        return Ok(Arc::new(Schema::new(fields)));
    }
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
        let dim_i32 = i32::try_from(conv.dim).map_err(|_| "vector dim is too large".to_string())?;
        let converted = Arc::new(Field::new(
            original.name(),
            DataType::FixedSizeList(child_field, dim_i32),
            original.is_nullable(),
        ));
        fields[idx] = reference
            .and_then(|reference| reference.iter().find(|f| f.name() == converted.name()))
            .and_then(|stored| align_extension_type(&converted, stored))
            .unwrap_or(converted);
    }
    Ok(Arc::new(Schema::new(fields)))
}

fn convert_record_batch(
    input_batch: &RecordBatch,
    output_schema: &SchemaRef,
    conversions: &[VectorConversion],
    blob_conversions: &[BlobColumnConversion],
    blob_context: &BlobWriteContext,
    source_options: &SourceOptionsMap,
) -> Result<RecordBatch, String> {
    if conversions.is_empty() && blob_conversions.is_empty() {
        return RecordBatch::try_new(output_schema.clone(), input_batch.columns().to_vec())
            .map_err(|e| e.to_string());
    }
    let mut cols = input_batch.columns().to_vec();
    // One shared ingest budget for the whole batch conversion, so N ingest
    // columns of one batch cannot accumulate N times the configured budget.
    let mut remaining_budget = blob_context.ingest_batch_budget;
    for conversion in blob_conversions {
        let arr = cols
            .get(conversion.col_idx)
            .ok_or_else(|| "blob column index is out of bounds".to_string())?
            .as_ref();
        let input_schema = input_batch.schema();
        let field = input_schema.field(conversion.col_idx);
        let converted = convert_blob_column(
            field.name(),
            arr,
            conversion.target,
            field.is_nullable(),
            blob_context,
            source_options,
            &mut remaining_budget,
            conversion.output_field.as_ref(),
        )?;
        cols[conversion.col_idx] = converted;
    }
    for conv in conversions {
        let arr = cols
            .get(conv.col_idx)
            .ok_or_else(|| "vector column index is out of bounds".to_string())?
            .as_ref();
        validate_list_vector_dim(arr, conv.list_kind, conv.dim)?;
        let fixed =
            convert_list_array_to_fixed_size(arr, conv.list_kind, conv.element_type, conv.dim)?;
        cols[conv.col_idx] = Arc::new(fixed);
    }
    RecordBatch::try_new(output_schema.clone(), cols).map_err(|e| e.to_string())
}

fn spawn_writer_thread(
    path: String,
    params: WriteParams,
    schema: SchemaRef,
    receiver: Receiver<RecordBatch>,
    writer_error: Arc<Mutex<Option<String>>>,
) -> JoinHandle<Result<WriterResult, String>> {
    std::thread::spawn(move || -> Result<WriterResult, String> {
        let reader = ReceiverRecordBatchReader::new(schema, receiver);
        // Every writer produces an uncommitted transaction. A dropped pipeline
        // simply drops that transaction, so a later batch failure can never
        // publish an earlier batch's data. Committed callers commit explicitly
        // in `writer_finish_inner`.
        let source: Box<dyn RecordBatchReader + Send> = Box::new(reader);
        let builder = InsertBuilder::new(path.as_str()).with_params(&params);
        let fut = builder.execute_uncommitted_stream(source);
        let result = match runtime::block_on(fut) {
            Ok(Ok(txn)) => Ok(WriterResult(Box::new(txn))),
            Ok(Err(err)) => Err(err.to_string()),
            Err(err) => Err(format!("runtime: {err}")),
        };
        if let Err(err) = &result {
            *writer_error
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(err.clone());
        }
        result
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
    data_storage_version: *const c_char,
    session: *mut c_void,
    schema: *const c_void,
    blob_columns: *const u8,
    blob_columns_len: usize,
    ingest_batch_budget: u64,
) -> *mut c_void {
    match open_writer_inner(
        WriterKind::Committed,
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        session,
        schema,
        blob_columns,
        blob_columns_len,
        ingest_batch_budget,
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
    data_storage_version: *const c_char,
    session: *mut c_void,
    schema: *const c_void,
    blob_columns: *const u8,
    blob_columns_len: usize,
    ingest_batch_budget: u64,
) -> *mut c_void {
    match open_writer_inner(
        WriterKind::Uncommitted,
        path,
        mode,
        option_keys,
        option_values,
        options_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        session,
        schema,
        blob_columns,
        blob_columns_len,
        ingest_batch_budget,
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

fn parse_data_storage_version_arg(
    data_storage_version: *const c_char,
) -> FfiResult<Option<String>> {
    if data_storage_version.is_null() {
        return Ok(None);
    }

    let raw = unsafe { cstr_to_str(data_storage_version, "data_storage_version")? };
    let token = raw.trim();
    if token.is_empty() {
        return Err(FfiError::new(
            ErrorCode::DatasetWriteOpen,
            "data_storage_version cannot be empty",
        ));
    }

    let lower = token.to_ascii_lowercase();
    let normalized = match lower.as_str() {
        "v2_0" | "v2.0" | "2_0" => "2.0".to_string(),
        "v2_1" | "v2.1" | "2_1" => "2.1".to_string(),
        "v2_2" | "v2.2" | "2_2" => "2.2".to_string(),
        _ => lower,
    };
    Ok(Some(normalized))
}

#[allow(clippy::too_many_arguments)]
fn open_writer_inner(
    kind: WriterKind,
    path: *const c_char,
    mode: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    data_storage_version: *const c_char,
    session: *mut c_void,
    schema: *const c_void,
    blob_columns: *const u8,
    blob_columns_len: usize,
    ingest_batch_budget: u64,
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
    let mut schema: SchemaRef = std::sync::Arc::new(Schema::new(fields.clone()));

    let write_mode = WriteMode::try_from(mode).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetWriteOpen,
            format!("invalid write mode '{mode}': {err}"),
        )
    })?;

    // The incoming Arrow arrays are imported against `data_type`, so realigning
    // the schema without it would fail RecordBatch validation.
    let mut data_type = data_type.clone();
    let mut append_reference = None;
    let mut existing_schema = None;
    let mut existing_version = None;
    if matches!(write_mode, WriteMode::Append) {
        if let Some((existing, version)) = existing_dataset_context(&path, &storage_options) {
            if let Some(fields) = align_append_fields(schema.fields(), existing.fields()) {
                data_type = DataType::Struct(fields.clone());
                schema = Arc::new(Schema::new(fields));
            }
            append_reference = Some(existing.fields().clone());
            existing_version = Some(version);
            existing_schema = Some(existing);
        }
    }

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
    let data_storage_version = parse_data_storage_version_arg(data_storage_version)?
        .map(|value| {
            value.parse().map_err(|err| {
                FfiError::new(
                    ErrorCode::DatasetWriteOpen,
                    format!("invalid data_storage_version '{value}': {err}"),
                )
            })
        })
        .transpose()?;
    let session = unsafe { optional_session_handle(session)? };

    let blob_flags = if blob_columns_len == 0 {
        Vec::new()
    } else if blob_columns.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "blob_columns is null with non-zero length",
        ));
    } else {
        let flags = unsafe { std::slice::from_raw_parts(blob_columns, blob_columns_len) };
        if flags.len() != schema.fields().len() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!(
                    "blob_columns length {} does not match the {}-column schema",
                    flags.len(),
                    schema.fields().len()
                ),
            ));
        }
        flags.to_vec()
    };

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }

    let params = WriteParams {
        mode: write_mode,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        data_storage_version,
        session: session.clone(),
        store_params: Some(store_params),
        // Absolute external URIs are part of the SQL reference contract:
        // references that match a registered base reuse it, others are stored
        // as absolute URIs through Lance's native outside-bases capability.
        allow_external_blob_outside_bases: true,
        external_blob_mode: ExternalBlobMode::Reference,
        ..Default::default()
    };

    // Blob v2 requires file version >= 2.2. Append writes follow the
    // destination's recorded version; create/overwrite use the resolved request
    // (an unspecified request resolves through Lance's stable selector).
    let new_fields_use_blob_v2 = match existing_version.as_deref() {
        Some(version) => super::blob::version_supports_blob_v2(version),
        None => super::blob::version_supports_blob_v2(
            params.storage_version_or_default().to_manifest_string(),
        ),
    };
    let blob_conversions = plan_blob_conversions(
        schema.as_ref(),
        existing_schema.as_ref(),
        new_fields_use_blob_v2,
        &blob_flags,
    )
    .map_err(|err| FfiError::new(ErrorCode::DatasetWriteOpen, err))?;
    let ingest_batch_budget = if ingest_batch_budget == 0 {
        super::blob::DEFAULT_INGEST_BATCH_BUDGET
    } else {
        usize::try_from(ingest_batch_budget).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetWriteOpen,
                format!("invalid ingest memory budget: {err}"),
            )
        })?
    };
    let blob_context = BlobWriteContext {
        registry: params.store_registry(),
        ingest_batch_budget,
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

    let writer_error = Arc::new(Mutex::new(None));
    Ok(WriterHandle {
        input_schema: schema.clone(),
        data_type,
        state: Mutex::new(WriterState {
            kind,
            path,
            params,
            append_reference,
            vector_candidates,
            blob_conversions,
            blob_context,
            buffered_batches: Vec::new(),
            writer_error: writer_error.clone(),
            output_schema: None,
            output_sender: None,
            output_join: None,
        }),
        batches_sent: AtomicU64::new(0),
        writer_error,
    })
}

#[no_mangle]
pub unsafe extern "C" fn lance_writer_write_batch(writer: *mut c_void, array: *mut c_void) -> i32 {
    match writer_write_batch_inner(writer, array, &HashMap::new()) {
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

pub(super) fn parse_batch_source_options(
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> FfiResult<SourceOptionsMap> {
    let mut map = HashMap::new();
    if source_uri_count == 0 {
        return Ok(map);
    }
    if source_uris.is_null()
        || source_uri_keys.is_null()
        || source_uri_values.is_null()
        || source_uri_counts.is_null()
    {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "source URI option arrays are null with non-zero count",
        ));
    }
    let uris = unsafe { slice_from_ptr(source_uris, source_uri_count, "source_uris")? };
    let all_keys = unsafe { slice_from_ptr(source_uri_keys, source_uri_count, "source_uri_keys")? };
    let all_values =
        unsafe { slice_from_ptr(source_uri_values, source_uri_count, "source_uri_values")? };
    let counts =
        unsafe { slice_from_ptr(source_uri_counts, source_uri_count, "source_uri_counts")? };
    for idx in 0..source_uri_count {
        if uris[idx].is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("source_uris[{idx}] is null"),
            ));
        }
        let uri = unsafe { CStr::from_ptr(uris[idx]) }
            .to_str()
            .map_err(|err| {
                FfiError::new(ErrorCode::Utf8, format!("source_uris[{idx}] utf8: {err}"))
            })?
            .to_string();
        let count = counts[idx];
        let mut options = HashMap::new();
        if count > 0 {
            if all_keys[idx].is_null() || all_values[idx].is_null() {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!("source_uri_options[{idx}] arrays are null with non-zero count"),
                ));
            }
            let keys = unsafe { slice_from_ptr(all_keys[idx], count, "source_uri_keys")? };
            let values = unsafe { slice_from_ptr(all_values[idx], count, "source_uri_values")? };
            for kv_idx in 0..count {
                if keys[kv_idx].is_null() || values[kv_idx].is_null() {
                    return Err(FfiError::new(
                        ErrorCode::InvalidArgument,
                        format!("source_uri_options[{idx}] option at index {kv_idx} is null"),
                    ));
                }
                let key = unsafe { CStr::from_ptr(keys[kv_idx]) }
                    .to_str()
                    .map_err(|err| {
                        FfiError::new(
                            ErrorCode::Utf8,
                            format!("source_uri_options[{idx}] key[{kv_idx}] utf8: {err}"),
                        )
                    })?
                    .to_string();
                let value = unsafe { CStr::from_ptr(values[kv_idx]) }
                    .to_str()
                    .map_err(|err| {
                        FfiError::new(
                            ErrorCode::Utf8,
                            format!("source_uri_options[{idx}] value[{kv_idx}] utf8: {err}"),
                        )
                    })?
                    .to_string();
                options.insert(key, value);
            }
        }
        map.insert(uri, options);
    }
    Ok(map)
}

/// Write one batch and the source-storage options resolved by DuckDB's secret
/// lookup for each ingest URI it carries.
#[no_mangle]
pub unsafe extern "C" fn lance_writer_write_batch_with_source_options(
    writer: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> i32 {
    let source_options = parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    );
    let result =
        source_options.and_then(|options| writer_write_batch_inner(writer, array, &options));
    match result {
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

fn writer_write_batch_inner(
    writer: *mut c_void,
    array: *mut c_void,
    source_options: &SourceOptionsMap,
) -> FfiResult<()> {
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

    let input_batch =
        RecordBatch::try_new(handle.input_schema.clone(), struct_array.columns().to_vec())
            .map_err(|err| {
                FfiError::new(ErrorCode::DatasetWriteBatch, format!("record batch: {err}"))
            })?;

    let (sender, to_send) = {
        let mut guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        if guard.output_sender.is_none() {
            guard
                .buffered_batches
                .push((input_batch, source_options.clone()));

            if !guard.vector_candidates.is_empty() {
                let batches = guard
                    .buffered_batches
                    .iter()
                    .map(|(batch, _)| batch.clone())
                    .collect::<Vec<_>>();
                for cand in guard.vector_candidates.iter_mut() {
                    if cand.dim != 0 {
                        continue;
                    }
                    for batch in batches.iter() {
                        let arr = batch.column(cand.col_idx).as_ref();
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

                let batches = guard
                    .buffered_batches
                    .iter()
                    .map(|(batch, _)| batch.clone())
                    .collect::<Vec<_>>();
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
                let out_batches = guard
                    .start(&handle.input_schema, false)
                    .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
                (guard.output_sender.clone(), out_batches)
            } else {
                (None, Vec::new())
            }
        } else {
            let sender = guard.output_sender.as_ref().cloned();
            let schema = guard
                .output_schema
                .as_ref()
                .ok_or_else(|| {
                    FfiError::new(ErrorCode::DatasetWriteBatch, "writer is not initialized")
                })?
                .clone();
            let conversions: Vec<VectorConversion> = guard
                .vector_candidates
                .iter()
                .filter(|c| c.dim != 0)
                .cloned()
                .collect();
            let out = convert_record_batch(
                &input_batch,
                &schema,
                &conversions,
                &guard.blob_conversions,
                &guard.blob_context,
                source_options,
            )
            .map_err(|e| FfiError::new(ErrorCode::DatasetWriteBatch, e))?;
            (sender, vec![out])
        }
    };

    if let Some(sender) = sender {
        for batch in to_send {
            sender.send(batch).map_err(|_| {
                let detail = handle
                    .writer_error
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                FfiError::new(
                    ErrorCode::DatasetWriteBatch,
                    detail
                        .map(|detail| format!("writer background task failed: {detail}"))
                        .unwrap_or_else(|| "writer background task exited".to_string()),
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
    let txn = handle
        .finish()
        .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinish, e))?
        .0;
    let (path, store_params, session) = {
        let guard = handle
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (
            guard.path.clone(),
            guard.params.store_params.clone().unwrap_or_default(),
            guard.params.session.clone(),
        )
    };
    let mut builder = CommitBuilder::new(path.as_str()).with_store_params(store_params);
    if let Some(session) = session {
        builder = builder.with_session(session);
    }
    match runtime::block_on(builder.execute(*txn)) {
        Ok(Ok(_)) => {
            record_commit();
            Ok(())
        }
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            err.to_string(),
        )),
        Err(err) => Err(FfiError::new(
            ErrorCode::DatasetWriteFinish,
            format!("runtime: {err}"),
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
    let txn = handle
        .finish()
        .map_err(|e| FfiError::new(ErrorCode::DatasetWriteFinishUncommitted, e))?
        .0;

    unsafe {
        *out_transaction = Box::into_raw(txn) as *mut c_void;
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
    session: *mut c_void,
    transaction: *mut c_void,
) -> i32 {
    match commit_transaction_inner(
        path,
        option_keys,
        option_values,
        options_len,
        session,
        transaction,
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

fn commit_transaction_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    session: *mut c_void,
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
        store_params.storage_options_accessor = Some(Arc::new(
            StorageOptionsAccessor::with_static_options(storage_options),
        ));
    }
    let session = unsafe { optional_session_handle(session)? };

    let txn =
        unsafe { Box::from_raw(transaction as *mut lance::dataset::transaction::Transaction) };
    let mut builder = CommitBuilder::new(path.as_str()).with_store_params(store_params);
    if let Some(session) = session {
        builder = builder.with_session(session);
    }
    let fut = builder.execute(*txn);
    match runtime::block_on(fut) {
        Ok(Ok(_)) => {
            record_commit();
            Ok(())
        }
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
pub unsafe extern "C" fn lance_commit_transaction_to_dataset(
    dataset: *mut c_void,
    transaction: *mut c_void,
) -> i32 {
    match commit_transaction_to_dataset_inner(dataset, transaction) {
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

fn commit_transaction_to_dataset_inner(
    dataset: *mut c_void,
    transaction: *mut c_void,
) -> FfiResult<()> {
    if transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "transaction is null",
        ));
    }
    // SAFETY: the caller transfers the transaction returned by an uncommitted
    // write. Consume it even if validating the destination fails.
    let transaction =
        unsafe { Box::from_raw(transaction as *mut lance::dataset::transaction::Transaction) };
    // SAFETY: the pending write retains the dataset cache entry until commit.
    let dataset = unsafe { dataset_handle(dataset)? }.dataset.clone();
    // A Dataset destination reuses its object store, base path, credentials,
    // commit handler and session, without re-resolving a mutable namespace.
    match runtime::block_on(CommitBuilder::new(dataset).execute(*transaction)) {
        Ok(Ok(_)) => {
            record_commit();
            Ok(())
        }
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

#[cfg(test)]
mod align_tests {
    use super::*;
    use lance_arrow::bfloat16::BFLOAT16_EXT_NAME;
    use lance_arrow::json::{ARROW_JSON_EXT_NAME, JSON_EXT_NAME};

    fn list_of(child_name: &str) -> DataType {
        DataType::List(Arc::new(Field::new(child_name, DataType::Utf8, true)))
    }

    #[test]
    fn renames_list_child_to_match_the_destination() {
        let incoming = Fields::from(vec![Field::new("tags", list_of("item"), true)]);
        let existing = Fields::from(vec![Field::new("tags", list_of("l"), true)]);

        let aligned = align_list_field_names(&incoming, &existing).expect("expected a rename");
        assert_eq!(aligned[0].data_type(), &list_of("l"));
    }

    #[test]
    fn leaves_matching_schemas_alone() {
        let fields = Fields::from(vec![Field::new("tags", list_of("item"), true)]);
        assert!(align_list_field_names(&fields, &fields.clone()).is_none());
    }

    #[test]
    fn ignores_columns_the_destination_does_not_have() {
        let incoming = Fields::from(vec![Field::new("tags", list_of("item"), true)]);
        let existing = Fields::from(vec![Field::new("other", list_of("l"), true)]);
        assert!(align_list_field_names(&incoming, &existing).is_none());
    }

    #[test]
    fn descends_into_struct_columns() {
        let incoming = Fields::from(vec![Field::new(
            "meta",
            DataType::Struct(Fields::from(vec![Field::new(
                "tags",
                list_of("item"),
                true,
            )])),
            true,
        )]);
        let existing = Fields::from(vec![Field::new(
            "meta",
            DataType::Struct(Fields::from(vec![Field::new("tags", list_of("l"), true)])),
            true,
        )]);

        let aligned = align_list_field_names(&incoming, &existing).expect("expected a rename");
        let DataType::Struct(children) = aligned[0].data_type() else {
            panic!("expected a struct");
        };
        assert_eq!(children[0].data_type(), &list_of("l"));
    }

    fn vector_of(child_name: &str) -> DataType {
        DataType::FixedSizeList(Arc::new(Field::new(child_name, DataType::Float32, true)), 4)
    }

    fn with_extension(field: Field, name: &str, metadata: Option<&str>) -> Field {
        let mut map = field.metadata().clone();
        map.insert(EXTENSION_TYPE_NAME_KEY.to_string(), name.to_string());
        if let Some(metadata) = metadata {
            map.insert(
                EXTENSION_TYPE_METADATA_KEY.to_string(),
                metadata.to_string(),
            );
        }
        field.with_metadata(map)
    }

    fn column(name: &str, data_type: DataType, extension: Option<(&str, Option<&str>)>) -> Fields {
        let field = Field::new(name, data_type, true);
        Fields::from(vec![match extension {
            Some((extension, metadata)) => with_extension(field, extension, metadata),
            None => field,
        }])
    }

    const TENSOR: &str = "arrow.fixed_shape_tensor";
    const TENSOR_META: &str = r#"{"shape":[4]}"#;
    const OPAQUE: &str = "arrow.opaque";
    const BIT_META: &str = r#"{"type_name":"bit","vendor_name":"DuckDB"}"#;
    const WKB: &str = "geoarrow.wkb";

    fn extension(field: &Field) -> (Option<&str>, Option<&str>) {
        (field.extension_type_name(), field.extension_type_metadata())
    }

    #[test]
    fn restates_the_destination_extension() {
        // DuckDB drops an extension it does not model (and names the child "")...
        let tensor = column("v", vector_of("item"), Some((TENSOR, Some(TENSOR_META))));
        let aligned = align_append_fields(&column("v", vector_of(""), None), &tensor)
            .expect("expected the tensor extension");
        assert_eq!(extension(&aligned[0]), (Some(TENSOR), Some(TENSOR_META)));
        assert_eq!(aligned[0].data_type(), &vector_of(""));

        // ...adds its own under arrow_lossless_conversion, other metadata staying...
        let comment = HashMap::from([("comment".to_string(), "kept".to_string())]);
        let bit = Field::new("b", DataType::Binary, true).with_metadata(comment.clone());
        let incoming = Fields::from(vec![with_extension(bit, OPAQUE, Some(BIT_META))]);
        let aligned = align_append_fields(&incoming, &column("b", DataType::Binary, None));
        assert_eq!(
            aligned.expect("expected no extension")[0].metadata(),
            &comment
        );

        // ...and spells extension metadata its own way.
        let geometry = |metadata| column("g", DataType::Binary, Some((WKB, metadata)));
        for stored in [None, Some(r#"{"crs":"OGC:CRS84"}"#)] {
            let aligned = align_append_fields(&geometry(Some("{}")), &geometry(stored));
            assert_eq!(aligned, Some(geometry(stored)));
        }

        // JSON text keeps its name, so Lance still converts it; the metadata follows.
        let text = |metadata| column("j", DataType::Utf8, Some((ARROW_JSON_EXT_NAME, metadata)));
        let json = |metadata| column("j", DataType::LargeBinary, Some((JSON_EXT_NAME, metadata)));
        assert_eq!(
            align_append_fields(&text(Some("")), &json(None)),
            Some(text(None))
        );
        assert!(align_append_fields(&text(Some("")), &json(Some(""))).is_none());
    }

    #[test]
    fn leaves_type_differences_to_lance() {
        let leaves = |incoming: Fields, stored: Fields| {
            assert!(align_append_fields(&incoming, &stored).is_none());
        };
        let tensor = column("v", vector_of("item"), Some((TENSOR, Some(TENSOR_META))));
        leaves(tensor.clone(), tensor);
        leaves(
            column("g", DataType::Binary, None),
            column("g", DataType::LargeBinary, Some((WKB, None))),
        );

        // Lance types JSON, Blob v2 and bfloat16 vectors by their extension.
        let json = column("j", DataType::LargeBinary, Some((JSON_EXT_NAME, None)));
        leaves(column("j", DataType::LargeBinary, None), json);
        let blob = DataType::Struct(Fields::from(vec![
            Field::new("data", DataType::LargeBinary, true),
            Field::new("uri", DataType::Utf8, true),
        ]));
        leaves(
            column("b", blob.clone(), None),
            column("b", blob, Some((BLOB_V2_EXT_NAME, None))),
        );
        let raw = Field::new("item", DataType::FixedSizeBinary(2), true);
        let bf16 = with_extension(raw.clone(), BFLOAT16_EXT_NAME, None);
        let halves = |item| column("h", DataType::FixedSizeList(Arc::new(item), 4), None);
        leaves(halves(raw), halves(bf16));

        // Lance compares the item of a fixed-size list of structs whole.
        let pairs = |child: Field| {
            let item = Field::new("item", DataType::Struct(Fields::from(vec![child])), true);
            column("p", DataType::FixedSizeList(Arc::new(item), 2), None)
        };
        let bit = Field::new("c", DataType::Binary, true);
        let opaque = with_extension(bit.clone(), OPAQUE, Some(BIT_META));
        leaves(pairs(bit), pairs(opaque));
    }

    #[test]
    fn restates_nested_extensions_after_list_names() {
        let bits = |item: Field| column("bits", DataType::List(Arc::new(item)), None);
        let item = Field::new("l", DataType::Binary, true);
        let stored = bits(with_extension(item, OPAQUE, Some(BIT_META)));
        let incoming = bits(Field::new("item", DataType::Binary, true));
        assert_eq!(align_append_fields(&incoming, &stored), Some(stored));
    }

    #[test]
    fn converted_vectors_take_the_destination_extension() {
        let floats = || Arc::new(Field::new("item", DataType::Float32, true));
        let input = Arc::new(Schema::new(vec![Field::new(
            "v",
            DataType::List(floats()),
            true,
        )]));
        let conversions = vec![VectorConversion {
            col_idx: 0,
            dim: 4,
            list_kind: VectorListKind::List,
            element_type: VectorElementType::Float32,
        }];
        let tensor = column("v", vector_of("item"), Some((TENSOR, Some(TENSOR_META))));

        let output = build_output_schema(&input, &conversions, Some(&tensor), &[]).unwrap();
        assert_eq!(output.field(0).extension_type_name(), Some(TENSOR));
        let offsets = arrow::buffer::OffsetBuffer::from_lengths([4]);
        let values = Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]));
        let list = ListArray::new(floats(), offsets, values, None);
        let batch = RecordBatch::try_new(input.clone(), vec![Arc::new(list)]).unwrap();
        let converted = convert_record_batch(
            &batch,
            &output,
            &conversions,
            &[],
            &BlobWriteContext {
                registry: Arc::new(lance::io::ObjectStoreRegistry::default()),
                ingest_batch_budget: 0,
            },
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            converted.schema().field(0).extension_type_name(),
            Some(TENSOR)
        );

        let plain = build_output_schema(&input, &conversions, None, &[]).unwrap();
        assert_eq!(plain.field(0).extension_type_name(), None);
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use arrow_array::types::Float32Type;
    use arrow_array::{
        BinaryArray, Int64Array, LargeBinaryArray, RecordBatchIterator, StringArray, UnionArray,
    };
    use arrow_schema::UnionFields;
    use futures::StreamExt as _;
    use lance::Dataset;
    use lance_core::datatypes::BlobHandling;
    use std::ffi::CString;

    fn blob_union_fields() -> UnionFields {
        let descriptor = |name: &str| {
            Field::new(
                name,
                DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                true,
            )
        };
        UnionFields::try_new(
            vec![0, 1, 2],
            vec![
                Arc::new(Field::new("bytes", DataType::Binary, true)),
                Arc::new(descriptor("reference")),
                Arc::new(descriptor("ingest")),
            ],
        )
        .unwrap()
    }

    fn blob_union_batch(source_uri: &str) -> RecordBatch {
        let fields = blob_union_fields();
        let uri_children = || {
            Fields::from(vec![
                Arc::new(Field::new("uri", DataType::Utf8, true)),
                Arc::new(Field::new("position", DataType::UInt64, true)),
                Arc::new(Field::new("size", DataType::UInt64, true)),
            ])
        };
        let descriptor = |uri_val: Option<&str>, len: usize, at: usize| {
            let mut uri_rows = vec![None; len];
            if let Some(uri) = uri_val {
                uri_rows[at] = Some(uri.to_string());
            }
            let position = arrow_array::UInt64Array::from(vec![None; len]);
            let size = arrow_array::UInt64Array::from(vec![None; len]);
            StructArray::try_new(
                uri_children(),
                vec![
                    Arc::new(StringArray::from(uri_rows)),
                    Arc::new(position),
                    Arc::new(size),
                ],
                None,
            )
            .unwrap()
        };
        let bytes = BinaryArray::from(vec![Some(b"abc".as_slice()), None, None]);
        let reference = descriptor(Some(source_uri), 3, 1);
        let ingest = descriptor(Some(source_uri), 3, 2);
        let payload = UnionArray::try_new(
            fields,
            vec![0_i8, 1, 2].into(),
            None,
            vec![Arc::new(bytes), Arc::new(reference), Arc::new(ingest)],
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("payload", payload.data_type().clone(), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![1, 2, 3])), Arc::new(payload)],
        )
        .unwrap()
    }

    fn scan_blob_payloads(dataset: &Dataset) -> Result<Vec<(i64, Vec<u8>)>, String> {
        let mut scanner = dataset.scan();
        scanner.blob_handling(BlobHandling::AllBinary);
        let mut stream = runtime::block_on(scanner.try_into_stream())
            .map_err(|err| err.to_string())?
            .map_err(|err| err.to_string())?;
        let mut out = Vec::new();
        loop {
            let Some(batch) = runtime::block_on(stream.next()).map_err(|err| err.to_string())?
            else {
                break;
            };
            let batch = batch.map_err(|err| err.to_string())?;
            let ids = batch
                .column_by_name("id")
                .ok_or_else(|| "missing id".to_string())?
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| "id not int64".to_string())?;
            let payloads = batch
                .column_by_name("payload")
                .ok_or_else(|| "missing payload".to_string())?
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| "payload not large binary".to_string())?;
            for i in 0..batch.num_rows() {
                out.push((ids.value(i), payloads.value(i).to_vec()));
            }
        }
        Ok(out)
    }

    struct TestDirectory(std::path::PathBuf);

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn check_publication(batch: RecordBatch, write_batch: bool) {
        let _counter_guard = super::super::session::debug_counter_test_lock();
        for kind in [WriterKind::Committed, WriterKind::Uncommitted] {
            let directory = TestDirectory(
                std::env::temp_dir().join(format!("ffi-writer-{}", rand::random::<u64>())),
            );
            let path = CString::new(directory.0.to_str().unwrap()).unwrap();
            let schema =
                super::super::util::schema_to_ffi_arrow_schema(batch.schema().as_ref()).unwrap();
            let mut writer = open_writer_inner(
                kind,
                path.as_ptr(),
                c"create".as_ptr(),
                ptr::null(),
                ptr::null(),
                0,
                1024,
                1024,
                1024 * 1024,
                ptr::null(),
                ptr::null_mut(),
                &schema as *const _ as *const c_void,
                ptr::null(),
                0,
                0,
            )
            .unwrap();
            let writer_ptr = &mut writer as *mut WriterHandle as *mut c_void;
            if write_batch {
                let array = StructArray::from(batch.clone());
                let mut exported = arrow::ffi::FFI_ArrowArray::new(&array.to_data());
                writer_write_batch_inner(
                    writer_ptr,
                    &mut exported as *mut _ as *mut c_void,
                    &HashMap::new(),
                )
                .unwrap();
            }
            match kind {
                WriterKind::Committed => writer_finish_inner(writer_ptr).unwrap(),
                WriterKind::Uncommitted => {
                    let mut transaction = ptr::null_mut();
                    writer_finish_uncommitted_inner(writer_ptr, &mut transaction).unwrap();
                    assert!(!transaction.is_null());
                    // Finishing an uncommitted writer must not publish a manifest.
                    assert!(runtime::block_on(Dataset::open(path.to_str().unwrap()))
                        .unwrap()
                        .is_err());
                    commit_transaction_inner(
                        path.as_ptr(),
                        ptr::null(),
                        ptr::null(),
                        0,
                        ptr::null_mut(),
                        transaction,
                    )
                    .unwrap();
                }
            }
            let dataset = runtime::block_on(Dataset::open(path.to_str().unwrap()))
                .unwrap()
                .unwrap();
            let count = runtime::block_on(dataset.count_rows(None))
                .unwrap()
                .unwrap();
            assert_eq!(count, if write_batch { batch.num_rows() } else { 0 });
            if write_batch && batch.column(0).null_count() == 0 {
                let schema: Schema = dataset.schema().into();
                assert!(matches!(
                    schema.field(0).data_type(),
                    DataType::FixedSizeList(_, 2)
                ));
            }
        }
    }

    fn vector_batch(values: Vec<Option<Vec<Option<f32>>>>) -> RecordBatch {
        let array = ListArray::from_iter_primitive::<Float32Type, _, _>(values);
        RecordBatch::try_from_iter(vec![("vector", Arc::new(array) as Arc<dyn Array>)]).unwrap()
    }

    #[test]
    fn empty_finish_preserves_publication_mode() {
        check_publication(vector_batch(vec![]), false);
    }

    #[test]
    fn dropped_writer_never_publishes_committed_data() {
        let _counter_guard = super::super::session::debug_counter_test_lock();
        let directory = TestDirectory(
            std::env::temp_dir().join(format!("ffi-writer-drop-{}", rand::random::<u64>())),
        );
        let path = CString::new(directory.0.to_str().unwrap()).unwrap();
        let batch = vector_batch(vec![Some(vec![Some(1.0), Some(2.0)])]);
        let schema =
            super::super::util::schema_to_ffi_arrow_schema(batch.schema().as_ref()).unwrap();
        let mut writer = open_writer_inner(
            WriterKind::Committed,
            path.as_ptr(),
            c"create".as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            1024,
            1024,
            1024 * 1024,
            ptr::null(),
            ptr::null_mut(),
            &schema as *const _ as *const c_void,
            ptr::null(),
            0,
            0,
        )
        .unwrap();
        let writer_ptr = &mut writer as *mut WriterHandle as *mut c_void;
        let array = StructArray::from(batch.clone());
        let mut exported = arrow::ffi::FFI_ArrowArray::new(&array.to_data());
        writer_write_batch_inner(
            writer_ptr,
            &mut exported as *mut _ as *mut c_void,
            &HashMap::new(),
        )
        .unwrap();
        // Dropping without finish (the failure path of a later batch) must
        // publish nothing.
        drop(writer);
        assert!(runtime::block_on(Dataset::open(path.to_str().unwrap()))
            .unwrap()
            .is_err());
    }

    #[test]
    fn blob_v2_ffi_roundtrip_preserves_modes_and_bytes() {
        let _counter_guard = super::super::session::debug_counter_test_lock();
        let directory = TestDirectory(
            std::env::temp_dir().join(format!("ffi-blob-v2-{}", rand::random::<u64>())),
        );
        std::fs::create_dir_all(&directory.0).unwrap();
        let source = directory.0.join("blob_src.bin");
        std::fs::write(&source, b"source-bytes").unwrap();
        let source_uri = format!("file://{}", source.display());
        let batch = blob_union_batch(&source_uri);
        let dataset_path = directory.0.join("ds.lance");
        let path = CString::new(dataset_path.to_str().unwrap()).unwrap();
        let schema =
            super::super::util::schema_to_ffi_arrow_schema(batch.schema().as_ref()).unwrap();
        let mut writer = open_writer_inner(
            WriterKind::Uncommitted,
            path.as_ptr(),
            c"create".as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            1024,
            1024,
            1024 * 1024,
            ptr::null(),
            ptr::null_mut(),
            &schema as *const _ as *const c_void,
            [0u8, 1u8].as_ptr(),
            2,
            0,
        )
        .unwrap();
        let writer_ptr = &mut writer as *mut WriterHandle as *mut c_void;
        let array = StructArray::from(batch);
        let mut exported = arrow::ffi::FFI_ArrowArray::new(&array.to_data());
        writer_write_batch_inner(
            writer_ptr,
            &mut exported as *mut _ as *mut c_void,
            &HashMap::new(),
        )
        .unwrap();
        let mut transaction = ptr::null_mut();
        writer_finish_uncommitted_inner(writer_ptr, &mut transaction).unwrap();
        commit_transaction_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap();

        let dataset = runtime::block_on(Dataset::open(dataset_path.to_str().unwrap()))
            .unwrap()
            .unwrap();
        let schema: Schema = dataset.schema().into();
        assert!(
            super::super::blob::is_blob_v2_field(
                schema.field_with_name("payload").expect("payload field"),
            ),
            "payload field must be Blob v2"
        );
        let rows = scan_blob_payloads(&dataset).unwrap();
        assert_eq!(
            rows,
            vec![
                (1, b"abc".to_vec()),
                (2, b"source-bytes".to_vec()),
                (3, b"source-bytes".to_vec()),
            ]
        );

        // References follow the external object; deleting it must make those
        // reads fail while managed rows stay in the dataset.
        std::fs::remove_file(&source).unwrap();
        assert!(scan_blob_payloads(&dataset).is_err());
    }

    #[test]
    fn blob_v2_append_accepts_minimal_two_field_datasets() {
        let _counter_guard = super::super::session::debug_counter_test_lock();
        let directory = TestDirectory(
            std::env::temp_dir().join(format!("ffi-blob-v2-min-{}", rand::random::<u64>())),
        );
        std::fs::create_dir_all(&directory.0).unwrap();
        let dataset_path = directory.0.join("ds.lance");

        // Lance's public blob_field/BlobArrayBuilder path creates the minimal
        // two-field (data, uri) logical shape; our writer emits the complete
        // four-field shape. The append must accept the existing dataset as is.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            lance::blob::blob_field("payload", true),
        ]));
        let data = LargeBinaryArray::from(vec![Some(b"initial".as_slice())]);
        let uri = StringArray::from(vec![None::<String>]);
        let payload = StructArray::try_new(
            Fields::from(vec![
                Arc::new(Field::new("data", DataType::LargeBinary, true)),
                Arc::new(Field::new("uri", DataType::Utf8, true)),
            ]),
            vec![Arc::new(data), Arc::new(uri)],
            None,
        )
        .unwrap();
        let initial = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(vec![1])), Arc::new(payload)],
        )
        .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(initial)], schema);
        runtime::block_on(Dataset::write(
            reader,
            dataset_path.to_str().unwrap(),
            Some(WriteParams::default()),
        ))
        .unwrap()
        .unwrap();

        let append_source = blob_union_batch("file:///unused");
        let payload = append_source.column(1).slice(0, 1);
        let append_batch = RecordBatch::try_new(
            append_source.schema(),
            vec![Arc::new(Int64Array::from(vec![2])), payload],
        )
        .unwrap();
        let path = CString::new(dataset_path.to_str().unwrap()).unwrap();
        let schema =
            super::super::util::schema_to_ffi_arrow_schema(append_batch.schema().as_ref()).unwrap();
        let mut writer = open_writer_inner(
            WriterKind::Uncommitted,
            path.as_ptr(),
            c"append".as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            1024,
            1024,
            1024 * 1024,
            ptr::null(),
            ptr::null_mut(),
            &schema as *const _ as *const c_void,
            [0u8, 1u8].as_ptr(),
            2,
            0,
        )
        .unwrap();
        let writer_ptr = &mut writer as *mut WriterHandle as *mut c_void;
        let array = StructArray::from(append_batch);
        let mut exported = arrow::ffi::FFI_ArrowArray::new(&array.to_data());
        writer_write_batch_inner(
            writer_ptr,
            &mut exported as *mut _ as *mut c_void,
            &HashMap::new(),
        )
        .unwrap();
        let mut transaction = ptr::null_mut();
        writer_finish_uncommitted_inner(writer_ptr, &mut transaction).unwrap();
        commit_transaction_inner(
            path.as_ptr(),
            ptr::null(),
            ptr::null(),
            0,
            ptr::null_mut(),
            transaction,
        )
        .unwrap();

        let dataset = runtime::block_on(Dataset::open(dataset_path.to_str().unwrap()))
            .unwrap()
            .unwrap();
        let rows = scan_blob_payloads(&dataset).unwrap();
        assert_eq!(rows, vec![(1, b"initial".to_vec()), (2, b"abc".to_vec())]);
    }

    #[test]
    fn finish_flushes_buffered_null_vectors() {
        check_publication(vector_batch(vec![None]), true);
    }

    #[test]
    fn finish_joins_started_vector_writer() {
        check_publication(vector_batch(vec![Some(vec![Some(1.0), Some(2.0)])]), true);
    }
}
