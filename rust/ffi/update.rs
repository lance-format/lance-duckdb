use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow_array::builder::{LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_array::cast::AsArray;
use arrow_array::types::{Int64Type, UInt64Type};
use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_buffer::NullBufferBuilder;
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance::dataset::rowids::load_row_id_sequence;
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_V2_EXT_NAME};
use lance_core::datatypes::BLOB_V2_LOGICAL_TYPE;
use lance_core::utils::address::RowAddress;
use lance_core::utils::deletion::DeletionVector;
use lance_table::format::Fragment;
use lance_table::rowids::{FragmentRowIdIndex, RowIdIndex};
use lance_table::transaction::UpdatedFragmentOffsets;
use roaring::RoaringBitmap;
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::blob::{
    apply_blob_output_fields, blob_input_column_to_binary, convert_blob_column, is_blob_input_type,
    is_blob_v2_field, plan_blob_conversions, BlobWriteContext, SourceOptionsMap,
};

use super::util::{cstr_to_str, optional_session_handle, slice_from_ptr, FfiError, FfiResult};

pub(super) async fn build_row_id_index(dataset: &lance::Dataset) -> Result<RowIdIndex, String> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Err("row id index requested for dataset without stable row ids".to_string());
    }

    let fragments = dataset.get_fragments();

    let mut indices = Vec::with_capacity(dataset.manifest.fragments.len());
    for fragment in dataset.manifest.fragments.iter() {
        // Lance owns the row id layout (inline bytes or a spilled column).
        let sequence = load_row_id_sequence(dataset, fragment)
            .await
            .map_err(|e| e.to_string())?;
        let deletion_vector = fragments
            .iter()
            .find(|f| f.id() as u32 == fragment.id as u32)
            .ok_or_else(|| "fragment missing from dataset fragments".to_string())?
            .get_deletion_vector()
            .await
            .map_err(|e| e.to_string())?
            .unwrap_or_else(|| Arc::new(DeletionVector::default()));

        indices.push(FragmentRowIdIndex {
            fragment_id: fragment.id as u32,
            row_id_sequence: sequence,
            deletion_vector,
        });
    }

    RowIdIndex::new(&indices).map_err(|e| e.to_string())
}

pub(super) fn lookup_row_address(row_id_index: &RowIdIndex, row_id: u64) -> Result<u64, String> {
    let addr = row_id_index
        .get(row_id)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("row id missing from row id index: {row_id}"))?;
    Ok(u64::from(addr))
}

pub(super) async fn apply_deletions(
    dataset: &lance::Dataset,
    row_addrs: &RoaringTreemap,
) -> Result<(Vec<lance_table::format::Fragment>, Vec<u64>), String> {
    let bitmaps: BTreeMap<u32, _> = row_addrs.bitmaps().collect();

    let mut updated_fragments = Vec::new();
    let mut removed_fragment_ids = Vec::new();

    for fragment in dataset.get_fragments() {
        let fragment_id = fragment.id() as u32;
        let Some(bitmap) = bitmaps.get(&fragment_id) else {
            continue;
        };

        match fragment
            .extend_deletions(bitmap.iter())
            .await
            .map_err(|e| e.to_string())?
        {
            Some(new_fragment) => updated_fragments.push(new_fragment.metadata().clone()),
            None => removed_fragment_ids.push(fragment_id as u64),
        }
    }

    Ok((updated_fragments, removed_fragment_ids))
}

// ---------------------------------------------------------------------------
// Value-execution UPDATE path
// ---------------------------------------------------------------------------
//
// DuckDB evaluates the SET expressions and streams (updated column values,
// "_rowid") here. Lance rewrites only the assigned top-level columns of the
// touched fragments, so unassigned columns (including Blob v2 values with
// their own ownership and external-reference semantics) are never read,
// re-encoded, or moved as part of the statement.

/// One batch of selected UPDATE values. `batch` has the assigned columns
/// followed by a trailing `_rowid` column; `columns` names the assigned
/// columns in batch order.
#[derive(Clone)]
pub(super) struct UpdateValueBatch {
    pub(super) columns: Vec<String>,
    pub(super) batch: RecordBatch,
    pub(super) source_options: SourceOptionsMap,
}

struct UpdateHandle {
    path: String,
    session: Option<Arc<lance::session::Session>>,
    storage_options: HashMap<String, String>,
    ingest_batch_budget: usize,
    batches: Vec<UpdateValueBatch>,
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

#[repr(C)]
struct RawArrowSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut RawArrowSchema,
    dictionary: *mut RawArrowSchema,
    release: Option<unsafe extern "C" fn(arg1: *mut RawArrowSchema)>,
    private_data: *mut c_void,
}

pub(super) fn import_struct_batch(
    array: *mut c_void,
    schema: *mut c_void,
) -> FfiResult<RecordBatch> {
    if array.is_null() || schema.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "array/schema is null",
        ));
    }

    let raw_array = unsafe { ptr::read(array as *mut RawArrowArray) };
    unsafe {
        (*(array as *mut RawArrowArray)).release = None;
    }
    let ffi_array: arrow::ffi::FFI_ArrowArray = unsafe { std::mem::transmute(raw_array) };

    let raw_schema = unsafe { ptr::read(schema as *mut RawArrowSchema) };
    unsafe {
        (*(schema as *mut RawArrowSchema)).release = None;
    }
    let ffi_schema: arrow_schema::ffi::FFI_ArrowSchema = unsafe { std::mem::transmute(raw_schema) };
    let arrow_schema = ArrowSchema::try_from(&ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("schema import: {err}"))
    })?;
    let data_type = DataType::try_from(&ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("schema import: {err}"))
    })?;
    let array_data = unsafe { arrow_array::ffi::from_ffi_and_data_type(ffi_array, data_type) }
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, format!("array import: {err}")))?;
    let array = arrow_array::make_array(array_data);
    let struct_array = array
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .ok_or_else(|| FfiError::new(ErrorCode::InvalidArgument, "array is not a struct"))?;

    RecordBatch::try_new(Arc::new(arrow_schema), struct_array.columns().to_vec())
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, format!("record batch: {err}")))
}

pub(super) fn take_update_batch_fields(batch: &RecordBatch) -> FfiResult<Vec<String>> {
    if batch.num_columns() < 2 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "UPDATE value batch must have at least one value column plus _rowid",
        ));
    }
    let value_count = batch.num_columns() - 1;
    let mut columns = Vec::with_capacity(value_count);
    for field in batch.schema().fields().iter().take(value_count) {
        columns.push(field.name().to_string());
    }
    let batch_schema = batch.schema();
    let rowid_field = batch_schema.fields()[value_count].name().to_string();
    if rowid_field != "_rowid" {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!(
                "UPDATE value batch last column must be _rowid, got '{}'",
                rowid_field
            ),
        ));
    }
    Ok(columns)
}

#[no_mangle]
pub unsafe extern "C" fn lance_update_begin_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_update_handle: *mut *mut c_void,
) -> i32 {
    match update_begin_with_storage_options_inner(
        path,
        option_keys,
        option_values,
        options_len,
        ingest_batch_budget,
        session,
        out_update_handle,
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

fn update_begin_with_storage_options_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_update_handle: *mut *mut c_void,
) -> FfiResult<()> {
    if out_update_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_update_handle is null",
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

    let handle = Box::new(UpdateHandle {
        path,
        session,
        storage_options,
        ingest_batch_budget,
        batches: Vec::new(),
    });
    unsafe {
        *out_update_handle = Box::into_raw(handle) as *mut c_void;
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_update_add_value_batch(
    update_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> i32 {
    match update_add_value_batch_inner(
        update_handle,
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
fn update_add_value_batch_inner(
    update_handle: *mut c_void,
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
) -> FfiResult<()> {
    if update_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "update_handle is null",
        ));
    }
    let handle = unsafe { &mut *(update_handle as *mut UpdateHandle) };
    let batch = import_struct_batch(array, schema)?;
    let columns = take_update_batch_fields(&batch)?;
    let source_options = super::write::parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    )?;
    handle.batches.push(UpdateValueBatch {
        columns,
        batch,
        source_options,
    });
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_update_abort(update_handle: *mut c_void) {
    if update_handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(update_handle as *mut UpdateHandle);
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_update_finish_uncommitted(
    update_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> i32 {
    match update_finish_uncommitted_inner(update_handle, out_transaction) {
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

fn update_finish_uncommitted_inner(
    update_handle: *mut c_void,
    out_transaction: *mut *mut c_void,
) -> FfiResult<()> {
    if update_handle.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "update_handle is null",
        ));
    }
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }

    let handle = unsafe { Box::from_raw(update_handle as *mut UpdateHandle) };
    let maybe_txn = match runtime::block_on(async move {
        let dataset = build_dataset(
            &handle.path,
            &handle.storage_options,
            handle.session.clone(),
        )
        .await?;
        let dataset = Arc::new(dataset);

        let existing_schema: ArrowSchema = dataset.schema().into();
        let blob_context = BlobWriteContext {
            registry: handle
                .session
                .as_ref()
                .map(|session| session.store_registry())
                .unwrap_or_default(),
            ingest_batch_budget: handle.ingest_batch_budget,
        };
        let mut batches = Vec::with_capacity(handle.batches.len());
        for mut item in handle.batches {
            item.batch = convert_blob_batch(
                &item.batch,
                &existing_schema,
                &blob_context,
                &item.source_options,
            )?;
            batches.push(item);
        }

        let (updated_fragments, fields_modified, fragment_offsets) = compose_column_updates(
            &dataset,
            &batches,
            &RoaringTreemap::new(),
            handle.ingest_batch_budget,
        )
        .await?;
        if updated_fragments.is_empty() {
            return Ok::<_, String>(None);
        }

        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments,
            new_fragments: Vec::new(),
            fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(fragment_offsets)),
        };
        let txn = Transaction::new(dataset.manifest.version, operation, None);
        Ok::<_, String>(Some(txn))
    }) {
        Ok(Ok(txn)) => txn,
        Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetUpdateOverwrite, message)),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetUpdateOverwrite,
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

async fn build_dataset(
    path: &str,
    storage_options: &HashMap<String, String>,
    session: Option<Arc<lance::session::Session>>,
) -> Result<lance::Dataset, String> {
    let mut builder = lance::dataset::builder::DatasetBuilder::from_uri(path)
        .with_storage_options(storage_options.clone());
    if let Some(session) = session {
        builder = builder.with_session(session);
    }
    builder.load().await.map_err(|e| e.to_string())
}

/// Convert one DuckDB-exported write batch into the Lance logical value form
/// shared with INSERT: bytes stay bytes, references stay logical URIs, and
/// ingest values are materialized under the per-batch budget.
pub(super) fn convert_blob_batch(
    batch: &RecordBatch,
    existing_schema: &ArrowSchema,
    blob_context: &BlobWriteContext,
    source_options: &SourceOptionsMap,
) -> Result<RecordBatch, String> {
    // DML has no SQL type vector at this boundary. A column is a blob write
    // target exactly when its value uses LANCE_BLOB_INPUT or the destination
    // field is already Blob v2; ordinary Binary/LargeBinary targets such as
    // BIT remain untouched.
    let blob_flags = batch
        .schema()
        .fields()
        .iter()
        .map(|field| {
            let existing_field = existing_schema.field_with_name(field.name()).ok();
            u8::from(
                is_blob_input_type(field.data_type())
                    || existing_field.is_some_and(is_blob_v2_field),
            )
        })
        .collect::<Vec<_>>();
    let conversions = plan_blob_conversions(
        batch.schema().as_ref(),
        Some(existing_schema),
        false,
        &blob_flags,
    )?;
    if conversions.is_empty() {
        return Ok(batch.clone());
    }
    let output_schema = Arc::new(apply_blob_output_fields(
        batch.schema().as_ref(),
        &conversions,
    ));
    let mut columns = batch.columns().to_vec();
    let batch_schema = batch.schema();
    let mut remaining_budget = blob_context.ingest_batch_budget;
    for conversion in &conversions {
        let field = batch_schema.field(conversion.col_idx);
        columns[conversion.col_idx] = convert_blob_column(
            field.name(),
            columns[conversion.col_idx].as_ref(),
            conversion.target,
            field.is_nullable(),
            blob_context,
            source_options,
            &mut remaining_budget,
            conversion.output_field.as_ref(),
        )?;
    }
    RecordBatch::try_new(output_schema, columns).map_err(|err| err.to_string())
}

/// Convert every blob input column of one DuckDB batch to raw bytes.
///
/// This is the RETURNING representation: references and ingest values are
/// materialized here, before the transaction commits, so the emitted row image
/// never exposes `LANCE_BLOB_INPUT` descriptors.
#[no_mangle]
pub unsafe extern "C" fn lance_blob_inputs_to_binary(
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_schema: *mut arrow::ffi::FFI_ArrowSchema,
    out_array: *mut arrow::ffi::FFI_ArrowArray,
) -> i32 {
    match blob_inputs_to_binary_inner(
        schema,
        array,
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
        ingest_batch_budget,
        session,
        out_schema,
        out_array,
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
fn blob_inputs_to_binary_inner(
    schema: *mut c_void,
    array: *mut c_void,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_schema: *mut arrow::ffi::FFI_ArrowSchema,
    out_array: *mut arrow::ffi::FFI_ArrowArray,
) -> FfiResult<()> {
    if out_schema.is_null() || out_array.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_schema/out_array is null",
        ));
    }
    let batch = import_struct_batch(array, schema)?;
    let source_options = super::write::parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    )?;
    let session = unsafe { optional_session_handle(session)? };
    let budget = if ingest_batch_budget == 0 {
        super::blob::DEFAULT_INGEST_BATCH_BUDGET
    } else {
        usize::try_from(ingest_batch_budget).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("invalid ingest memory budget: {err}"),
            )
        })?
    };
    let context = BlobWriteContext {
        registry: session
            .as_ref()
            .map(|session| session.store_registry())
            .unwrap_or_default(),
        ingest_batch_budget: budget,
    };

    let mut columns = batch.columns().to_vec();
    let mut fields = batch.schema().fields().as_ref().to_vec();
    let mut remaining_budget = budget;
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        if !is_blob_input_type(field.data_type()) {
            continue;
        }
        columns[idx] = blob_input_column_to_binary(
            field.name(),
            columns[idx].as_ref(),
            &context,
            &source_options,
            &mut remaining_budget,
        )
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, err))?;
        fields[idx] = Arc::new(ArrowField::new(
            field.name(),
            DataType::LargeBinary,
            field.is_nullable(),
        ));
    }
    let output = RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, err.to_string()))?;
    let struct_array: Arc<dyn Array> = Arc::new(StructArray::from(output));
    let data = struct_array.to_data();
    let exported_array = arrow::ffi::FFI_ArrowArray::new(&data);
    let exported_schema = arrow_schema::ffi::FFI_ArrowSchema::try_from(data.data_type())
        .map_err(|err| FfiError::new(ErrorCode::BatchExport, format!("batch export: {err}")))?;
    unsafe {
        std::ptr::write_unaligned(out_array, exported_array);
        std::ptr::write_unaligned(out_schema, exported_schema);
    }
    Ok(())
}

/// Field describing one update value column as the fragment updater expects
/// it. Blob v2 targets use the complete logical shape, because the updater
/// materializes the left side that way regardless of the manifest's minimal
/// two-child form.
fn update_value_field(column_name: &str, target_field: &ArrowField) -> ArrowField {
    if !is_blob_v2_field(target_field) {
        return target_field.clone();
    }
    let metadata = HashMap::from([(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]);
    ArrowField::new(
        column_name,
        BLOB_V2_LOGICAL_TYPE.clone(),
        target_field.is_nullable(),
    )
    .with_metadata(metadata)
}

fn read_row_ids(array: &ArrayRef) -> Result<Vec<u64>, String> {
    if let Some(values) = array.as_primitive_opt::<UInt64Type>() {
        return Ok(values.values().to_vec());
    }
    if let Some(values) = array.as_primitive_opt::<Int64Type>() {
        let mut out = Vec::with_capacity(values.len());
        for row in 0..values.len() {
            if values.is_null(row) {
                return Err("UPDATE _rowid is NULL".to_string());
            }
            let value = values.value(row);
            if value < 0 {
                return Err(format!("UPDATE _rowid must be non-negative, got {value}"));
            }
            out.push(value as u64);
        }
        return Ok(out);
    }
    Err(format!(
        "UPDATE _rowid must be UInt64 or Int64, got {}",
        array.data_type()
    ))
}

fn build_blob_bytes_array(column_name: &str, array: &ArrayRef) -> Result<ArrayRef, String> {
    // The fragment updater materializes the left side with the complete
    // logical shape, so the right stream must use the same four children even
    // when the manifest stores the minimal two-child logical field.
    let DataType::Struct(child_fields) = BLOB_V2_LOGICAL_TYPE.clone() else {
        return Err("canonical blob v2 logical type is not a struct".to_string());
    };

    let mut data = LargeBinaryBuilder::new();
    let mut uri = StringBuilder::new();
    let mut position = PrimitiveBuilder::<UInt64Type>::new();
    let mut size = PrimitiveBuilder::<UInt64Type>::new();
    let mut nulls = NullBufferBuilder::new(array.len());

    let read_bytes = |row: usize| -> Result<Option<Vec<u8>>, String> {
        if let Some(binary) = array.as_any().downcast_ref::<arrow_array::BinaryArray>() {
            return Ok((!binary.is_null(row)).then(|| binary.value(row).to_vec()));
        }
        if let Some(binary) = array
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
        {
            return Ok((!binary.is_null(row)).then(|| binary.value(row).to_vec()));
        }
        Err(format!(
            "blob update column '{column_name}' is not a binary array, got {}",
            array.data_type()
        ))
    };

    for row in 0..array.len() {
        let bytes = read_bytes(row)?;
        match &bytes {
            Some(bytes) => data.append_value(bytes),
            None => data.append_null(),
        }
        uri.append_null();
        position.append_null();
        size.append_null();
        nulls.append(bytes.is_some());
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(child_fields.len());
    for child in child_fields.iter() {
        match child.name().as_str() {
            "data" => arrays.push(Arc::new(data.finish()) as ArrayRef),
            "uri" => arrays.push(Arc::new(uri.finish()) as ArrayRef),
            "position" => arrays.push(Arc::new(position.finish()) as ArrayRef),
            "size" => arrays.push(Arc::new(size.finish()) as ArrayRef),
            other => {
                return Err(format!(
                    "blob update column '{column_name}' target has unexpected child '{}'",
                    other
                ))
            }
        }
    }
    let struct_array =
        arrow_array::StructArray::try_new(child_fields.clone(), arrays, nulls.finish())
            .map_err(|err| format!("failed to build blob update column '{column_name}': {err}"))?;
    Ok(Arc::new(struct_array))
}

/// Normalize one update value column to the target field's logical Arrow form.
///
/// Blob v2 targets accept raw bytes (wrapped into the canonical logical
/// struct) or an already-logical blob array (e.g. produced by the writer's
/// input conversion for reference/ingest values). Non-blob columns are cast
/// when the DuckDB-exported type differs from the stored field.
fn normalize_value_column(
    column_name: &str,
    array: &ArrayRef,
    target_field: &ArrowField,
) -> Result<ArrayRef, String> {
    if is_blob_v2_field(target_field) {
        match array.data_type() {
            DataType::Struct(_) => Ok(array.clone()),
            DataType::Binary | DataType::LargeBinary => build_blob_bytes_array(column_name, array),
            other => Err(format!(
                "blob update column '{column_name}' has unsupported value type {other}"
            )),
        }
    } else if array.data_type() == target_field.data_type() {
        Ok(array.clone())
    } else {
        arrow::compute::cast(array, target_field.data_type())
            .map_err(|err| format!("cannot cast update column '{column_name}': {err}"))
    }
}

/// Group a single column's selected rows per fragment.
///
/// Returns fragment id -> (row ids, values) for the fragments actually touched
/// by this column, with duplicate row ids collapsed (first occurrence wins).
fn group_column_rows(
    column_name: &str,
    batch: &RecordBatch,
    row_id_index: Option<&RowIdIndex>,
    deleted_addrs: &RoaringTreemap,
) -> Result<HashMap<u32, (Vec<u64>, ArrayRef)>, String> {
    let value_column = &batch.columns()[0];
    let rowid_column = &batch.columns()[1];
    let row_ids = read_row_ids(rowid_column)?;

    let mut groups: HashMap<u32, (Vec<u64>, Vec<bool>)> = HashMap::new();
    let mut seen: HashSet<u64> = HashSet::new();
    for &row_id in row_ids.iter() {
        if !seen.insert(row_id) {
            continue;
        }
        let addr = match row_id_index {
            Some(index) => lookup_row_address(index, row_id)?,
            None => row_id,
        };
        if deleted_addrs.contains(addr) {
            continue;
        }
        let fragment_id = RowAddress::from(addr).fragment_id();
        groups
            .entry(fragment_id)
            .or_insert_with(|| (Vec::new(), Vec::new()));
        let entry = groups.get_mut(&fragment_id).expect("entry just inserted");
        entry.0.push(row_id);
        entry.1.push(true);
    }

    let mut result = HashMap::with_capacity(groups.len());
    for (fragment_id, (ids, mask)) in groups {
        let mask_array: arrow_array::BooleanArray = mask.into();
        let sliced = arrow::compute::filter(value_column, &mask_array)
            .map_err(|err| format!("failed to slice update column '{column_name}': {err}"))?;
        result.insert(fragment_id, (ids, sliced));
    }
    Ok(result)
}

/// Rewrite the assigned columns of the touched fragments using Lance's native
/// fragment updater and return the composed update metadata.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(super) async fn compose_column_updates(
    dataset: &Arc<lance::Dataset>,
    batches: &[UpdateValueBatch],
    deleted_addrs: &RoaringTreemap,
    ingest_batch_budget: usize,
) -> Result<(Vec<Fragment>, Vec<u32>, HashMap<u64, RoaringBitmap>), String> {
    if batches.is_empty() {
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }

    let arrow_schema: Arc<ArrowSchema> = Arc::new(dataset.schema().into());
    let stable_row_ids = dataset.manifest.uses_stable_row_ids();
    let row_id_index = if stable_row_ids {
        Some(build_row_id_index(dataset).await?)
    } else {
        None
    };

    // Per (fragment, column): concat all value rows across statement batches.
    let mut groups: HashMap<u32, HashMap<String, Vec<RecordBatch>>> = HashMap::new();
    for item in batches {
        for (column_idx, column_name) in item.columns.iter().enumerate() {
            let projected = item
                .batch
                .project(&[column_idx, item.batch.num_columns() - 1])
                .map_err(|err| format!("failed to project update column '{column_name}': {err}"))?;
            let per_fragment = group_column_rows(
                column_name,
                &projected,
                row_id_index.as_ref(),
                deleted_addrs,
            )?;
            for (fragment_id, (row_ids, values)) in per_fragment {
                // The intermediate batch is typed by the DuckDB-exported value
                // array; normalization to the target field happens per column
                // after concatenation.
                let value_schema = ArrowSchema::new(vec![
                    ArrowField::new("_rowid", DataType::UInt64, false),
                    ArrowField::new(column_name.as_str(), values.data_type().clone(), true),
                ]);
                let batch = RecordBatch::try_new(
                    Arc::new(value_schema),
                    vec![Arc::new(arrow_array::UInt64Array::from(row_ids)), values],
                )
                .map_err(|err| {
                    format!("failed to build update batch for column '{column_name}': {err}")
                })?;
                groups
                    .entry(fragment_id)
                    .or_default()
                    .entry(column_name.clone())
                    .or_default()
                    .push(batch);
            }
        }
    }

    if groups.is_empty() {
        return Ok((Vec::new(), Vec::new(), HashMap::new()));
    }

    let fragments = dataset.get_fragments();
    let mut fragment_map: HashMap<u32, lance::dataset::fragment::FileFragment> = fragments
        .into_iter()
        .map(|fragment| (fragment.id() as u32, fragment))
        .collect();

    let mut updated_fragments = Vec::new();
    let mut fields_modified = BTreeSet::<u32>::new();
    let mut fragment_offsets = HashMap::<u64, RoaringBitmap>::new();

    for (fragment_id, column_batches) in groups {
        let mut fragment = fragment_map
            .remove(&fragment_id)
            .ok_or_else(|| format!("fragment {fragment_id} disappeared while preparing update"))?;
        let base_metadata = fragment.metadata().clone();

        let mut changed_fields: BTreeSet<i32> = BTreeSet::new();
        let mut matched = RoaringBitmap::new();
        let mut new_files = Vec::new();

        for (column_name, batches) in &column_batches {
            let target_field = arrow_schema.field_with_name(column_name).map_err(|err| {
                format!("column '{column_name}' missing from dataset schema: {err}")
            })?;
            let input_schema = batches[0].schema();
            let concat = arrow::compute::concat_batches(&input_schema, batches)
                .map_err(|err| format!("failed to concat update batches: {err}"))?;

            let value_array = normalize_value_column(column_name, concat.column(1), target_field)?;
            let blob_rows = if is_blob_v2_field(target_field) {
                blob_struct_rows(&value_array)?
            } else {
                Vec::new()
            };
            let original_paths: HashSet<&str> = base_metadata
                .files
                .iter()
                .map(|file| file.path.as_str())
                .collect();

            if is_blob_v2_field(target_field) && contains_blob_reference(&blob_rows) {
                // Brand-new absolute external URIs cannot pass the native column
                // updater's registered-base check. Rewrite the complete column
                // through Lance's update writer instead, deriving the fallback
                // rows from stored descriptors.
                let rowids = read_row_ids(concat.column(0))?;
                if rowids.len() != blob_rows.len() {
                    return Err(format!(
                        "UPDATE value/row-id length mismatch for column '{column_name}'"
                    ));
                }
                let mut assigned_rows = HashMap::with_capacity(rowids.len());
                for (row, row_id) in rowids.iter().enumerate() {
                    let address = match row_id_index.as_ref() {
                        Some(index) => lookup_row_address(index, *row_id)?,
                        None => *row_id,
                    };
                    assigned_rows.insert(address, blob_rows[row].clone());
                }
                let field = dataset
                    .schema()
                    .field(column_name.as_str())
                    .ok_or_else(|| format!("column '{column_name}' missing from dataset schema"))?;
                let (combined, field_id, offsets) = write_full_blob_column(
                    dataset,
                    &fragment,
                    column_name,
                    field.id,
                    &assigned_rows,
                    deleted_addrs,
                    ingest_batch_budget,
                )
                .await?;
                changed_fields.insert(field_id as i32);
                matched |= offsets;
                for file in combined.files.iter() {
                    if !original_paths.contains(file.path.as_str())
                        && !new_files
                            .iter()
                            .any(|existing: &lance_table::format::DataFile| {
                                existing.path == file.path
                            })
                    {
                        new_files.push(file.clone());
                    }
                }
                continue;
            }

            let value_schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("_rowid", DataType::UInt64, false),
                update_value_field(column_name, target_field),
            ]));
            let update_batch = RecordBatch::try_new(
                value_schema.clone(),
                vec![concat.column(0).clone(), value_array],
            )
            .map_err(|err| {
                format!("failed to build update batch for column '{column_name}': {err}")
            })?;

            let reader = arrow_array::RecordBatchIterator::new(
                vec![Ok(update_batch)].into_iter(),
                value_schema.clone(),
            );
            let result = fragment
                .update_columns_with_offsets(reader, "_rowid", "_rowid")
                .await
                .map_err(|err| format!("fragment {fragment_id} update '{column_name}': {err}"))?;

            changed_fields.extend(
                result
                    .fields_modified
                    .iter()
                    .map(|field_id| *field_id as i32),
            );
            matched |= result.matched_offsets.clone();
            for file in result.fragment.files.iter() {
                if !original_paths.contains(file.path.as_str())
                    && !new_files
                        .iter()
                        .any(|existing: &lance_table::format::DataFile| existing.path == file.path)
                {
                    new_files.push(file.clone());
                }
            }
        }

        let mut combined = base_metadata.clone();
        for file in combined.files.iter_mut() {
            let new_fields: Vec<i32> = file
                .fields
                .iter()
                .map(|field_id| {
                    if changed_fields.contains(field_id) {
                        -2 // Tombstone: superseded by the rewrite below
                    } else {
                        *field_id
                    }
                })
                .collect();
            file.fields = Arc::from(new_fields);
        }
        combined
            .files
            .retain(|file| file.fields.iter().any(|field_id| *field_id != -2));
        combined.files.extend(new_files);

        let mut active: Vec<i32> = combined
            .files
            .iter()
            .flat_map(|file| file.fields.iter().copied())
            .filter(|field_id| *field_id >= 0)
            .collect();
        let unique_active: std::collections::HashSet<i32> = active.iter().copied().collect();
        if active.len() != unique_active.len() {
            return Err(format!(
                "internal error: conflicting column update files for fragment {fragment_id}"
            ));
        }
        active.sort_unstable();

        fields_modified.extend(
            changed_fields
                .into_iter()
                .filter_map(|field_id| u32::try_from(field_id).ok()),
        );
        fragment_offsets.insert(fragment_id as u64, matched);
        updated_fragments.push(combined);
    }

    Ok((
        updated_fragments,
        fields_modified.into_iter().collect(),
        fragment_offsets,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::LargeBinaryBuilder;
    use arrow_array::{Int64Array, RecordBatchIterator, UInt64Array};
    use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};
    use futures::StreamExt;
    use lance::blob::BlobArrayBuilder;
    use lance::dataset::{CommitBuilder, ExternalBlobMode, WriteMode, WriteParams};
    use lance_core::{ROW_ID, ROW_LAST_UPDATED_AT_VERSION};

    fn temp_uri(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("lance-dml-{tag}-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn compose_column_updates_rewrites_blob_v2_payload() {
        let dir = temp_uri("blobv2");
        let uri = dir.to_string_lossy().to_string();
        let external = dir.join("external.bin");
        std::fs::write(&external, b"external-payload").unwrap();
        let other_external = dir.join("other-external.bin");
        std::fs::write(&other_external, b"other-external-payload").unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            lance::blob::blob_field("payload", true),
            lance::blob::blob_field("other", true),
        ]));

        let mut payload = BlobArrayBuilder::new(4);
        payload.push_bytes(b"one").unwrap();
        payload.push_bytes(vec![b'p'; 100_000]).unwrap();
        payload.push_null().unwrap();
        payload
            .push_uri(format!("file://{}", external.display()))
            .unwrap();
        let mut other = BlobArrayBuilder::new(4);
        other
            .push_uri(format!("file://{}", other_external.display()))
            .unwrap();
        other.push_bytes(b"keep-2").unwrap();
        other.push_bytes(b"keep-3").unwrap();
        other.push_bytes(b"keep-4").unwrap();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                payload.finish().unwrap(),
                other.finish().unwrap(),
            ],
        )
        .unwrap();
        let params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: true,
            external_blob_mode: ExternalBlobMode::Reference,
            allow_external_blob_outside_bases: true,
            ..Default::default()
        };
        let dataset = Arc::new(
            runtime::block_on(lance::Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone()),
                &uri,
                Some(params),
            ))
            .unwrap()
            .unwrap(),
        );
        let version = dataset.version().version;

        // Make the unassigned column's external source unavailable before the
        // update: it must not be read, and the update must still commit.
        std::fs::remove_file(&other_external).unwrap();

        // Read stable row ids for rows 0 and 2.
        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let mut stream = runtime::block_on(scanner.try_into_stream())
            .unwrap()
            .unwrap();
        let first = runtime::block_on(stream.next()).unwrap().unwrap().unwrap();
        let row_ids = first
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();
        assert_eq!(row_ids.len(), 4);

        let mut values = LargeBinaryBuilder::new();
        values.append_value(b"ONE");
        values.append_null();
        let update_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("payload", DataType::LargeBinary, true),
                ArrowField::new("_rowid", DataType::UInt64, false),
            ])),
            vec![
                Arc::new(values.finish()),
                Arc::new(UInt64Array::from(vec![row_ids[0], row_ids[2]])),
            ],
        )
        .unwrap();
        let batches = vec![UpdateValueBatch {
            columns: vec!["payload".to_string()],
            batch: update_batch,
            source_options: HashMap::new(),
        }];

        let (fragments, fields_modified, fragment_offsets) =
            runtime::block_on(compose_column_updates(
                &dataset,
                &batches,
                &RoaringTreemap::new(),
                crate::ffi::blob::DEFAULT_INGEST_BATCH_BUDGET,
            ))
            .unwrap()
            .unwrap();
        assert_eq!(fragments.len(), 1);
        assert_eq!(fields_modified.len(), 1);
        assert_eq!(fragment_offsets.len(), 1);
        let offsets = fragment_offsets.values().next().unwrap();
        assert_eq!(offsets.len(), 2);

        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: fragments,
            new_fragments: Vec::new(),
            fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(fragment_offsets)),
        };
        let txn = Transaction::new(version, operation, None);
        runtime::block_on(CommitBuilder::new(&uri).execute(txn))
            .unwrap()
            .unwrap();

        let after = Arc::new(
            runtime::block_on(lance::Dataset::open(&uri))
                .unwrap()
                .unwrap(),
        );
        assert_eq!(after.version().version, version + 1);

        // The unassigned column's external reference is preserved as a
        // reference; materializing it while its source is missing fails instead
        // of exposing silently re-ingested bytes.
        let mut missing_scan = after.scan();
        missing_scan.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
        let _ = missing_scan.project(&["other"]);
        let missing_scan_failed = match runtime::block_on(missing_scan.try_into_stream()).unwrap() {
            Ok(mut stream) => matches!(runtime::block_on(stream.next()).unwrap(), Some(Err(_))),
            Err(_) => true,
        };
        assert!(missing_scan_failed);

        let mut scanner = after.scan();
        scanner.blob_handling(lance_core::datatypes::BlobHandling::BlobsDescriptions);
        let mut stream = runtime::block_on(scanner.try_into_stream())
            .unwrap()
            .unwrap();
        let batch = runtime::block_on(stream.next()).unwrap().unwrap().unwrap();
        let other = batch
            .column_by_name("other")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .unwrap();
        let other_uri = other
            .column_by_name("blob_uri")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
            .unwrap();
        let other_kind = other
            .column_by_name("kind")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::UInt8Array>()
            .unwrap();
        assert_eq!(other_kind.value(0), 3, "external kind");
        assert_eq!(
            other_uri.value(0),
            format!("file://{}", other_external.display())
        );

        // Restore the source, then the preserved reference and the rewritten
        // payload are both readable with real bytes.
        std::fs::write(&other_external, b"other-external-payload").unwrap();
        let after = Arc::new(
            runtime::block_on(lance::Dataset::open(&uri))
                .unwrap()
                .unwrap(),
        );
        runtime::block_on(after.validate()).unwrap().unwrap();

        let mut scanner = after.scan();
        scanner.blob_handling(lance_core::datatypes::BlobHandling::AllBinary);
        let mut stream = runtime::block_on(scanner.try_into_stream())
            .unwrap()
            .unwrap();
        let batch = runtime::block_on(stream.next()).unwrap().unwrap().unwrap();
        let read_payload = batch
            .column_by_name("payload")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        assert_eq!(read_payload.value(0), b"ONE");
        assert_eq!(read_payload.value(1), vec![b'p'; 100_000].as_slice());
        assert!(read_payload.is_null(2));
        assert_eq!(read_payload.value(3), b"external-payload");
        let read_other = batch
            .column_by_name("other")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
            .unwrap();
        assert_eq!(read_other.value(0), b"other-external-payload");
    }

    #[test]
    fn blob_reference_update_versions_only_assigned_rows() {
        let dir = temp_uri("blobrefmeta");
        let uri = dir.to_string_lossy().to_string();
        let replacement = dir.join("replacement.bin");
        std::fs::write(&replacement, b"replacement-payload").unwrap();

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            lance::blob::blob_field("payload", true),
        ]));
        let mut payload = BlobArrayBuilder::new(4);
        for value in [b"zero".as_slice(), b"one", b"two", b"three"] {
            payload.push_bytes(value).unwrap();
        }
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
                payload.finish().unwrap(),
            ],
        )
        .unwrap();
        let params = WriteParams {
            mode: WriteMode::Create,
            enable_stable_row_ids: true,
            ..Default::default()
        };
        let dataset = Arc::new(
            runtime::block_on(lance::Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
                &uri,
                Some(params),
            ))
            .unwrap()
            .unwrap(),
        );
        let version = dataset.version().version;
        assert!(dataset.get_fragments()[0]
            .metadata()
            .last_updated_at_version_meta
            .is_some());

        let mut scanner = dataset.scan();
        scanner.with_row_id();
        let mut stream = runtime::block_on(scanner.try_into_stream())
            .unwrap()
            .unwrap();
        let rows = runtime::block_on(stream.next()).unwrap().unwrap().unwrap();
        let row_ids = rows
            .column_by_name(ROW_ID)
            .unwrap()
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        let replacement_uri = format!("file://{}", replacement.display());
        let mut payload = BlobArrayBuilder::new(2);
        payload.push_uri(replacement_uri.as_str()).unwrap();
        payload.push_uri(replacement_uri.as_str()).unwrap();
        let update_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                lance::blob::blob_field("payload", true),
                ArrowField::new("_rowid", DataType::UInt64, false),
            ])),
            vec![
                payload.finish().unwrap(),
                Arc::new(UInt64Array::from(vec![row_ids[0], row_ids[2]])),
            ],
        )
        .unwrap();
        let batches = vec![UpdateValueBatch {
            columns: vec!["payload".to_string()],
            batch: update_batch,
            source_options: HashMap::new(),
        }];

        let (fragments, fields_modified, fragment_offsets) =
            runtime::block_on(compose_column_updates(
                &dataset,
                &batches,
                &RoaringTreemap::new(),
                crate::ffi::blob::DEFAULT_INGEST_BATCH_BUDGET,
            ))
            .unwrap()
            .unwrap();
        let offsets = fragment_offsets.values().next().unwrap();
        assert_eq!(
            offsets.iter().collect::<Vec<_>>(),
            vec![0, 2],
            "only assigned physical offsets may advance row version metadata"
        );

        let operation = Operation::Update {
            removed_fragment_ids: Vec::new(),
            updated_fragments: fragments,
            new_fragments: Vec::new(),
            fields_modified,
            compacted_sstables: Vec::new(),
            fields_for_preserving_frag_bitmap: Vec::new(),
            update_mode: Some(UpdateMode::RewriteColumns),
            inserted_rows_filter: None,
            updated_fragment_offsets: Some(UpdatedFragmentOffsets(fragment_offsets)),
        };
        runtime::block_on(
            CommitBuilder::new(&uri).execute(Transaction::new(version, operation, None)),
        )
        .unwrap()
        .unwrap();

        let after = Arc::new(
            runtime::block_on(lance::Dataset::open(&uri))
                .unwrap()
                .unwrap(),
        );
        let mut meta_scan = after.scan();
        meta_scan
            .project(&["id", ROW_LAST_UPDATED_AT_VERSION])
            .unwrap();
        let mut stream = runtime::block_on(meta_scan.try_into_stream())
            .unwrap()
            .unwrap();
        let rows = runtime::block_on(stream.next()).unwrap().unwrap().unwrap();
        let versions = rows
            .column_by_name(ROW_LAST_UPDATED_AT_VERSION)
            .unwrap()
            .as_primitive::<UInt64Type>();
        assert_eq!(versions.value(0), version + 1);
        assert_eq!(versions.value(1), version);
        assert_eq!(versions.value(2), version + 1);
        assert_eq!(versions.value(3), version);
    }

    #[test]
    fn blob_reference_rewrite_skips_deleted_payloads() {
        let dir = temp_uri("blobrefdelete");
        let uri = dir.to_string_lossy().to_string();
        let replacement = dir.join("replacement.bin");
        std::fs::write(&replacement, b"replacement-payload").unwrap();
        let missing = dir.join("missing.bin");

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("id", DataType::Int64, false),
            lance::blob::blob_field("payload", true),
        ]));
        let mut payload = BlobArrayBuilder::new(2);
        payload.push_bytes(b"managed").unwrap();
        payload
            .push_uri(format!("file://{}", missing.display()))
            .unwrap();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                payload.finish().unwrap(),
            ],
        )
        .unwrap();
        let dataset = Arc::new(
            runtime::block_on(lance::Dataset::write(
                RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema),
                &uri,
                Some(WriteParams {
                    mode: WriteMode::Create,
                    external_blob_mode: ExternalBlobMode::Reference,
                    allow_external_blob_outside_bases: true,
                    ..Default::default()
                }),
            ))
            .unwrap()
            .unwrap(),
        );

        let mut replacement_payload = BlobArrayBuilder::new(1);
        replacement_payload
            .push_uri(format!("file://{}", replacement.display()))
            .unwrap();
        let update_batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                lance::blob::blob_field("payload", true),
                ArrowField::new("_rowid", DataType::UInt64, false),
            ])),
            vec![
                replacement_payload.finish().unwrap(),
                Arc::new(UInt64Array::from(vec![0])),
            ],
        )
        .unwrap();
        let batches = vec![UpdateValueBatch {
            columns: vec!["payload".to_string()],
            batch: update_batch,
            source_options: HashMap::new(),
        }];

        let mut deleted_addrs = RoaringTreemap::new();
        deleted_addrs.insert(1);
        let (fragments, _, fragment_offsets) = runtime::block_on(compose_column_updates(
            &dataset,
            &batches,
            &deleted_addrs,
            crate::ffi::blob::DEFAULT_INGEST_BATCH_BUDGET,
        ))
        .unwrap()
        .unwrap();
        assert_eq!(fragments.len(), 1);
        assert_eq!(
            fragment_offsets
                .values()
                .next()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            vec![0]
        );
    }
}

/// One logical Blob v2 value used by the full-column rewrite path.
#[derive(Clone, Debug, Default)]
pub(super) struct LogicalBlobValue {
    pub(super) data: Option<Vec<u8>>,
    pub(super) uri: Option<String>,
    pub(super) position: Option<u64>,
    pub(super) size: Option<u64>,
}

fn blob_struct_rows(array: &ArrayRef) -> Result<Vec<Option<LogicalBlobValue>>, String> {
    let values = array
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .ok_or_else(|| "blob value column is not a struct".to_string())?;
    let data = values.column_by_name("data").and_then(|column| {
        column
            .as_any()
            .downcast_ref::<arrow_array::LargeBinaryArray>()
    });
    let uri = values
        .column_by_name("uri")
        .and_then(|column| column.as_any().downcast_ref::<arrow_array::StringArray>());
    let position = values
        .column_by_name("position")
        .and_then(|column| column.as_any().downcast_ref::<arrow_array::UInt64Array>());
    let size = values
        .column_by_name("size")
        .and_then(|column| column.as_any().downcast_ref::<arrow_array::UInt64Array>());
    let mut rows = Vec::with_capacity(values.len());
    for row in 0..values.len() {
        if values.is_null(row) {
            rows.push(None);
            continue;
        }
        rows.push(Some(LogicalBlobValue {
            data: data.and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_vec())),
            uri: uri.and_then(|a| (!a.is_null(row)).then(|| a.value(row).to_string())),
            position: position.and_then(|a| (!a.is_null(row)).then(|| a.value(row))),
            size: size.and_then(|a| (!a.is_null(row)).then(|| a.value(row))),
        }));
    }
    Ok(rows)
}

fn contains_blob_reference(rows: &[Option<LogicalBlobValue>]) -> bool {
    rows.iter().flatten().any(|value| value.uri.is_some())
}

/// Build the canonical four-child logical Blob v2 array for a complete
/// replacement column.
fn build_logical_blob_column(
    name: &str,
    rows: &[Option<LogicalBlobValue>],
    nullable: bool,
) -> Result<ArrayRef, String> {
    let mut data = LargeBinaryBuilder::new();
    let mut uri = StringBuilder::new();
    let mut position = PrimitiveBuilder::<UInt64Type>::new();
    let mut size = PrimitiveBuilder::<UInt64Type>::new();
    let mut nulls = NullBufferBuilder::new(rows.len());
    for value in rows {
        match value {
            None => {
                data.append_null();
                uri.append_null();
                position.append_null();
                size.append_null();
                nulls.append(false);
            }
            Some(value) => {
                match &value.data {
                    Some(bytes) => data.append_value(bytes),
                    None => data.append_null(),
                }
                match &value.uri {
                    Some(uri_value) => uri.append_value(uri_value),
                    None => uri.append_null(),
                }
                match value.position {
                    Some(position_value) => position.append_value(position_value),
                    None => position.append_null(),
                }
                match value.size {
                    Some(size_value) => size.append_value(size_value),
                    None => size.append_null(),
                }
                nulls.append(true);
            }
        }
    }
    let DataType::Struct(children) = BLOB_V2_LOGICAL_TYPE.clone() else {
        return Err("canonical blob v2 type is not a struct".to_string());
    };
    let _ = name;
    let _ = nullable;
    let array = arrow_array::StructArray::try_new(
        children,
        vec![
            Arc::new(data.finish()) as ArrayRef,
            Arc::new(uri.finish()) as ArrayRef,
            Arc::new(position.finish()) as ArrayRef,
            Arc::new(size.finish()) as ArrayRef,
        ],
        nulls.finish(),
    )
    .map_err(|err| format!("failed to build logical blob column: {err}"))?;
    Ok(Arc::new(array))
}

struct FragmentDescriptor {
    kind: u8,
    base_id: u32,
    uri: String,
    position: u64,
    size: u64,
}

fn resolve_descriptor_uri(dataset: &lance::Dataset, descriptor: &FragmentDescriptor) -> String {
    if descriptor.base_id == 0 {
        descriptor.uri.clone()
    } else {
        let base = &dataset.manifest.base_paths[&descriptor.base_id];
        format!("{}/{}", base.path, descriptor.uri)
    }
}

/// Rewrite one complete Blob v2 column through Lance's `write_columns`, which
/// accepts brand-new absolute external URIs. Unassigned rows are derived from
/// their stored descriptors: external references are preserved without reading
/// payload bytes, managed values are materialized through native blob reads.
#[allow(clippy::too_many_arguments)]
async fn write_full_blob_column(
    dataset: &Arc<lance::Dataset>,
    fragment: &lance::dataset::fragment::FileFragment,
    column_name: &str,
    field_id: i32,
    assigned_rows: &HashMap<u64, Option<LogicalBlobValue>>,
    deleted_addrs: &RoaringTreemap,
    ingest_budget: usize,
) -> Result<(Fragment, u32, RoaringBitmap), String> {
    use futures::StreamExt;

    let physical_rows = fragment.physical_rows().await.map_err(|e| e.to_string())?;
    let lance_projection = dataset
        .schema()
        .project(&[column_name.to_string()])
        .map_err(|err| format!("project '{column_name}': {err}"))?;

    // Descriptor scan includes deleted physical rows so the replacement column
    // is aligned positionally with the fragment.
    let mut scanner = dataset.scan();
    scanner.with_fragments(vec![fragment.metadata().clone()]);
    scanner
        .project(&[column_name])
        .map_err(|err| format!("project '{column_name}': {err}"))?;
    scanner.blob_handling(lance_core::datatypes::BlobHandling::BlobsDescriptions);
    scanner.include_deleted_rows();
    scanner.with_row_id();
    scanner.with_row_address();
    let mut stream = scanner
        .try_into_stream()
        .await
        .map_err(|err| format!("scan fragment for '{column_name}': {err}"))?;

    let mut descriptors: HashMap<u64, Option<FragmentDescriptor>> = HashMap::new();
    let mut order: Vec<u64> = Vec::with_capacity(physical_rows);
    while let Some(batch) = stream.next().await {
        let batch = batch.map_err(|err| format!("read fragment descriptors: {err}"))?;
        let column = batch
            .column_by_name(column_name)
            .ok_or_else(|| format!("column '{column_name}' missing from descriptor scan"))?;
        let addresses = batch
            .column_by_name(lance_core::ROW_ADDR)
            .ok_or_else(|| "descriptor scan is missing _rowaddr".to_string())?;
        let addresses = addresses
            .as_any()
            .downcast_ref::<arrow_array::UInt64Array>()
            .ok_or_else(|| "descriptor scan row address is not UInt64".to_string())?;
        let values = column
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| format!("column '{column_name}' descriptor is not a struct"))?;
        let kind = values
            .column_by_name("kind")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt8Array>());
        let blob_id = values
            .column_by_name("blob_id")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt32Array>());
        let blob_uri = values
            .column_by_name("blob_uri")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>());
        let position = values
            .column_by_name("position")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>());
        let size = values
            .column_by_name("size")
            .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>());
        for row in 0..batch.num_rows() {
            let address = addresses.value(row);
            order.push(address);
            if values.is_null(row) {
                descriptors.insert(address, None);
            } else {
                descriptors.insert(
                    address,
                    Some(FragmentDescriptor {
                        kind: kind.map(|a| a.value(row)).unwrap_or(0),
                        base_id: blob_id.map(|a| a.value(row)).unwrap_or(0),
                        uri: blob_uri
                            .map(|a| a.value(row).to_string())
                            .unwrap_or_default(),
                        position: position.map(|a| a.value(row)).unwrap_or(0),
                        size: size.map(|a| a.value(row)).unwrap_or(0),
                    }),
                );
            }
        }
    }
    if order.len() != physical_rows {
        return Err(format!(
            "descriptor scan for '{column_name}' returned {} rows, expected {physical_rows}",
            order.len()
        ));
    }

    // Materialize managed rows that have no assigned replacement value.
    let mut managed_addresses = Vec::new();
    for address in &order {
        if assigned_rows.contains_key(address) || deleted_addrs.contains(*address) {
            continue;
        }
        if let Some(Some(descriptor)) = descriptors.get(address) {
            if descriptor.kind != 3 {
                managed_addresses.push(*address);
            }
        }
    }
    let mut managed_bytes: HashMap<u64, Vec<u8>> = HashMap::new();
    let mut remaining_budget = ingest_budget;
    for chunk in managed_addresses.chunks(256) {
        let blobs = dataset
            .take_blobs_by_addresses(chunk, column_name)
            .await
            .map_err(|err| format!("read managed blob values: {err}"))?;
        for (address, blob) in chunk.iter().zip(blobs) {
            let Some(blob) = blob else {
                continue;
            };
            let bytes = blob
                .read()
                .await
                .map_err(|err| format!("read managed blob at {address}: {err}"))?;
            if bytes.len() > remaining_budget {
                return Err(format!(
                    "managed blob column '{column_name}' requires {} bytes, exceeding the remaining materialization budget of {remaining_budget} bytes",
                    bytes.len()
                ));
            }
            remaining_budget -= bytes.len();
            managed_bytes.insert(*address, bytes.to_vec());
        }
    }

    let mut values = Vec::with_capacity(order.len());
    for address in &order {
        if let Some(assigned) = assigned_rows.get(address) {
            values.push(assigned.clone());
            continue;
        }
        if deleted_addrs.contains(*address) {
            values.push(None);
            continue;
        }
        match descriptors.get(address) {
            None | Some(None) => values.push(None),
            Some(Some(descriptor)) => {
                if descriptor.kind == 3 {
                    values.push(Some(LogicalBlobValue {
                        data: None,
                        uri: Some(resolve_descriptor_uri(dataset, descriptor)),
                        position: (descriptor.position != 0 || descriptor.size != 0)
                            .then_some(descriptor.position),
                        size: (descriptor.position != 0 || descriptor.size != 0)
                            .then_some(descriptor.size),
                    }));
                } else if let Some(bytes) = managed_bytes.remove(address) {
                    values.push(Some(LogicalBlobValue {
                        data: Some(bytes),
                        uri: None,
                        position: None,
                        size: None,
                    }));
                } else {
                    values.push(None);
                }
            }
        }
    }

    let array = build_logical_blob_column(column_name, &values, true)?;
    let field = ArrowField::new(column_name, BLOB_V2_LOGICAL_TYPE.clone(), true).with_metadata(
        HashMap::from([(ARROW_EXT_NAME_KEY.to_string(), BLOB_V2_EXT_NAME.to_string())]),
    );
    let batch = RecordBatch::try_new(Arc::new(ArrowSchema::new(vec![field])), vec![array])
        .map_err(|err| format!("failed to build '{column_name}' replacement batch: {err}"))?;
    let stream = futures::stream::iter(vec![Ok(batch)]);
    let replacement = fragment
        .write_columns(stream, &lance_projection)
        .await
        .map_err(|err| format!("write full Blob column '{column_name}': {err}"))?;

    let base = fragment.metadata().clone();
    let original_paths: HashSet<&str> = base.files.iter().map(|file| file.path.as_str()).collect();
    let mut combined = base.clone();
    for file in combined.files.iter_mut() {
        let new_fields: Vec<i32> = file
            .fields
            .iter()
            .map(|id| if *id == field_id { -2 } else { *id })
            .collect();
        file.fields = Arc::from(new_fields);
    }
    combined
        .files
        .retain(|file| file.fields.iter().any(|id| *id != -2));
    if !original_paths.contains(replacement.1.path.as_str()) {
        combined.files.push(replacement.1.clone());
    }

    let mut offsets = RoaringBitmap::new();
    for address in assigned_rows.keys() {
        let address = RowAddress::new_from_u64(*address);
        if address.fragment_id() != fragment.id() as u32 {
            return Err(format!(
                "assigned row address '{}' belongs to fragment {}, not fragment {}",
                address,
                address.fragment_id(),
                fragment.id()
            ));
        }
        offsets.insert(address.row_offset());
    }
    Ok((combined, field_id as u32, offsets))
}
