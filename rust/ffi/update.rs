use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_void, CStr};
use std::sync::{Arc, Mutex};

use arrow_array::cast::AsArray;
use arrow_array::types::UInt64Type;
use datafusion::common::DFSchema;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::ExprSchemable;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::SendableRecordBatchStream;
use datafusion_common::{DataFusionError, Result as DFResult};
use futures::{StreamExt, TryStreamExt};
use lance::dataset::transaction::{Operation, Transaction, UpdateMode};
use lance::dataset::{InsertBuilder, WriteMode, WriteParams};
use lance::io::exec::Planner;
use lance::io::ObjectStoreParams;
use lance_arrow::RecordBatchExt;
use lance_core::utils::deletion::DeletionVector;
use lance_core::ROW_ID;
use lance_table::format::RowIdMeta;
use lance_table::rowids::{
    read_row_ids, rechunk_sequences, write_row_ids, FragmentRowIdIndex, RowIdIndex, RowIdSequence,
};
use roaring::RoaringTreemap;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, slice_from_ptr, FfiError, FfiResult};

#[no_mangle]
pub unsafe extern "C" fn lance_overwrite_update_transaction_with_storage_options(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    predicate: *const c_char,
    set_columns: *const *const c_char,
    set_expressions: *const *const c_char,
    set_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    out_transaction: *mut *mut c_void,
    out_rows_updated: *mut u64,
) -> i32 {
    match rewrite_rows_update_transaction_inner(
        path,
        option_keys,
        option_values,
        options_len,
        predicate,
        set_columns,
        set_expressions,
        set_len,
        max_rows_per_file,
        max_rows_per_group,
        max_bytes_per_file,
        out_transaction,
        out_rows_updated,
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

#[derive(Debug, Clone)]
enum CapturedRowIds {
    AddressStyle(RoaringTreemap),
    SequenceStyle(RowIdSequence),
}

impl CapturedRowIds {
    fn new(stable_row_ids: bool) -> Self {
        if stable_row_ids {
            Self::SequenceStyle(RowIdSequence::new())
        } else {
            Self::AddressStyle(RoaringTreemap::new())
        }
    }

    fn capture(&mut self, row_ids: &[u64]) -> DFResult<()> {
        match self {
            Self::AddressStyle(addrs) => {
                addrs
                    .append(row_ids.iter().copied())
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }
            Self::SequenceStyle(sequence) => {
                sequence.extend(row_ids.into());
            }
        }
        Ok(())
    }

    fn len(&self) -> u64 {
        match self {
            Self::AddressStyle(addrs) => addrs.len(),
            Self::SequenceStyle(sequence) => sequence.len(),
        }
    }
}

fn rewrite_rows_update_transaction_inner(
    path: *const c_char,
    option_keys: *const *const c_char,
    option_values: *const *const c_char,
    options_len: usize,
    predicate: *const c_char,
    set_columns: *const *const c_char,
    set_expressions: *const *const c_char,
    set_len: usize,
    max_rows_per_file: u64,
    max_rows_per_group: u64,
    max_bytes_per_file: u64,
    out_transaction: *mut *mut c_void,
    out_rows_updated: *mut u64,
) -> FfiResult<()> {
    if out_transaction.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_transaction is null",
        ));
    }
    if out_rows_updated.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_rows_updated is null",
        ));
    }

    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let predicate = unsafe { cstr_to_str(predicate, "predicate")? }.to_string();

    if predicate.trim().is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "predicate cannot be empty",
        ));
    }

    if set_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "set_len must be > 0",
        ));
    }
    if set_columns.is_null() || set_expressions.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "set_columns/set_expressions is null with non-zero length",
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

    let set_cols_slice = unsafe { slice_from_ptr(set_columns, set_len, "set_columns")? };
    let set_exprs_slice = unsafe { slice_from_ptr(set_expressions, set_len, "set_expressions")? };

    let mut set_pairs = Vec::with_capacity(set_len);
    for (idx, (&col_ptr, &expr_ptr)) in set_cols_slice
        .iter()
        .zip(set_exprs_slice.iter())
        .enumerate()
    {
        if col_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("set_columns[{idx}] is null"),
            ));
        }
        if expr_ptr.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("set_expressions[{idx}] is null"),
            ));
        }

        let col = unsafe { CStr::from_ptr(col_ptr) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("set_columns[{idx}] utf8: {err}"))
        })?;
        let expr = unsafe { CStr::from_ptr(expr_ptr) }
            .to_str()
            .map_err(|err| {
                FfiError::new(
                    ErrorCode::Utf8,
                    format!("set_expressions[{idx}] utf8: {err}"),
                )
            })?;
        set_pairs.push((col.to_string(), expr.to_string()));
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

    let mut store_params = ObjectStoreParams::default();
    if !storage_options.is_empty() {
        store_params.storage_options = Some(storage_options.clone());
    }

    let (maybe_txn, rows_updated) = match runtime::block_on(async {
        let dataset = lance::dataset::builder::DatasetBuilder::from_uri(path.as_str())
            .with_storage_options(storage_options)
            .load()
            .await
            .map_err(|e| e.to_string())?;
        let dataset = Arc::new(dataset);

        let arrow_schema: Arc<arrow_schema::Schema> = Arc::new(dataset.schema().into());
        let planner = Planner::new(arrow_schema.clone());
        let predicate_expr = planner
            .parse_filter(predicate.as_str())
            .map_err(|e| e.to_string())?;
        let predicate_expr = planner
            .optimize_expr(predicate_expr)
            .map_err(|e| e.to_string())?;

        let df_schema =
            DFSchema::try_from(arrow_schema.as_ref().clone()).map_err(|e| e.to_string())?;
        let session_ctx = SessionContext::new();
        let mut update_exprs =
            HashMap::<String, Arc<dyn datafusion::physical_expr::PhysicalExpr>>::new();

        for (column, value_sql) in &set_pairs {
            if column.contains('.') {
                return Err(format!(
                    "nested column references are not supported: {}",
                    column
                ));
            }

            let field = dataset
                .schema()
                .field(column.as_str())
                .ok_or_else(|| format!("column does not exist: {}", column))?;

            let mut value_expr = planner
                .parse_expr(value_sql.as_str())
                .map_err(|e| e.to_string())?;

            let dest_type = field.data_type();
            let src_type = value_expr.get_type(&df_schema).map_err(|e| e.to_string())?;
            if dest_type != src_type {
                value_expr = value_expr
                    .cast_to(&dest_type, &df_schema)
                    .map_err(|e| e.to_string())?;
            }

            let value_expr = planner
                .optimize_expr(value_expr)
                .map_err(|e| e.to_string())?;
            let physical_expr = session_ctx
                .create_physical_expr(value_expr, &df_schema)
                .map_err(|e| e.to_string())?;
            update_exprs.insert(column.clone(), physical_expr);
        }

        let stable_row_ids = dataset.manifest.uses_stable_row_ids();

        let mut scanner = dataset.scan();
        scanner.with_row_id();
        scanner.filter_expr(predicate_expr);

        let mut input_stream: SendableRecordBatchStream = scanner
            .try_into_stream()
            .await
            .map_err(|e| e.to_string())?
            .into();

        let input_schema = input_stream.schema();
        let first_batch = input_stream.try_next().await.map_err(|e| e.to_string())?;
        let Some(first_batch) = first_batch else {
            return Ok::<_, String>((None, 0));
        };

        let base_stream = futures::stream::iter(Some(Ok(first_batch)))
            .chain(input_stream)
            .boxed();
        let base_stream = RecordBatchStreamAdapter::new(input_schema.clone(), base_stream);

        let update_exprs_ref = Arc::new(update_exprs);
        let captured_row_ids = Arc::new(Mutex::new(CapturedRowIds::new(stable_row_ids)));
        let captured_row_ids_ref = captured_row_ids.clone();
        let stream = base_stream
            .map(move |batch_res| {
                let batch = batch_res?;
                let row_ids = batch
                    .column_by_name(ROW_ID)
                    .ok_or_else(|| DataFusionError::Execution("missing _rowid column".to_string()))?
                    .as_primitive::<UInt64Type>()
                    .values();

                // Capture row ids for later deletion and stable row id preservation.
                // For non-stable row ids, these are row addresses (RowAddress as u64).
                // For stable row ids, these are stable row ids (requiring a RowIdIndex to map).
                captured_row_ids_ref.lock().unwrap().capture(row_ids)?;

                let mut batch = batch
                    .drop_column(ROW_ID)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                // Evaluate all SET expressions against the original batch to match
                // SQL semantics (assignments are based on pre-update values).
                let mut new_values = Vec::with_capacity(update_exprs_ref.len());
                for (col, expr) in update_exprs_ref.iter() {
                    let arr = expr.evaluate(&batch)?.into_array(batch.num_rows())?;
                    new_values.push((col.as_str(), arr));
                }
                for (col, arr) in new_values {
                    batch = batch
                        .replace_column_by_name(col, arr)
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                }
                Ok(batch)
            })
            .boxed();

        let updated_stream: SendableRecordBatchStream =
            Box::pin(RecordBatchStreamAdapter::new(arrow_schema.clone(), stream));

        let write_params = WriteParams {
            mode: WriteMode::Append,
            max_rows_per_file,
            max_rows_per_group,
            max_bytes_per_file,
            store_params: Some(store_params),
            ..Default::default()
        };

        let append_txn = InsertBuilder::new(dataset.clone())
            .with_params(&write_params)
            .execute_uncommitted_stream(updated_stream)
            .await
            .map_err(|e| e.to_string())?;

        let captured_row_ids = captured_row_ids.lock().unwrap().clone();
        let rows_updated = captured_row_ids.len();

        let Operation::Append { fragments } = append_txn.operation else {
            return Err("unexpected transaction operation for update write".to_string());
        };

        let mut new_fragments = fragments;

        if stable_row_ids {
            let CapturedRowIds::SequenceStyle(sequence) = &captured_row_ids else {
                return Err(
                    "stable row ids enabled but captured row ids are not sequence style"
                        .to_string(),
                );
            };

            let fragment_sizes = new_fragments
                .iter()
                .map(|f| f.physical_rows.unwrap_or_default() as u64);
            let sequences = rechunk_sequences(vec![sequence.clone()], fragment_sizes, false)
                .map_err(|e| e.to_string())?;
            for (fragment, seq) in new_fragments.iter_mut().zip(sequences) {
                fragment.row_id_meta = Some(RowIdMeta::Inline(write_row_ids(&seq)));
            }
        }

        let row_addrs = match &captured_row_ids {
            CapturedRowIds::AddressStyle(addrs) => addrs.clone(),
            CapturedRowIds::SequenceStyle(sequence) => {
                let row_id_index = build_row_id_index(dataset.as_ref())
                    .await
                    .map_err(|e| e.to_string())?;
                let mut addrs = RoaringTreemap::new();
                for row_id in sequence.iter() {
                    let addr = row_id_index
                        .get(row_id)
                        .ok_or_else(|| format!("row id missing from row id index: {row_id}"))?;
                    addrs.insert(u64::from(addr));
                }
                addrs
            }
        };

        let (updated_fragments, removed_fragment_ids) =
            apply_deletions(dataset.as_ref(), &row_addrs)
                .await
                .map_err(|e| e.to_string())?;

        let mut fields_for_preserving_frag_bitmap = Vec::new();
        for (column_name, _) in set_pairs {
            if let Ok(field_id) = dataset.schema().field_id(column_name.as_str()) {
                fields_for_preserving_frag_bitmap.push(field_id as u32);
            }
        }

        let operation = Operation::Update {
            removed_fragment_ids,
            updated_fragments,
            new_fragments,
            fields_modified: vec![],
            mem_wal_to_merge: None,
            fields_for_preserving_frag_bitmap,
            update_mode: Some(UpdateMode::RewriteRows),
        };

        let txn = Transaction::new(dataset.manifest.version, operation, None);

        Ok::<_, String>((Some(txn), rows_updated))
    }) {
        Ok(Ok(v)) => v,
        Ok(Err(message)) => return Err(FfiError::new(ErrorCode::DatasetUpdateOverwrite, message)),
        Err(err) => {
            return Err(FfiError::new(
                ErrorCode::DatasetUpdateOverwrite,
                format!("runtime: {err}"),
            ))
        }
    };

    unsafe {
        *out_rows_updated = rows_updated;
        if let Some(txn) = maybe_txn {
            let boxed = Box::new(txn);
            *out_transaction = Box::into_raw(boxed) as *mut c_void;
        } else {
            *out_transaction = std::ptr::null_mut();
        }
    };

    Ok(())
}

async fn build_row_id_index(dataset: &lance::Dataset) -> Result<RowIdIndex, String> {
    if !dataset.manifest.uses_stable_row_ids() {
        return Err("row id index requested for dataset without stable row ids".to_string());
    }

    let base = dataset.branch_location().path;
    let fragments = dataset.get_fragments();

    let mut indices = Vec::with_capacity(dataset.manifest.fragments.len());
    for fragment in dataset.manifest.fragments.iter() {
        let row_id_meta = fragment
            .row_id_meta
            .as_ref()
            .ok_or_else(|| "missing row id meta".to_string())?;
        let row_id_bytes = match row_id_meta {
            RowIdMeta::Inline(data) => data.clone(),
            RowIdMeta::External(file) => {
                let path = base.child(file.path.as_str());
                let range = file.offset as usize..(file.offset + file.size) as usize;
                dataset
                    .object_store
                    .open(&path)
                    .await
                    .map_err(|e| e.to_string())?
                    .get_range(range)
                    .await
                    .map_err(|e| e.to_string())?
                    .to_vec()
            }
        };

        let sequence = read_row_ids(&row_id_bytes).map_err(|e| e.to_string())?;
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
            row_id_sequence: Arc::new(sequence),
            deletion_vector,
        });
    }

    RowIdIndex::new(&indices).map_err(|e| e.to_string())
}

async fn apply_deletions(
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
