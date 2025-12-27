use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

use arrow::compute;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Schema as ArrowSchema};
use chrono::Duration;
use lance::dataset::optimize::{compact_files, CompactionOptions};
use lance::dataset::{BatchUDF, ColumnAlteration, NewColumnTransform};
use lance::Dataset;
use lance_index::scalar::ScalarIndexParams;
use lance_index::DatasetIndexExt;
use lance_index::IndexType;
use snafu::location;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, optional_cstr_array, to_c_string, FfiError, FfiResult};

fn parse_batch_size_from_config(dataset: &Dataset) -> Option<u32> {
    dataset
        .config()
        .get("lance.add_columns.batch_size")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| *v > 0)
}

fn parse_arrow_schema(schema: *const c_void, what: &'static str) -> FfiResult<Arc<ArrowSchema>> {
    if schema.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }

    let ffi_schema = unsafe { &*(schema as *const arrow_schema::ffi::FFI_ArrowSchema) };
    let data_type = DataType::try_from(ffi_schema).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("{what} import: {err}"))
    })?;
    let DataType::Struct(fields) = data_type else {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must be a struct"),
        ));
    };
    Ok(Arc::new(ArrowSchema::new(fields)))
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_add_columns(
    dataset: *mut c_void,
    new_columns_schema: *const c_void,
    expressions: *const *const c_char,
    expressions_len: usize,
    batch_size: u32,
) -> i32 {
    match dataset_add_columns_inner(dataset, new_columns_schema, expressions, expressions_len, batch_size) {
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

fn dataset_add_columns_inner(
    dataset: *mut c_void,
    new_columns_schema: *const c_void,
    expressions: *const *const c_char,
    expressions_len: usize,
    batch_size: u32,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let mut ds = (*handle.dataset).clone();

    let output_schema = parse_arrow_schema(new_columns_schema, "new_columns_schema")?;
    if output_schema.fields().is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "new_columns_schema must have at least one field",
        ));
    }

    let batch_size = if batch_size == 0 {
        parse_batch_size_from_config(&ds)
    } else {
        Some(batch_size)
    };

    if expressions_len == 0 {
        let transforms = NewColumnTransform::AllNulls(output_schema);
        match runtime::block_on(ds.add_columns(transforms, None, batch_size)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(FfiError::new(
                ErrorCode::DatasetAddColumns,
                format!("dataset add_columns(all_nulls): {err}"),
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    } else {
        let exprs = unsafe { optional_cstr_array(expressions, expressions_len, "expressions")? };
        if exprs.len() != output_schema.fields().len() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "expressions_len must match new_columns_schema field count",
            ));
        }

        let full_read_schema = Arc::new(ArrowSchema::from(ds.schema()));
        let planner = lance::io::exec::Planner::new(full_read_schema);

        let parsed = exprs
            .iter()
            .enumerate()
            .map(|(idx, expr)| {
                let expr = planner.parse_expr(expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] parse: {err}"),
                    )
                })?;
                let expr = planner.optimize_expr(expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] optimize: {err}"),
                    )
                })?;
                Ok(expr)
            })
            .collect::<FfiResult<Vec<_>>>()?;

        let mut needed = HashSet::<String>::new();
        for expr in parsed.iter() {
            for col in lance::io::exec::Planner::column_names_in_expr(expr) {
                needed.insert(col);
            }
        }
        let needed_columns = needed.into_iter().collect::<Vec<_>>();
        let read_schema = ds.schema().project(&needed_columns).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetAddColumns,
                format!("read schema project: {err}"),
            )
        })?;
        let read_schema = Arc::new(ArrowSchema::from(&read_schema));
        let planner = lance::io::exec::Planner::new(read_schema.clone());

        let physical = parsed
            .into_iter()
            .enumerate()
            .map(|(idx, expr)| {
                planner.create_physical_expr(&expr).map_err(|err| {
                    FfiError::new(
                        ErrorCode::DatasetAddColumns,
                        format!("expression[{idx}] physical: {err}"),
                    )
                })
            })
            .collect::<FfiResult<Vec<_>>>()?;

        let output_fields = output_schema.fields().to_vec();
        let schema_ref = output_schema.clone();
        let mapper = move |batch: &RecordBatch| {
            let num_rows = batch.num_rows();
            let mut arrays = Vec::with_capacity(physical.len());
            for (idx, (field, expr)) in output_fields.iter().zip(physical.iter()).enumerate() {
                let arr = expr
                    .evaluate(batch)
                    .map_err(|err| {
                        lance::Error::invalid_input(
                            format!("expression[{idx}] evaluate: {err}"),
                            location!(),
                        )
                    })?
                    .into_array(num_rows)
                    .map_err(|err| {
                        lance::Error::invalid_input(
                            format!("expression[{idx}] into_array: {err}"),
                            location!(),
                        )
                    })?;
                let arr = if arr.data_type() != field.data_type() {
                    compute::cast(&arr, field.data_type())
                        .map_err(|err| {
                            lance::Error::invalid_input(
                                format!("expression[{idx}] cast: {err}"),
                                location!(),
                            )
                        })?
                } else {
                    arr
                };
                arrays.push(arr);
            }
            RecordBatch::try_new(schema_ref.clone(), arrays).map_err(|err| {
                lance::Error::invalid_input(format!("output batch: {err}"), location!())
            })
        };

        let transforms = NewColumnTransform::BatchUDF(BatchUDF {
            mapper: Box::new(mapper),
            output_schema,
            result_checkpoint: None,
        });

        let read_columns = Some(
            read_schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect::<Vec<_>>(),
        );
        match runtime::block_on(ds.add_columns(transforms, read_columns, batch_size)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(FfiError::new(
                ErrorCode::DatasetAddColumns,
                format!("dataset add_columns(sql): {err}"),
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_drop_columns(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
) -> i32 {
    match dataset_drop_columns_inner(dataset, columns, columns_len) {
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

fn dataset_drop_columns_inner(
    dataset: *mut c_void,
    columns: *const *const c_char,
    columns_len: usize,
) -> FfiResult<()> {
    if columns_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "columns_len must be > 0",
        ));
    }
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let cols = unsafe { optional_cstr_array(columns, columns_len, "columns")? };

    let mut ds = (*handle.dataset).clone();
    let col_refs = cols.iter().map(|c| c.as_str()).collect::<Vec<_>>();
    match runtime::block_on(ds.drop_columns(&col_refs)) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetDropColumns,
            format!("dataset drop_columns: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_rename(
    dataset: *mut c_void,
    path: *const c_char,
    new_name: *const c_char,
) -> i32 {
    match dataset_alter_columns_rename_inner(dataset, path, new_name) {
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

fn dataset_alter_columns_rename_inner(
    dataset: *mut c_void,
    path: *const c_char,
    new_name: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();
    let new_name = unsafe { cstr_to_str(new_name, "new_name")? }.to_string();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).rename(new_name);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetAlterColumns,
            format!("dataset alter_columns(rename): {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_set_nullable(
    dataset: *mut c_void,
    path: *const c_char,
    nullable: u8,
) -> i32 {
    match dataset_alter_columns_set_nullable_inner(dataset, path, nullable != 0) {
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

fn dataset_alter_columns_set_nullable_inner(
    dataset: *mut c_void,
    path: *const c_char,
    nullable: bool,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).set_nullable(nullable);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetAlterColumns,
            format!("dataset alter_columns(nullable): {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_alter_columns_cast(
    dataset: *mut c_void,
    path: *const c_char,
    new_type_schema: *const c_void,
) -> i32 {
    match dataset_alter_columns_cast_inner(dataset, path, new_type_schema) {
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

fn dataset_alter_columns_cast_inner(
    dataset: *mut c_void,
    path: *const c_char,
    new_type_schema: *const c_void,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let path = unsafe { cstr_to_str(path, "path")? }.to_string();

    let type_schema = parse_arrow_schema(new_type_schema, "new_type_schema")?;
    if type_schema.fields().len() != 1 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "new_type_schema must have exactly one field",
        ));
    }
    let new_type = type_schema.fields()[0].data_type().clone();

    let mut ds = (*handle.dataset).clone();
    let alteration = ColumnAlteration::new(path).cast_to(new_type);
    match runtime::block_on(ds.alter_columns(&[alteration])) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetAlterColumns,
            format!("dataset alter_columns(cast): {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_table_metadata(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_table_metadata_inner(dataset, key, value) {
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

fn dataset_update_table_metadata_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    let value = if value.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(value) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}"))
        })?)
    };

    let mut ds = (*handle.dataset).clone();
    let updates = [(key.as_str(), value)];
    match runtime::block_on(async { ds.update_metadata(updates).await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetUpdateMetadata,
            format!("dataset update_metadata: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_config(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_config_inner(dataset, key, value) {
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

fn dataset_update_config_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    let value = if value.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(value) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}"))
        })?)
    };

    let mut ds = (*handle.dataset).clone();
    let updates = [(key.as_str(), value)];
    match runtime::block_on(async { ds.update_config(updates).await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetUpdateConfig,
            format!("dataset update_config: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_schema_metadata(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_schema_metadata_inner(dataset, key, value) {
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

fn dataset_update_schema_metadata_inner(
    dataset: *mut c_void,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    let value = if value.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(value) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}"))
        })?)
    };

    let mut ds = (*handle.dataset).clone();
    let updates = [(key.as_str(), value)];
    match runtime::block_on(async { ds.update_schema_metadata(updates).await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetUpdateSchemaMetadata,
            format!("dataset update_schema_metadata: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_update_field_metadata(
    dataset: *mut c_void,
    field_path: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> i32 {
    match dataset_update_field_metadata_inner(dataset, field_path, key, value) {
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

fn dataset_update_field_metadata_inner(
    dataset: *mut c_void,
    field_path: *const c_char,
    key: *const c_char,
    value: *const c_char,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let field_path = unsafe { cstr_to_str(field_path, "field_path")? }.to_string();
    let key = unsafe { cstr_to_str(key, "key")? }.to_string();
    let value = if value.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(value) }.to_str().map_err(|err| {
            FfiError::new(ErrorCode::Utf8, format!("value utf8: {err}"))
        })?)
    };

    let mut ds = (*handle.dataset).clone();
    let mut builder = ds.update_field_metadata();
    builder = builder
        .update(field_path.as_str(), [(key.as_str(), value)])
        .map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetUpdateFieldMetadata,
                format!("update_field_metadata: {err}"),
            )
        })?;

    match runtime::block_on(async { builder.await }) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetUpdateFieldMetadata,
            format!("dataset update_field_metadata: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_compact_files(dataset: *mut c_void) -> i32 {
    match dataset_compact_files_inner(dataset) {
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

fn dataset_compact_files_inner(dataset: *mut c_void) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let mut ds = (*handle.dataset).clone();

    match runtime::block_on(compact_files(&mut ds, CompactionOptions::default(), None)) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCompactFiles,
            format!("dataset compact_files: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_cleanup_old_versions(
    dataset: *mut c_void,
    older_than_seconds: i64,
    delete_unverified: u8,
) -> i32 {
    match dataset_cleanup_old_versions_inner(dataset, older_than_seconds, delete_unverified != 0) {
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

fn dataset_cleanup_old_versions_inner(
    dataset: *mut c_void,
    older_than_seconds: i64,
    delete_unverified: bool,
) -> FfiResult<()> {
    if older_than_seconds < 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "older_than_seconds must be >= 0",
        ));
    }
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();

    let duration = Duration::seconds(older_than_seconds);
    match runtime::block_on(ds.cleanup_old_versions(duration, Some(delete_unverified), None)) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCleanupOldVersions,
            format!("dataset cleanup_old_versions: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_config(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "config") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_table_metadata(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "metadata") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_schema_metadata(dataset: *mut c_void) -> *const c_char {
    match dataset_list_kv_inner(dataset, "schema_metadata") {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_field_metadata(
    dataset: *mut c_void,
    field_path: *const c_char,
) -> *const c_char {
    match dataset_list_field_metadata_inner(dataset, field_path) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_field_metadata_inner(
    dataset: *mut c_void,
    field_path: *const c_char,
) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let field_path = unsafe { cstr_to_str(field_path, "field_path")? };

    let field = ds.schema().field(field_path).ok_or_else(|| {
        FfiError::new(
            ErrorCode::DatasetListKeyValues,
            format!("field not found: '{field_path}'"),
        )
    })?;
    let mut out = String::new();
    for (k, v) in field.metadata.iter() {
        out.push_str(k);
        out.push('\t');
        out.push_str(v);
        out.push('\n');
    }
    Ok(out)
}

fn dataset_list_kv_inner(dataset: *mut c_void, which: &'static str) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let mut out = String::new();

    match which {
        "config" => {
            for (k, v) in ds.config().iter() {
                out.push_str(k);
                out.push('\t');
                out.push_str(v);
                out.push('\n');
            }
        }
        "metadata" => {
            for (k, v) in ds.metadata().iter() {
                out.push_str(k);
                out.push('\t');
                out.push_str(v);
                out.push('\n');
            }
        }
        "schema_metadata" => {
            for (k, v) in ds.schema().metadata.iter() {
                out.push_str(k);
                out.push('\t');
                out.push_str(v);
                out.push('\n');
            }
        }
        _ => {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "unknown kv type",
            ))
        }
    }

    Ok(out)
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_indices(dataset: *mut c_void) -> *const c_char {
    match dataset_list_indices_inner(dataset) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_indices_inner(dataset: *mut c_void) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();

    let indices = match runtime::block_on(ds.load_indices()) {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetListIndices,
                format!("dataset load_indices: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let schema = ds.schema();
    let mut out = String::new();
    for idx in indices.iter() {
        out.push_str(&idx.name);
        out.push('\t');
        let cols = idx
            .fields
            .iter()
            .filter_map(|id| schema.field_by_id(*id).map(|f| f.name.as_str()))
            .collect::<Vec<_>>()
            .join(",");
        out.push_str(&cols);
        out.push('\n');
    }
    Ok(out)
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_create_scalar_index(
    dataset: *mut c_void,
    column: *const c_char,
    index_name: *const c_char,
    replace: u8,
) -> i32 {
    match dataset_create_scalar_index_inner(dataset, column, index_name, replace != 0) {
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

fn dataset_create_scalar_index_inner(
    dataset: *mut c_void,
    column: *const c_char,
    index_name: *const c_char,
    replace: bool,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let column = unsafe { cstr_to_str(column, "column")? };
    let index_name = unsafe { cstr_to_str(index_name, "index_name")? };

    let mut ds = (*handle.dataset).clone();
    let cols = [column];
    match runtime::block_on(ds.create_index(
        &cols,
        IndexType::Scalar,
        Some(index_name.to_string()),
        &ScalarIndexParams::default(),
        replace,
    )) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetCreateScalarIndex,
            format!("dataset create_index(scalar): {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}
