use std::ffi::c_void;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StructArray};
use arrow::datatypes::DataType;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::builder::Float32Builder;
use arrow_array::Float32Array;

use crate::constants::DISTANCE_COLUMN;
use crate::error::{clear_last_error, set_last_error, ErrorCode};

use super::types::SchemaHandle;
use super::util::{batch_handle, schema_handle, schema_to_ffi_arrow_schema, FfiError, FfiResult};

#[no_mangle]
pub unsafe extern "C" fn lance_free_schema(schema: *mut c_void) {
    if !schema.is_null() {
        unsafe {
            let _ = Box::from_raw(schema as *mut SchemaHandle);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_schema_to_arrow(
    schema: *mut c_void,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    match schema_to_arrow_inner(schema, out_schema) {
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

fn schema_to_arrow_inner(schema: *mut c_void, out_schema: *mut FFI_ArrowSchema) -> FfiResult<()> {
    if out_schema.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_schema is null",
        ));
    }
    let schema = unsafe { schema_handle(schema)? };
    let ffi_schema = schema_to_ffi_arrow_schema(schema)?;
    unsafe {
        std::ptr::write_unaligned(out_schema, ffi_schema);
    }
    Ok(())
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_batch(batch: *mut c_void) {
    if !batch.is_null() {
        unsafe {
            let _ = Box::from_raw(batch as *mut RecordBatch);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_batch_to_arrow(
    batch: *mut c_void,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    match batch_to_arrow_inner(batch, out_array, out_schema) {
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

fn batch_to_arrow_inner(
    batch: *mut c_void,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> FfiResult<()> {
    if out_array.is_null() || out_schema.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "out_array/out_schema is null",
        ));
    }
    let batch = unsafe { batch_handle(batch)? };
    let batch = normalize_distance_column(batch)?;

    let struct_array: Arc<dyn Array> = Arc::new(StructArray::from(batch));
    let data = struct_array.to_data();
    let array = FFI_ArrowArray::new(&data);
    let schema = FFI_ArrowSchema::try_from(data.data_type())
        .map_err(|err| FfiError::new(ErrorCode::BatchExport, format!("batch export: {err}")))?;

    unsafe {
        std::ptr::write_unaligned(out_array, array);
        std::ptr::write_unaligned(out_schema, schema);
    }
    Ok(())
}

fn normalize_distance_column(batch: &RecordBatch) -> FfiResult<RecordBatch> {
    let schema = batch.schema();
    let idx = match schema.index_of(DISTANCE_COLUMN) {
        Ok(v) => v,
        Err(_) => return Ok(batch.clone()),
    };

    let col = batch.column(idx);
    if col.data_type() != &DataType::Float32 {
        return Ok(batch.clone());
    }

    let col = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| FfiError::new(ErrorCode::BatchExport, "distance column type mismatch"))?;

    let mut builder = Float32Builder::with_capacity(col.len());
    for i in 0..col.len() {
        if col.is_null(i) {
            builder.append_null();
        } else {
            builder.append_value(col.value(i));
        }
    }
    let normalized = Arc::new(builder.finish()) as Arc<dyn Array>;

    let mut cols: Vec<Arc<dyn Array>> = batch.columns().to_vec();
    cols[idx] = normalized;

    RecordBatch::try_new(schema.clone(), cols)
        .map_err(|err| FfiError::new(ErrorCode::BatchExport, format!("batch export: {err}")))
}
