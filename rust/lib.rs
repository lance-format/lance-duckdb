#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StructArray};
use arrow::datatypes::{DataType, Schema};
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use lance::Dataset;

mod runtime;
mod scanner;

use scanner::LanceStream;

// Dataset operations - just holds the dataset
struct DatasetHandle {
    dataset: Arc<Dataset>,
}

#[no_mangle]
pub unsafe extern "C" fn lance_open_dataset(path: *const c_char) -> *mut c_void {
    if path.is_null() {
        return ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    let dataset = match runtime::block_on(Dataset::open(path_str)) {
        Ok(Ok(ds)) => Arc::new(ds),
        _ => return ptr::null_mut(),
    };

    let handle = Box::new(DatasetHandle { dataset });

    Box::into_raw(handle) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_dataset(dataset: *mut c_void) {
    if !dataset.is_null() {
        unsafe {
            let _ = Box::from_raw(dataset as *mut DatasetHandle);
        }
    }
}

// Schema operations
#[no_mangle]
pub unsafe extern "C" fn lance_get_schema(dataset: *mut c_void) -> *mut c_void {
    if dataset.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };
    let schema = handle.dataset.schema();

    let arrow_schema: Schema = schema.into();

    Box::into_raw(Box::new(Arc::new(arrow_schema))) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_schema(schema: *mut c_void) {
    if !schema.is_null() {
        unsafe {
            let _ = Box::from_raw(schema as *mut Arc<Schema>);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_schema_to_arrow(
    schema: *mut c_void,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    if schema.is_null() || out_schema.is_null() {
        return -1;
    }

    let schema = unsafe { &*(schema as *const Arc<Schema>) };
    let data_type = DataType::Struct(schema.fields().clone());

    let ffi_schema = match FFI_ArrowSchema::try_from(&data_type) {
        Ok(schema) => schema,
        Err(_) => return -1,
    };

    std::ptr::write_unaligned(out_schema, ffi_schema);
    0
}

// Stream operations
#[no_mangle]
pub unsafe extern "C" fn lance_create_stream(dataset: *mut c_void) -> *mut c_void {
    if dataset.is_null() {
        return ptr::null_mut();
    }

    let handle = unsafe { &*(dataset as *const DatasetHandle) };

    match LanceStream::new(&handle.dataset) {
        Ok(stream) => Box::into_raw(Box::new(stream)) as *mut c_void,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_stream_next(stream: *mut c_void) -> *mut c_void {
    if stream.is_null() {
        return ptr::null_mut();
    }

    let stream = unsafe { &mut *(stream as *mut LanceStream) };

    match stream.next() {
        Some(batch) => Box::into_raw(Box::new(batch)) as *mut c_void,
        None => ptr::null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_stream(stream: *mut c_void) {
    if !stream.is_null() {
        unsafe {
            let _ = Box::from_raw(stream as *mut LanceStream);
        }
    }
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
pub unsafe extern "C" fn lance_batch_num_rows(batch: *mut c_void) -> i64 {
    if batch.is_null() {
        return 0;
    }

    let batch = unsafe { &*(batch as *const RecordBatch) };
    batch.num_rows() as i64
}

// Export RecordBatch to Arrow C Data Interface
#[no_mangle]
pub unsafe extern "C" fn lance_batch_to_arrow(
    batch: *mut c_void,
    out_array: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> i32 {
    if batch.is_null() || out_array.is_null() || out_schema.is_null() {
        return -1;
    }

    let batch = unsafe { &*(batch as *const RecordBatch) };

    // Convert RecordBatch to StructArray for FFI export
    let struct_array: Arc<dyn Array> = Arc::new(StructArray::from(batch.clone()));

    let data = struct_array.to_data();
    let array = FFI_ArrowArray::new(&data);
    let schema = match FFI_ArrowSchema::try_from(data.data_type()) {
        Ok(schema) => schema,
        Err(_) => return -1,
    };

    std::ptr::write_unaligned(out_array, array);
    std::ptr::write_unaligned(out_schema, schema);

    0
}
