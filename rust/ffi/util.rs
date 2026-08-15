use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::Arc;

use anyhow::Context;
use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use datafusion_expr::Expr;
use lance::session::Session;
use lance_core::datatypes::Schema as LanceSchema;

use crate::error::{clear_last_error, set_last_error, ErrorCode};

use super::types::{DatasetHandle, SchemaHandle, SessionHandle, StreamHandle};

#[derive(Debug)]
pub(crate) struct FfiError {
    pub(crate) code: ErrorCode,
    pub(crate) message: String,
}

impl FfiError {
    pub(crate) fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

pub(crate) type FfiResult<T> = Result<T, FfiError>;

#[repr(C)]
pub struct LanceStringList {
    pub items: *mut *mut c_char,
    pub count: usize,
}

pub(crate) unsafe fn write_string_list(
    values: Vec<String>,
    out: *mut LanceStringList,
) -> FfiResult<()> {
    if out.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "string list output is null",
        ));
    }

    let count = values.len();
    let items = if values.is_empty() {
        std::ptr::null_mut()
    } else {
        let pointers = values
            .into_iter()
            .map(|value| to_c_string(value).into_raw())
            .collect::<Vec<_>>();
        let mut boxed = pointers.into_boxed_slice();
        let items = boxed.as_mut_ptr();
        std::mem::forget(boxed);
        items
    };
    unsafe { std::ptr::write_unaligned(out, LanceStringList { items, count }) };
    Ok(())
}

pub(crate) unsafe fn export_string_list(
    result: FfiResult<Vec<String>>,
    out: *mut LanceStringList,
) -> i32 {
    match result.and_then(|values| unsafe { write_string_list(values, out) }) {
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

#[no_mangle]
pub unsafe extern "C" fn lance_free_string_list(list: *mut LanceStringList) {
    if list.is_null() {
        return;
    }
    let list = unsafe { &mut *list };
    if !list.items.is_null() {
        let items =
            unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(list.items, list.count)) };
        for item in items.iter().copied() {
            if !item.is_null() {
                unsafe { drop(CString::from_raw(item)) };
            }
        }
    }
    list.items = std::ptr::null_mut();
    list.count = 0;
}

pub(crate) fn to_c_string(s: impl AsRef<str>) -> CString {
    match CString::new(s.as_ref()) {
        Ok(v) => v,
        Err(_) => CString::new(s.as_ref().replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("invalid string").unwrap()),
    }
}

pub(crate) fn canonicalize_lance_field_path(
    schema: &LanceSchema,
    column: &str,
    what: &'static str,
) -> FfiResult<String> {
    let column = column.trim();
    if column.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} cannot be empty"),
        ));
    }

    let field = schema.field_case_insensitive(column).ok_or_else(|| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} not found: '{column}'"),
        )
    })?;

    schema.field_path(field.id).map_err(|err| {
        FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} path normalize: {err}"),
        )
    })
}

pub(crate) unsafe fn cstr_to_str<'a>(ptr: *const c_char, what: &'static str) -> FfiResult<&'a str> {
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }
    let s = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .context("utf8 decode")
        .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what} utf8: {err}")))?;
    Ok(s)
}

pub(crate) unsafe fn slice_from_ptr<'a, T>(
    ptr: *const T,
    len: usize,
    what: &'static str,
) -> FfiResult<&'a [T]> {
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null"),
        ));
    }
    // SAFETY: Caller guarantees ptr points to at least len elements.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

pub(crate) unsafe fn optional_cstr_array(
    ptr: *const *const c_char,
    len: usize,
    what: &'static str,
) -> FfiResult<Vec<String>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        ));
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut out = Vec::with_capacity(len);
    for (idx, &item) in slice.iter().enumerate() {
        if item.is_null() {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("{what}[{idx}] is null"),
            ));
        }
        let s = unsafe { CStr::from_ptr(item) }
            .to_str()
            .context("utf8 decode")
            .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("{what}[{idx}] utf8: {err}")))?;
        out.push(s.to_string());
    }
    Ok(out)
}

pub(crate) unsafe fn parse_optional_filter_ir(
    filter_ir: *const u8,
    filter_ir_len: usize,
    code: ErrorCode,
    what: &'static str,
) -> FfiResult<Option<Expr>> {
    if filter_ir_len == 0 {
        return Ok(None);
    }
    if filter_ir.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} is null with non-zero length"),
        ));
    }

    let bytes = unsafe { std::slice::from_raw_parts(filter_ir, filter_ir_len) };
    crate::filter_ir::parse_filter_ir(bytes)
        .map(Some)
        .map_err(|err| FfiError::new(code, format!("{what} parse: {err}")))
}

pub(crate) fn u64_to_usize(v: u64, what: &'static str) -> FfiResult<usize> {
    usize::try_from(v)
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, format!("invalid {what}: {err}")))
}

pub(crate) fn nonzero_u64_to_usize(v: u64, what: &'static str) -> FfiResult<usize> {
    let v = u64_to_usize(v, what)?;
    if v == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must be > 0"),
        ));
    }
    Ok(v)
}

pub(crate) fn nonzero_u64_to_i64(v: u64, what: &'static str) -> FfiResult<i64> {
    let v = i64::try_from(v).map_err(|err| {
        FfiError::new(ErrorCode::InvalidArgument, format!("invalid {what}: {err}"))
    })?;
    if v <= 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("{what} must be > 0"),
        ));
    }
    Ok(v)
}

pub(crate) unsafe fn dataset_handle<'a>(dataset: *mut c_void) -> FfiResult<&'a DatasetHandle> {
    if dataset.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "dataset is null"));
    }
    // SAFETY: Caller guarantees dataset points to a valid DatasetHandle.
    Ok(unsafe { &*(dataset as *const DatasetHandle) })
}

pub(crate) unsafe fn optional_session_handle(
    session: *mut c_void,
) -> FfiResult<Option<Arc<Session>>> {
    if session.is_null() {
        return Ok(None);
    }
    let handle = unsafe { &*(session as *const SessionHandle) };
    Ok(Some(handle.session.clone()))
}

pub(crate) unsafe fn stream_handle_mut<'a>(stream: *mut c_void) -> FfiResult<&'a mut StreamHandle> {
    if stream.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "stream is null"));
    }
    // SAFETY: Caller guarantees stream points to a valid StreamHandle.
    Ok(unsafe { &mut *(stream as *mut StreamHandle) })
}

pub(crate) unsafe fn schema_handle<'a>(schema: *mut c_void) -> FfiResult<&'a SchemaHandle> {
    if schema.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "schema is null"));
    }
    // SAFETY: Caller guarantees schema points to a valid SchemaHandle.
    Ok(unsafe { &*(schema as *const SchemaHandle) })
}

pub(crate) unsafe fn batch_handle<'a>(batch: *mut c_void) -> FfiResult<&'a RecordBatch> {
    if batch.is_null() {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "batch is null"));
    }
    // SAFETY: Caller guarantees batch points to a valid RecordBatch.
    Ok(unsafe { &*(batch as *const RecordBatch) })
}

pub(crate) fn schema_to_ffi_arrow_schema(
    schema: &Schema,
) -> FfiResult<arrow::ffi::FFI_ArrowSchema> {
    let data_type = arrow::datatypes::DataType::Struct(schema.fields().clone());
    arrow::ffi::FFI_ArrowSchema::try_from(&data_type)
        .map_err(|err| FfiError::new(ErrorCode::SchemaExport, format!("schema export: {err}")))
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::{lance_free_string_list, write_string_list, LanceStringList};

    #[test]
    fn string_list_preserves_element_boundaries() {
        let mut list = LanceStringList {
            items: std::ptr::null_mut(),
            count: 0,
        };
        unsafe {
            write_string_list(vec!["a\nb".to_string(), "销售".to_string()], &mut list).unwrap();
            assert_eq!(list.count, 2);
            assert_eq!(CStr::from_ptr(*list.items).to_str().unwrap(), "a\nb");
            assert_eq!(CStr::from_ptr(*list.items.add(1)).to_str().unwrap(), "销售");
            lance_free_string_list(&mut list);
        }
        assert!(list.items.is_null());
        assert_eq!(list.count, 0);

        unsafe {
            write_string_list(Vec::new(), &mut list).unwrap();
            assert!(list.items.is_null());
            assert_eq!(list.count, 0);
            lance_free_string_list(&mut list);
            assert!(write_string_list(Vec::new(), std::ptr::null_mut()).is_err());
        }
    }
}
