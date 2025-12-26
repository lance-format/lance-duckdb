use std::cell::RefCell;
use std::ffi::{c_char, CString};
use std::ptr;

#[repr(i32)]
#[derive(Clone, Copy, Debug)]
pub enum ErrorCode {
    InvalidArgument = 1,
    Utf8 = 2,
    Runtime = 3,
    DatasetOpen = 4,
    DatasetCountRows = 5,
    FragmentScan = 6,
    StreamCreate = 7,
    StreamNext = 8,
    SchemaExport = 9,
    BatchExport = 10,
    KnnSchema = 11,
    KnnStreamCreate = 12,
    ExplainPlan = 13,
    FtsSchema = 14,
    FtsStreamCreate = 15,
    HybridStreamCreate = 17,
    DatasetWriteOpen = 18,
    DatasetWriteBatch = 19,
    DatasetWriteFinish = 20,
    NamespaceListTables = 21,
    NamespaceDescribeTable = 22,
    DirNamespaceListTables = 23,
    DirNamespaceDropTable = 24,
}

struct LastError {
    code: i32,
    message: CString,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<LastError>> = const { RefCell::new(None) };
}

fn sanitize_message(message: &str) -> CString {
    match CString::new(message) {
        Ok(v) => v,
        Err(_) => CString::new(message.replace('\0', "\\0"))
            .unwrap_or_else(|_| CString::new("invalid error message").unwrap()),
    }
}

pub fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

pub fn set_last_error(code: ErrorCode, message: impl AsRef<str>) {
    let code = code as i32;
    let message = sanitize_message(message.as_ref());
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(LastError { code, message });
    });
}

#[no_mangle]
pub extern "C" fn lance_last_error_code() -> i32 {
    LAST_ERROR.with(|e| e.borrow().as_ref().map(|v| v.code).unwrap_or(0))
}

#[no_mangle]
pub extern "C" fn lance_last_error_message() -> *const c_char {
    LAST_ERROR.with(|e| match e.borrow_mut().take() {
        Some(err) => err.message.into_raw() as *const c_char,
        None => ptr::null(),
    })
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_string(s: *const c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s as *mut c_char);
        }
    }
}
