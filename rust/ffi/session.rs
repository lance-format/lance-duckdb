use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lance::session::Session;

use crate::error::{clear_last_error, set_last_error, ErrorCode};

use super::types::SessionHandle;
use super::util::{FfiError, FfiResult, optional_session_handle, u64_to_usize};

static DATASET_OPEN_COUNT: AtomicU64 = AtomicU64::new(0);
static NAMESPACE_DESCRIBE_COUNT: AtomicU64 = AtomicU64::new(0);
static COMMIT_COUNT: AtomicU64 = AtomicU64::new(0);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LanceSessionStats {
    pub size_bytes: u64,
    pub approx_num_items: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LanceDebugCounters {
    pub dataset_open_count: u64,
    pub namespace_describe_count: u64,
    pub commit_count: u64,
}

pub(crate) fn record_dataset_open() {
    DATASET_OPEN_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_namespace_describe() {
    NAMESPACE_DESCRIBE_COUNT.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_commit() {
    COMMIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_session(
    index_cache_size_bytes: u64,
    metadata_cache_size_bytes: u64,
) -> *mut c_void {
    match create_session_inner(index_cache_size_bytes, metadata_cache_size_bytes) {
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

fn create_session_inner(
    index_cache_size_bytes: u64,
    metadata_cache_size_bytes: u64,
) -> FfiResult<SessionHandle> {
    let session = if index_cache_size_bytes == 0 && metadata_cache_size_bytes == 0 {
        Arc::new(Session::default())
    } else {
        Arc::new(Session::new(
            u64_to_usize(index_cache_size_bytes, "index_cache_size_bytes")?,
            u64_to_usize(metadata_cache_size_bytes, "metadata_cache_size_bytes")?,
            Default::default(),
        ))
    };
    Ok(SessionHandle { session })
}

#[no_mangle]
pub unsafe extern "C" fn lance_close_session(session: *mut c_void) {
    if !session.is_null() {
        unsafe {
            let _ = Box::from_raw(session as *mut SessionHandle);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_session_get_stats(
    session: *mut c_void,
    out_stats: *mut LanceSessionStats,
) -> i32 {
    if !out_stats.is_null() {
        unsafe {
            std::ptr::write_unaligned(out_stats, LanceSessionStats::default());
        }
    }

    match session_get_stats_inner(session) {
        Ok(stats) => {
            clear_last_error();
            if !out_stats.is_null() {
                unsafe {
                    std::ptr::write_unaligned(out_stats, stats);
                }
            }
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

fn session_get_stats_inner(session: *mut c_void) -> FfiResult<LanceSessionStats> {
    let Some(session) = (unsafe { optional_session_handle(session)? }) else {
        return Err(FfiError::new(ErrorCode::InvalidArgument, "session is null"));
    };
    Ok(LanceSessionStats {
        size_bytes: session.size_bytes(),
        approx_num_items: session.approx_num_items() as u64,
    })
}

#[no_mangle]
pub unsafe extern "C" fn lance_debug_get_counters(out_counters: *mut LanceDebugCounters) -> i32 {
    if out_counters.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_counters is null".to_string());
        return -1;
    }

    clear_last_error();
    let counters = LanceDebugCounters {
        dataset_open_count: DATASET_OPEN_COUNT.load(Ordering::Relaxed),
        namespace_describe_count: NAMESPACE_DESCRIBE_COUNT.load(Ordering::Relaxed),
        commit_count: COMMIT_COUNT.load(Ordering::Relaxed),
    };
    unsafe {
        std::ptr::write_unaligned(out_counters, counters);
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn lance_debug_reset_counters() {
    DATASET_OPEN_COUNT.store(0, Ordering::Relaxed);
    NAMESPACE_DESCRIBE_COUNT.store(0, Ordering::Relaxed);
    COMMIT_COUNT.store(0, Ordering::Relaxed);
    clear_last_error();
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::fs;
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch, RecordBatchIterator};
    use arrow_schema::{DataType, Field, Schema};
    use lance::dataset::WriteParams;
    use lance::Dataset;

    use crate::runtime;

    use super::*;
    use super::super::dataset::{lance_close_dataset, lance_open_dataset_with_session};

    #[test]
    fn test_create_session_and_get_stats() {
        unsafe {
            let session = lance_create_session(0, 0);
            assert!(!session.is_null());

            let mut stats = LanceSessionStats::default();
            assert_eq!(lance_session_get_stats(session, &mut stats), 0);
            assert_eq!(stats.approx_num_items, 0);

            lance_close_session(session);
        }
    }

    #[test]
    fn test_open_dataset_with_session_records_debug_counters() {
        let dataset_dir = std::env::temp_dir().join(format!("ffi-session-{}", rand::random::<u64>()));
        let uri = dataset_dir.to_string_lossy().to_string();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
                .unwrap();
        let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

        unsafe {
            lance_debug_reset_counters();
            let session = lance_create_session(0, 0);
            assert!(!session.is_null());

            runtime::block_on(Dataset::write(
                reader,
                &uri,
                Some(WriteParams::default()),
            ))
            .unwrap()
            .unwrap();

            let uri_c = CString::new(uri.clone()).unwrap();
            let first = lance_open_dataset_with_session(uri_c.as_ptr(), session);
            assert!(!first.is_null());
            lance_close_dataset(first);

            let second = lance_open_dataset_with_session(uri_c.as_ptr(), session);
            assert!(!second.is_null());
            lance_close_dataset(second);

            let mut counters = LanceDebugCounters::default();
            assert_eq!(lance_debug_get_counters(&mut counters), 0);
            assert_eq!(counters.dataset_open_count, 2);

            let mut stats = LanceSessionStats::default();
            assert_eq!(lance_session_get_stats(session, &mut stats), 0);

            lance_close_session(session);
        }

        let _ = fs::remove_dir_all(dataset_dir);
    }
}
