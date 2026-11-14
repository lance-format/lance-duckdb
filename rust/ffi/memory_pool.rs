use std::ffi::{c_char, c_void, CStr};
use std::fmt;

use datafusion::execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation};
use datafusion_common::{DataFusionError, Result};

#[repr(C)]
#[derive(Debug)]
pub struct LanceMemoryPoolCallbacks {
    pub pool: *mut c_void,
    pub grow: unsafe extern "C" fn(*mut c_void, usize),
    pub shrink: unsafe extern "C" fn(*mut c_void, usize),
    pub try_grow: unsafe extern "C" fn(*mut c_void, usize, *mut c_char, usize) -> i32,
    pub reserved: unsafe extern "C" fn(*mut c_void) -> usize,
    pub memory_limit: unsafe extern "C" fn(*mut c_void) -> usize,
    pub release: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: The host serializes accesses to the reservation and retains the
// database owning its buffer pool until release. Callbacks do not throw.
unsafe impl Send for LanceMemoryPoolCallbacks {}
unsafe impl Sync for LanceMemoryPoolCallbacks {}

impl Drop for LanceMemoryPoolCallbacks {
    fn drop(&mut self) {
        // SAFETY: This object exclusively owns the host pool. MemoryReservation
        // retains its Arc, so release cannot race with a callback.
        let release = || unsafe { (self.release)(self.pool) };
        if tokio::runtime::Handle::try_current().is_ok() {
            // Exec tasks use our multi-thread runtime. Host release can destroy
            // DuckDB and re-enter Rust to clear session caches with block_on.
            tokio::task::block_in_place(release);
        } else {
            release();
        }
    }
}

impl fmt::Display for LanceMemoryPoolCallbacks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DuckDBMemoryPool")
    }
}

impl MemoryPool for LanceMemoryPoolCallbacks {
    fn name(&self) -> &str {
        "DuckDBMemoryPool"
    }

    fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
        // SAFETY: The host pool remains alive for this call.
        unsafe { (self.grow)(self.pool, additional) }
    }

    fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
        // SAFETY: DataFusion only releases previously reserved bytes.
        unsafe { (self.shrink)(self.pool, shrink) }
    }

    fn try_grow(&self, _reservation: &MemoryReservation, additional: usize) -> Result<()> {
        let mut error = [0 as c_char; 1024];
        // SAFETY: The host pool is alive and the error buffer is writable for
        // its entire capacity. The callback always NUL-terminates on failure.
        let status =
            unsafe { (self.try_grow)(self.pool, additional, error.as_mut_ptr(), error.len()) };
        if status == 0 {
            return Ok(());
        }
        // SAFETY: The callback initialized a NUL-terminated message above.
        let message = unsafe { CStr::from_ptr(error.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        match status {
            1 => Err(DataFusionError::ResourcesExhausted(message)),
            _ => Err(DataFusionError::Execution(message)),
        }
    }

    fn reserved(&self) -> usize {
        // SAFETY: The host pool remains alive for this call.
        unsafe { (self.reserved)(self.pool) }
    }

    fn memory_limit(&self) -> MemoryLimit {
        // SAFETY: The host pool remains alive for this call.
        MemoryLimit::Finite(unsafe { (self.memory_limit)(self.pool) })
    }
}
