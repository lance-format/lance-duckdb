use std::cell::Cell;
use std::ffi::c_void;
use std::future::Future;
use std::io;
use std::sync::OnceLock;
use std::time::Duration;

use tokio::runtime::{Handle, Runtime};

static RUNTIME: OnceLock<Result<Runtime, io::Error>> = OnceLock::new();

const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Reports whether the query that owns `ctx` was interrupted; non-zero means interrupted.
pub type InterruptCheck = unsafe extern "C" fn(ctx: *mut c_void) -> i32;

#[derive(Clone, Copy)]
pub struct InterruptHook {
    check: InterruptCheck,
    ctx: *mut c_void,
}

// The hook is only called while the registering C++ scope is alive, and the C++ callback
// reads an atomic flag.
unsafe impl Send for InterruptHook {}

thread_local! {
    static INTERRUPT_HOOK: Cell<Option<InterruptHook>> = const { Cell::new(None) };
}

pub fn interrupt_hook() -> Option<InterruptHook> {
    INTERRUPT_HOOK.with(Cell::get)
}

pub fn set_interrupt_hook(hook: Option<InterruptHook>) {
    INTERRUPT_HOOK.with(|slot| slot.set(hook));
}

impl InterruptHook {
    pub fn new(check: InterruptCheck, ctx: *mut c_void) -> Self {
        Self { check, ctx }
    }

    fn interrupted(&self) -> bool {
        unsafe { (self.check)(self.ctx) != 0 }
    }
}

pub fn runtime() -> Result<&'static Runtime, io::Error> {
    match RUNTIME.get_or_init(Runtime::new) {
        Ok(rt) => Ok(rt),
        Err(err) => Err(io::Error::new(err.kind(), err.to_string())),
    }
}

pub fn initialized_runtime() -> Option<&'static Runtime> {
    RUNTIME.get()?.as_ref().ok()
}

pub fn handle() -> Result<Handle, io::Error> {
    Ok(runtime()?.handle().clone())
}

/// Runs `future` to completion on the shared runtime.
///
/// When the calling thread has an interrupt hook, the future is dropped at its next await point
/// once the hook reports an interrupt, and the call fails with `io::ErrorKind::Interrupted`.
pub fn block_on<F: Future>(future: F) -> Result<F::Output, io::Error> {
    let rt = runtime()?;
    let Some(hook) = interrupt_hook() else {
        return Ok(rt.block_on(future));
    };
    rt.block_on(async move {
        let wait_for_interrupt = async {
            while !hook.interrupted() {
                tokio::time::sleep(INTERRUPT_POLL_INTERVAL).await;
            }
        };
        // Biased: a query that is already interrupted must not start any work.
        tokio::select! {
            biased;
            () = wait_for_interrupt => Err(io::Error::new(io::ErrorKind::Interrupted, "interrupted")),
            out = future => Ok(out),
        }
    })
}
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    unsafe extern "C" fn interrupted_after_polls(ctx: *mut c_void) -> i32 {
        let polls = unsafe { &*(ctx as *const AtomicUsize) };
        i32::from(polls.fetch_add(1, Ordering::SeqCst) >= 2)
    }

    unsafe extern "C" fn never_interrupted(_ctx: *mut c_void) -> i32 {
        0
    }

    #[test]
    fn interrupt_hook_drops_a_pending_future() {
        let polls = AtomicUsize::new(0);
        set_interrupt_hook(Some(InterruptHook::new(
            interrupted_after_polls,
            &polls as *const AtomicUsize as *mut c_void,
        )));
        let out = block_on(std::future::pending::<()>());
        set_interrupt_hook(None);

        let err = out.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::Interrupted);
        assert_eq!(polls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn without_an_interrupt_the_future_completes() {
        set_interrupt_hook(Some(InterruptHook::new(
            never_interrupted,
            std::ptr::null_mut(),
        )));
        let hooked = block_on(async { 7 });
        set_interrupt_hook(None);
        assert_eq!(hooked.unwrap(), 7);
        assert_eq!(block_on(async { 8 }).unwrap(), 8);
    }
}
