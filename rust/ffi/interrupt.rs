use std::ffi::c_void;

use crate::runtime::{set_interrupt_hook, InterruptCheck, InterruptHook};

/// Registers `check` as the interrupt hook of the calling thread; FFI calls made on this thread
/// then fail with a runtime error once `check(ctx)` returns non-zero. Passing NULL clears it.
#[no_mangle]
pub unsafe extern "C" fn lance_set_interrupt_check(
    check: Option<InterruptCheck>,
    ctx: *mut c_void,
) {
    set_interrupt_hook(check.map(|check| InterruptHook::new(check, ctx)));
}
