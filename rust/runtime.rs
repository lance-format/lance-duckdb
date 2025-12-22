use std::future::Future;
use std::io;
use std::sync::OnceLock;

use tokio::runtime::{Handle, Runtime};

static RUNTIME: OnceLock<Result<Runtime, io::Error>> = OnceLock::new();

pub fn runtime() -> Result<&'static Runtime, io::Error> {
    match RUNTIME.get_or_init(Runtime::new) {
        Ok(rt) => Ok(rt),
        Err(err) => Err(io::Error::new(err.kind(), err.to_string())),
    }
}

pub fn handle() -> Result<Handle, io::Error> {
    Ok(runtime()?.handle().clone())
}

pub fn block_on<F: Future>(future: F) -> Result<F::Output, io::Error> {
    Ok(runtime()?.block_on(future))
}
