use anyhow::{bail, Result};
use datafusion_expr::Expr;

use crate::expr_ir::parse_expr_ir_payload;

const MAGIC: &[u8; 4] = b"LFT1";

pub fn parse_filter_ir(filter_ir: &[u8]) -> Result<Expr> {
    if filter_ir.len() < MAGIC.len() {
        bail!("filter_ir is too short");
    }
    if &filter_ir[..MAGIC.len()] != MAGIC {
        bail!("filter_ir magic mismatch");
    }
    parse_expr_ir_payload(&filter_ir[MAGIC.len()..], None)
        .map_err(anyhow::Error::msg)
}
