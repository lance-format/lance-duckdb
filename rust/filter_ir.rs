use std::sync::{Arc, OnceLock};

use anyhow::{anyhow, bail, Context, Result};
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::expr::{InList, ScalarFunction};
use datafusion_expr::{Expr, ScalarUDF};
use datafusion_functions::core::getfield::GetFieldFunc;

const MAGIC: &[u8; 4] = b"LFT1";

const TAG_COLUMN_REF: u8 = 1;
const TAG_LITERAL: u8 = 2;
const TAG_AND: u8 = 3;
const TAG_OR: u8 = 4;
const TAG_NOT: u8 = 5;
const TAG_COMPARISON: u8 = 6;
const TAG_IS_NULL: u8 = 7;
const TAG_IS_NOT_NULL: u8 = 8;
const TAG_IN_LIST: u8 = 9;

const LIT_NULL: u8 = 0;
const LIT_BOOL: u8 = 1;
const LIT_I64: u8 = 2;
const LIT_U64: u8 = 3;
const LIT_F32: u8 = 4;
const LIT_F64: u8 = 5;
const LIT_STRING: u8 = 6;
const LIT_DATE32: u8 = 7;

const OP_EQ: u8 = 0;
const OP_NOT_EQ: u8 = 1;
const OP_LT: u8 = 2;
const OP_LT_EQ: u8 = 3;
const OP_GT: u8 = 4;
const OP_GT_EQ: u8 = 5;

static GETFIELD_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();

fn getfield_udf() -> Arc<ScalarUDF> {
    GETFIELD_UDF
        .get_or_init(|| Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::default())))
        .clone()
}

pub fn parse_filter_ir(filter_ir: &[u8]) -> Result<Expr> {
    if filter_ir.len() < MAGIC.len() {
        bail!("filter_ir is too short");
    }
    if &filter_ir[0..MAGIC.len()] != MAGIC {
        bail!("filter_ir magic mismatch");
    }

    let mut cursor = Cursor::new(&filter_ir[MAGIC.len()..]);
    let expr = parse_node(&mut cursor)?;
    if cursor.remaining() != 0 {
        bail!("trailing bytes in filter_ir: {}", cursor.remaining());
    }
    Ok(expr)
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.remaining() < 1 {
            bail!("unexpected end of input");
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i64_le(&mut self) -> Result<i64> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_i32_le(&mut self) -> Result<i32> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_u64_le(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f32_le(&mut self) -> Result<f32> {
        let bytes = self.read_exact(4)?;
        Ok(f32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_f64_le(&mut self) -> Result<f64> {
        let bytes = self.read_exact(8)?;
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        if self.remaining() < len {
            bail!("unexpected end of input");
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.buf[start..start + len])
    }

    fn read_len_prefixed_slice(&mut self) -> Result<&'a [u8]> {
        let len = usize::try_from(self.read_u32_le()?)?;
        self.read_exact(len)
    }

    fn read_len_prefixed_string(&mut self) -> Result<String> {
        let bytes = self.read_len_prefixed_slice()?;
        Ok(std::str::from_utf8(bytes)
            .context("invalid utf8 string")?
            .to_string())
    }
}

fn parse_subexpr(bytes: &[u8]) -> Result<Expr> {
    let mut cursor = Cursor::new(bytes);
    let expr = parse_node(&mut cursor)?;
    if cursor.remaining() != 0 {
        bail!("trailing bytes in subexpr: {}", cursor.remaining());
    }
    Ok(expr)
}

fn parse_node(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let tag = cursor.read_u8()?;
    match tag {
        TAG_COLUMN_REF => parse_column_ref(cursor),
        TAG_LITERAL => parse_literal(cursor),
        TAG_AND => parse_conjunction(cursor, true),
        TAG_OR => parse_conjunction(cursor, false),
        TAG_NOT => {
            let child = parse_len_prefixed_node(cursor)?;
            Ok(Expr::Not(Box::new(child)))
        }
        TAG_COMPARISON => parse_comparison(cursor),
        TAG_IS_NULL => {
            let child = parse_len_prefixed_node(cursor)?;
            Ok(Expr::IsNull(Box::new(child)))
        }
        TAG_IS_NOT_NULL => {
            let child = parse_len_prefixed_node(cursor)?;
            Ok(Expr::IsNotNull(Box::new(child)))
        }
        TAG_IN_LIST => parse_in_list(cursor),
        other => Err(anyhow!("unknown node tag: {other}")),
    }
}

fn parse_len_prefixed_node(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let bytes = cursor.read_len_prefixed_slice()?;
    parse_subexpr(bytes)
}

fn parse_column_ref(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let segments_len = usize::try_from(cursor.read_u32_le()?)?;
    if segments_len == 0 {
        bail!("column ref has no segments");
    }
    let mut segments = Vec::with_capacity(segments_len);
    for _ in 0..segments_len {
        segments.push(cursor.read_len_prefixed_string()?);
    }

    let mut expr = Expr::Column(Column::new_unqualified(segments[0].clone()));
    for segment in segments.into_iter().skip(1) {
        expr = Expr::ScalarFunction(ScalarFunction {
            func: getfield_udf(),
            args: vec![
                std::mem::take(&mut expr),
                Expr::Literal(ScalarValue::Utf8(Some(segment)), None),
            ],
        });
    }
    Ok(expr)
}

fn parse_literal(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let lit_tag = cursor.read_u8()?;
    let scalar = match lit_tag {
        LIT_NULL => ScalarValue::Null,
        LIT_BOOL => ScalarValue::Boolean(Some(cursor.read_u8()? != 0)),
        LIT_I64 => ScalarValue::Int64(Some(cursor.read_i64_le()?)),
        LIT_U64 => ScalarValue::UInt64(Some(cursor.read_u64_le()?)),
        LIT_F32 => ScalarValue::Float32(Some(cursor.read_f32_le()?)),
        LIT_F64 => ScalarValue::Float64(Some(cursor.read_f64_le()?)),
        LIT_STRING => ScalarValue::Utf8(Some(cursor.read_len_prefixed_string()?)),
        LIT_DATE32 => ScalarValue::Date32(Some(cursor.read_i32_le()?)),
        other => return Err(anyhow!("unknown literal tag: {other}")),
    };
    Ok(Expr::Literal(scalar, None))
}

fn parse_conjunction(cursor: &mut Cursor<'_>, is_and: bool) -> Result<Expr> {
    let children_len = usize::try_from(cursor.read_u32_le()?)?;
    if children_len == 0 {
        bail!("conjunction has no children");
    }
    let mut children = Vec::with_capacity(children_len);
    for _ in 0..children_len {
        children.push(parse_len_prefixed_node(cursor)?);
    }

    let mut iter = children.into_iter();
    let mut expr = iter.next().unwrap();
    for child in iter {
        expr = if is_and {
            expr.and(child)
        } else {
            expr.or(child)
        };
    }
    Ok(expr)
}

fn parse_comparison(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let op = cursor.read_u8()?;
    let left = parse_len_prefixed_node(cursor)?;
    let right = parse_len_prefixed_node(cursor)?;
    Ok(match op {
        OP_EQ => left.eq(right),
        OP_NOT_EQ => left.not_eq(right),
        OP_LT => left.lt(right),
        OP_LT_EQ => left.lt_eq(right),
        OP_GT => left.gt(right),
        OP_GT_EQ => left.gt_eq(right),
        other => return Err(anyhow!("unknown comparison op: {other}")),
    })
}

fn parse_in_list(cursor: &mut Cursor<'_>) -> Result<Expr> {
    let negated = cursor.read_u8()? != 0;
    let expr = parse_len_prefixed_node(cursor)?;
    let list_len = usize::try_from(cursor.read_u32_le()?)?;
    let mut list = Vec::with_capacity(list_len);
    for _ in 0..list_len {
        list.push(parse_len_prefixed_node(cursor)?);
    }
    Ok(Expr::InList(InList::new(Box::new(expr), list, negated)))
}
