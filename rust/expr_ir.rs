use datafusion::execution::context::SessionContext;
use datafusion_common::{Column, ScalarValue};
use datafusion_expr::expr::{BinaryExpr, Case, InList, Like, ScalarFunction};
use datafusion_expr::registry::FunctionRegistry;
use datafusion_expr::{Expr, Operator, ScalarUDF};
use std::sync::{Arc, OnceLock};

use datafusion_functions::core::getfield::GetFieldFunc;

const MAGIC: &[u8; 4] = b"LUE1";

const TAG_COLUMN_REF: u8 = 1;
const TAG_LITERAL: u8 = 2;
const TAG_CAST: u8 = 3;
const TAG_BINARY: u8 = 4;
const TAG_COMPARISON: u8 = 5;
const TAG_CONJUNCTION: u8 = 6;
const TAG_NOT: u8 = 7;
const TAG_SCALAR_FUNCTION: u8 = 8;
const TAG_CASE: u8 = 9;
const TAG_IS_NULL: u8 = 10;
const TAG_IS_NOT_NULL: u8 = 11;
const TAG_IN_LIST: u8 = 12;
const TAG_LIKE: u8 = 13;
const TAG_REGEXP: u8 = 14;

const LIT_NULL: u8 = 0;
const LIT_BOOL: u8 = 1;
const LIT_I64: u8 = 2;
const LIT_U64: u8 = 3;
const LIT_F32: u8 = 4;
const LIT_F64: u8 = 5;
const LIT_STRING: u8 = 6;
const LIT_DATE32: u8 = 7;
const LIT_TIMESTAMP: u8 = 8;
const LIT_DECIMAL128: u8 = 9;

const TYPE_BOOL: u8 = 1;
const TYPE_INT8: u8 = 2;
const TYPE_INT16: u8 = 3;
const TYPE_INT32: u8 = 4;
const TYPE_INT64: u8 = 5;
const TYPE_UINT8: u8 = 6;
const TYPE_UINT16: u8 = 7;
const TYPE_UINT32: u8 = 8;
const TYPE_UINT64: u8 = 9;
const TYPE_FLOAT32: u8 = 10;
const TYPE_FLOAT64: u8 = 11;
const TYPE_VARCHAR: u8 = 12;
const TYPE_DATE32: u8 = 13;
const TYPE_TIMESTAMP: u8 = 14;
const TYPE_DECIMAL128: u8 = 15;

const TS_UNIT_SECOND: u8 = 0;
const TS_UNIT_MILLISECOND: u8 = 1;
const TS_UNIT_MICROSECOND: u8 = 2;
const TS_UNIT_NANOSECOND: u8 = 3;

const BIN_ADD: u8 = 1;
const BIN_SUB: u8 = 2;
const BIN_MUL: u8 = 3;
const BIN_DIV: u8 = 4;

const CMP_EQ: u8 = 0;
const CMP_NOT_EQ: u8 = 1;
const CMP_LT: u8 = 2;
const CMP_LT_EQ: u8 = 3;
const CMP_GT: u8 = 4;
const CMP_GT_EQ: u8 = 5;
const CMP_DISTINCT_FROM: u8 = 6;
const CMP_NOT_DISTINCT_FROM: u8 = 7;

const CONJ_AND: u8 = 1;
const CONJ_OR: u8 = 2;

const LIKE_FLAG_CASE_INSENSITIVE: u8 = 1;
const LIKE_FLAG_HAS_ESCAPE: u8 = 2;

const REGEXP_MODE_PARTIAL_MATCH: u8 = 0;
const REGEXP_MODE_FULL_MATCH: u8 = 1;

static GETFIELD_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static REGEXP_LIKE_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static STARTS_WITH_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static ENDS_WITH_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static CONTAINS_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static LOWER_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();
static UPPER_UDF: OnceLock<Arc<ScalarUDF>> = OnceLock::new();

fn getfield_udf() -> Arc<ScalarUDF> {
    GETFIELD_UDF
        .get_or_init(|| Arc::new(ScalarUDF::new_from_impl(GetFieldFunc::default())))
        .clone()
}

fn regexp_like_udf() -> Arc<ScalarUDF> {
    REGEXP_LIKE_UDF
        .get_or_init(datafusion_functions::regex::regexp_like)
        .clone()
}

fn starts_with_udf() -> Arc<ScalarUDF> {
    STARTS_WITH_UDF
        .get_or_init(datafusion_functions::string::starts_with)
        .clone()
}

fn ends_with_udf() -> Arc<ScalarUDF> {
    ENDS_WITH_UDF
        .get_or_init(datafusion_functions::string::ends_with)
        .clone()
}

fn contains_udf() -> Arc<ScalarUDF> {
    CONTAINS_UDF
        .get_or_init(datafusion_functions::string::contains)
        .clone()
}

fn lower_udf() -> Arc<ScalarUDF> {
    LOWER_UDF
        .get_or_init(datafusion_functions::string::lower)
        .clone()
}

fn upper_udf() -> Arc<ScalarUDF> {
    UPPER_UDF
        .get_or_init(datafusion_functions::string::upper)
        .clone()
}

pub fn parse_expr_ir(bytes: &[u8], ctx: Option<&SessionContext>) -> Result<Expr, String> {
    if bytes.len() < MAGIC.len() {
        return Err("expr_ir is too short".to_string());
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return Err("expr_ir magic mismatch".to_string());
    }

    parse_expr_ir_payload(&bytes[MAGIC.len()..], ctx)
}

pub(crate) fn parse_expr_ir_payload(
    bytes: &[u8],
    ctx: Option<&SessionContext>,
) -> Result<Expr, String> {
    let mut cursor = Cursor::new(bytes);
    let expr = parse_node(&mut cursor, ctx)?;
    if !cursor.is_eof() {
        return Err("trailing bytes in expr_ir".to_string());
    }
    Ok(expr)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| "unexpected EOF".to_string())?;
        self.pos += 1;
        Ok(b)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut out = 0u32;
        for i in 0..4 {
            out |= (self.read_u8()? as u32) << (i * 8);
        }
        Ok(out)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut out = 0u64;
        for i in 0..8 {
            out |= (self.read_u8()? as u64) << (i * 8);
        }
        Ok(out)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_i128(&mut self) -> Result<i128, String> {
        Ok(i128::from_le_bytes(
            self.read_bytes(16)?.try_into().unwrap(),
        ))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap()))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap()))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| "length overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected EOF".to_string());
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_len_prefixed_bytes(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_u32()? as usize;
        self.read_bytes(len)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let bytes = self.read_len_prefixed_bytes()?;
        let s = std::str::from_utf8(bytes).map_err(|e| format!("utf8: {e}"))?;
        Ok(s.to_string())
    }
}

fn parse_subexpr(bytes: &[u8], ctx: Option<&SessionContext>) -> Result<Expr, String> {
    let mut cursor = Cursor::new(bytes);
    let expr = parse_node(&mut cursor, ctx)?;
    if !cursor.is_eof() {
        return Err("trailing bytes in expr_ir subexpr".to_string());
    }
    Ok(expr)
}

fn parse_node(cursor: &mut Cursor<'_>, ctx: Option<&SessionContext>) -> Result<Expr, String> {
    match cursor.read_u8()? {
        TAG_COLUMN_REF => parse_column_ref(cursor),
        TAG_LITERAL => parse_literal(cursor),
        TAG_CAST => {
            let data_type = parse_type(cursor)?;
            let child = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::Cast(datafusion_expr::Cast::new(
                Box::new(child),
                data_type,
            )))
        }
        TAG_BINARY => {
            let op = match cursor.read_u8()? {
                BIN_ADD => Operator::Plus,
                BIN_SUB => Operator::Minus,
                BIN_MUL => Operator::Multiply,
                BIN_DIV => Operator::Divide,
                other => return Err(format!("invalid expr_ir binary op: {other}")),
            };
            let lhs = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            let rhs = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(lhs),
                op,
                Box::new(rhs),
            )))
        }
        TAG_COMPARISON => {
            let op = match cursor.read_u8()? {
                CMP_EQ => Operator::Eq,
                CMP_NOT_EQ => Operator::NotEq,
                CMP_LT => Operator::Lt,
                CMP_LT_EQ => Operator::LtEq,
                CMP_GT => Operator::Gt,
                CMP_GT_EQ => Operator::GtEq,
                CMP_DISTINCT_FROM => Operator::IsDistinctFrom,
                CMP_NOT_DISTINCT_FROM => Operator::IsNotDistinctFrom,
                other => return Err(format!("invalid expr_ir comparison op: {other}")),
            };
            let lhs = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            let rhs = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(lhs),
                op,
                Box::new(rhs),
            )))
        }
        TAG_CONJUNCTION => {
            let op = match cursor.read_u8()? {
                CONJ_AND => Operator::And,
                CONJ_OR => Operator::Or,
                other => return Err(format!("invalid expr_ir conjunction op: {other}")),
            };
            let child_count = cursor.read_u32()? as usize;
            if child_count == 0 {
                return Err("expr_ir conjunction has no children".to_string());
            }
            let first = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            let mut expr = first;
            for _ in 1..child_count {
                let rhs = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
                expr = Expr::BinaryExpr(BinaryExpr::new(Box::new(expr), op, Box::new(rhs)));
            }
            Ok(expr)
        }
        TAG_NOT => {
            let child = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::Not(Box::new(child)))
        }
        TAG_SCALAR_FUNCTION => {
            let name = cursor.read_string()?;
            let arg_count = cursor.read_u32()? as usize;
            let mut args = Vec::with_capacity(arg_count);
            for _ in 0..arg_count {
                args.push(parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?);
            }
            let udf = resolve_scalar_function(&name, ctx)?;
            Ok(Expr::ScalarFunction(ScalarFunction { func: udf, args }))
        }
        TAG_CASE => {
            let check_count = cursor.read_u32()? as usize;
            if check_count == 0 {
                return Err("expr_ir CASE has no branches".to_string());
            }
            let mut checks = Vec::with_capacity(check_count);
            for _ in 0..check_count {
                let when_expr = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
                let then_expr = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
                checks.push((Box::new(when_expr), Box::new(then_expr)));
            }
            let has_else = cursor.read_u8()? != 0;
            let else_expr = if has_else {
                Some(Box::new(parse_subexpr(
                    cursor.read_len_prefixed_bytes()?,
                    ctx,
                )?))
            } else {
                None
            };
            Ok(Expr::Case(Case::new(None, checks, else_expr)))
        }
        TAG_IS_NULL => {
            let child = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::IsNull(Box::new(child)))
        }
        TAG_IS_NOT_NULL => {
            let child = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            Ok(Expr::IsNotNull(Box::new(child)))
        }
        TAG_IN_LIST => {
            let negated = cursor.read_u8()? != 0;
            let expr = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
            let item_count = cursor.read_u32()? as usize;
            let mut list = Vec::with_capacity(item_count);
            for _ in 0..item_count {
                list.push(parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?);
            }
            Ok(Expr::InList(InList::new(Box::new(expr), list, negated)))
        }
        TAG_LIKE => parse_like(cursor, ctx),
        TAG_REGEXP => parse_regexp(cursor, ctx),
        other => Err(format!("invalid expr_ir tag: {other}")),
    }
}

fn parse_column_ref(cursor: &mut Cursor<'_>) -> Result<Expr, String> {
    let segments_len = cursor.read_u32()? as usize;
    if segments_len == 0 {
        return Err("expr_ir column ref has no segments".to_string());
    }
    let mut segments = Vec::with_capacity(segments_len);
    for _ in 0..segments_len {
        segments.push(cursor.read_string()?);
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

fn resolve_scalar_function(
    name: &str,
    ctx: Option<&SessionContext>,
) -> Result<Arc<ScalarUDF>, String> {
    match name {
        "starts_with" => Ok(starts_with_udf()),
        "ends_with" => Ok(ends_with_udf()),
        "contains" => Ok(contains_udf()),
        "lower" => Ok(lower_udf()),
        "upper" => Ok(upper_udf()),
        _ => match ctx {
            Some(ctx) => ctx
                .udf(name)
                .map_err(|e| format!("resolve udf {name}: {e}")),
            None => Err(format!("resolve udf {name}: registry unavailable")),
        },
    }
}

fn parse_like(cursor: &mut Cursor<'_>, ctx: Option<&SessionContext>) -> Result<Expr, String> {
    let flags = cursor.read_u8()?;
    let expr = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
    let pattern = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
    let escape_char = if (flags & LIKE_FLAG_HAS_ESCAPE) != 0 {
        Some(cursor.read_u8()? as char)
    } else {
        None
    };
    Ok(Expr::Like(Like::new(
        false,
        Box::new(expr),
        Box::new(pattern),
        escape_char,
        (flags & LIKE_FLAG_CASE_INSENSITIVE) != 0,
    )))
}

fn parse_regexp(cursor: &mut Cursor<'_>, ctx: Option<&SessionContext>) -> Result<Expr, String> {
    let mode = cursor.read_u8()?;
    let has_flags = cursor.read_u8()? != 0;
    let value = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
    let mut pattern = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
    let normalized_flags = if has_flags {
        let flags_expr = parse_subexpr(cursor.read_len_prefixed_bytes()?, ctx)?;
        let flags_str = match flags_expr {
            Expr::Literal(ScalarValue::Utf8(Some(v)), _) => v,
            other => {
                return Err(format!(
                    "regexp_matches expects a non-null string literal options, got: {other:?}"
                ))
            }
        };

        let mut has_i = false;
        let mut has_s = false;
        for c in flags_str.chars() {
            match c {
                'i' => has_i = true,
                's' => has_s = true,
                ' ' | '\t' | '\n' => {}
                other => return Err(format!("unsupported regexp option: {other}")),
            }
        }

        let mut normalized = String::new();
        if has_i {
            normalized.push('i');
        }
        if has_s {
            normalized.push('s');
        }
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    } else {
        None
    };

    if mode == REGEXP_MODE_FULL_MATCH {
        let pattern_str = match &pattern {
            Expr::Literal(ScalarValue::Utf8(Some(v)), _) => v,
            other => {
                return Err(format!(
                    "regexp_full_match expects a non-null string literal pattern, got: {other:?}"
                ))
            }
        };
        pattern = Expr::Literal(
            ScalarValue::Utf8(Some(format!("^(?:{pattern_str})$"))),
            None,
        );
    } else if mode != REGEXP_MODE_PARTIAL_MATCH {
        return Err(format!("invalid expr_ir regexp mode: {mode}"));
    }

    let pattern_str = match &pattern {
        Expr::Literal(ScalarValue::Utf8(Some(v)), _) => v,
        other => {
            return Err(format!(
                "regexp_matches expects a non-null string literal pattern, got: {other:?}"
            ))
        }
    };
    datafusion_functions::regex::compile_regex(pattern_str, normalized_flags.as_deref())
        .map_err(|e| format!("regexp compile failed: {e}"))?;

    let mut args = vec![value, pattern];
    if let Some(flags) = normalized_flags {
        args.push(Expr::Literal(ScalarValue::Utf8(Some(flags)), None));
    }
    Ok(Expr::ScalarFunction(ScalarFunction {
        func: regexp_like_udf(),
        args,
    }))
}

fn parse_literal(cursor: &mut Cursor<'_>) -> Result<Expr, String> {
    let lit = match cursor.read_u8()? {
        LIT_NULL => ScalarValue::Null,
        LIT_BOOL => ScalarValue::Boolean(Some(cursor.read_u8()? != 0)),
        LIT_I64 => ScalarValue::Int64(Some(cursor.read_i64()?)),
        LIT_U64 => ScalarValue::UInt64(Some(cursor.read_u64()?)),
        LIT_F32 => ScalarValue::Float32(Some(cursor.read_f32()?)),
        LIT_F64 => ScalarValue::Float64(Some(cursor.read_f64()?)),
        LIT_STRING => ScalarValue::Utf8(Some(cursor.read_string()?)),
        LIT_DATE32 => ScalarValue::Date32(Some(cursor.read_i32()?)),
        LIT_TIMESTAMP => {
            let unit = cursor.read_u8()?;
            let value = cursor.read_i64()?;
            match unit {
                TS_UNIT_SECOND => ScalarValue::TimestampSecond(Some(value), None),
                TS_UNIT_MILLISECOND => ScalarValue::TimestampMillisecond(Some(value), None),
                TS_UNIT_MICROSECOND => ScalarValue::TimestampMicrosecond(Some(value), None),
                TS_UNIT_NANOSECOND => ScalarValue::TimestampNanosecond(Some(value), None),
                other => return Err(format!("invalid expr_ir timestamp unit: {other}")),
            }
        }
        LIT_DECIMAL128 => {
            let precision = cursor.read_u8()?;
            let scale = cursor.read_u8()? as i8;
            ScalarValue::Decimal128(Some(cursor.read_i128()?), precision, scale)
        }
        other => return Err(format!("invalid expr_ir literal tag: {other}")),
    };
    Ok(Expr::Literal(lit, None))
}

fn parse_type(cursor: &mut Cursor<'_>) -> Result<arrow::datatypes::DataType, String> {
    use arrow::datatypes::{DataType, TimeUnit};

    match cursor.read_u8()? {
        TYPE_BOOL => Ok(DataType::Boolean),
        TYPE_INT8 => Ok(DataType::Int8),
        TYPE_INT16 => Ok(DataType::Int16),
        TYPE_INT32 => Ok(DataType::Int32),
        TYPE_INT64 => Ok(DataType::Int64),
        TYPE_UINT8 => Ok(DataType::UInt8),
        TYPE_UINT16 => Ok(DataType::UInt16),
        TYPE_UINT32 => Ok(DataType::UInt32),
        TYPE_UINT64 => Ok(DataType::UInt64),
        TYPE_FLOAT32 => Ok(DataType::Float32),
        TYPE_FLOAT64 => Ok(DataType::Float64),
        TYPE_VARCHAR => Ok(DataType::Utf8),
        TYPE_DATE32 => Ok(DataType::Date32),
        TYPE_TIMESTAMP => {
            let unit = match cursor.read_u8()? {
                TS_UNIT_SECOND => TimeUnit::Second,
                TS_UNIT_MILLISECOND => TimeUnit::Millisecond,
                TS_UNIT_MICROSECOND => TimeUnit::Microsecond,
                TS_UNIT_NANOSECOND => TimeUnit::Nanosecond,
                other => return Err(format!("invalid expr_ir timestamp unit: {other}")),
            };
            Ok(DataType::Timestamp(unit, None))
        }
        TYPE_DECIMAL128 => {
            let precision = cursor.read_u8()?;
            let scale = cursor.read_u8()? as i8;
            Ok(DataType::Decimal128(precision, scale))
        }
        other => Err(format!("invalid expr_ir type tag: {other}")),
    }
}
