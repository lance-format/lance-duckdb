use datafusion::functions_aggregate::expr_fn::{avg, max, min, sum};
use datafusion::prelude::{col, lit};
use datafusion_common::ScalarValue;
use datafusion_expr::{Expr, Operator};

#[derive(Debug, Clone)]
pub struct ExecIr {
    pub filter_ir: Vec<u8>,
    pub scan_projection: Vec<String>,
    pub groups: Vec<GroupIr>,
    pub aggs: Vec<AggIr>,
    pub order_by: Vec<OrderByIr>,
}

#[derive(Debug, Clone)]
pub struct GroupIr {
    pub output_name: String,
    pub expr: ExprIr,
    pub output_type: Option<OutputTypeHint>,
}

#[derive(Debug, Clone)]
pub struct AggIr {
    pub func: AggFunc,
    pub output_name: String,
    pub args: Vec<ExprIr>,
    pub output_type: Option<OutputTypeHint>,
}

#[derive(Debug, Clone)]
pub struct OrderByIr {
    pub column: String,
    pub asc: bool,
    pub nulls_first: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum AggFunc {
    Sum,
    Count,
    CountStar,
    Min,
    Max,
    Avg,
}

#[derive(Debug, Clone)]
pub enum OutputTypeHint {
    Bool,
    Int64,
    Double,
    Date,
    TimestampUs,
    Utf8,
    Decimal { precision: u8, scale: u8 },
}

#[derive(Debug, Clone)]
pub enum ExprIr {
    ColumnRef(String),
    Literal(LiteralIr),
    Binary {
        op: BinaryOp,
        lhs: Box<ExprIr>,
        rhs: Box<ExprIr>,
    },
    Cast {
        to: OutputTypeHint,
        expr: Box<ExprIr>,
    },
}

#[derive(Debug, Clone)]
pub enum LiteralIr {
    Null,
    Int64(i64),
    Double(f64),
    Bool(bool),
    String(String),
    Decimal128 {
        value: i128,
        precision: u8,
        scale: u8,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

pub fn parse_exec_ir_v1(bytes: &[u8]) -> Result<ExecIr, String> {
    let mut c = Cursor::new(bytes);
    let magic = c.read_bytes(4)?.to_vec();
    if magic != b"LEX1" {
        return Err("invalid ExecIR magic".to_string());
    }
    let version = c.read_u32()?;
    if version != 1 && version != 2 && version != 3 && version != 4 {
        return Err(format!("unsupported ExecIR version: {version}"));
    }
    let _flags = c.read_u32()?;

    let filter_len = c.read_u32()? as usize;
    let filter_ir = c.read_bytes(filter_len)?.to_vec();

    let proj_len = c.read_u32()? as usize;
    let mut scan_projection = Vec::with_capacity(proj_len);
    for _ in 0..proj_len {
        scan_projection.push(c.read_string()?);
    }

    let mut groups = Vec::new();
    if version >= 3 {
        let group_len = c.read_u32()? as usize;
        groups = Vec::with_capacity(group_len);
        for _ in 0..group_len {
            let output_name = c.read_string()?;
            let expr = parse_expr_ir(&mut c)?;
            let output_type = Some(parse_output_type_hint(&mut c)?);
            groups.push(GroupIr {
                output_name,
                expr,
                output_type,
            });
        }
    }

    let agg_len = c.read_u32()? as usize;
    let mut aggs = Vec::with_capacity(agg_len);
    for _ in 0..agg_len {
        let func = match c.read_u8()? {
            1 => AggFunc::Sum,
            2 => AggFunc::Count,
            3 => AggFunc::CountStar,
            4 => AggFunc::Min,
            5 => AggFunc::Max,
            6 => AggFunc::Avg,
            v => return Err(format!("invalid AggFunc tag: {v}")),
        };
        let output_name = c.read_string()?;
        let arg_count = c.read_u32()? as usize;
        let mut args = Vec::with_capacity(arg_count);
        for _ in 0..arg_count {
            args.push(parse_expr_ir(&mut c)?);
        }
        let output_type = if version >= 2 {
            Some(parse_output_type_hint(&mut c)?)
        } else {
            None
        };
        aggs.push(AggIr {
            func,
            output_name,
            args,
            output_type,
        });
    }

    let mut order_by = Vec::new();
    if version >= 3 {
        let order_len = c.read_u32()? as usize;
        order_by = Vec::with_capacity(order_len);
        for _ in 0..order_len {
            let column = c.read_string()?;
            let asc = c.read_u8()? != 0;
            let nulls_first = c.read_u8()? != 0;
            order_by.push(OrderByIr {
                column,
                asc,
                nulls_first,
            });
        }
    }

    if !c.is_eof() {
        return Err("trailing bytes in ExecIR".to_string());
    }

    Ok(ExecIr {
        filter_ir,
        scan_projection,
        groups,
        aggs,
        order_by,
    })
}

fn parse_output_type_hint(c: &mut Cursor<'_>) -> Result<OutputTypeHint, String> {
    match c.read_u8()? {
        1 => Ok(OutputTypeHint::Bool),
        2 => Ok(OutputTypeHint::Int64),
        3 => Ok(OutputTypeHint::Double),
        4 => Ok(OutputTypeHint::Date),
        5 => Ok(OutputTypeHint::TimestampUs),
        6 => Ok(OutputTypeHint::Utf8),
        7 => {
            let precision = c.read_u8()?;
            let scale = c.read_u8()?;
            Ok(OutputTypeHint::Decimal { precision, scale })
        }
        v => Err(format!("invalid OutputType tag: {v}")),
    }
}

fn parse_expr_ir(c: &mut Cursor<'_>) -> Result<ExprIr, String> {
    match c.read_u8()? {
        1 => Ok(ExprIr::ColumnRef(c.read_string()?)),
        2 => {
            let lit_tag = c.read_u8()?;
            let lit = match lit_tag {
                0 => LiteralIr::Null,
                1 => LiteralIr::Int64(c.read_i64()?),
                2 => LiteralIr::Double(c.read_f64()?),
                3 => LiteralIr::Bool(c.read_u8()? != 0),
                4 => LiteralIr::String(c.read_string()?),
                5 => {
                    let precision = c.read_u8()?;
                    let scale = c.read_u8()?;
                    let bytes = c.read_bytes(16)?;
                    let mut buf = [0u8; 16];
                    buf.copy_from_slice(bytes);
                    let value = i128::from_le_bytes(buf);
                    LiteralIr::Decimal128 {
                        value,
                        precision,
                        scale,
                    }
                }
                v => return Err(format!("invalid Literal tag: {v}")),
            };
            Ok(ExprIr::Literal(lit))
        }
        3 => {
            let op = match c.read_u8()? {
                1 => BinaryOp::Add,
                2 => BinaryOp::Sub,
                3 => BinaryOp::Mul,
                4 => BinaryOp::Div,
                v => return Err(format!("invalid BinaryOp tag: {v}")),
            };
            let lhs = parse_expr_ir(c)?;
            let rhs = parse_expr_ir(c)?;
            Ok(ExprIr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            })
        }
        4 => {
            let to = parse_output_type_hint(c)?;
            let expr = parse_expr_ir(c)?;
            Ok(ExprIr::Cast {
                to,
                expr: Box::new(expr),
            })
        }
        v => Err(format!("invalid ExprIR tag: {v}")),
    }
}

pub fn expr_ir_to_df_expr(expr: &ExprIr) -> Result<Expr, String> {
    match expr {
        ExprIr::ColumnRef(name) => Ok(col(name)),
        ExprIr::Literal(lit_ir) => Ok(lit(literal_ir_to_scalar(lit_ir)?)),
        ExprIr::Binary { op, lhs, rhs } => {
            let lhs = expr_ir_to_df_expr(lhs)?;
            let rhs = expr_ir_to_df_expr(rhs)?;
            let op = match op {
                BinaryOp::Add => Operator::Plus,
                BinaryOp::Sub => Operator::Minus,
                BinaryOp::Mul => Operator::Multiply,
                BinaryOp::Div => Operator::Divide,
            };
            Ok(Expr::BinaryExpr(datafusion_expr::BinaryExpr::new(
                Box::new(lhs),
                op,
                Box::new(rhs),
            )))
        }
        ExprIr::Cast { to, expr } => {
            let dtype = output_type_to_arrow(to)?;
            let child = expr_ir_to_df_expr(expr)?;
            Ok(Expr::Cast(datafusion_expr::Cast::new(
                Box::new(child),
                dtype,
            )))
        }
    }
}

pub fn agg_ir_to_df_expr(agg: &AggIr) -> Result<Expr, String> {
    let base_expr: Expr = match agg.func {
        AggFunc::CountStar => {
            if !agg.args.is_empty() {
                return Err("count_star must have 0 args".to_string());
            }
            datafusion::functions_aggregate::expr_fn::count(lit(ScalarValue::Int64(Some(1))))
        }
        AggFunc::Count => {
            if agg.args.len() != 1 {
                return Err("count must have 1 arg".to_string());
            }
            let arg = expr_ir_to_df_expr(&agg.args[0])?;
            datafusion::functions_aggregate::expr_fn::count(arg)
        }
        AggFunc::Sum => {
            if agg.args.len() != 1 {
                return Err("sum must have 1 arg".to_string());
            }
            let arg = expr_ir_to_df_expr(&agg.args[0])?;
            sum(arg)
        }
        AggFunc::Min => {
            if agg.args.len() != 1 {
                return Err("min must have 1 arg".to_string());
            }
            let arg = expr_ir_to_df_expr(&agg.args[0])?;
            min(arg)
        }
        AggFunc::Max => {
            if agg.args.len() != 1 {
                return Err("max must have 1 arg".to_string());
            }
            let arg = expr_ir_to_df_expr(&agg.args[0])?;
            max(arg)
        }
        AggFunc::Avg => {
            if agg.args.len() != 1 {
                return Err("avg must have 1 arg".to_string());
            }
            let arg = expr_ir_to_df_expr(&agg.args[0])?;
            avg(arg)
        }
    };
    Ok(base_expr.alias(&agg.output_name))
}

pub fn output_type_to_arrow(hint: &OutputTypeHint) -> Result<arrow::datatypes::DataType, String> {
    use arrow::datatypes::{DataType, TimeUnit};
    match hint {
        OutputTypeHint::Bool => Ok(DataType::Boolean),
        OutputTypeHint::Int64 => Ok(DataType::Int64),
        OutputTypeHint::Double => Ok(DataType::Float64),
        OutputTypeHint::Date => Ok(DataType::Date32),
        OutputTypeHint::TimestampUs => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        OutputTypeHint::Utf8 => Ok(DataType::Utf8),
        OutputTypeHint::Decimal { precision, scale } => {
            Ok(DataType::Decimal128(*precision, *scale as i8))
        }
    }
}

fn literal_ir_to_scalar(lit: &LiteralIr) -> Result<ScalarValue, String> {
    match lit {
        LiteralIr::Null => Ok(ScalarValue::Null),
        LiteralIr::Int64(v) => Ok(ScalarValue::Int64(Some(*v))),
        LiteralIr::Double(v) => Ok(ScalarValue::Float64(Some(*v))),
        LiteralIr::Bool(v) => Ok(ScalarValue::Boolean(Some(*v))),
        LiteralIr::String(v) => Ok(ScalarValue::Utf8(Some(v.clone()))),
        LiteralIr::Decimal128 {
            value,
            precision,
            scale,
        } => Ok(ScalarValue::Decimal128(
            Some(*value),
            *precision,
            *scale as i8,
        )),
    }
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

    fn read_i64(&mut self) -> Result<i64, String> {
        let mut out = 0u64;
        for i in 0..8 {
            out |= (self.read_u8()? as u64) << (i * 8);
        }
        Ok(out as i64)
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        let mut bits = 0u64;
        for i in 0..8 {
            bits |= (self.read_u8()? as u64) << (i * 8);
        }
        let v = f64::from_bits(bits);
        Ok(v)
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

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u32()? as usize;
        let bytes = self.read_bytes(len)?;
        let s = std::str::from_utf8(bytes).map_err(|e| format!("utf8: {e}"))?;
        Ok(s.to_string())
    }
}
