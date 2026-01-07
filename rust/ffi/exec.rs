use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow::datatypes::{Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::dataframe::DataFrame;
use datafusion::prelude::SessionContext;
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_expr::Expr;
use datafusion::physical_plan::streaming::{PartitionStream, StreamingTableExec};
use datafusion::physical_plan::SendableRecordBatchStream;
use futures::StreamExt;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::exec_ir::{agg_ir_to_df_expr, output_type_to_arrow, parse_exec_ir_v1};
use crate::runtime;

use super::types::{DatasetHandle, SchemaHandle, StreamHandle};
use super::util::{dataset_handle, slice_from_ptr, FfiError, FfiResult};

#[derive(Debug)]
struct LanceExecPartition {
    dataset: Arc<lance::Dataset>,
    schema: Arc<Schema>,
    projection: Arc<[String]>,
    filter: Option<Expr>,
}

impl PartitionStream for LanceExecPartition {
    fn schema(&self) -> &arrow::datatypes::SchemaRef {
        &self.schema
    }

    fn execute(&self, ctx: Arc<datafusion::execution::TaskContext>) -> SendableRecordBatchStream {
        let schema = self.schema.clone();
        let handle = match crate::runtime::handle() {
            Ok(h) => h,
            Err(err) => {
                let stream = futures::stream::iter(vec![Err(DataFusionError::Execution(format!(
                    "runtime: {err}"
                )))]);
                let adapter =
                    datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream);
                return Box::pin(adapter);
            }
        };

        let mut builder = datafusion::physical_plan::stream::RecordBatchReceiverStream::builder(
            schema.clone(),
            2,
        );
        let tx = builder.tx();
        let dataset = self.dataset.clone();
        let projection = self.projection.clone();
        let filter = self.filter.clone();

        builder.spawn_on(
            async move {
                let mut scan = dataset.scan();
                scan.project(projection.as_ref())
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                if let Some(filter) = &filter {
                    scan.filter_expr(filter.clone());
                }
                scan.scan_in_order(false);

                let mut stream = scan
                    .try_into_stream()
                    .await
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                while let Some(item) = stream.next().await {
                    match item {
                        Ok(batch) => {
                            if tx.send(Ok(batch)).await.is_err() {
                                // Consumer dropped early.
                                return Ok(());
                            }
                        }
                        Err(err) => {
                            let _ = tx
                                .send(Err(DataFusionError::Execution(err.to_string())))
                                .await;
                            return Ok(());
                        }
                    }
                }
                Ok(())
            },
            &handle,
        );

        let _ = ctx;
        builder.build()
    }
}

#[derive(Debug)]
struct LanceExecTableProvider {
    schema: Arc<Schema>,
    partition: Arc<LanceExecPartition>,
}

#[async_trait::async_trait]
impl TableProvider for LanceExecTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> datafusion::datasource::TableType {
        datafusion::datasource::TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        limit: Option<usize>,
    ) -> DFResult<Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
        Ok(Arc::new(StreamingTableExec::try_new(
            self.schema(),
            vec![self.partition.clone()],
            projection,
            vec![],
            false,
            limit,
        )?))
    }
}

fn projected_schema(handle: &DatasetHandle, projection: &[String]) -> Result<Arc<Schema>, String> {
    let base_schema = handle.arrow_schema.as_ref();
    let mut fields = Vec::with_capacity(projection.len());
    for name in projection.iter() {
        let idx = base_schema
            .index_of(name)
            .map_err(|_| format!("projection column not found: {name}"))?;
        let field = base_schema.field(idx);
        fields.push(Field::new(
            field.name(),
            field.data_type().clone(),
            field.is_nullable(),
        ));
    }
    Ok(Arc::new(Schema::new(fields)))
}

async fn build_exec_df(handle: &DatasetHandle, exec_ir: &[u8]) -> Result<DataFrame, String> {
    let exec_ir = parse_exec_ir_v1(exec_ir).map_err(|e| format!("exec_ir parse: {e}"))?;

    let filter = if exec_ir.filter_ir.is_empty() {
        None
    } else {
        Some(
            crate::filter_ir::parse_filter_ir(&exec_ir.filter_ir)
                .map_err(|e| format!("filter_ir parse: {e}"))?,
        )
    };

    let projection: Arc<[String]> = exec_ir.scan_projection.clone().into();
    let schema = projected_schema(handle, projection.as_ref())?;

    let partition = Arc::new(LanceExecPartition {
        dataset: handle.dataset.clone(),
        schema: schema.clone(),
        projection,
        filter,
    });

    let provider = LanceExecTableProvider {
        schema: schema.clone(),
        partition,
    };

    let ctx = SessionContext::new();
    ctx.register_table("t", Arc::new(provider))
        .map_err(|e| e.to_string())?;
    let df = ctx.table("t").await.map_err(|e| e.to_string())?;

    let group_exprs = exec_ir
        .groups
        .iter()
        .map(|g| crate::exec_ir::expr_ir_to_df_expr(&g.expr).map(|e| e.alias(&g.output_name)))
        .collect::<Result<Vec<_>, _>>()?;

    let agg_exprs = exec_ir
        .aggs
        .iter()
        .map(agg_ir_to_df_expr)
        .collect::<Result<Vec<_>, _>>()?;

    let df = df
        .aggregate(group_exprs, agg_exprs)
        .map_err(|e| e.to_string())?;

    let mut select_exprs = Vec::with_capacity(exec_ir.groups.len() + exec_ir.aggs.len());

    for g in exec_ir.groups.iter() {
        let name = g.output_name.as_str();
        let expr = if let Some(hint) = &g.output_type {
            let dtype = output_type_to_arrow(hint)?;
            Expr::Cast(datafusion_expr::Cast::new(
                Box::new(datafusion::prelude::col(name)),
                dtype,
            ))
            .alias(name)
        } else {
            datafusion::prelude::col(name)
        };
        select_exprs.push(expr);
    }
    for agg in exec_ir.aggs.iter() {
        let name = agg.output_name.as_str();
        let expr = if let Some(hint) = &agg.output_type {
            let dtype = output_type_to_arrow(hint)?;
            Expr::Cast(datafusion_expr::Cast::new(Box::new(datafusion::prelude::col(name)), dtype))
                .alias(name)
        } else {
            datafusion::prelude::col(name)
        };
        select_exprs.push(expr);
    }

    let df = df.select(select_exprs).map_err(|e| e.to_string())?;

    if exec_ir.order_by.is_empty() {
        return Ok(df);
    }

    let sort_exprs = exec_ir
        .order_by
        .iter()
        .map(|o| datafusion::prelude::col(&o.column).sort(o.asc, o.nulls_first))
        .collect();

    df.sort(sort_exprs).map_err(|e| e.to_string())
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_exec_schema(
    dataset: *mut c_void,
    exec_ir: *const u8,
    exec_ir_len: usize,
) -> *mut c_void {
    match get_exec_schema_inner(dataset, exec_ir, exec_ir_len) {
        Ok(schema) => {
            clear_last_error();
            Box::into_raw(Box::new(schema)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn get_exec_schema_inner(
    dataset: *mut c_void,
    exec_ir: *const u8,
    exec_ir_len: usize,
) -> FfiResult<SchemaHandle> {
    let handle = unsafe { dataset_handle(dataset)? };
    let bytes = unsafe { slice_from_ptr(exec_ir, exec_ir_len, "exec_ir")? };

    let schema_res = runtime::block_on(async {
        let df = build_exec_df(&handle, bytes).await?;
        let plan = df
            .create_physical_plan()
            .await
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(plan.schema())
    })
    .map_err(|e| FfiError::new(ErrorCode::Exec, format!("runtime: {e}")))?;

    let schema = schema_res
        .map_err(|e| FfiError::new(ErrorCode::Exec, format!("exec validate: {e}")))?;
    Ok(Arc::new(Schema::new(schema.fields().clone())))
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_dataset_exec_stream_ir(
    dataset: *mut c_void,
    exec_ir: *const u8,
    exec_ir_len: usize,
) -> *mut c_void {
    match create_dataset_exec_stream_ir_inner(dataset, exec_ir, exec_ir_len) {
        Ok(stream) => {
            clear_last_error();
            Box::into_raw(Box::new(stream)) as *mut c_void
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            ptr::null_mut()
        }
    }
}

fn create_dataset_exec_stream_ir_inner(
    dataset: *mut c_void,
    exec_ir: *const u8,
    exec_ir_len: usize,
) -> FfiResult<StreamHandle> {
    let handle = unsafe { dataset_handle(dataset)? };
    let bytes = unsafe { slice_from_ptr(exec_ir, exec_ir_len, "exec_ir")? };

    let stream_res = runtime::block_on(async {
        let df = build_exec_df(&handle, bytes).await?;
        df.execute_stream().await.map_err(|e| e.to_string())
    })
    .map_err(|e| FfiError::new(ErrorCode::Exec, format!("runtime: {e}")))?;

    let stream = stream_res
        .map_err(|e| FfiError::new(ErrorCode::Exec, format!("exec stream: {e}")))?;

    let df_stream = crate::datafusion_stream::DataFusionStream::try_new(stream)
        .map_err(|e| FfiError::new(ErrorCode::Exec, format!("runtime: {e}")))?;

    Ok(StreamHandle::DataFusion(df_stream))
}
