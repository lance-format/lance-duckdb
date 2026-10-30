use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use lance::dataset::rowids::get_row_id_index;
use lance::Dataset;
use lance_core::datatypes::{BlobHandling, Schema};
use lance_datafusion::projection::ProjectionPlan;

use crate::constants::{DISTANCE_COLUMN, HYBRID_SCORE_COLUMN, ROW_ID_COLUMN, SCORE_COLUMN};

pub(crate) fn build_base_projection(schema: &ArrowSchema) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        cols.push(field.name().to_string());
    }
    cols.into()
}

pub(crate) fn build_fts_projection(base_projection: &Arc<[String]>) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 2);
    cols.extend(base_projection.iter().cloned());
    cols.push(SCORE_COLUMN.to_string());
    cols.push(ROW_ID_COLUMN.to_string());
    cols.into()
}

pub(crate) fn build_knn_projection(base_projection: &Arc<[String]>) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 1);
    cols.extend(base_projection.iter().cloned());
    cols.push(DISTANCE_COLUMN.to_string());
    cols.into()
}

fn append_search_metric(schema: &ArrowSchema, metric: &str) -> Arc<ArrowSchema> {
    let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new(metric, DataType::Float32, true)));
    Arc::new(ArrowSchema::new(fields))
}

pub(crate) fn build_knn_schema(schema: &ArrowSchema) -> Arc<ArrowSchema> {
    append_search_metric(schema, DISTANCE_COLUMN)
}

pub(crate) fn build_fts_schema(schema: &ArrowSchema) -> Arc<ArrowSchema> {
    let mut fields = schema.fields().iter().cloned().collect::<Vec<_>>();
    fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)));
    fields.push(Arc::new(Field::new(ROW_ID_COLUMN, DataType::UInt64, false)));
    Arc::new(ArrowSchema::new(fields))
}

pub(crate) fn build_hybrid_schema(schema: &ArrowSchema) -> Arc<ArrowSchema> {
    let mut fields = Vec::with_capacity(schema.fields().len() + 4);
    for field in schema.fields() {
        fields.push(field.clone());
    }
    fields.push(Arc::new(Field::new(
        DISTANCE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(Field::new(SCORE_COLUMN, DataType::Float32, true)));
    fields.push(Arc::new(Field::new(
        HYBRID_SCORE_COLUMN,
        DataType::Float32,
        true,
    )));
    fields.push(Arc::new(Field::new(ROW_ID_COLUMN, DataType::UInt64, false)));
    Arc::new(ArrowSchema::new(fields))
}

/// Convert every Lance blob field in `schema` to its binary payload view.
///
/// SQL contracts bind BLOB and every read path returns the actual bytes.
/// Non-blob fields (including ordinary binary and nested non-blob structure)
/// are preserved unchanged.
pub(crate) fn materialized_schema(schema: &Schema) -> Schema {
    let fields = schema
        .fields
        .iter()
        .map(|field| BlobHandling::AllBinary.unload_if_needed(field.clone()))
        .collect();
    Schema {
        fields,
        metadata: schema.metadata.clone(),
    }
}

fn field_contains_blob_v2(field: &lance_core::datatypes::Field) -> bool {
    field.is_blob_v2() || field.children.iter().any(field_contains_blob_v2)
}

/// Whether the requested top-level columns contain a Blob v2 leaf.
///
/// Blob v1 on legacy storage remains on the native TakeBuilder path because
/// Lance 13 does not support row-address prefilter scans for legacy storage.
pub(crate) fn projection_contains_blob_v2(
    dataset: &Dataset,
    columns: &[String],
) -> Result<bool, String> {
    let projected = dataset
        .schema()
        .project_preserve_system_columns(columns)
        .map_err(|err| format!("take projection: {err}"))?;
    Ok(projected.fields.iter().any(field_contains_blob_v2))
}

/// Build a projection plan whose blob fields are materialized as binary.
pub(crate) fn binary_projection_plan(
    dataset: Arc<Dataset>,
    columns: &[String],
) -> Result<ProjectionPlan, String> {
    let projected = dataset
        .schema()
        .project_preserve_system_columns(columns)
        .map_err(|err| format!("take projection: {err}"))?;
    let projected = materialized_schema(&projected);
    let mut plan = ProjectionPlan::from_schema(dataset, &projected)
        .map_err(|err| format!("take projection plan: {err}"))?;
    plan.physical_projection = plan
        .physical_projection
        .clone()
        .with_blob_handling(BlobHandling::AllBinary);
    Ok(plan)
}

/// Resolve stable row IDs to physical row addresses in input order.
///
/// Datasets without stable row IDs already use row addresses as their
/// `_rowid` surface, so the input is returned unchanged.
pub(crate) async fn row_ids_to_addresses(
    dataset: &Dataset,
    row_ids: &[u64],
) -> Result<Vec<u64>, String> {
    match get_row_id_index(dataset)
        .await
        .map_err(|err| format!("row id index: {err}"))?
    {
        Some(index) => Ok(index
            .get_many(row_ids)
            .map_err(|err| format!("row id index: {err}"))?
            .into_iter()
            .filter_map(|address| address.map(u64::from))
            .collect()),
        None => Ok(row_ids.to_vec()),
    }
}
