use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

use crate::constants::{DISTANCE_COLUMN, HYBRID_SCORE_COLUMN, ROW_ID_COLUMN, SCORE_COLUMN};

pub(crate) fn build_base_projection(schema: &Schema) -> Arc<[String]> {
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

pub(crate) fn build_hybrid_schema(schema: &Schema) -> Arc<Schema> {
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
    Arc::new(Schema::new(fields))
}
