use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};

use crate::constants::{DISTANCE_COLUMN, HYBRID_SCORE_COLUMN, SCORE_COLUMN};

pub(crate) fn is_default_excluded_field(field: &Field) -> bool {
    matches!(field.data_type(), DataType::FixedSizeList(_, _))
}

pub(crate) fn build_base_projection(schema: &Schema) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        if is_default_excluded_field(field) {
            continue;
        }
        cols.push(field.name().to_string());
    }
    cols.into()
}

pub(crate) fn build_fts_projection(base_projection: &Arc<[String]>) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 1);
    cols.extend(base_projection.iter().cloned());
    cols.push(SCORE_COLUMN.to_string());
    cols.into()
}

pub(crate) fn build_knn_projection(
    base_projection: &Arc<[String]>,
    vector_column: &str,
) -> Arc<[String]> {
    let mut cols = Vec::with_capacity(base_projection.len() + 1);
    for col in base_projection.iter() {
        if col == vector_column {
            continue;
        }
        cols.push(col.clone());
    }
    cols.push(DISTANCE_COLUMN.to_string());
    cols.into()
}

pub(crate) fn build_hybrid_schema(schema: &Schema) -> Arc<Schema> {
    let mut fields = Vec::with_capacity(schema.fields().len() + 3);
    for field in schema.fields() {
        if is_default_excluded_field(field) {
            continue;
        }
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
    Arc::new(Schema::new(fields))
}
