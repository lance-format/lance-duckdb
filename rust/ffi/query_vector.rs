use std::ffi::c_void;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, UInt8Array};
use arrow::datatypes::DataType;
use lance::index::vector::utils::get_vector_type;
use lance_linalg::distance::DistanceType;

use crate::error::ErrorCode;

use super::types::DatasetHandle;
use super::util::{slice_from_ptr, FfiError, FfiResult};

pub(crate) const QUERY_ELEMENT_FLOAT32: u8 = 0;
pub(crate) const QUERY_ELEMENT_UINT8: u8 = 1;

pub(crate) unsafe fn query_array(
    query_values: *const c_void,
    query_len: usize,
    element_type: u8,
) -> FfiResult<ArrayRef> {
    if query_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "query vector must be non-empty",
        ));
    }
    match element_type {
        QUERY_ELEMENT_FLOAT32 => {
            let values =
                unsafe { slice_from_ptr(query_values.cast::<f32>(), query_len, "query_values")? };
            Ok(Arc::new(Float32Array::from_iter_values(
                values.iter().copied(),
            )))
        }
        QUERY_ELEMENT_UINT8 => {
            let values =
                unsafe { slice_from_ptr(query_values.cast::<u8>(), query_len, "query_values")? };
            Ok(Arc::new(UInt8Array::from_iter_values(
                values.iter().copied(),
            )))
        }
        _ => Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("unsupported query element type {element_type}"),
        )),
    }
}

pub(crate) fn validate_query_vector(
    handle: &DatasetHandle,
    vector_column: &str,
    query: ArrayRef,
    distance_type: Option<&DistanceType>,
) -> FfiResult<ArrayRef> {
    let (_, stored_type) = get_vector_type(handle.dataset.schema(), vector_column)
        .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, err.to_string()))?;
    let query_type = query.data_type().clone();

    // Preserve the existing numeric-cast behavior on floating-point columns.
    // UInt8 keeps its native binary meaning only on a UInt8 column.
    if stored_type.is_floating() && query_type == DataType::UInt8 {
        let values = query.as_any().downcast_ref::<UInt8Array>().unwrap();
        return Ok(Arc::new(Float32Array::from_iter_values(
            values.values().iter().copied().map(f32::from),
        )));
    }

    let float_coercion = stored_type.is_floating() && query_type == DataType::Float32;
    if stored_type != query_type && !float_coercion {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!(
                "Column {vector_column} has element type {stored_type} and the query vector is {query_type}"
            ),
        ));
    }
    if stored_type == DataType::UInt8
        && distance_type.is_some_and(|metric| *metric != DistanceType::Hamming)
    {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!(
                "UINT8 binary vector column {vector_column} requires distance_type='hamming', got {}",
                distance_type.unwrap()
            ),
        ));
    }
    Ok(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_array_preserves_uint8_values() {
        let values = [0_u8, 15, 255];
        let array =
            unsafe { query_array(values.as_ptr().cast(), values.len(), QUERY_ELEMENT_UINT8) }
                .unwrap();
        assert_eq!(array.data_type(), &DataType::UInt8);
        let array = array.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(array.value(0), 0);
        assert_eq!(array.value(1), 15);
        assert_eq!(array.value(2), 255);
    }

    #[test]
    fn query_array_rejects_unknown_element_type() {
        let values = [0.0_f32];
        let error = unsafe { query_array(values.as_ptr().cast(), 1, 99) }.unwrap_err();
        assert!(matches!(error.code, ErrorCode::InvalidArgument));
        assert!(error.message.contains("unsupported query element type 99"));
    }
}
