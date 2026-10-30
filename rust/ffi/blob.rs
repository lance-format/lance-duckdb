// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared Blob v2 write-input conversion for the DuckDB SQL surface.
//!
//! DuckDB exposes one write input type (`LANCE_BLOB_INPUT`, a named UNION with
//! `bytes` / `reference` / `ingest` members) plus ordinary `BLOB` columns.
//! Lance consumes logical blob v2 input as `Struct<data: LargeBinary, uri:
//! Utf8, position: UInt64, size: UInt64>` tagged `lance.blob.v2`. This module
//! is the single conversion point used by INSERT/COPY/CTAS and by the DML
//! (UPDATE/MERGE) paths: classify each top-level column against the
//! authoritative target schema, then turn DuckDB values into Lance logical
//! blob arrays.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::ops::Range;
use std::ptr;
use std::sync::Arc;

use arrow_array::builder::{BinaryBuilder, LargeBinaryBuilder, PrimitiveBuilder, StringBuilder};
use arrow_array::types::UInt64Type;
use arrow_array::{
    make_array, Array, ArrayRef, BinaryArray, LargeBinaryArray, StringArray, StructArray,
    UInt64Array, UnionArray,
};
use arrow_schema::{DataType, Field, Fields, Schema};
use lance::io::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry, StorageOptionsAccessor};
use lance_arrow::{ARROW_EXT_NAME_KEY, BLOB_META_KEY, BLOB_V2_EXT_NAME};
use lance_core::datatypes::BLOB_V2_LOGICAL_TYPE;
use lance_io::object_store::uri_to_url;
use object_store::path::Path as ObjectStorePath;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, optional_session_handle, FfiError, FfiResult};
use arrow_buffer::NullBufferBuilder;

/// Fallback ingest batch budget when the caller does not supply one. The C++
/// write paths derive the budget from DuckDB's memory limit; this value only
/// applies to direct FFI callers (e.g. tests).
pub(crate) const DEFAULT_INGEST_BATCH_BUDGET: usize = 256 * 1024 * 1024;

/// Whether a Lance manifest file version supports Blob v2 (>= 2.2).
pub(crate) fn version_supports_blob_v2(version: &str) -> bool {
    let mut parts = version.split('.');
    let major = parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);
    let minor = parts
        .next()
        .and_then(|part| part.parse::<u32>().ok())
        .unwrap_or(0);
    major > 2 || (major == 2 && minor >= 2)
}

/// Whether each row of a blob input column is raw bytes, an external reference,
/// or a managed ingest request. Tags match the DuckDB UNION member order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlobInputMode {
    Bytes,
    Reference,
    Ingest,
}

/// The storage treatment of one top-level write column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlobColumnTarget {
    /// The persisted field is Blob v2; URI inputs are references and ingest
    /// inputs are materialized before storage.
    BlobV2,
    /// The persisted field keeps its existing binary encoding; URI inputs are
    /// rejected instead of silently migrating the field.
    Binary,
}

#[derive(Debug, Clone)]
pub(crate) struct BlobColumnConversion {
    pub(crate) col_idx: usize,
    pub(crate) target: BlobColumnTarget,
    /// The authoritative destination field for binary targets: the existing
    /// dataset field (Binary/LargeBinary plus legacy metadata) when present,
    /// otherwise the input field. Blob v2 targets derive their output field
    /// from the logical blob shape.
    pub(crate) output_field: Option<Field>,
}

/// Per-write context for materializing ingest values.
#[derive(Clone)]
pub(crate) struct BlobWriteContext {
    pub(crate) registry: Arc<ObjectStoreRegistry>,
    /// Total ingest bytes that may be materialized for one batch. The budget
    /// is consumed across every ingest row of every blob column in that batch
    /// and checked before each read, so a failed (oversized) ingest never
    /// publishes partial state. Buffered batches waiting for vector-dimension
    /// inference (at most `MAX_VECTOR_DIM_INFERENCE_BATCHES`) and the writer
    /// channel window are additional bounded copies; no end-to-end query/RSS
    /// bound is implied.
    pub(crate) ingest_batch_budget: usize,
}

/// Per-batch source storage options keyed by the exact ingest URI string.
/// DuckDB's native secret lookup performs the selection; Rust only consumes
/// the resolved options.
pub(crate) type SourceOptionsMap = HashMap<String, HashMap<String, String>>;

fn normalize_source_uri(uri: &str) -> Result<String, String> {
    let url = uri_to_url(uri).map_err(|err| format!("invalid source URI '{uri}': {err}"))?;
    if url.scheme() == "file" {
        let path = url
            .to_file_path()
            .map_err(|_| format!("invalid file source URI '{uri}'"))?;
        ObjectStorePath::from_absolute_path(&path)
            .or_else(|_| ObjectStorePath::from_url_path(url.path()))
            .map_err(|err| format!("Failed to parse path '{}': {err}", url.path()))?;
        return path
            .to_str()
            .map(str::to_string)
            .ok_or_else(|| format!("file source URI '{uri}' is not valid UTF-8"));
    }
    Ok(url.as_str().to_string())
}

/// Normalize an ingest source through the same URI parser Lance uses before
/// opening an object store. DuckDB's admission and secret lookup must use this
/// resource identity, not the raw URI spelling.
#[no_mangle]
pub unsafe extern "C" fn lance_normalize_source_uri(
    uri: *const c_char,
    out_uri: *mut *const c_char,
) -> i32 {
    if out_uri.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_uri is null");
        return -1;
    }
    let uri = match unsafe { cstr_to_str(uri, "uri") } {
        Ok(uri) => uri,
        Err(err) => {
            set_last_error(err.code, err.message);
            return -1;
        }
    };
    let normalized = match normalize_source_uri(uri) {
        Ok(normalized) => normalized,
        Err(message) => {
            set_last_error(ErrorCode::InvalidArgument, message);
            return -1;
        }
    };
    let normalized = match CString::new(normalized) {
        Ok(normalized) => normalized,
        Err(_) => {
            set_last_error(
                ErrorCode::InvalidArgument,
                "normalized source URI contains a NUL byte",
            );
            return -1;
        }
    };
    unsafe {
        *out_uri = normalized.into_raw();
    }
    clear_last_error();
    0
}

/// Build the complete logical blob v2 input field used by the Lance writer.
pub(crate) fn blob_v2_logical_field(name: &str, nullable: bool) -> Field {
    Field::new(name, BLOB_V2_LOGICAL_TYPE.clone(), nullable).with_metadata(HashMap::from([(
        ARROW_EXT_NAME_KEY.to_string(),
        BLOB_V2_EXT_NAME.to_string(),
    )]))
}

/// True when an Arrow field is a Lance blob v2 logical field.
pub(crate) fn is_blob_v2_field(field: &Field) -> bool {
    field
        .metadata()
        .get(ARROW_EXT_NAME_KEY)
        .is_some_and(|extension_name| extension_name == BLOB_V2_EXT_NAME)
}

/// True when a binary Arrow field carries storage semantics of its own (Arrow
/// extension types such as arrow.opaque/arrow.json or the legacy Blob marker).
/// Such fields must keep their encoding instead of becoming Blob v2.
fn has_non_blob_storage_extension(field: &Field) -> bool {
    let extension = field.metadata().get(ARROW_EXT_NAME_KEY);
    if extension.is_some_and(|name| name != BLOB_V2_EXT_NAME) {
        return true;
    }
    field.metadata().contains_key(BLOB_META_KEY)
}

/// True when an Arrow (DuckDB exported) type is the named LANCE_BLOB_INPUT
/// union. The member names are the contract; DuckDB expression composition may
/// drop the alias, so a structural match is the durable check.
pub(crate) fn is_blob_input_type(data_type: &DataType) -> bool {
    let DataType::Union(fields, _) = data_type else {
        return false;
    };
    let names = fields
        .iter()
        .map(|(_, field)| field.name().as_str())
        .collect::<Vec<_>>();
    names == ["bytes", "reference", "ingest"]
}

fn find_nested_blob_input(field: &Field) -> Option<String> {
    match field.data_type() {
        DataType::Union(_, _) if is_blob_input_type(field.data_type()) => {
            Some(field.name().clone())
        }
        DataType::Struct(fields) => fields
            .iter()
            .find_map(|child| find_nested_blob_input(child)),
        DataType::Map(entries, _) => match entries.data_type() {
            DataType::Struct(fields) => fields
                .iter()
                .find_map(|child| find_nested_blob_input(child)),
            _ => None,
        },
        DataType::List(child) | DataType::LargeList(child) | DataType::FixedSizeList(child, _) => {
            find_nested_blob_input(child)
        }
        _ => None,
    }
}

/// Decide per top-level column how the Lance writer must receive its values.
///
/// `existing` is the authoritative `dataset.schema()` of the destination
/// (append mode); `new_fields_use_blob_v2` is true only when the effective
/// storage version of the write is >= 2.2.
pub(crate) fn plan_blob_conversions(
    input_schema: &Schema,
    existing: Option<&Schema>,
    new_fields_use_blob_v2: bool,
    blob_flags: &[u8],
) -> Result<Vec<BlobColumnConversion>, String> {
    let mut conversions = Vec::new();
    for (idx, field) in input_schema.fields().iter().enumerate() {
        // DuckDB decides which top-level columns are BLOB write targets by
        // their SQL type; Arrow cannot distinguish BLOB from e.g. BIT when
        // lossless conversion is off. Only flagged columns participate.
        if blob_flags.get(idx).copied().unwrap_or(0) == 0 {
            continue;
        }
        match field.data_type() {
            DataType::Union(_, _) => {
                let is_input = is_blob_input_type(field.data_type());
                if !is_input {
                    continue;
                }
                let existing_field = existing.and_then(|schema| {
                    schema
                        .fields()
                        .iter()
                        .find(|candidate| candidate.name() == field.name())
                });
                let target = match existing_field {
                    Some(existing_field) if is_blob_v2_field(existing_field) => {
                        BlobColumnTarget::BlobV2
                    }
                    Some(_) => BlobColumnTarget::Binary,
                    None => {
                        if new_fields_use_blob_v2 {
                            BlobColumnTarget::BlobV2
                        } else {
                            BlobColumnTarget::Binary
                        }
                    }
                };
                if is_input || target == BlobColumnTarget::BlobV2 {
                    let output_field = if target == BlobColumnTarget::Binary {
                        Some(
                            existing_field
                                .map(|existing_field| existing_field.as_ref().clone())
                                .unwrap_or_else(|| {
                                    Field::new(field.name(), DataType::Binary, field.is_nullable())
                                }),
                        )
                    } else {
                        None
                    };
                    conversions.push(BlobColumnConversion {
                        col_idx: idx,
                        target,
                        output_field,
                    });
                }
            }
            DataType::Binary | DataType::LargeBinary => {
                if has_non_blob_storage_extension(field) {
                    continue;
                }
                let existing_field = existing.and_then(|schema| {
                    schema
                        .fields()
                        .iter()
                        .find(|candidate| candidate.name() == field.name())
                });
                let target = match existing_field {
                    Some(existing_field) if is_blob_v2_field(existing_field) => {
                        BlobColumnTarget::BlobV2
                    }
                    Some(_) => BlobColumnTarget::Binary,
                    None => {
                        if new_fields_use_blob_v2 {
                            BlobColumnTarget::BlobV2
                        } else {
                            BlobColumnTarget::Binary
                        }
                    }
                };
                if target == BlobColumnTarget::BlobV2 {
                    conversions.push(BlobColumnConversion {
                        col_idx: idx,
                        target,
                        output_field: None,
                    });
                }
            }
            DataType::Struct(fields) => {
                if let Some(nested) = fields
                    .iter()
                    .find_map(|child| find_nested_blob_input(child))
                {
                    return Err(format!(
                        "nested LANCE_BLOB_INPUT values are not supported (column '{nested}')"
                    ));
                }
            }
            DataType::Map(entries, _) => {
                if let Some(nested) = match entries.data_type() {
                    DataType::Struct(fields) => fields
                        .iter()
                        .find_map(|child| find_nested_blob_input(child)),
                    _ => None,
                } {
                    return Err(format!(
                        "nested LANCE_BLOB_INPUT values are not supported (column '{nested}')"
                    ));
                }
            }
            DataType::List(child)
            | DataType::LargeList(child)
            | DataType::FixedSizeList(child, _) => {
                if let Some(nested) = find_nested_blob_input(child) {
                    return Err(format!(
                        "nested LANCE_BLOB_INPUT values are not supported (column '{nested}')"
                    ));
                }
            }
            _ => {}
        }
    }
    Ok(conversions)
}

/// Build the output (Lance-side) schema for the given blob conversions.
pub(crate) fn apply_blob_output_fields(
    input_schema: &Schema,
    conversions: &[BlobColumnConversion],
) -> Schema {
    if conversions.is_empty() {
        return input_schema.clone();
    }
    let mut fields = input_schema.fields().as_ref().to_vec();
    for conversion in conversions {
        let idx = conversion.col_idx;
        let output = match &conversion.output_field {
            Some(field) => field.clone(),
            None => blob_v2_logical_field(fields[idx].name(), fields[idx].is_nullable()),
        };
        fields[idx] = Arc::new(output);
    }
    Schema::new(fields)
}

pub(crate) struct RowValue {
    pub(crate) mode: BlobInputMode,
    pub(crate) data: Option<Vec<u8>>,
    pub(crate) uri: Option<String>,
    pub(crate) position: Option<u64>,
    pub(crate) size: Option<u64>,
}

fn cast_u64_array<'a>(
    array: &'a ArrayRef,
    column_name: &str,
    field_name: &str,
) -> Result<&'a UInt64Array, String> {
    array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| format!("blob input column '{column_name}' has invalid {field_name} array"))
}

fn cast_uri_array<'a>(array: &'a ArrayRef, column_name: &str) -> Result<&'a StringArray, String> {
    array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| format!("blob input column '{column_name}' has invalid uri array"))
}

pub(crate) fn read_union_row_value(
    union: &UnionArray,
    row: usize,
    column_name: &str,
) -> Result<RowValue, String> {
    let type_id = union.type_id(row);
    let value_offset = union.value_offset(row);
    let member_name = union
        .fields()
        .iter()
        .find(|(id, _)| *id == type_id)
        .map(|(_, field)| field.name().as_str())
        .ok_or_else(|| {
            format!("blob input column '{column_name}' row {row} has unknown union tag {type_id}")
        })?;
    let child = union.child(type_id);
    let value = child.slice(value_offset, 1);
    match member_name {
        "bytes" => {
            if value.is_null(0) {
                // DuckDB represents a SQL NULL cast to the write input as a
                // bytes variant holding NULL. Normalize it to the NULL blob.
                return Ok(RowValue {
                    mode: BlobInputMode::Bytes,
                    data: None,
                    uri: None,
                    position: None,
                    size: None,
                });
            }
            if let Some(binary) = value.as_any().downcast_ref::<BinaryArray>() {
                Ok(RowValue {
                    mode: BlobInputMode::Bytes,
                    data: Some(binary.value(0).to_vec()),
                    uri: None,
                    position: None,
                    size: None,
                })
            } else if let Some(large_binary) = value.as_any().downcast_ref::<LargeBinaryArray>() {
                Ok(RowValue {
                    mode: BlobInputMode::Bytes,
                    data: Some(large_binary.value(0).to_vec()),
                    uri: None,
                    position: None,
                    size: None,
                })
            } else {
                Err(format!(
                    "blob input column '{column_name}' row {row} has an invalid bytes member"
                ))
            }
        }
        "reference" | "ingest" => {
            let mode = if member_name == "ingest" {
                BlobInputMode::Ingest
            } else {
                BlobInputMode::Reference
            };
            let struct_value = value
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| {
                    format!("blob input column '{column_name}' row {row} has an invalid URI member")
                })?;
            if struct_value.is_null(0) {
                return Err(format!(
                    "blob input column '{column_name}' row {row} has a non-NULL input with a NULL URI descriptor"
                ));
            }
            let Some(uri) = struct_value.column_by_name("uri") else {
                return Err(format!(
                    "blob input column '{column_name}' row {row} has a URI descriptor without a uri field"
                ));
            };
            let uri_array = cast_uri_array(uri, column_name)?;
            if uri_array.is_null(0) {
                return Err(format!(
                    "blob input column '{column_name}' row {row} has a NULL URI"
                ));
            }
            let uri = uri_array.value(0).to_string();
            validate_absolute_uri(&uri, column_name, row)?;
            let position = struct_value
                .column_by_name("position")
                .map(|column| {
                    cast_u64_array(column, column_name, "position")
                        .map(|array| (!array.is_null(0)).then(|| array.value(0)))
                })
                .transpose()?
                .flatten();
            let size = struct_value
                .column_by_name("size")
                .map(|column| {
                    cast_u64_array(column, column_name, "size")
                        .map(|array| (!array.is_null(0)).then(|| array.value(0)))
                })
                .transpose()?
                .flatten();
            match (position, size) {
                (None, None) => {}
                (Some(position), Some(size)) => {
                    if size == 0 {
                        return Err(format!(
                            "blob input column '{column_name}' row {row} has an external range with zero size"
                        ));
                    }
                    position.checked_add(size).ok_or_else(|| {
                        format!(
                            "blob input column '{column_name}' row {row} has an overflowing external range"
                        )
                    })?;
                }
                _ => {
                    return Err(format!(
                        "blob input column '{column_name}' row {row} must set both `position` and `size`, or neither"
                    ));
                }
            }
            Ok(RowValue {
                mode,
                data: None,
                uri: Some(uri),
                position,
                size,
            })
        }
        other => Err(format!(
            "blob input column '{column_name}' row {row} has unknown union member '{other}'"
        )),
    }
}

/// The write boundary enforces the same absolute-URI invariant the SQL
/// constructors promise. A hand-built LANCE_BLOB_INPUT union value must not
/// persist a relative path just because it never passed `lance_blob_ref`.
fn validate_absolute_uri(uri: &str, column_name: &str, row: usize) -> Result<(), String> {
    if uri.is_empty() {
        return Err(format!(
            "blob input column '{column_name}' row {row} has an empty URI"
        ));
    }
    let Some(scheme_end) = uri.find("://") else {
        return Err(format!(
            "blob input column '{column_name}' row {row} requires an absolute URI with a scheme (e.g. file:///path or s3://bucket/key), got '{uri}'"
        ));
    };
    if scheme_end == 0 || !uri.as_bytes()[0].is_ascii_alphabetic() {
        return Err(format!(
            "blob input column '{column_name}' row {row} requires an absolute URI with a scheme, got '{uri}'"
        ));
    }
    for byte in uri.as_bytes()[1..scheme_end].iter() {
        if !(byte.is_ascii_alphanumeric() || *byte == b'+' || *byte == b'-' || *byte == b'.') {
            return Err(format!(
                "blob input column '{column_name}' row {row} requires an absolute URI with a scheme, got '{uri}'"
            ));
        }
    }
    Ok(())
}

fn read_binary_row_value(
    array: &dyn Array,
    row: usize,
    column_name: &str,
) -> Result<RowValue, String> {
    if let Some(binary) = array.as_any().downcast_ref::<BinaryArray>() {
        if binary.is_null(row) {
            return Ok(RowValue {
                mode: BlobInputMode::Bytes,
                data: None,
                uri: None,
                position: None,
                size: None,
            });
        }
        return Ok(RowValue {
            mode: BlobInputMode::Bytes,
            data: Some(binary.value(row).to_vec()),
            uri: None,
            position: None,
            size: None,
        });
    }
    if let Some(large_binary) = array.as_any().downcast_ref::<LargeBinaryArray>() {
        if large_binary.is_null(row) {
            return Ok(RowValue {
                mode: BlobInputMode::Bytes,
                data: None,
                uri: None,
                position: None,
                size: None,
            });
        }
        return Ok(RowValue {
            mode: BlobInputMode::Bytes,
            data: Some(large_binary.value(row).to_vec()),
            uri: None,
            position: None,
            size: None,
        });
    }
    Err(format!(
        "blob input column '{column_name}' row {row} is not a binary array"
    ))
}

pub(crate) async fn read_ingest_bytes(
    uri: &str,
    position: Option<u64>,
    size: Option<u64>,
    remaining_budget: usize,
    registry: &Arc<ObjectStoreRegistry>,
    source_options: &SourceOptionsMap,
) -> Result<Vec<u8>, String> {
    let store_params = source_options
        .get(uri)
        .map(|options| ObjectStoreParams {
            storage_options_accessor: Some(Arc::new(StorageOptionsAccessor::with_static_options(
                options.clone(),
            ))),
            ..Default::default()
        })
        .unwrap_or_default();
    let (store, path) = ObjectStore::from_uri_and_params(registry.clone(), uri, &store_params)
        .await
        .map_err(|err| format!("failed to open ingest source '{uri}': {err}"))?;
    let range: Range<usize> = match (position, size) {
        (Some(position), Some(size)) => {
            let end = position.checked_add(size).ok_or_else(|| {
                format!(
                    "ingest source '{uri}' range overflows u64: position={position}, size={size}"
                )
            })?;
            let end = usize::try_from(end)
                .map_err(|_| format!("ingest source '{uri}' range is too large"))?;
            let start = usize::try_from(position)
                .map_err(|_| format!("ingest source '{uri}' position {position} is too large"))?;
            start..end
        }
        (None, None) => {
            let length = store
                .size(&path)
                .await
                .map_err(|err| format!("failed to size ingest source '{uri}': {err}"))?;
            let length = usize::try_from(length)
                .map_err(|_| format!("ingest source '{uri}' is too large to materialize"))?;
            0..length
        }
        _ => {
            return Err(format!(
                "ingest source '{uri}' must set both position and size when slicing"
            ))
        }
    };
    let range_len = range.end - range.start;
    if range_len > remaining_budget {
        return Err(format!(
            "ingest source '{uri}' requires {range_len} bytes, exceeding the remaining ingest memory budget of {remaining_budget} bytes for this batch; use lance_blob_ref to store an external reference instead"
        ));
    }
    let bytes = store
        .read_one_range(&path, range)
        .await
        .map_err(|err| format!("failed to read ingest source '{uri}': {err}"))?;
    validate_ingest_read_len(bytes.len(), range_len, uri)?;
    Ok(bytes.to_vec())
}

/// Remote object stores clamp a requested range whose end exceeds the object
/// length, so a successful read does not by itself prove the range was valid.
/// The write contract requires an explicit out-of-range error.
fn validate_ingest_read_len(actual: usize, expected: usize, uri: &str) -> Result<(), String> {
    if actual != expected {
        return Err(format!(
            "ingest source '{uri}' returned {actual} bytes for a {expected}-byte range; the range exceeds the source object"
        ));
    }
    Ok(())
}

fn build_bytes_target_array(
    column_name: &str,
    rows: &[RowValue],
    _nullable: bool,
) -> Result<ArrayRef, String> {
    let mut data = LargeBinaryBuilder::new();
    let mut uri = StringBuilder::new();
    let mut position = PrimitiveBuilder::<UInt64Type>::new();
    let mut size = PrimitiveBuilder::<UInt64Type>::new();
    let mut nulls = NullBufferBuilder::new(rows.len());
    for value in rows.iter() {
        data.append_option(value.data.as_deref());
        uri.append_option(value.uri.as_deref());
        position.append_option(value.position);
        size.append_option(value.size);
        nulls.append(value.data.is_some() || value.uri.is_some());
    }
    finish_blob_v2_struct(column_name, data, uri, position, size, nulls)
}

fn finish_blob_v2_struct(
    column_name: &str,
    mut data: LargeBinaryBuilder,
    mut uri: StringBuilder,
    mut position: PrimitiveBuilder<UInt64Type>,
    mut size: PrimitiveBuilder<UInt64Type>,
    mut nulls: NullBufferBuilder,
) -> Result<ArrayRef, String> {
    let fields = Fields::from(vec![
        Arc::new(Field::new("data", DataType::LargeBinary, true)),
        Arc::new(Field::new("uri", DataType::Utf8, true)),
        Arc::new(Field::new("position", DataType::UInt64, true)),
        Arc::new(Field::new("size", DataType::UInt64, true)),
    ]);
    let struct_array = StructArray::try_new(
        fields,
        vec![
            Arc::new(data.finish()),
            Arc::new(uri.finish()),
            Arc::new(position.finish()),
            Arc::new(size.finish()),
        ],
        nulls.finish(),
    )
    .map_err(|err| format!("failed to build blob input column '{column_name}': {err}"))?;
    // The child fields keep Lance's canonical logical names and types; the
    // blob v2 extension marker is carried by the output-schema field.
    Ok(Arc::new(struct_array))
}

/// Wrap plain bytes as the logical Blob v2 struct used by schema evolution.
///
/// All rows carry `data` and leave the external-reference coordinates empty.
/// `ALTER TABLE ... ADD COLUMN ... DEFAULT` is the only caller: URI values are
/// a DML input contract, not a stored-default contract.
pub(crate) fn blob_v2_array_from_bytes(
    column_name: &str,
    array: &dyn Array,
) -> Result<ArrayRef, String> {
    let mut data = LargeBinaryBuilder::new();
    let mut uri = StringBuilder::new();
    let mut position = PrimitiveBuilder::<UInt64Type>::new();
    let mut size = PrimitiveBuilder::<UInt64Type>::new();
    let mut nulls = NullBufferBuilder::new(array.len());
    match array.data_type() {
        DataType::Binary => {
            let values = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| format!("blob column '{column_name}' is not a binary array"))?;
            for row in 0..values.len() {
                data.append_option((!values.is_null(row)).then(|| values.value(row)));
                uri.append_null();
                position.append_null();
                size.append_null();
                nulls.append(!values.is_null(row));
            }
        }
        DataType::LargeBinary => {
            let values = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| {
                    format!("blob column '{column_name}' is not a large binary array")
                })?;
            for row in 0..values.len() {
                data.append_option((!values.is_null(row)).then(|| values.value(row)));
                uri.append_null();
                position.append_null();
                size.append_null();
                nulls.append(!values.is_null(row));
            }
        }
        DataType::Null => {
            for _ in 0..array.len() {
                data.append_null();
                uri.append_null();
                position.append_null();
                size.append_null();
                nulls.append(false);
            }
        }
        other => {
            return Err(format!(
                "blob column '{column_name}' default expression produced {other}, expected binary"
            ))
        }
    }
    finish_blob_v2_struct(column_name, data, uri, position, size, nulls)
}

/// Read one blob source (or range) and hand the bytes to the caller.
///
/// Used by the C++ write boundary to normalize ingest rows exactly once before
/// the same batch is shared by the writer and RETURNING.
#[no_mangle]
pub unsafe extern "C" fn lance_blob_read_source(
    uri: *const c_char,
    position: *const u8,
    size: *const u8,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_bytes: *mut *const u8,
    out_len: *mut usize,
) -> i32 {
    match blob_read_source_inner(
        uri,
        position,
        size,
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
        ingest_batch_budget,
        session,
        out_bytes,
        out_len,
    ) {
        Ok(()) => {
            clear_last_error();
            0
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_blob_bytes(ptr: *const u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr as *mut u8, len, len));
    }
}

#[allow(clippy::too_many_arguments)]
fn blob_read_source_inner(
    uri: *const c_char,
    position: *const u8,
    size: *const u8,
    source_uris: *const *const c_char,
    source_uri_keys: *const *const *const c_char,
    source_uri_values: *const *const *const c_char,
    source_uri_counts: *const usize,
    source_uri_count: usize,
    ingest_batch_budget: u64,
    session: *mut c_void,
    out_bytes: *mut *const u8,
    out_len: *mut usize,
) -> FfiResult<()> {
    if uri.is_null() || out_bytes.is_null() || out_len.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "uri/out_bytes/out_len is null",
        ));
    }
    let uri = unsafe { cstr_to_str(uri, "uri")? }.to_string();
    let position = if position.is_null() {
        None
    } else {
        Some(unsafe { ptr::read_unaligned(position as *const u64) })
    };
    let size = if size.is_null() {
        None
    } else {
        Some(unsafe { ptr::read_unaligned(size as *const u64) })
    };
    if position.is_some() != size.is_some() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "position/size must be supplied together",
        ));
    }
    let source_options = super::write::parse_batch_source_options(
        source_uris,
        source_uri_keys,
        source_uri_values,
        source_uri_counts,
        source_uri_count,
    )?;
    let session = unsafe { optional_session_handle(session)? };
    let budget = if ingest_batch_budget == 0 {
        DEFAULT_INGEST_BATCH_BUDGET
    } else {
        usize::try_from(ingest_batch_budget).map_err(|err| {
            FfiError::new(
                ErrorCode::InvalidArgument,
                format!("invalid ingest memory budget: {err}"),
            )
        })?
    };
    let registry = session
        .as_ref()
        .map(|session| session.store_registry())
        .unwrap_or_default();
    let bytes = runtime::block_on(read_ingest_bytes(
        &uri,
        position,
        size,
        budget,
        &registry,
        &source_options,
    ))
    .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, err.to_string()))?
    .map_err(|err| FfiError::new(ErrorCode::InvalidArgument, err))?;
    let mut boxed = bytes.into_boxed_slice();
    let len = boxed.len();
    let data = boxed.as_mut_ptr();
    std::mem::forget(boxed);
    unsafe {
        *out_bytes = data;
        *out_len = len;
    }
    Ok(())
}

/// Materialize every row of one blob input column as raw bytes.
///
/// Used by RETURNING row images: bytes pass through, while reference and
/// ingest values are read from their sources under the same per-batch budget
/// and source credentials as the write conversion.
pub(crate) fn blob_input_column_to_binary(
    column_name: &str,
    array: &dyn Array,
    context: &BlobWriteContext,
    source_options: &SourceOptionsMap,
    remaining_budget: &mut usize,
) -> Result<ArrayRef, String> {
    if let Some(union) = array.as_any().downcast_ref::<UnionArray>() {
        let mut values: Vec<Option<Vec<u8>>> = Vec::with_capacity(array.len());
        let mut reads = Vec::new();
        for row in 0..array.len() {
            if union.is_null(row) {
                values.push(None);
                continue;
            }
            let value = read_union_row_value(union, row, column_name)?;
            if value.mode == BlobInputMode::Bytes {
                values.push(value.data);
            } else {
                reads.push((
                    row,
                    value.uri.expect("reference/ingest rows carry a uri"),
                    value.position,
                    value.size,
                ));
                values.push(None);
            }
        }
        if !reads.is_empty() {
            let registry = context.registry.clone();
            runtime::block_on(async {
                for (row, uri, position, size) in reads {
                    let bytes = read_ingest_bytes(
                        &uri,
                        position,
                        size,
                        *remaining_budget,
                        &registry,
                        source_options,
                    )
                    .await?;
                    *remaining_budget = remaining_budget.saturating_sub(bytes.len());
                    values[row] = Some(bytes);
                }
                Ok::<_, String>(())
            })
            .map_err(|err| err.to_string())??;
        }
        let mut builder = LargeBinaryBuilder::new();
        for value in &values {
            builder.append_option(value.as_deref());
        }
        return Ok(Arc::new(builder.finish()));
    }
    if array.data_type() == &DataType::LargeBinary {
        return Ok(make_array(array.to_data()));
    }
    if array.data_type() == &DataType::Binary {
        let binary = array
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| format!("blob column '{column_name}' is not a BinaryArray"))?;
        let mut builder = LargeBinaryBuilder::new();
        for row in 0..binary.len() {
            if binary.is_null(row) {
                builder.append_null();
            } else {
                builder.append_value(binary.value(row));
            }
        }
        return Ok(Arc::new(builder.finish()));
    }
    Err(format!(
        "blob column '{column_name}' is neither a LANCE_BLOB_INPUT union nor binary"
    ))
}

fn build_binary_target_array(
    column_name: &str,
    rows: &[RowValue],
    target_field: &Field,
) -> Result<ArrayRef, String> {
    // The persisted representation is authoritative: an existing Binary or
    // LargeBinary field keeps its own type, and rows stay bytes only.
    match target_field.data_type() {
        DataType::Binary => {
            let mut data = BinaryBuilder::new();
            for (row, value) in rows.iter().enumerate() {
                reject_uri_row(column_name, row, value)?;
                data.append_option(value.data.as_deref());
            }
            Ok(Arc::new(data.finish()))
        }
        DataType::LargeBinary => {
            let mut data = LargeBinaryBuilder::new();
            for (row, value) in rows.iter().enumerate() {
                reject_uri_row(column_name, row, value)?;
                data.append_option(value.data.as_deref());
            }
            Ok(Arc::new(data.finish()))
        }
        other => Err(format!(
            "blob input column '{column_name}' targets binary storage of type {other}"
        )),
    }
}

fn reject_uri_row(column_name: &str, row: usize, value: &RowValue) -> Result<(), String> {
    if value.mode == BlobInputMode::Reference || value.mode == BlobInputMode::Ingest {
        return Err(format!(
            "blob input column '{column_name}' row {row} uses URI input, but the target column keeps its existing binary encoding; URI assignments are not supported without migrating the column to Blob v2"
        ));
    }
    Ok(())
}

/// Convert one blob column's DuckDB values into the Lance writer representation.
///
/// `nullable` describes the target/input column and is preserved on the
/// produced logical field. Ingest rows are materialized here with a bounded
/// per-object read; reference rows keep their URI so Lance owns the external
/// reference semantics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn convert_blob_column(
    column_name: &str,
    array: &dyn Array,
    target: BlobColumnTarget,
    nullable: bool,
    context: &BlobWriteContext,
    source_options: &SourceOptionsMap,
    remaining_budget: &mut usize,
    binary_target_field: Option<&Field>,
) -> Result<ArrayRef, String> {
    let mut rows = Vec::with_capacity(array.len());
    let mut ingest_uris = Vec::new();
    if let Some(union) = array.as_any().downcast_ref::<UnionArray>() {
        if target == BlobColumnTarget::Binary {
            // Existing binary/legacy columns keep their encoding: reject URI
            // assignments before any I/O instead of silently migrating them.
            for row in 0..array.len() {
                if union.is_null(row) {
                    rows.push(RowValue {
                        mode: BlobInputMode::Bytes,
                        data: None,
                        uri: None,
                        position: None,
                        size: None,
                    });
                    continue;
                }
                let type_id = union.type_id(row);
                let member = union
                    .fields()
                    .iter()
                    .find(|(id, _)| *id == type_id)
                    .map(|(_, field)| field.name().as_str());
                if member != Some("bytes") {
                    return Err(format!(
                        "blob input column '{column_name}' row {row} uses URI input, but the target column keeps its existing binary encoding; URI assignments are not supported without migrating the column to Blob v2"
                    ));
                }
                let value = read_union_row_value(union, row, column_name)?;
                if value.mode != BlobInputMode::Bytes {
                    return Err(format!(
                        "blob input column '{column_name}' row {row} uses URI input, but the target column keeps its existing binary encoding; URI assignments are not supported without migrating the column to Blob v2"
                    ));
                }
                rows.push(value);
            }
            let target_field = binary_target_field.ok_or_else(|| {
                format!("blob input column '{column_name}' has no binary target field")
            })?;
            return build_binary_target_array(column_name, &rows, target_field);
        }
        for row in 0..array.len() {
            if union.is_null(row) {
                rows.push(RowValue {
                    mode: BlobInputMode::Bytes,
                    data: None,
                    uri: None,
                    position: None,
                    size: None,
                });
                continue;
            }
            let value = read_union_row_value(union, row, column_name)?;
            if value.mode == BlobInputMode::Ingest {
                ingest_uris.push(row);
            }
            rows.push(value);
        }
        if !ingest_uris.is_empty() {
            let registry = context.registry.clone();
            runtime::block_on(async {
                for row in ingest_uris {
                    let uri = rows[row].uri.clone().expect("ingest row has uri");
                    let position = rows[row].position;
                    let size = rows[row].size;
                    let bytes = read_ingest_bytes(
                        &uri,
                        position,
                        size,
                        *remaining_budget,
                        &registry,
                        source_options,
                    )
                    .await?;
                    *remaining_budget -= bytes.len();
                    rows[row].data = Some(bytes);
                    rows[row].uri = None;
                    rows[row].position = None;
                    rows[row].size = None;
                    rows[row].mode = BlobInputMode::Bytes;
                }
                Ok::<(), String>(())
            })
            .map_err(|err| format!("failed to run ingest read: {err}"))??;
        }
    } else {
        for row in 0..array.len() {
            rows.push(read_binary_row_value(array, row, column_name)?);
        }
    }
    match target {
        BlobColumnTarget::BlobV2 => build_bytes_target_array(column_name, &rows, nullable),
        BlobColumnTarget::Binary => {
            let target_field = binary_target_field.ok_or_else(|| {
                format!("blob input column '{column_name}' has no binary target field")
            })?;
            build_binary_target_array(column_name, &rows, target_field)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{new_null_array, BinaryArray, RecordBatch, UnionArray};
    use arrow_schema::UnionMode;

    fn members() -> arrow_schema::UnionFields {
        arrow_schema::UnionFields::try_new(
            vec![0, 1, 2],
            vec![
                Arc::new(Field::new("bytes", DataType::Binary, true)),
                Arc::new(Field::new(
                    "reference",
                    DataType::Struct(Fields::from(vec![
                        Arc::new(Field::new("uri", DataType::Utf8, true)),
                        Arc::new(Field::new("position", DataType::UInt64, true)),
                        Arc::new(Field::new("size", DataType::UInt64, true)),
                    ])),
                    true,
                )),
                Arc::new(Field::new(
                    "ingest",
                    DataType::Struct(Fields::from(vec![
                        Arc::new(Field::new("uri", DataType::Utf8, true)),
                        Arc::new(Field::new("position", DataType::UInt64, true)),
                        Arc::new(Field::new("size", DataType::UInt64, true)),
                    ])),
                    true,
                )),
            ],
        )
        .unwrap()
    }

    fn schema_with_input() -> Schema {
        Schema::new(vec![Field::new(
            "payload",
            DataType::Union(members(), UnionMode::Sparse),
            true,
        )])
    }

    #[test]
    fn recognizes_the_blob_input_union() {
        let schema = schema_with_input();
        assert!(is_blob_input_type(schema.field(0).data_type()));
        assert!(!is_blob_input_type(&DataType::Utf8));
    }

    #[test]
    fn source_uri_uses_lance_uri_normalization() {
        assert_eq!(
            normalize_source_uri("file:///tmp/allowed/%2e%2e/secret").unwrap(),
            "/tmp/secret"
        );
        assert_eq!(
            normalize_source_uri("file:///tmp/allowed/public%20bytes").unwrap(),
            "/tmp/allowed/public bytes"
        );
        let error =
            normalize_source_uri("file:///tmp/allowed/nested%2F..%2F..%2Fsecret").unwrap_err();
        assert!(error.contains("illegal character sequence"), "{error}");
    }

    #[test]
    fn plans_blob_v2_for_new_fields() {
        let schema = schema_with_input();
        let conversions = plan_blob_conversions(&schema, None, true, &[1]).unwrap();
        assert_eq!(conversions.len(), 1);
        assert_eq!(conversions[0].target, BlobColumnTarget::BlobV2);
        let conversions = plan_blob_conversions(&schema, None, false, &[1]).unwrap();
        assert_eq!(conversions[0].target, BlobColumnTarget::Binary);
    }

    #[test]
    fn plans_binary_for_existing_plain_binary_fields() {
        let input = Schema::new(vec![Field::new("payload", DataType::Binary, true)]);
        let existing = Schema::new(vec![Field::new("payload", DataType::Binary, true)]);
        let conversions = plan_blob_conversions(&input, Some(&existing), true, &[1]).unwrap();
        assert!(conversions.is_empty());
    }

    #[test]
    fn rejects_nested_blob_input() {
        let nested = Schema::new(vec![Field::new(
            "payload",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Union(members(), UnionMode::Sparse),
                true,
            ))),
            true,
        )]);
        let error = plan_blob_conversions(&nested, None, true, &[1]).unwrap_err();
        assert!(error.contains("nested LANCE_BLOB_INPUT"), "{error}");
    }

    #[test]
    fn wraps_plain_bytes_as_blob_v2_logical_struct() {
        let bytes = BinaryArray::from(vec![Some(b"a\0\xff".as_slice()), None, Some(b"")]);
        let array = blob_v2_array_from_bytes("payload", &bytes).unwrap();
        let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_array.len(), 3);
        assert!(struct_array.is_null(1));
        let data = struct_array
            .column(0)
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(data.value(0), b"a\0\xff");
        assert!(data.is_null(1));
        assert_eq!(data.value(2).len(), 0);

        let nulls = new_null_array(&DataType::Null, 2);
        let array = blob_v2_array_from_bytes("payload", nulls.as_ref()).unwrap();
        let struct_array = array.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(struct_array.null_count(), 2);
    }

    #[test]
    fn converts_bytes_members_into_logical_struct() {
        let children = vec![
            Arc::new(BinaryArray::from(vec![b"abc".as_slice(), b""])) as ArrayRef,
            Arc::new(new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                2,
            )) as ArrayRef,
            Arc::new(new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                2,
            )) as ArrayRef,
        ];
        let type_ids = vec![0_i8, 0].into();
        let union = UnionArray::try_new(members(), type_ids, None, children).unwrap();
        let context = BlobWriteContext {
            registry: Arc::new(ObjectStoreRegistry::default()),
            ingest_batch_budget: 1024,
        };
        let converted = convert_blob_column(
            "payload",
            &union,
            BlobColumnTarget::BlobV2,
            true,
            &context,
            &HashMap::new(),
            &mut 1024,
            None,
        )
        .unwrap();
        let struct_array = converted.as_any().downcast_ref::<StructArray>().unwrap();
        let data = struct_array
            .column_by_name("data")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(data.value(0), b"abc");
        assert_eq!(data.value(1), b"");
    }

    #[test]
    fn converts_reference_members_into_logical_struct() {
        let uri = StringArray::from(vec![Some("file:///ref")]);
        let position = UInt64Array::from(vec![7]);
        let size = UInt64Array::from(vec![3]);
        let reference = StructArray::from(vec![
            (
                Arc::new(Field::new("uri", DataType::Utf8, false)),
                Arc::new(uri) as ArrayRef,
            ),
            (
                Arc::new(Field::new("position", DataType::UInt64, false)),
                Arc::new(position) as ArrayRef,
            ),
            (
                Arc::new(Field::new("size", DataType::UInt64, false)),
                Arc::new(size) as ArrayRef,
            ),
        ]);
        let children = vec![
            Arc::new(BinaryArray::from(vec![Some(b"".as_slice())])) as ArrayRef,
            Arc::new(reference) as ArrayRef,
            Arc::new(new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                1,
            )) as ArrayRef,
        ];
        let type_ids = vec![1_i8].into();
        let union = UnionArray::try_new(members(), type_ids, None, children).unwrap();
        let context = BlobWriteContext {
            registry: Arc::new(ObjectStoreRegistry::default()),
            ingest_batch_budget: 1024,
        };
        let converted = convert_blob_column(
            "payload",
            &union,
            BlobColumnTarget::BlobV2,
            true,
            &context,
            &HashMap::new(),
            &mut 1024,
            None,
        )
        .unwrap();
        let struct_array = converted.as_any().downcast_ref::<StructArray>().unwrap();
        let uri = struct_array
            .column_by_name("uri")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(uri.value(0), "file:///ref");
        let position = struct_array
            .column_by_name("position")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap();
        assert_eq!(position.value(0), 7);
    }

    #[test]
    fn converts_plain_binary_into_logical_struct() {
        let binary = BinaryArray::from(vec![Some(b"abc".as_slice()), None]);
        let context = BlobWriteContext {
            registry: Arc::new(ObjectStoreRegistry::default()),
            ingest_batch_budget: 1024,
        };
        let converted = convert_blob_column(
            "payload",
            &binary,
            BlobColumnTarget::BlobV2,
            true,
            &context,
            &HashMap::new(),
            &mut 1024,
            None,
        )
        .unwrap();
        let struct_array = converted.as_any().downcast_ref::<StructArray>().unwrap();
        assert!(struct_array.is_null(1));
        let data = struct_array
            .column_by_name("data")
            .unwrap()
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .unwrap();
        assert_eq!(data.value(0), b"abc");
    }

    #[test]
    fn rejects_uri_input_for_binary_target() {
        let children = vec![
            Arc::new(BinaryArray::from(vec![Some(b"".as_slice())])) as ArrayRef,
            Arc::new(new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                1,
            )) as ArrayRef,
            Arc::new(new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                1,
            )) as ArrayRef,
        ];
        let type_ids = vec![1_i8].into();
        let union = UnionArray::try_new(members(), type_ids, None, children).unwrap();
        let context = BlobWriteContext {
            registry: Arc::new(ObjectStoreRegistry::default()),
            ingest_batch_budget: 1024,
        };
        let error = convert_blob_column(
            "payload",
            &union,
            BlobColumnTarget::Binary,
            true,
            &context,
            &HashMap::new(),
            &mut 1024,
            None,
        )
        .unwrap_err();
        assert!(
            error.contains("keeps its existing binary encoding"),
            "{error}"
        );
    }

    #[test]
    fn truncated_remote_reads_are_rejected() {
        // object_store's `GetRange` clamps `end > len` for remote stores; the
        // write boundary must detect the shortened payload instead of silently
        // committing a truncated blob.
        assert!(validate_ingest_read_len(3, 10, "s3://bucket/key").is_err());
        assert!(validate_ingest_read_len(10, 10, "file:///obj").is_ok());
    }

    #[test]
    fn binary_target_keeps_schema_and_values_coherent() {
        let input = schema_with_input();
        let existing_field = Field::new("payload", DataType::LargeBinary, true).with_metadata(
            HashMap::from([(lance_arrow::BLOB_META_KEY.to_string(), "true".to_string())]),
        );
        let existing = Schema::new(vec![existing_field.clone()]);
        let conversions = plan_blob_conversions(&input, Some(&existing), true, &[1]).unwrap();
        assert_eq!(conversions.len(), 1);
        assert_eq!(conversions[0].target, BlobColumnTarget::Binary);
        let output = apply_blob_output_fields(&input, &conversions);
        assert_eq!(output.field(0).data_type(), &DataType::LargeBinary);
        assert_eq!(
            output
                .field(0)
                .metadata()
                .get(lance_arrow::BLOB_META_KEY)
                .map(String::as_str),
            Some("true")
        );

        let children = vec![
            Arc::new(BinaryArray::from(vec![
                Some(b"abc".as_slice()),
                Some(b"xyz".as_slice()),
            ])) as ArrayRef,
            new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                2,
            ),
            new_null_array(
                &DataType::Struct(Fields::from(vec![
                    Arc::new(Field::new("uri", DataType::Utf8, true)),
                    Arc::new(Field::new("position", DataType::UInt64, true)),
                    Arc::new(Field::new("size", DataType::UInt64, true)),
                ])),
                2,
            ),
        ];
        let union = UnionArray::try_new(members(), vec![0_i8, 0].into(), None, children).unwrap();
        let context = BlobWriteContext {
            registry: Arc::new(ObjectStoreRegistry::default()),
            ingest_batch_budget: 1024,
        };
        let converted = convert_blob_column(
            "payload",
            &union,
            BlobColumnTarget::Binary,
            true,
            &context,
            &HashMap::new(),
            &mut 1024,
            conversions[0].output_field.as_ref(),
        )
        .unwrap();
        let batch = RecordBatch::try_new(Arc::new(output), vec![converted]).unwrap();
        assert_eq!(batch.num_rows(), 2);
    }

    #[test]
    fn extension_binary_fields_keep_their_encoding() {
        let opaque = Field::new("b", DataType::LargeBinary, true).with_metadata(HashMap::from([
            (ARROW_EXT_NAME_KEY.to_string(), "arrow.opaque".to_string()),
            (
                "ARROW:extension:metadata".to_string(),
                r#"{"ARROW:extension:name":"arrow.bit"}"#.to_string(),
            ),
        ]));
        let json = Field::new("j", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            ARROW_EXT_NAME_KEY.to_string(),
            "arrow.json".to_string(),
        )]));
        let legacy = Field::new("l", DataType::LargeBinary, true).with_metadata(HashMap::from([(
            BLOB_META_KEY.to_string(),
            "true".to_string(),
        )]));
        let schema = Schema::new(vec![opaque, json, legacy]);
        assert!(plan_blob_conversions(&schema, None, true, &[1, 1, 1])
            .unwrap()
            .is_empty());
    }
}
