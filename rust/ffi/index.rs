use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance::index::vector::{IndexFileVersion, VectorIndexParams};
use lance::index::DatasetIndexExt;
use lance::Dataset;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::IndexType;
use lance_linalg::distance::DistanceType;
use serde::{Deserialize, Serialize};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::{SchemaHandle, StreamHandle};
use super::util::{
    canonicalize_lance_field_path, cstr_to_str, dataset_handle, to_c_string, FfiError, FfiResult,
};

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OptimizeIndexOptionsInput {
    mode: Option<String>,
    retrain: Option<bool>,
    num_indices_to_merge: Option<usize>,
}

#[derive(Debug, Serialize)]
struct OptimizeIndexMetricsOutput {
    index_name: String,
    mode: String,
    retrain: bool,
    num_indices_to_merge: Option<usize>,
}

#[no_mangle]
pub unsafe extern "C" fn lance_get_index_list_schema(dataset: *mut c_void) -> *mut c_void {
    match get_index_list_schema_inner(dataset) {
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

fn get_index_list_schema_inner(dataset: *mut c_void) -> FfiResult<SchemaHandle> {
    let _ = unsafe { dataset_handle(dataset)? };
    Ok(index_list_schema())
}

#[no_mangle]
pub unsafe extern "C" fn lance_create_index_list_stream(dataset: *mut c_void) -> *mut c_void {
    match create_index_list_stream_inner(dataset) {
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

fn create_index_list_stream_inner(dataset: *mut c_void) -> FfiResult<StreamHandle> {
    let handle = unsafe { dataset_handle(dataset)? };

    let dataset = handle.dataset.as_ref().clone();
    let descs = match runtime::block_on(async { dataset.describe_indices(None).await }) {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDescribeIndices,
                format!("dataset describe_indices: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let mut index_names = Vec::with_capacity(descs.len());
    let mut index_types = Vec::with_capacity(descs.len());
    let mut fields = Vec::with_capacity(descs.len());
    let mut rows_indexed = Vec::with_capacity(descs.len());
    let mut details: Vec<Option<String>> = Vec::with_capacity(descs.len());

    let mut frag_rows = std::collections::HashMap::<u32, u64>::new();
    let mut dataset_frag_ids = roaring::RoaringBitmap::new();
    for frag in dataset.get_fragments() {
        dataset_frag_ids.insert(frag.id() as u32);
        if let Some(rows) = frag.metadata().num_rows() {
            if let Ok(rows_u64) = u64::try_from(rows) {
                frag_rows.insert(frag.id() as u32, rows_u64);
            }
        }
    }
    let dataset_total_rows = match runtime::block_on(async { dataset.count_rows(None).await }) {
        Ok(Ok(v)) => u64::try_from(v).unwrap_or(u64::MAX),
        _ => u64::MAX,
    };

    for d in descs {
        index_names.push(d.name().to_string());
        index_types.push(d.index_type().to_string());

        let mut cols = Vec::new();
        for &field_id in d.field_ids() {
            cols.push(
                dataset
                    .schema()
                    .field_path(field_id as i32)
                    .unwrap_or_else(|_| format!("_field_id_{field_id}")),
            );
        }
        fields.push(cols.join(","));

        rows_indexed.push(estimate_rows_indexed(
            dataset_total_rows,
            &dataset_frag_ids,
            &frag_rows,
            d.as_ref(),
        ));
        match d.details() {
            Ok(v) => details.push(Some(v)),
            Err(_) => details.push(None),
        }
    }

    let schema = index_list_schema();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(index_names)),
            Arc::new(StringArray::from(index_types)),
            Arc::new(StringArray::from(fields)),
            Arc::new(UInt64Array::from(rows_indexed)),
            Arc::new(StringArray::from(details)),
        ],
    )
    .map_err(|err| {
        FfiError::new(
            ErrorCode::IndexStreamCreate,
            format!("index list batch: {err}"),
        )
    })?;

    Ok(StreamHandle::Batches(vec![batch].into_iter()))
}

/// Returns an array of column names that have scalar indices.
/// The caller must free the returned array with `lance_free_scalar_indexed_columns`.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_scalar_indexed_columns(
    dataset: *mut c_void,
    out_len: *mut usize,
) -> *mut *mut c_char {
    match list_scalar_indexed_columns_inner(dataset) {
        Ok(cols) => {
            clear_last_error();
            unsafe { *out_len = cols.len() };
            if cols.is_empty() {
                return std::ptr::null_mut();
            }
            let ptrs: Vec<*mut c_char> = cols
                .into_iter()
                .map(|s| to_c_string(s).into_raw())
                .collect();
            let mut boxed = ptrs.into_boxed_slice();
            let ptr = boxed.as_mut_ptr();
            std::mem::forget(boxed);
            ptr
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            unsafe { *out_len = 0 };
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_free_scalar_indexed_columns(ptr: *mut *mut c_char, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let slice = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len));
        for &p in slice.iter() {
            if !p.is_null() {
                drop(std::ffi::CString::from_raw(p));
            }
        }
    }
}

fn list_scalar_indexed_columns_inner(dataset: *mut c_void) -> FfiResult<Vec<String>> {
    let handle = unsafe { dataset_handle(dataset)? };
    let dataset = handle.dataset.as_ref().clone();

    let descs = match runtime::block_on(async { dataset.describe_indices(None).await }) {
        Ok(Ok(v)) => v,
        Ok(Err(err)) => {
            return Err(FfiError::new(
                ErrorCode::DatasetDescribeIndices,
                format!("dataset describe_indices: {err}"),
            ))
        }
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    let schema = dataset.schema();
    let mut cols = Vec::new();
    for d in descs {
        let idx_type = d.index_type().to_string();
        let normalized = normalize_index_type(&idx_type);
        if is_vector_index_type(&normalized) {
            continue;
        }
        for &field_id in d.field_ids() {
            if let Ok(path) = schema.field_path(field_id as i32) {
                cols.push(path);
            }
        }
    }
    Ok(cols)
}

fn estimate_rows_indexed(
    dataset_total_rows: u64,
    dataset_frag_ids: &roaring::RoaringBitmap,
    frag_rows: &std::collections::HashMap<u32, u64>,
    desc: &dyn lance_index::IndexDescription,
) -> u64 {
    let mut bitmap = roaring::RoaringBitmap::new();
    for meta in desc.metadata() {
        if let Some(b) = &meta.fragment_bitmap {
            bitmap |= b;
        }
    }
    if bitmap.is_empty() {
        // Prefer an explicit 0 for indices without fragment coverage (e.g. untrained scalar).
        // Fall back to the description-provided value if we don't have any bitmap information.
        if desc
            .metadata()
            .iter()
            .all(|m| m.fragment_bitmap.as_ref().is_some_and(|b| b.is_empty()))
        {
            return 0;
        }
        return desc.rows_indexed();
    }

    // If we can't get per-fragment row counts, but the index covers all fragments, then we can
    // still provide a reasonable estimate.
    if dataset_total_rows != u64::MAX && bitmap == *dataset_frag_ids {
        return dataset_total_rows;
    }

    let mut total = 0u64;
    for frag_id in bitmap.iter() {
        if let Some(rows) = frag_rows.get(&frag_id) {
            total = total.saturating_add(*rows);
        }
    }
    if total > 0 {
        total
    } else {
        // If fragment row counts are unknown, fall back to whatever the plugin reports.
        // This can be approximate but should at least be non-zero for trained indices.
        desc.rows_indexed()
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_create_index(
    dataset: *mut c_void,
    index_name: *const c_char,
    columns: *const *const c_char,
    columns_len: usize,
    index_type: *const c_char,
    params_json: *const c_char,
    replace: u8,
    train: u8,
) -> i32 {
    match dataset_create_index_inner(
        dataset,
        index_name,
        columns,
        columns_len,
        index_type,
        params_json,
        replace,
        train,
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

#[allow(clippy::too_many_arguments)]
fn dataset_create_index_inner(
    dataset: *mut c_void,
    index_name: *const c_char,
    columns: *const *const c_char,
    columns_len: usize,
    index_type: *const c_char,
    params_json: *const c_char,
    replace: u8,
    train: u8,
) -> FfiResult<()> {
    let handle = unsafe { dataset_handle(dataset)? };

    if columns_len == 0 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "columns must be non-empty",
        ));
    }
    if columns.is_null() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "columns is null with non-zero length",
        ));
    }

    let index_type = unsafe { cstr_to_str(index_type, "index_type")? };
    let index_type_norm = normalize_index_type(index_type);
    let params_json = if params_json.is_null() {
        None
    } else {
        let s = unsafe { cstr_to_str(params_json, "params_json")? };
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    };

    let columns = unsafe { super::util::optional_cstr_array(columns, columns_len, "columns")? };
    if columns.len() != 1 {
        return Err(FfiError::new(
            ErrorCode::DatasetCreateIndex,
            "only single-column indices are supported",
        ));
    }

    let index_name = if index_name.is_null() {
        None
    } else {
        let s = unsafe { cstr_to_str(index_name, "index_name")? };
        if s.is_empty() {
            None
        } else {
            Some(s.to_string())
        }
    };

    let canonical_columns = canonicalize_index_columns(handle.dataset.schema(), &columns)?;

    let (lance_index_type, params) = build_index_params(&index_type_norm, params_json.as_deref())?;

    let mut ds: Dataset = handle.dataset.as_ref().clone();
    let replace = replace != 0;
    let train = train != 0;

    run_with_large_stack(move || {
        match runtime::block_on(async {
            let cols: [&str; 1] = [canonical_columns[0].as_str()];
            let mut builder = ds.create_index_builder(&cols, lance_index_type, params.as_ref());
            if let Some(name) = index_name {
                builder = builder.name(name);
            }
            builder.replace(replace).train(train).await
        }) {
            Ok(Ok(_)) => Ok(()),
            Ok(Err(err)) => Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("dataset create_index: {err}"),
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    })?
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_drop_index(
    dataset: *mut c_void,
    index_name: *const c_char,
) -> i32 {
    match dataset_drop_index_inner(dataset, index_name) {
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

fn dataset_drop_index_inner(dataset: *mut c_void, index_name: *const c_char) -> FfiResult<()> {
    let handle = unsafe { dataset_handle(dataset)? };
    let index_name = unsafe { cstr_to_str(index_name, "index_name")? };
    if index_name.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "index_name must be non-empty",
        ));
    }

    let mut ds: Dataset = handle.dataset.as_ref().clone();
    match runtime::block_on(async { ds.drop_index(index_name).await }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(FfiError::new(
            ErrorCode::DatasetDropIndex,
            format!("dataset drop_index: {err}"),
        )),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

#[no_mangle]
pub unsafe extern "C" fn lance_dataset_optimize_index(
    dataset: *mut c_void,
    index_name: *const c_char,
    retrain: u8,
) -> i32 {
    match dataset_optimize_index_with_options_inner(
        dataset,
        index_name,
        ptr::null(),
        ptr::null_mut(),
        retrain != 0,
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
pub unsafe extern "C" fn lance_dataset_optimize_index_with_options(
    dataset: *mut c_void,
    index_name: *const c_char,
    options_json: *const c_char,
    out_metrics_json: *mut *const c_char,
) -> i32 {
    if !out_metrics_json.is_null() {
        unsafe {
            ptr::write_unaligned(out_metrics_json, ptr::null());
        }
    }
    match dataset_optimize_index_with_options_inner(
        dataset,
        index_name,
        options_json,
        out_metrics_json,
        false,
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

fn parse_optimize_index_options_json(
    options_json: *const c_char,
    legacy_retrain: bool,
) -> FfiResult<(OptimizeOptions, String, bool, Option<usize>)> {
    let input = if options_json.is_null() {
        OptimizeIndexOptionsInput::default()
    } else {
        let text = unsafe { CStr::from_ptr(options_json) }
            .to_str()
            .map_err(|err| FfiError::new(ErrorCode::Utf8, format!("options_json utf8: {err}")))?;
        if text.trim().is_empty() {
            OptimizeIndexOptionsInput::default()
        } else {
            serde_json::from_str(text).map_err(|err| {
                FfiError::new(
                    ErrorCode::InvalidArgument,
                    format!("optimize_index options_json parse: {err}"),
                )
            })?
        }
    };

    let mode = if let Some(mode) = input.mode.as_ref() {
        mode.trim().to_ascii_lowercase()
    } else if input.retrain.unwrap_or(false) || legacy_retrain {
        String::from("retrain")
    } else {
        String::from("append")
    };

    let (options, retrain, num_indices_to_merge) = match mode.as_str() {
        "append" => {
            if input.num_indices_to_merge.is_some() {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    "num_indices_to_merge is only valid for mode='merge'",
                ));
            }
            (OptimizeOptions::append(), false, Some(0))
        }
        "merge" => {
            let num = input.num_indices_to_merge.unwrap_or(1);
            if num == 0 {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    "num_indices_to_merge must be > 0 for mode='merge'",
                ));
            }
            (OptimizeOptions::merge(num), false, Some(num))
        }
        "retrain" => {
            if input.num_indices_to_merge.is_some() {
                return Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    "num_indices_to_merge is invalid for mode='retrain'",
                ));
            }
            (OptimizeOptions::retrain(), true, None)
        }
        other => {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!("unsupported optimize mode: {other}"),
            ))
        }
    };

    Ok((options, mode, retrain, num_indices_to_merge))
}

fn dataset_optimize_index_with_options_inner(
    dataset: *mut c_void,
    index_name: *const c_char,
    options_json: *const c_char,
    out_metrics_json: *mut *const c_char,
    legacy_retrain: bool,
) -> FfiResult<()> {
    let handle = unsafe { dataset_handle(dataset)? };
    let index_name = unsafe { cstr_to_str(index_name, "index_name")? };
    if index_name.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "index_name must be non-empty",
        ));
    }
    let index_name_owned = index_name.to_string();

    let (mut options, mode, retrain, num_indices_to_merge) =
        parse_optimize_index_options_json(options_json, legacy_retrain)?;
    options = options.index_names(vec![index_name_owned.clone()]);

    let mut ds: Dataset = handle.dataset.as_ref().clone();
    let metrics = run_with_large_stack(move || {
        match runtime::block_on(async { ds.optimize_indices(&options).await }) {
            Ok(Ok(())) => Ok(OptimizeIndexMetricsOutput {
                index_name: index_name_owned,
                mode,
                retrain,
                num_indices_to_merge,
            }),
            Ok(Err(err)) => Err(FfiError::new(
                ErrorCode::DatasetOptimizeIndices,
                format!("dataset optimize_indices: {err}"),
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    })??;

    if !out_metrics_json.is_null() {
        let payload = serde_json::to_string(&metrics).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetOptimizeIndices,
                format!("optimize_index metrics_json serialize: {err}"),
            )
        })?;
        unsafe {
            ptr::write_unaligned(
                out_metrics_json,
                to_c_string(payload).into_raw() as *const c_char,
            );
        }
    }
    Ok(())
}

fn normalize_index_type(index_type: &str) -> String {
    index_type
        .trim()
        .to_ascii_uppercase()
        .replace(['-', ' '], "_")
}

fn canonicalize_index_columns(
    schema: &lance_core::datatypes::Schema,
    columns: &[String],
) -> FfiResult<Vec<String>> {
    columns
        .iter()
        .map(|column| canonicalize_lance_field_path(schema, column, "index column"))
        .collect()
}

fn build_index_params(
    index_type: &str,
    params_json: Option<&str>,
) -> FfiResult<(IndexType, Box<dyn lance_index::IndexParams + Send + Sync>)> {
    if is_vector_index_type(index_type) {
        let params = build_vector_params(index_type, params_json)?;
        return Ok((IndexType::Vector, Box::new(params)));
    }

    let scalar_type = match index_type {
        "BTREE" => IndexType::BTree,
        "BITMAP" => IndexType::Bitmap,
        "ZONEMAP" => IndexType::ZoneMap,
        "BLOOMFILTER" | "BLOOM_FILTER" => IndexType::BloomFilter,
        "INVERTED" => IndexType::Inverted,
        "NGRAM" | "N_GRAM" => IndexType::NGram,
        "LABELLIST" | "LABEL_LIST" => IndexType::LabelList,
        other => {
            return Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("unsupported index_type: {other}"),
            ))
        }
    };

    let mut params = ScalarIndexParams::for_builtin(scalar_type.try_into().map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetCreateIndex,
            format!("scalar index type: {err}"),
        )
    })?);
    if let Some(json) = params_json {
        params.params = Some(json.to_string());
    } else if scalar_type == IndexType::Inverted {
        // lance-index's InvertedIndexParams requires at least `base_tokenizer` and `language`.
        // Provide a stable default that is also eligible for query acceleration.
        params.params =
            Some(r#"{"base_tokenizer":"simple","language":"English","stem":false}"#.to_string());
    }
    Ok((scalar_type, Box::new(params)))
}

fn is_vector_index_type(index_type: &str) -> bool {
    matches!(
        index_type,
        "IVF_FLAT"
            | "IVF_PQ"
            | "IVF_SQ"
            | "IVF_RQ"
            | "IVF_HNSW_FLAT"
            | "IVF_HNSW_PQ"
            | "IVF_HNSW_SQ"
    )
}

/// Options a vector index reads from `params_json`. An unknown name or a value
/// of the wrong type is an error instead of a silent default, as for the
/// index optimize options above.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct VectorIndexOptions {
    metric_type: Option<String>,
    version: Option<String>,
    num_partitions: Option<usize>,
    num_bits: Option<u8>,
    num_sub_vectors: Option<usize>,
    max_iterations: Option<usize>,
    sample_rate: Option<usize>,
    hnsw_m: Option<usize>,
    hnsw_ef_construction: Option<usize>,
    hnsw_max_level: Option<u16>,
    /// `null` disables prefetching; leaving the option out keeps Lance's default.
    #[serde(deserialize_with = "deserialize_present")]
    hnsw_prefetch_distance: Option<Option<usize>>,
}

/// Wraps a present value, `null` included, in `Some`; a missing field keeps
/// the container default (`None`).
fn deserialize_present<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

impl VectorIndexOptions {
    /// The first option set here that `index_type` does not read. The
    /// destructuring names every field, so a new option does not compile until
    /// it is listed here.
    fn unused_by(&self, index_type: &str) -> Option<&'static str> {
        let Self {
            metric_type: _,
            version: _,
            num_partitions: _,
            num_bits,
            num_sub_vectors,
            max_iterations,
            sample_rate,
            hnsw_m,
            hnsw_ef_construction,
            hnsw_max_level,
            hnsw_prefetch_distance,
        } = self;
        let pq = matches!(index_type, "IVF_PQ" | "IVF_HNSW_PQ");
        let sq = matches!(index_type, "IVF_SQ" | "IVF_HNSW_SQ");
        let hnsw = index_type.starts_with("IVF_HNSW_");
        let reads_num_bits = pq || sq || index_type == "IVF_RQ";
        [
            ("num_bits", num_bits.is_some(), reads_num_bits),
            ("num_sub_vectors", num_sub_vectors.is_some(), pq),
            ("max_iterations", max_iterations.is_some(), pq),
            ("sample_rate", sample_rate.is_some(), sq),
            ("hnsw_m", hnsw_m.is_some(), hnsw),
            ("hnsw_ef_construction", hnsw_ef_construction.is_some(), hnsw),
            ("hnsw_max_level", hnsw_max_level.is_some(), hnsw),
            (
                "hnsw_prefetch_distance",
                hnsw_prefetch_distance.is_some(),
                hnsw,
            ),
        ]
        .into_iter()
        .find(|(_, set, read)| *set && !*read)
        .map(|(name, _, _)| name)
    }

    fn hnsw(&self) -> HnswBuildParams {
        let mut hnsw = HnswBuildParams::default();
        if let Some(m) = self.hnsw_m {
            hnsw.m = m;
        }
        if let Some(ef_construction) = self.hnsw_ef_construction {
            hnsw.ef_construction = ef_construction;
        }
        if let Some(max_level) = self.hnsw_max_level {
            hnsw.max_level = max_level;
        }
        if let Some(prefetch_distance) = self.hnsw_prefetch_distance {
            hnsw.prefetch_distance = prefetch_distance;
        }
        hnsw
    }

    fn sq(&self) -> lance_index::vector::sq::builder::SQBuildParams {
        lance_index::vector::sq::builder::SQBuildParams {
            num_bits: self.num_bits.unwrap_or(8).into(),
            sample_rate: self.sample_rate.unwrap_or(256),
        }
    }
}

fn build_vector_params(
    index_type: &str,
    params_json: Option<&str>,
) -> FfiResult<VectorIndexParams> {
    let options: VectorIndexOptions = match params_json {
        // serde also fills a struct from a JSON array, by field position, so
        // only an object names its options.
        Some(json) if !json.trim_start().starts_with('{') => {
            return Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("{index_type} options must be a JSON object"),
            ))
        }
        Some(json) => serde_json::from_str(json).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("{index_type} options: {err}"),
            )
        })?,
        None => VectorIndexOptions::default(),
    };
    if let Some(name) = options.unused_by(index_type) {
        return Err(FfiError::new(
            ErrorCode::DatasetCreateIndex,
            format!("{index_type} does not accept option '{name}'"),
        ));
    }

    let metric = options.metric_type.as_deref().unwrap_or("l2");
    let metric_type = DistanceType::try_from(metric).map_err(|err| {
        FfiError::new(ErrorCode::DatasetCreateIndex, format!("metric_type: {err}"))
    })?;
    let version = IndexFileVersion::try_from(options.version.as_deref().unwrap_or("v3"))
        .map_err(|err| FfiError::new(ErrorCode::DatasetCreateIndex, format!("version: {err}")))?;
    // Lance reads the file version only for IVF_PQ and builds V3 for the rest.
    if matches!(version, IndexFileVersion::Legacy) && index_type != "IVF_PQ" {
        return Err(FfiError::new(
            ErrorCode::DatasetCreateIndex,
            format!("{index_type} does not accept version 'legacy'"),
        ));
    }
    let num_partitions = options.num_partitions.unwrap_or(256);
    let ivf = || lance_index::vector::ivf::IvfBuildParams::new(num_partitions);

    let mut out = match index_type {
        "IVF_FLAT" => VectorIndexParams::ivf_flat(num_partitions, metric_type),
        "IVF_PQ" => VectorIndexParams::ivf_pq(
            num_partitions,
            options.num_bits.unwrap_or(8),
            options.num_sub_vectors.unwrap_or(16),
            metric_type,
            options.max_iterations.unwrap_or(50),
        ),
        "IVF_RQ" => {
            VectorIndexParams::ivf_rq(num_partitions, options.num_bits.unwrap_or(8), metric_type)
        }
        "IVF_SQ" => VectorIndexParams::with_ivf_sq_params(metric_type, ivf(), options.sq()),
        "IVF_HNSW_FLAT" => VectorIndexParams::ivf_hnsw(metric_type, ivf(), options.hnsw()),
        "IVF_HNSW_PQ" => {
            let mut pq = PQBuildParams::default();
            if let Some(num_bits) = options.num_bits {
                pq.num_bits = num_bits.into();
            }
            if let Some(num_sub_vectors) = options.num_sub_vectors {
                pq.num_sub_vectors = num_sub_vectors;
            }
            if let Some(max_iterations) = options.max_iterations {
                pq.max_iters = max_iterations;
            }
            VectorIndexParams::with_ivf_hnsw_pq_params(metric_type, ivf(), options.hnsw(), pq)
        }
        "IVF_HNSW_SQ" => VectorIndexParams::with_ivf_hnsw_sq_params(
            metric_type,
            ivf(),
            options.hnsw(),
            options.sq(),
        ),
        other => {
            return Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("unsupported vector index_type: {other}"),
            ))
        }
    };
    out.version(version);
    Ok(out)
}

fn run_with_large_stack<T>(f: impl FnOnce() -> T + Send + 'static) -> FfiResult<T>
where
    T: Send + 'static,
{
    const STACK_SIZE: usize = 32 * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .name("lance_duckdb_index".to_string())
        .stack_size(STACK_SIZE)
        .spawn(f)
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("spawn: {err}")))?;
    handle
        .join()
        .map_err(|_| FfiError::new(ErrorCode::Runtime, "thread panicked"))
}

fn index_list_schema() -> SchemaHandle {
    let schema = Schema::new(vec![
        Field::new("index_name", DataType::Utf8, false),
        Field::new("index_type", DataType::Utf8, false),
        Field::new("fields", DataType::Utf8, false),
        Field::new("rows_indexed", DataType::UInt64, false),
        Field::new("details", DataType::Utf8, true),
    ]);
    Arc::new(schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType, Field as ArrowField, Fields, Schema as ArrowSchema};

    fn nested_schema() -> lance_core::datatypes::Schema {
        let arrow = ArrowSchema::new(vec![
            ArrowField::new(
                "left_struct",
                DataType::Struct(Fields::from(vec![ArrowField::new(
                    "value",
                    DataType::Int32,
                    true,
                )])),
                true,
            ),
            ArrowField::new(
                "right_struct",
                DataType::Struct(Fields::from(vec![ArrowField::new(
                    "value",
                    DataType::Int32,
                    true,
                )])),
                true,
            ),
            ArrowField::new(
                "dotted",
                DataType::Struct(Fields::from(vec![ArrowField::new(
                    "literal.dot",
                    DataType::Int32,
                    true,
                )])),
                true,
            ),
        ]);
        let mut schema = lance_core::datatypes::Schema::try_from(&arrow).unwrap();
        schema.set_field_id(None);
        schema
    }

    #[test]
    fn canonicalizes_nested_index_columns() {
        let schema = nested_schema();

        assert_eq!(
            canonicalize_lance_field_path(&schema, "left_struct.value", "index column").unwrap(),
            "left_struct.value"
        );
        assert_eq!(
            canonicalize_lance_field_path(&schema, "right_struct.value", "index column").unwrap(),
            "right_struct.value"
        );
        assert_eq!(
            canonicalize_lance_field_path(&schema, "dotted.`literal.dot`", "index column").unwrap(),
            "dotted.`literal.dot`"
        );
    }

    #[test]
    fn canonicalizes_case_insensitive_index_columns() {
        let schema = nested_schema();

        assert_eq!(
            canonicalize_lance_field_path(&schema, "LEFT_STRUCT.VALUE", "index column").unwrap(),
            "left_struct.value"
        );
        assert_eq!(
            canonicalize_lance_field_path(&schema, "DOTTED.`LITERAL.DOT`", "index column").unwrap(),
            "dotted.`literal.dot`"
        );
    }

    #[test]
    fn rejects_ambiguous_leaf_without_parent_path() {
        let schema = nested_schema();
        let err = canonicalize_lance_field_path(&schema, "value", "index column").unwrap_err();

        assert!(err.message.contains("index column not found"));
    }

    fn hnsw_values(hnsw: &HnswBuildParams) -> String {
        format!(
            "hnsw {} {} {} {:?}",
            hnsw.m, hnsw.ef_construction, hnsw.max_level, hnsw.prefetch_distance
        )
    }

    /// The option values each stage of `params` carries.
    fn stage_values(params: &VectorIndexParams) -> Vec<String> {
        use lance::index::vector::StageParams;
        params
            .stages
            .iter()
            .map(|stage| match stage {
                StageParams::Ivf(ivf) => format!("ivf {:?}", ivf.num_partitions),
                StageParams::PQ(pq) => {
                    format!("pq {} {} {}", pq.num_bits, pq.num_sub_vectors, pq.max_iters)
                }
                StageParams::SQ(sq) => format!("sq {} {}", sq.num_bits, sq.sample_rate),
                StageParams::RQ(rq) => format!("rq {}", rq.num_bits),
                StageParams::Hnsw(hnsw) => hnsw_values(hnsw),
            })
            .collect()
    }

    #[test]
    fn vector_index_options_are_applied() {
        let params = build_vector_params(
            "IVF_FLAT",
            Some(r#"{"metric_type":"cosine","num_partitions":2,"version":"v3"}"#),
        )
        .unwrap();
        assert_eq!(params.metric_type, DistanceType::Cosine);
        assert_eq!(stage_values(&params), ["ivf Some(2)"]);

        let params = build_vector_params("IVF_PQ", Some(r#"{"version":"legacy"}"#)).unwrap();
        assert!(matches!(params.version, IndexFileVersion::Legacy));

        // Every option a type reads reaches its stage, and an HNSW option left
        // out keeps Lance's default. `null` disables HNSW prefetching.
        let hnsw_default = hnsw_values(&HnswBuildParams::default());
        let hnsw_m8 = hnsw_values(&HnswBuildParams {
            m: 8,
            ..Default::default()
        });
        let hnsw_prefetch4 = hnsw_values(&HnswBuildParams {
            prefetch_distance: Some(4),
            ..Default::default()
        });
        for (index_type, json, expected) in [
            (
                "IVF_PQ",
                r#"{"num_partitions":2,"num_bits":4,"num_sub_vectors":4,"max_iterations":10}"#,
                &["ivf Some(2)", "pq 4 4 10"][..],
            ),
            ("IVF_RQ", r#"{"num_bits":1}"#, &["ivf Some(256)", "rq 1"]),
            (
                "IVF_SQ",
                r#"{"num_bits":4,"sample_rate":128}"#,
                &["ivf Some(256)", "sq 4 128"],
            ),
            (
                "IVF_HNSW_FLAT",
                "{}",
                &["ivf Some(256)", hnsw_default.as_str()],
            ),
            (
                "IVF_HNSW_FLAT",
                r#"{"hnsw_m":8,"hnsw_ef_construction":64,"hnsw_max_level":5,"hnsw_prefetch_distance":null}"#,
                &["ivf Some(256)", "hnsw 8 64 5 None"],
            ),
            (
                "IVF_HNSW_PQ",
                r#"{"hnsw_m":8,"num_bits":4,"num_sub_vectors":4,"max_iterations":10}"#,
                &["ivf Some(256)", hnsw_m8.as_str(), "pq 4 4 10"],
            ),
            (
                "IVF_HNSW_SQ",
                r#"{"hnsw_prefetch_distance":4,"num_bits":4,"sample_rate":128}"#,
                &["ivf Some(256)", hnsw_prefetch4.as_str(), "sq 4 128"],
            ),
        ] {
            let params = build_vector_params(index_type, Some(json))
                .unwrap_or_else(|err| panic!("{index_type} {json}: {}", err.message));
            assert_eq!(stage_values(&params), expected, "{index_type} {json}");
        }
    }

    #[test]
    fn vector_index_options_reject_values_they_would_ignore() {
        for (index_type, json, expected) in [
            (
                "IVF_FLAT",
                r#"{"version":"legacy"}"#,
                "IVF_FLAT does not accept version 'legacy'",
            ),
            (
                "IVF_FLAT",
                r#"{"metric":"cosine"}"#,
                "unknown field `metric`",
            ),
            (
                "IVF_FLAT",
                r#"{"num_partition":2}"#,
                "unknown field `num_partition`",
            ),
            (
                "IVF_FLAT",
                r#"{"num_sub_vectors":4}"#,
                "IVF_FLAT does not accept option 'num_sub_vectors'",
            ),
            (
                "IVF_HNSW_FLAT",
                r#"{"sample_rate":4}"#,
                "IVF_HNSW_FLAT does not accept option 'sample_rate'",
            ),
            (
                "IVF_FLAT",
                r#"{"num_partitions":"2"}"#,
                "invalid type: string \"2\", expected usize",
            ),
            (
                "IVF_FLAT",
                r#"{"num_partitions":-1}"#,
                "invalid value: integer `-1`, expected usize",
            ),
            (
                "IVF_FLAT",
                r#"{"metric_type":1}"#,
                "invalid type: integer `1`, expected a string",
            ),
            (
                "IVF_PQ",
                r#"{"num_bits":300}"#,
                "invalid value: integer `300`, expected u8",
            ),
            ("IVF_FLAT", "5", "IVF_FLAT options must be a JSON object"),
            ("IVF_FLAT", "[]", "IVF_FLAT options must be a JSON object"),
            (
                "IVF_FLAT",
                r#"["cosine", "v3", 1]"#,
                "IVF_FLAT options must be a JSON object",
            ),
            (
                "IVF_FLAT",
                r#"{"metric_type":"cosine","metric_type":"l2"}"#,
                "duplicate field `metric_type`",
            ),
        ] {
            let err = build_vector_params(index_type, Some(json)).unwrap_err();
            assert!(
                err.message.contains(expected),
                "{index_type} {json}: {}",
                err.message
            );
        }
    }
}
