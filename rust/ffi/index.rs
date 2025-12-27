use std::ffi::{c_char, c_void};
use std::ptr;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use lance::index::vector::{IndexFileVersion, VectorIndexParams};
use lance::Dataset;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::ScalarIndexParams;
use lance_index::vector::hnsw::builder::HnswBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::DistanceType;

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::types::{SchemaHandle, StreamHandle};
use super::util::{cstr_to_str, dataset_handle, FfiError, FfiResult};

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
            if let Some(field) = dataset.schema().field_by_id(field_id as i32) {
                cols.push(field.name.clone());
            } else {
                cols.push(format!("_field_id_{field_id}"));
            }
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
    .map_err(|err| FfiError::new(ErrorCode::IndexStreamCreate, format!("index list batch: {err}")))?;

    Ok(StreamHandle::Batches(vec![batch].into_iter()))
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
        if s.is_empty() { None } else { Some(s.to_string()) }
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
        if s.is_empty() { None } else { Some(s.to_string()) }
    };

    let (lance_index_type, params) =
        build_index_params(&index_type_norm, params_json.as_deref())?;

    let mut ds: Dataset = handle.dataset.as_ref().clone();
    let replace = replace != 0;
    let train = train != 0;

    run_with_large_stack(move || {
        match runtime::block_on(async {
            let cols: [&str; 1] = [columns[0].as_str()];
            let mut builder = ds.create_index_builder(&cols, lance_index_type, params.as_ref());
            if let Some(name) = index_name {
                builder = builder.name(name);
            }
            builder.replace(replace).train(train).await
        }) {
            Ok(Ok(())) => Ok(()),
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
    match dataset_optimize_index_inner(dataset, index_name, retrain) {
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

fn dataset_optimize_index_inner(
    dataset: *mut c_void,
    index_name: *const c_char,
    retrain: u8,
) -> FfiResult<()> {
    let handle = unsafe { dataset_handle(dataset)? };
    let index_name = unsafe { cstr_to_str(index_name, "index_name")? };
    if index_name.is_empty() {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "index_name must be non-empty",
        ));
    }

    let mut options = if retrain != 0 {
        OptimizeOptions::retrain()
    } else {
        OptimizeOptions::append()
    };
    options = options.index_names(vec![index_name.to_string()]);

    let mut ds: Dataset = handle.dataset.as_ref().clone();
    run_with_large_stack(move || {
        match runtime::block_on(async { ds.optimize_indices(&options).await }) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => Err(FfiError::new(
                ErrorCode::DatasetOptimizeIndices,
                format!("dataset optimize_indices: {err}"),
            )),
            Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
        }
    })?
}

fn normalize_index_type(index_type: &str) -> String {
    index_type
        .trim()
        .to_ascii_uppercase()
        .replace('-', "_")
        .replace(' ', "_")
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
        FfiError::new(ErrorCode::DatasetCreateIndex, format!("scalar index type: {err}"))
    })?);
    if let Some(json) = params_json {
        params.params = Some(json.to_string());
    } else if scalar_type == IndexType::Inverted {
        // lance-index's InvertedIndexParams requires at least `base_tokenizer` and `language`.
        // Provide a stable default that is also eligible for query acceleration.
        params.params = Some(
            r#"{"base_tokenizer":"simple","language":"English","stem":false}"#.to_string(),
        );
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

fn build_vector_params(index_type: &str, params_json: Option<&str>) -> FfiResult<VectorIndexParams> {
    let mut params = serde_json::Map::<String, serde_json::Value>::new();
    if let Some(json) = params_json {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|err| {
            FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("params_json is not valid json: {err}"),
            )
        })?;
        if let serde_json::Value::Object(m) = v {
            params = m;
        } else {
            return Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                "params_json must be a JSON object",
            ));
        }
    }

    let metric = params
        .get("metric_type")
        .and_then(|v| v.as_str())
        .unwrap_or("l2");
    let metric_type = DistanceType::try_from(metric).map_err(|err| {
        FfiError::new(
            ErrorCode::DatasetCreateIndex,
            format!("metric_type: {err}"),
        )
    })?;

    let version = params
        .get("version")
        .and_then(|v| v.as_str())
        .unwrap_or("v3");
    let version = IndexFileVersion::try_from(version).map_err(|err| {
        FfiError::new(ErrorCode::DatasetCreateIndex, format!("version: {err}"))
    })?;

    let num_partitions = params
        .get("num_partitions")
        .and_then(|v| v.as_u64())
        .unwrap_or(256) as usize;

    let out = match index_type {
        "IVF_FLAT" => VectorIndexParams::ivf_flat(num_partitions, metric_type),
        "IVF_PQ" => {
            let num_bits = params
                .get("num_bits")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;
            let num_sub_vectors = params
                .get("num_sub_vectors")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize;
            let max_iterations = params
                .get("max_iterations")
                .and_then(|v| v.as_u64())
                .unwrap_or(50) as usize;
            VectorIndexParams::ivf_pq(
                num_partitions,
                num_bits,
                num_sub_vectors,
                metric_type,
                max_iterations,
            )
        }
        "IVF_RQ" => {
            let num_bits = params
                .get("num_bits")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;
            VectorIndexParams::ivf_rq(num_partitions, num_bits, metric_type)
        }
        "IVF_SQ" => {
            let num_bits = params
                .get("num_bits")
                .and_then(|v| v.as_u64())
                .unwrap_or(8) as u8;
            let ivf = lance_index::vector::ivf::IvfBuildParams::new(num_partitions);
            let sq = lance_index::vector::sq::builder::SQBuildParams {
                num_bits: num_bits as u16,
                sample_rate: params
                    .get("sample_rate")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(256) as usize,
            };
            VectorIndexParams::with_ivf_sq_params(metric_type, ivf, sq)
        }
        "IVF_HNSW_FLAT" => {
            let ivf = lance_index::vector::ivf::IvfBuildParams::new(num_partitions);
            let mut hnsw = HnswBuildParams::default();
            if let Some(v) = params.get("hnsw_m").and_then(|v| v.as_u64()) {
                hnsw.m = v as usize;
            }
            if let Some(v) = params.get("hnsw_ef_construction").and_then(|v| v.as_u64()) {
                hnsw.ef_construction = v as usize;
            }
            if let Some(v) = params.get("hnsw_max_level").and_then(|v| v.as_u64()) {
                hnsw.max_level = v as u16;
            }
            if let Some(v) = params.get("hnsw_prefetch_distance") {
                if v.is_null() {
                    hnsw.prefetch_distance = None;
                } else if let Some(u) = v.as_u64() {
                    hnsw.prefetch_distance = Some(u as usize);
                }
            }
            VectorIndexParams::ivf_hnsw(metric_type, ivf, hnsw)
        }
        "IVF_HNSW_PQ" => {
            let ivf = lance_index::vector::ivf::IvfBuildParams::new(num_partitions);
            let mut hnsw = HnswBuildParams::default();
            if let Some(v) = params.get("hnsw_m").and_then(|v| v.as_u64()) {
                hnsw.m = v as usize;
            }
            if let Some(v) = params.get("hnsw_ef_construction").and_then(|v| v.as_u64()) {
                hnsw.ef_construction = v as usize;
            }
            if let Some(v) = params.get("hnsw_max_level").and_then(|v| v.as_u64()) {
                hnsw.max_level = v as u16;
            }
            if let Some(v) = params.get("hnsw_prefetch_distance") {
                if v.is_null() {
                    hnsw.prefetch_distance = None;
                } else if let Some(u) = v.as_u64() {
                    hnsw.prefetch_distance = Some(u as usize);
                }
            }

            let mut pq = PQBuildParams::default();
            if let Some(v) = params.get("num_bits").and_then(|v| v.as_u64()) {
                pq.num_bits = v as usize;
            }
            if let Some(v) = params.get("num_sub_vectors").and_then(|v| v.as_u64()) {
                pq.num_sub_vectors = v as usize;
            }
            if let Some(v) = params.get("max_iterations").and_then(|v| v.as_u64()) {
                pq.max_iters = v as usize;
            }
            VectorIndexParams::with_ivf_hnsw_pq_params(metric_type, ivf, hnsw, pq)
        }
        "IVF_HNSW_SQ" => {
            let ivf = lance_index::vector::ivf::IvfBuildParams::new(num_partitions);
            let mut hnsw = HnswBuildParams::default();
            if let Some(v) = params.get("hnsw_m").and_then(|v| v.as_u64()) {
                hnsw.m = v as usize;
            }
            if let Some(v) = params.get("hnsw_ef_construction").and_then(|v| v.as_u64()) {
                hnsw.ef_construction = v as usize;
            }
            if let Some(v) = params.get("hnsw_max_level").and_then(|v| v.as_u64()) {
                hnsw.max_level = v as u16;
            }
            if let Some(v) = params.get("hnsw_prefetch_distance") {
                if v.is_null() {
                    hnsw.prefetch_distance = None;
                } else if let Some(u) = v.as_u64() {
                    hnsw.prefetch_distance = Some(u as usize);
                }
            }

            let mut sq = lance_index::vector::sq::builder::SQBuildParams {
                num_bits: 8,
                sample_rate: 256,
            };
            if let Some(v) = params.get("num_bits").and_then(|v| v.as_u64()) {
                sq.num_bits = v as u16;
            }
            if let Some(v) = params.get("sample_rate").and_then(|v| v.as_u64()) {
                sq.sample_rate = v as usize;
            }
            VectorIndexParams::with_ivf_hnsw_sq_params(metric_type, ivf, hnsw, sq)
        }
        other => {
            return Err(FfiError::new(
                ErrorCode::DatasetCreateIndex,
                format!("unsupported vector index_type: {other}"),
            ))
        }
    };

    let mut out = out;
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
