use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, dataset_handle, to_c_string, FfiError, FfiResult};

fn version_error(message: String) -> FfiError {
    FfiError::new(ErrorCode::DatasetVersions, message)
}

async fn drop_dataset_tag(dataset: &lance::Dataset, tag: &str) -> lance::Result<()> {
    let store = dataset.object_store(None).await?;
    if store.is_local() || store.scheme() == "file-object-store" {
        // A case-insensitive volume can open another spelling of this name.
        // Compare the stored directory entry before asking Lance to delete it.
        let root = dataset.refs.root()?.path;
        let path = lance::dataset::refs::tag_path(&root, tag);
        let dir = lance::dataset::refs::base_tags_path(&root);
        let entries = store.list_with_delimiter(Some(&dir)).await?;
        if !entries
            .objects
            .iter()
            .any(|e| e.location.filename() == path.filename())
        {
            return Err(lance::Error::RefNotFound {
                message: format!("tag {tag} does not exist"),
            });
        }
    }
    dataset.tags().delete(tag).await
}

/// Create a tag at the supplied version, or delete it when version is null.
///
/// # Safety
/// The dataset handle, NUL-terminated tag and optional version must be valid
/// for this call.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_mutate_tag(
    dataset: *mut c_void,
    tag: *const c_char,
    version: *const i64,
) -> i32 {
    match mutate_tag_inner(dataset, tag, version) {
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

fn mutate_tag_inner(
    dataset: *mut c_void,
    tag: *const c_char,
    version: *const i64,
) -> FfiResult<()> {
    // SAFETY: The caller keeps all inputs alive until the operation completes.
    let handle = unsafe { dataset_handle(dataset)? };
    let tag = unsafe { cstr_to_str(tag, "tag")? };
    let version = unsafe { version.as_ref() }.copied();
    lance::dataset::refs::check_valid_tag(tag).map_err(|err| version_error(err.to_string()))?;
    // A dispatched object store write can outlive a dropped future. The caller
    // checks interruption before dispatch and holds its catalog lock until completion.
    let rt = runtime::runtime()
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?;
    match version {
        // Lance 14.0.3 re-encodes non-ASCII tag filenames while listing them.
        Some(_) if !tag.is_ascii() => Err(version_error(
            "CREATE TAG requires an ASCII tag name".into(),
        )),
        Some(v) if v <= 0 => Err(version_error("tag version must be positive".into())),
        Some(v) => rt
            .block_on(handle.dataset.tags().create(tag, v as u64))
            .map_err(|err| version_error(err.to_string())),
        None => rt
            .block_on(drop_dataset_tag(&handle.dataset, tag))
            .map_err(|err| version_error(err.to_string())),
    }
}

/// The branch a dataset handle or tag belongs to; `None` is the main branch.
fn branch_of(branch: Option<&str>) -> Option<&str> {
    branch.filter(|name| !name.is_empty() && *name != "main")
}

fn timestamp_micros(timestamp: chrono::DateTime<chrono::Utc>) -> i64 {
    timestamp.timestamp_micros()
}

/// List every version of the dataset as `version\ttimestamp_micros\ttags` lines.
///
/// `tags` is a comma separated list of the tags that point at the version; Lance
/// restricts tag names to characters that never include a comma.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_versions(dataset: *mut c_void) -> *const c_char {
    match dataset_list_versions_inner(dataset) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_versions_inner(dataset: *mut c_void) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();

    let (mut versions, tags) = match runtime::block_on(async {
        let versions = ds.versions().await?;
        let tags = ds.tags().list().await?;
        Ok::<_, lance::Error>((versions, tags))
    }) {
        Ok(Ok(pair)) => pair,
        Ok(Err(err)) => return Err(version_error(format!("dataset versions: {err}"))),
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };

    // Tags are global to the dataset root and carry a branch; only the ones on
    // this handle's branch name one of its versions.
    let current = ds.branch_location().branch;
    let mut tags_by_version: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for (name, contents) in tags {
        if branch_of(contents.branch.as_deref()) != branch_of(current.as_deref()) {
            continue;
        }
        tags_by_version
            .entry(contents.version)
            .or_default()
            .push(name);
    }
    for names in tags_by_version.values_mut() {
        names.sort();
    }

    versions.sort_by_key(|version| version.version);
    let mut out = String::new();
    for version in versions {
        let tags = tags_by_version
            .get(&version.version)
            .map(|names| names.join(","))
            .unwrap_or_default();
        out.push_str(&format!(
            "{}\t{}\t{}\n",
            version.version,
            timestamp_micros(version.timestamp),
            tags
        ));
    }
    Ok(out)
}

/// List the dataset's tags as `tag\tversion\tbranch\tcreated_micros\tupdated_micros`
/// lines. The branch is empty for the main branch; a timestamp Lance did not
/// record is left empty.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_list_tags(dataset: *mut c_void) -> *const c_char {
    match dataset_list_tags_inner(dataset) {
        Ok(s) => {
            clear_last_error();
            to_c_string(s).into_raw()
        }
        Err(err) => {
            set_last_error(err.code, err.message);
            std::ptr::null()
        }
    }
}

fn dataset_list_tags_inner(dataset: *mut c_void) -> FfiResult<String> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();

    let mut tags = match runtime::block_on(async { ds.tags().list().await }) {
        Ok(Ok(tags)) => tags.into_iter().collect::<Vec<_>>(),
        Ok(Err(err)) => return Err(version_error(format!("dataset tags: {err}"))),
        Err(err) => return Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    };
    tags.sort_by(|a, b| a.1.version.cmp(&b.1.version).then_with(|| a.0.cmp(&b.0)));

    let micros = |timestamp: Option<chrono::DateTime<chrono::Utc>>| {
        timestamp
            .map(|t| timestamp_micros(t).to_string())
            .unwrap_or_default()
    };
    let mut out = String::new();
    for (name, contents) in tags {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            name,
            contents.version,
            branch_of(contents.branch.as_deref()).unwrap_or_default(),
            micros(contents.created_at),
            micros(contents.updated_at)
        ));
    }
    Ok(out)
}
