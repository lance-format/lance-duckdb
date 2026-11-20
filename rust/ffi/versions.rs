use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};

use lance::dataset::refs::{Ref, TagContents};
use lance_io::object_store::ObjectStore;
use object_store::{GetResult, ObjectStoreExt};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, dataset_handle, to_c_string, FfiError, FfiResult};

fn version_error(message: String) -> FfiError {
    FfiError::new(ErrorCode::DatasetVersions, message)
}

async fn check_local_tag_name(
    store: &ObjectStore,
    root: &object_store::path::Path,
    tag: &str,
) -> lance::Result<object_store::ObjectMeta> {
    let path = lance::dataset::refs::tag_path(root, tag);
    let dir = lance::dataset::refs::base_tags_path(root);
    let entries = store.list_with_delimiter(Some(&dir)).await?;
    entries
        .objects
        .into_iter()
        .find(|e| e.location.filename() == path.filename())
        .ok_or_else(|| lance::Error::RefNotFound {
            message: format!("tag {tag} does not exist"),
        })
}

pub(super) async fn resolve_tag(dataset: &lance::Dataset, tag: &str) -> lance::Result<Ref> {
    let store = dataset.object_store(None).await?;
    if !store.is_local() && store.scheme() != "file-object-store" {
        return Ok(Ref::from(tag));
    }
    lance::dataset::refs::check_valid_tag(tag)?;
    let root = dataset.refs.root()?.path;
    let path = lance::dataset::refs::tag_path(&root, tag);
    let missing = || lance::Error::RefNotFound {
        message: format!("tag {tag} does not exist"),
    };
    let object = loop {
        let object = match store.inner.get(&path).await {
            Ok(object) => object,
            Err(object_store::Error::NotFound { .. }) => return Err(missing()),
            Err(err) => return Err(err.into()),
        };
        if let Some(name) = opened_tag_name(&object) {
            if Some(name.as_str()) != path.filename() {
                return Err(missing());
            }
            break object;
        }
        // A wrapped store may return a stream. A differently cased extension
        // cannot resolve on a case-sensitive volume, even in a search-only directory.
        let dir = lance::dataset::refs::base_tags_path(&root);
        let entry = if store.exists(&dir.join(format!("{tag}.JSON"))).await? {
            check_local_tag_name(&store, &root, tag).await?
        } else {
            // A case-insensitive alias can also disappear due to a concurrent
            // delete. Verify that the original object still exists.
            match store.inner.head(&path).await {
                Ok(entry) => entry,
                Err(object_store::Error::NotFound { .. }) => continue,
                Err(err) => return Err(err.into()),
            }
        };
        if object.meta.e_tag.is_none() || entry.e_tag.is_none() {
            return Err(lance::Error::io(
                "The local store did not return an ETag for the tag file",
            ));
        }
        if entry.e_tag == object.meta.e_tag {
            break object;
        }
        // An atomic replacement between GET and listing invalidates the name
        // check. Read the current object with the next poll.
    };
    let bytes = object.bytes().await?;
    let contents: TagContents = serde_json::from_slice(&bytes)?;
    Ok(Ref::Version(contents.branch, Some(contents.version)))
}

// The name and the bytes come from the same open file even across an atomic
// tag update. The path supplied by GET itself is merely the requested spelling.
fn opened_tag_name(_object: &GetResult) -> Option<String> {
    #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios", windows))]
    if let object_store::GetResultPayload::File(file, _) = &_object.payload {
        use filepath::FilePath;
        let path = file.path().ok()?;
        let name = path.file_name()?.to_str()?;
        #[cfg(target_os = "linux")]
        let name = name.strip_suffix(" (deleted)").unwrap_or(name);
        if name.ends_with(".json") {
            return Some(name.to_owned());
        }
    }
    None
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
    match version {
        // Lance 14.0.3 re-encodes non-ASCII tag filenames while listing them.
        Some(_) if !tag.is_ascii() => {
            return Err(version_error(
                "CREATE TAG requires an ASCII tag name".into(),
            ))
        }
        Some(v) if v <= 0 => return Err(version_error("tag version must be positive".into())),
        _ => {}
    }
    runtime::block_on(async {
        let store = handle.dataset.object_store(None).await?;
        if version.is_none() {
            if store.is_local() || store.scheme() == "file-object-store" {
                check_local_tag_name(&store, &handle.dataset.refs.root()?.path, tag).await?;
            }
        } else if store.is_local() {
            // Lance 14.0.3 re-encodes the full key when staging a local create.
            // Remove this check after adopting lance-format/lance#11220.
            let root = handle.dataset.refs.root()?.path;
            if object_store::path::Path::from(root.as_ref()) != root {
                return Err(lance::Error::invalid_input(
                    "CREATE TAG cannot stage a tag under this local dataset path with Lance 14.0.3",
                ));
            }
        }
        Ok::<_, lance::Error>(())
    })
    .map_err(|err| FfiError::new(ErrorCode::Runtime, err.to_string()))?
    .map_err(|err| version_error(err.to_string()))?;
    let rt = runtime::runtime()
        .map_err(|err| FfiError::new(ErrorCode::Runtime, format!("runtime: {err}")))?;
    // Check again after the read-only work. Once dispatched, a store write can
    // outlive a cancelled future, so retain the catalog lock until Lance returns.
    if runtime::interrupt_hook().is_some_and(|hook| hook.interrupted()) {
        return Err(FfiError::new(ErrorCode::Runtime, "interrupted"));
    }
    let tags = handle.dataset.tags();
    rt.block_on(async {
        match version {
            Some(v) => tags.create(tag, v as u64).await,
            None => tags.delete(tag).await,
        }
    })
    .map_err(|err| version_error(err.to_string()))
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
