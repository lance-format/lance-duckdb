use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};

use crate::error::{clear_last_error, set_last_error, ErrorCode};
use crate::runtime;

use super::util::{cstr_to_str, to_c_string, FfiError, FfiResult};

fn version_error(message: String) -> FfiError {
    FfiError::new(ErrorCode::DatasetVersionAdmin, message)
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

    let mut tags_by_version: BTreeMap<u64, Vec<String>> = BTreeMap::new();
    for (name, contents) in tags {
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

/// List the dataset's tags as `tag\tversion\tcreated_micros\tupdated_micros` lines.
/// A timestamp Lance did not record is left empty.
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
            "{}\t{}\t{}\t{}\n",
            name,
            contents.version,
            micros(contents.created_at),
            micros(contents.updated_at)
        ));
    }
    Ok(out)
}

/// Tag `version` as `tag`. A tag that already exists is an error.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_create_tag(
    dataset: *mut c_void,
    tag: *const c_char,
    version: u64,
) -> i32 {
    finish(dataset_create_tag_inner(dataset, tag, version))
}

fn dataset_create_tag_inner(
    dataset: *mut c_void,
    tag: *const c_char,
    version: u64,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let tag = unsafe { cstr_to_str(tag, "tag")? }.to_string();
    match runtime::block_on(async { ds.tags().create(tag.as_str(), version).await }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(version_error(format!("create tag '{tag}': {err}"))),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

/// Report the version `tag` points at.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_tag_version(
    dataset: *mut c_void,
    tag: *const c_char,
    out_version: *mut u64,
) -> i32 {
    if out_version.is_null() {
        set_last_error(ErrorCode::InvalidArgument, "out_version must not be null");
        return -1;
    }
    finish(dataset_tag_version_inner(dataset, tag, out_version))
}

fn dataset_tag_version_inner(
    dataset: *mut c_void,
    tag: *const c_char,
    out_version: *mut u64,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let tag = unsafe { cstr_to_str(tag, "tag")? }.to_string();
    match runtime::block_on(async { ds.tags().get_version(tag.as_str()).await }) {
        Ok(Ok(version)) => {
            unsafe {
                *out_version = version;
            }
            Ok(())
        }
        Ok(Err(err)) => Err(version_error(format!("tag '{tag}': {err}"))),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

/// Delete `tag`. A tag that does not exist is an error.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_delete_tag(dataset: *mut c_void, tag: *const c_char) -> i32 {
    finish(dataset_delete_tag_inner(dataset, tag))
}

fn dataset_delete_tag_inner(dataset: *mut c_void, tag: *const c_char) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let tag = unsafe { cstr_to_str(tag, "tag")? }.to_string();
    match runtime::block_on(async { ds.tags().delete(tag.as_str()).await }) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(version_error(format!("delete tag '{tag}': {err}"))),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

/// Commit the contents of `version` as a new latest version. History is kept.
/// Reports the new version through `out_version`.
#[no_mangle]
pub unsafe extern "C" fn lance_dataset_restore_version(
    dataset: *mut c_void,
    version: u64,
    out_version: *mut u64,
) -> i32 {
    finish(dataset_restore_version_inner(dataset, version, out_version))
}

fn dataset_restore_version_inner(
    dataset: *mut c_void,
    version: u64,
    out_version: *mut u64,
) -> FfiResult<()> {
    let handle = unsafe { super::util::dataset_handle(dataset)? };
    let ds = (*handle.dataset).clone();
    let restored = runtime::block_on(async {
        let mut target = ds.checkout_version(version).await?;
        target.restore().await?;
        target.checkout_latest().await?;
        Ok::<_, lance::Error>(target.manifest.version)
    });
    match restored {
        Ok(Ok(new_version)) => {
            if !out_version.is_null() {
                unsafe {
                    *out_version = new_version;
                }
            }
            Ok(())
        }
        Ok(Err(err)) => Err(version_error(format!("restore version {version}: {err}"))),
        Err(err) => Err(FfiError::new(ErrorCode::Runtime, format!("runtime: {err}"))),
    }
}

fn finish(result: FfiResult<()>) -> i32 {
    match result {
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
