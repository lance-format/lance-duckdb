//! Regenerate test/data/blob_range_fixture.lance with the pinned Lance
//! dependency. The fixture is committed so SQL tests run without a writer that
//! can produce Blob v2 columns yet; use this binary to refresh it after
//! dependency changes:
//!
//! ```sh
//! cargo run --example generate_blob_range_fixture -- test/data/blob_range_fixture.lance
//! ```

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, RecordBatchIterator, StringArray, UInt64Array};
use arrow_schema::{DataType, Field as ArrowField, Schema as ArrowSchema};
use lance::blob::{blob_field_with_options, BlobArrayBuilder, BlobFieldOptions};
use lance::dataset::{Dataset, WriteParams};
use lance_file::version::LanceFileVersion;

enum Payload {
    Bytes(Vec<u8>),
    Null,
    Empty,
}

fn v1_payloads() -> Vec<Payload> {
    vec![
        Payload::Bytes(b"hello world".to_vec()),
        Payload::Bytes(vec![0x42; 100_000]),
        Payload::Null,
        Payload::Empty,
        Payload::Bytes(vec![0xA5; 1024 * 1024]),
    ]
}

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        // Keep one row past the dedicated threshold so the committed SQL
        // fixture exercises inline, packed and dedicated blob layouts.
        blob_field_with_options(
            "payload",
            true,
            BlobFieldOptions::default()
                .with_inline_size_threshold(1024)
                .with_dedicated_size_threshold(std::num::NonZeroUsize::new(512 * 1024).unwrap()),
        ),
        ArrowField::new("id", DataType::UInt64, false),
        ArrowField::new("tag", DataType::Utf8, false),
    ]))
}

fn v1_batch() -> RecordBatch {
    let schema = schema();
    let payloads = v1_payloads();
    let mut builder = BlobArrayBuilder::new(payloads.len());
    for payload in &payloads {
        match payload {
            Payload::Bytes(bytes) => builder.push_bytes(bytes).unwrap(),
            Payload::Null => builder.push_null().unwrap(),
            Payload::Empty => builder.push_empty().unwrap(),
        }
    }
    let blob_array: ArrayRef = builder.finish().unwrap();
    RecordBatch::try_new(
        schema,
        vec![
            blob_array,
            Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                "inline",
                "packed",
                "null",
                "empty",
                "dedicated",
            ])),
        ],
    )
    .unwrap()
}

fn v2_batch() -> RecordBatch {
    let schema = schema();
    let mut builder = BlobArrayBuilder::new(1);
    builder.push_bytes(b"appended payload").unwrap();
    let blob_array: ArrayRef = builder.finish().unwrap();
    RecordBatch::try_new(
        schema,
        vec![
            blob_array,
            Arc::new(UInt64Array::from(vec![5])),
            Arc::new(StringArray::from(vec!["appended"])),
        ],
    )
    .unwrap()
}

#[tokio::main]
async fn main() -> lance::Result<()> {
    let output = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "test/data/blob_range_fixture.lance".to_string());
    if output.starts_with("file://") {
        return Err(lance::Error::invalid_input(
            "pass a plain local path; file:// URIs are not supported here".to_string(),
        ));
    }
    if std::path::Path::new(&output).exists() {
        std::fs::remove_dir_all(&output).ok();
    }

    let v1 = v1_batch();
    let v1_schema = v1.schema();
    let mut dataset = Dataset::write(
        RecordBatchIterator::new(vec![Ok(v1)], v1_schema),
        &output,
        Some(WriteParams {
            data_storage_version: Some(LanceFileVersion::V2_2),
            ..Default::default()
        }),
    )
    .await?;
    let v2 = v2_batch();
    let v2_schema = v2.schema();
    dataset
        .append(RecordBatchIterator::new(vec![Ok(v2)], v2_schema), None)
        .await?;
    assert_eq!(dataset.manifest.version, 2);
    println!("wrote {} at version {}", output, dataset.manifest.version);
    Ok(())
}
