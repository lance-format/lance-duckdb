"""Generate a remote Blob v2 dataset with a source nested under its own URI."""

from pathlib import Path

import lance
import pyarrow as pa
from lance.blob import blob_array, blob_field


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            blob_field("payload"),
        ]
    )
    table = pa.Table.from_arrays(
        [
            pa.array([7], pa.int64()),
            blob_array(
                [
                    "s3://lance-test/nested_target.lance/source/object.bin",
                ]
            ),
        ],
        schema=schema,
    )
    lance.write_dataset(
        table,
        ROOT / "data" / "s3_nested_source_target.lance",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )


if __name__ == "__main__":
    main()
