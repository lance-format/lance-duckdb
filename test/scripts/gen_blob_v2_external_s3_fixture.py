"""Generate the local Lance fixture used by the MinIO external-source test."""

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
    # Row order is intentional: the limited reader is used first. Reusing its
    # same-bucket object store for row 2 would make the root-only source fail.
    table = pa.Table.from_arrays(
        [
            pa.array([1, 2], pa.int64()),
            blob_array(
                [
                    "s3://lance-test/blob_v2_external/b/object.bin",
                    "s3://lance-test/blob_v2_external/a/object.bin",
                ]
            ),
        ],
        schema=schema,
    )
    lance.write_dataset(
        table,
        ROOT / "data" / "blob_v2_external_source.lance",
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )


if __name__ == "__main__":
    main()
