"""Generate the Blob v2 read-path fixture.

The committed fixture uses small values so all managed layouts fit in the
repository:

* inline: a value at or below ``inline_size_threshold``
* packed: a value above the inline threshold and below the dedicated threshold
* dedicated: a value above ``dedicated_size_threshold``

It also includes NULL, an empty blob, ordinary binary, nested STRUCT/LIST
columns, a vector column, and an inverted index for FTS.

Usage:

    uv run --with pylance==13.0.0 --with pyarrow==26.0.0 \\
        python test/scripts/gen_blob_v2_read_fixture.py

Use ``--output`` to generate a temporary dataset and ``--external`` to add
an external blob whose URI points at a sibling file. The committed fixture does
not use external blobs because an external URI cannot be portable across
checkouts.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil

import lance
import pyarrow as pa
from lance.blob import Blob, blob_array, blob_field


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "test" / "data" / "blob_v2_reads.lance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Dataset output directory",
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Add an external blob backed by a sibling file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payloads: list[object] = [
        b"inline",
        b"p" * 96,
        b"d" * 256,
        None,
        b"",
    ]
    if args.external:
        external = output.with_suffix(".external")
        external.write_bytes(b"external-payload")
        payloads.append(Blob.from_uri(external.as_uri()))

    row_count = len(payloads)
    nested_values = [b"nested", None, b"", b"nested-3", b"nested-4"]
    item_values = [b"first", None, b"", b"list-a", b"list-b", b"list-c"]
    if args.external:
        nested_values.append(b"nested-external")
        item_values.append(b"list-external")
    nested_blob = pa.StructArray.from_arrays(
        [blob_array(nested_values)],
        fields=[blob_field("payload")],
    )
    blob_items = pa.ListArray.from_arrays(
        pa.array([0, 2, 3, 3, 5, 6], type=pa.int32()),
        blob_array(item_values),
    )
    if args.external:
        blob_items = pa.ListArray.from_arrays(
            pa.array([0, 2, 3, 3, 5, 6, 7], type=pa.int32()),
            blob_array(item_values),
        )
    table = pa.Table.from_arrays(
        [
            pa.array(range(row_count), type=pa.int64()),
            pa.array(
                [[1.0, 0.0, float(row)] for row in range(row_count)],
                type=pa.list_(pa.float32(), 3),
            ),
            pa.array(
                [
                    "alpha document",
                    "beta document",
                    "gamma document",
                    "delta document",
                    "epsilon document",
                ]
                + (["external document"] if args.external else [])
            ),
            pa.array(
                [
                    b"ordinary-binary",
                    b"ordinary-binary-pack",
                    b"ordinary-binary-dedicated",
                    None,
                    b"",
                ]
                + ([b"ordinary-binary-external"] if args.external else []),
                type=pa.binary(),
            ),
            blob_array(payloads),
            pa.array(
                [{"a": index, "b": f"nested-{index}"} for index in range(row_count)],
                type=pa.struct(
                    [
                        pa.field("a", pa.int32(), nullable=False),
                        pa.field("b", pa.string(), nullable=False),
                    ]
                ),
            ),
            pa.array(
                [[index, index + 1] for index in range(row_count)],
                type=pa.list_(pa.int32()),
            ),
            nested_blob,
            blob_items,
        ],
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("vec", pa.list_(pa.float32(), 3), nullable=False),
                pa.field("text", pa.string(), nullable=False),
                pa.field("ordinary", pa.binary(), nullable=True),
                blob_field(
                    "payload",
                    inline_size_threshold=16,
                    dedicated_size_threshold=128,
                    pack_file_size_threshold=1024 * 1024,
                ),
                pa.field(
                    "nested",
                    pa.struct(
                        [
                            pa.field("a", pa.int32(), nullable=False),
                            pa.field("b", pa.string(), nullable=False),
                        ]
                    ),
                    nullable=False,
                ),
                pa.field("items", pa.list_(pa.int32()), nullable=False),
                pa.field(
                    "nested_blob",
                    pa.struct([blob_field("payload")]),
                    nullable=False,
                ),
                pa.field("blob_items", pa.list_(blob_field("item")), nullable=False),
            ]
        ),
    )

    if output.exists():
        shutil.rmtree(output)
    dataset = lance.write_dataset(
        table,
        str(output),
        data_storage_version="2.2",
        allow_external_blob_outside_bases=args.external,
    )
    dataset.create_scalar_index("text", index_type="INVERTED")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
