"""Generate the stored-types test fixture.

Produces ``test/data/stored_types_fixture.lance``, a table written by pylance
whose columns use Arrow types DuckDB does not export for their DuckDB types,
as tables written by pyarrow-based writers commonly do:

* ``id``   : int64
* ``s``    : large_string (DuckDB exports VARCHAR as string)
* ``tags`` : large_list<large_string>
* ``cat``  : dictionary<int32, string> (DuckDB reads it as VARCHAR)
* ``d``    : date64 (DuckDB exports DATE as date32)
* ``t``    : time32[s] (DuckDB exports TIME as time64[us])
* ``ts``   : timestamp[ms, tz=UTC] (DuckDB exports TIMESTAMPTZ as timestamp[us])
* ``dur``  : duration[s] (DuckDB exports INTERVAL as month_day_nano)
* ``fsb``  : fixed_size_binary(4) (DuckDB exports BLOB as binary)

Run once before committing the fixture. The sqllogic test at
``test/sql/dml_stored_types.test`` copies it byte for byte before writing.

Usage
-----

::

    uv run --with pylance --with pyarrow \\
        python test/scripts/gen_stored_types_fixture.py

Delete ``test/data/stored_types_fixture.lance`` first for a clean run.
"""

from __future__ import annotations

import datetime
import pathlib
import shutil

import lance
import pyarrow as pa


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "test" / "data" / "stored_types_fixture.lance"

UTC = datetime.timezone.utc

table = pa.table(
    {
        "id": pa.array([1, 2], type=pa.int64()),
        "s": pa.array(["a", "b"], type=pa.large_string()),
        "tags": pa.array([["x"], ["y", "z"]], type=pa.large_list(pa.large_string())),
        "cat": pa.array(["red", "blue"]).dictionary_encode(),
        "d": pa.array(
            [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)],
            type=pa.date64(),
        ),
        "t": pa.array([datetime.time(0, 0, 1), datetime.time(0, 0, 2)], type=pa.time32("s")),
        "ts": pa.array(
            [
                datetime.datetime(2024, 1, 1, 0, 0, 0, 1000, tzinfo=UTC),
                datetime.datetime(2024, 1, 2, 0, 0, 0, 2000, tzinfo=UTC),
            ],
            type=pa.timestamp("ms", tz="UTC"),
        ),
        "dur": pa.array([1, 2], type=pa.duration("s")),
        "fsb": pa.array([b"\x01\x02\x03\x04", b"\x05\x06\x07\x08"], type=pa.binary(4)),
    }
)

if OUT_PATH.exists():
    shutil.rmtree(OUT_PATH)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

lance.write_dataset(table, str(OUT_PATH))
print(f"Wrote {OUT_PATH}")
