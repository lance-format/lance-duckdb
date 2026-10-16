"""Generate the float16 test fixture.

Produces ``test/data/float16_fixture.lance`` with Float16 exercised across
complex nested shapes to cover every recursion branch in the widening hook:

* ``id``           : int64
* ``half_scalar``  : float16 scalar
* ``half_vec``     : fixed-size list of float16, list size 3
* ``half_list``    : variable-size list of float16
* ``half_struct``  : struct containing a float16 and a float32 field
* ``half_in_list`` : list of struct<half : float16>

(Lance's Rust writer does not yet encode ``map<_, float16>``, so the
map-of-half case is exercised in the schema-conversion Scala tests on the
lance-spark side instead of here.)

Run once before committing the fixture. The sqllogic test at
``test/sql/float16_widening.test`` reads from this directory.

Usage
-----

::

    uv run --with pylance --with pyarrow --with numpy \\
        python test/scripts/gen_float16_fixture.py

Delete ``test/data/float16_fixture.lance`` first for a clean run.
"""

from __future__ import annotations

import pathlib
import shutil

import lance
import numpy as np
import pyarrow as pa


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "test" / "data" / "float16_fixture.lance"

# Deterministic, non-trivial values. Exactly representable in half to keep
# sqllogic assertions stable (65504.0 is the largest finite half).
SCALAR = np.array([1.5, -2.25, 3.0, 0.0, 65504.0], dtype=np.float16)
VEC = np.array(
    [
        [0.5, 1.5, 2.5],
        [-1.0, -2.0, -3.0],
        [10.0, 20.0, 30.0],
        [0.125, 0.25, 0.5],
        [100.0, 200.0, 300.0],
    ],
    dtype=np.float16,
)

scalar_arr = pa.array(SCALAR, type=pa.float16())
vec_arr = pa.FixedSizeListArray.from_arrays(pa.array(VEC.reshape(-1), type=pa.float16()), 3)

# Variable-size list<float16>: [[1.0, 2.0], [3.0], [], [-1.0, -2.0, -3.0], [0.0]]
list_values = pa.array(
    np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.0], dtype=np.float16),
    type=pa.float16(),
)
list_offsets = pa.array([0, 2, 3, 3, 6, 7], type=pa.int32())
half_list = pa.ListArray.from_arrays(list_offsets, list_values)

# Struct<half : float16, f32 : float32>.
half_struct = pa.StructArray.from_arrays(
    [
        scalar_arr,
        pa.array([10.0, 20.0, 30.0, 40.0, 50.0], type=pa.float32()),
    ],
    names=["half", "f32"],
)

# List<Struct<half : float16>> — struct-inside-list.
inner_struct = pa.StructArray.from_arrays(
    [pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], type=pa.float16())],
    names=["h"],
)
inner_offsets = pa.array([0, 1, 3, 3, 5, 6], type=pa.int32())
half_in_list = pa.ListArray.from_arrays(inner_offsets, inner_struct)

table = pa.table(
    {
        "id": pa.array(range(len(SCALAR)), type=pa.int64()),
        "half_scalar": scalar_arr,
        "half_vec": vec_arr,
        "half_list": half_list,
        "half_struct": half_struct,
        "half_in_list": half_in_list,
    }
)

if OUT_PATH.exists():
    shutil.rmtree(OUT_PATH)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

lance.write_dataset(table, str(OUT_PATH))
print(f"Wrote {OUT_PATH}")
