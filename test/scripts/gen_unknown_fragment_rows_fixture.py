"""Generate the legacy fragment-count fixture for scan_batch_index.test.

Run with pylance 14.0.2 and pyarrow 26.0.1. The first commit intentionally
omits physical row counts, as historical manifests did. Appending to an
existing dataset would migrate its metadata and lose this test condition.
"""

from pathlib import Path

import lance
import pyarrow as pa
from lance.fragment import LanceFragment


OUT_PATH = Path(__file__).resolve().parents[2] / "test/data/unknown_fragment_rows.lance"
if OUT_PATH.exists():
    raise FileExistsError(f"Remove {OUT_PATH} before regenerating the fixture")

schema = pa.schema([("id", pa.int64()), ("label", pa.string())])
fragments = []
for start, end in [(0, 5000), (5000, 5011), (5011, 10023)]:
    table = pa.Table.from_arrays(
        [pa.array(range(start, end)), pa.array([f"row-{i}" for i in range(start, end)])], schema=schema
    )
    fragment = LanceFragment.create(str(OUT_PATH), table, data_storage_version="legacy")
    fragment.physical_rows = None
    fragments.append(fragment)

dataset = lance.LanceDataset.commit(
    str(OUT_PATH), lance.LanceOperation.Overwrite(schema, fragments), enable_stable_row_ids=False
)
assert all(fragment.metadata.physical_rows is None for fragment in dataset.get_fragments())
assert dataset.to_table()["id"].to_pylist() == list(range(10023))
print(f"Wrote {OUT_PATH}")
