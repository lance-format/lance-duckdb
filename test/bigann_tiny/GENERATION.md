# bigann_tiny generation notes

This directory contains a tiny, repo-committed Big-ANN style fixture:

- `base.lance`: base vectors + scalar columns + a vector index on `vec`
- `queries.lance`: query vectors
- `groundtruth_all_top10.lance`: exact top-10 ids for each query over all rows
- `groundtruth_label1_top10.lance`: exact top-10 ids for each query restricted to `label = 1`

Parameters (see `metadata.json` for the exact values):

- `dim=16`, `rows=2048`, `queries=32`, `k=10`, `metric=L2`
- `label=1` iff `id % 8 == 0`
- Vectors are generated from two far-apart Gaussian clusters:
  - `label=0` near `0.0`, `label=1` near `10.0` (same `noise_stddev`)

Index:

- Built on `base.lance` column `vec` with `index_type=IVF_PQ` and `metric=L2`.

Regeneration (one-off):

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install numpy pyarrow pylance
python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import lance

OUT_DIR = Path("test/bigann_tiny")
BASE_PATH = OUT_DIR / "base.lance"
QUERIES_PATH = OUT_DIR / "queries.lance"
TRUTH_ALL_PATH = OUT_DIR / "groundtruth_all_top10.lance"
TRUTH_LABEL1_PATH = OUT_DIR / "groundtruth_label1_top10.lance"

SEED = 1337
DIM = 16
N_ROWS = 2048
N_QUERIES = 32
K = 10

rng = np.random.default_rng(SEED)
ids = np.arange(N_ROWS, dtype=np.int64)
labels = (ids % 8 == 0).astype(np.int8)
buckets = (ids % 16).astype(np.int32)
centers = np.where(labels[:, None] == 1, 10.0, 0.0).astype(np.float32)
noise = rng.normal(loc=0.0, scale=0.35, size=(N_ROWS, DIM)).astype(np.float32)
base_vecs = centers + noise
queries = rng.normal(loc=0.0, scale=0.35, size=(N_QUERIES, DIM)).astype(np.float32)
qids = np.arange(N_QUERIES, dtype=np.int32)

def fsl(arr: np.ndarray) -> pa.FixedSizeListArray:
    values = pa.array(arr.reshape(-1).tolist(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, arr.shape[1])

def write(table: pa.Table, path: Path) -> lance.LanceDataset:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    lance.write_dataset(table, str(path), mode="create")
    return lance.dataset(str(path))

OUT_DIR.mkdir(parents=True, exist_ok=True)
base = write(pa.table({"id": ids, "label": labels, "bucket": buckets, "vec": fsl(base_vecs)}), BASE_PATH)
_ = write(pa.table({"qid": qids, "vec": fsl(queries)}), QUERIES_PATH)

truth_all, truth_label1 = [], []
for qid, q in enumerate(queries):
    diffs = base_vecs - q[None, :]
    dists = np.sum(diffs * diffs, axis=1)
    topk = np.argsort(dists, kind="stable")[:K]
    truth_all.extend((qid, rank, int(ids[i]), float(dists[i])) for rank, i in enumerate(topk))
    mask = labels == 1
    d1 = dists[mask]
    i1 = ids[mask]
    topk1 = np.argsort(d1, kind="stable")[:K]
    truth_label1.extend((qid, rank, int(i1[p]), float(d1[p])) for rank, p in enumerate(topk1))

schema = pa.schema([("qid", pa.int32()), ("rank", pa.int32()), ("id", pa.int64()), ("distance", pa.float32())])
_ = write(pa.Table.from_pylist([{"qid": q, "rank": r, "id": i, "distance": d} for q, r, i, d in truth_all], schema=schema), TRUTH_ALL_PATH)
_ = write(pa.Table.from_pylist([{"qid": q, "rank": r, "id": i, "distance": d} for q, r, i, d in truth_label1], schema=schema), TRUTH_LABEL1_PATH)

base.create_index("vec", index_type="IVF_PQ", metric="L2", replace=True, num_partitions=8, num_sub_vectors=4)

meta = {
    "seed": SEED,
    "dim": DIM,
    "rows": N_ROWS,
    "queries": N_QUERIES,
    "k": K,
    "metric": "L2",
    "label_rule": "label=1 iff id % 8 == 0",
    "bucket_rule": "bucket = id % 16",
    "clusters": {"label0_center": 0.0, "label1_center": 10.0, "noise_stddev": 0.35},
    "packages": {"numpy": np.__version__, "pyarrow": pa.__version__, "pylance": getattr(lance, "__version__", "unknown")},
}
(OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
print("done:", OUT_DIR)
PY
```
