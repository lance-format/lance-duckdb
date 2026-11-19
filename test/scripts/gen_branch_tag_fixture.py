"""Generate the branch-tag test fixture.

Produces ``test/data/branch_tag_fixture.lance``: a dataset whose main branch has
versions 1..3 (``id`` = 1, 2, 3), a tag ``main_tag`` on main version 1, and a
branch ``dev`` that holds the tag ``dev_tag`` on a snapshot with ``id`` = 99.
Lance tags are global to the dataset root, so ``dev_tag`` is visible from main
while its version number belongs to the ``dev`` branch.

The sqllogic test at ``test/sql/version_admin_branch_tag.test`` reads from this
directory.

Usage
-----

::

    uv run --with pylance==14.0.3 --with pyarrow \\
        python test/scripts/gen_branch_tag_fixture.py

Delete ``test/data/branch_tag_fixture.lance`` first for a clean run.
"""

from __future__ import annotations

import pathlib
import shutil

import lance
import pyarrow as pa


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "test" / "data" / "branch_tag_fixture.lance"


def main() -> None:
    if OUT_PATH.exists():
        shutil.rmtree(OUT_PATH)
    path = str(OUT_PATH)

    lance.write_dataset(pa.table({"id": [1]}), path)
    ds = lance.write_dataset(pa.table({"id": [2]}), path, mode="overwrite")
    ds.tags.create("main_tag", 1)

    branch = ds.create_branch("dev")
    branch.update({"id": "99"})
    branch.tags.create("dev_tag")

    lance.write_dataset(pa.table({"id": [3]}), path, mode="overwrite")


if __name__ == "__main__":
    main()
