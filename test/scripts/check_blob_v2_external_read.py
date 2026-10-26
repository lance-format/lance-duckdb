"""Exercise external Blob v2 reads through the built DuckDB extension.

This is an acceptance check rather than a committed fixture: external blob
URIs are absolute and cannot be portable across checkouts. The script generates
a temporary dataset, changes and then removes the referenced source object, and
checks the SQL-visible bytes and failure behavior through the real CLI.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import tempfile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def run_sql(duckdb: pathlib.Path, extension: pathlib.Path, sql: str) -> str:
    command = [str(duckdb), "-unsigned", "-csv", "-noheader", "-c", sql]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def load_extension(extension: pathlib.Path) -> str:
    escaped = str(extension).replace("'", "''")
    return f"LOAD '{escaped}';"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb",
        type=pathlib.Path,
        default=REPO_ROOT / "build" / "release" / "duckdb",
    )
    parser.add_argument(
        "--extension",
        type=pathlib.Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="lance-blob-read-") as tmp:
        root = pathlib.Path(tmp)
        output = root / "external.lance"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "test" / "scripts" / "gen_blob_v2_read_fixture.py"),
                "--output",
                str(output),
                "--external",
            ],
            check=True,
        )
        source = output.with_suffix(".external")

        prefix = load_extension(args.extension)
        escaped_dataset = str(output).replace("'", "''")
        initial = run_sql(
            args.duckdb,
            args.extension,
            prefix + f" SELECT id, octet_length(payload), payload = 'external-payload'::BLOB "
            f"FROM '{escaped_dataset}' WHERE id = 5;",
        )
        assert initial == "5,16,true", initial

        source.write_bytes(b"changed-external")
        changed = run_sql(
            args.duckdb,
            args.extension,
            prefix + f" SELECT payload = 'changed-external'::BLOB " f"FROM '{escaped_dataset}' WHERE id = 5;",
        )
        assert changed == "true", changed

        source.unlink()
        survivors = run_sql(
            args.duckdb,
            args.extension,
            prefix + f" SELECT id, octet_length(payload) FROM '{escaped_dataset}' "
            f"WHERE _rowid IN (0, 4) ORDER BY id;",
        )
        assert survivors == "0,6\n4,0", survivors

        search_calls = {
            "fts": f"lance_fts('{escaped_dataset}', 'text', 'external', k = 1)",
            "vector": (
                f"lance_vector_search('{escaped_dataset}', 'vec', "
                f"[1.0, 0.0, 5.0]::FLOAT[3], k = 1, use_index = false)"
            ),
            "hybrid": (
                f"lance_hybrid_search('{escaped_dataset}', 'vec', "
                f"[1.0, 0.0, 5.0]::FLOAT[3], 'text', 'external', "
                f"k = 1, use_index = false)"
            ),
        }
        for name, search_call in search_calls.items():
            rows = run_sql(
                args.duckdb,
                args.extension,
                prefix + f" SELECT id FROM {search_call};",
            )
            assert rows == "5", f"{name} unexpectedly read the removed blob: {rows}"
            projected = subprocess.run(
                [
                    str(args.duckdb),
                    "-unsigned",
                    "-csv",
                    "-noheader",
                    "-c",
                    prefix + f" SELECT id, hex(payload) FROM {search_call};",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            assert projected.returncode != 0, f"{name} projected the removed external blob without an error"

        result = subprocess.run(
            [
                str(args.duckdb),
                "-unsigned",
                "-csv",
                "-noheader",
                "-c",
                prefix + f" SELECT payload FROM '{escaped_dataset}' WHERE _rowid = 5;",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0, "missing external source unexpectedly succeeded"
        print("external Blob v2 read, source change, and source failure checks passed")


if __name__ == "__main__":
    main()
