"""Check search `_rowid` on REST namespace tables against the rows it addresses.

The fixture namespace describes table `t` at a local dataset A. Its
query_table answers every search with rows of another table B whose `_rowid`
values are the row addresses of two other rows of A, as a namespace repointed
after a DELETE pinned A would. Outside DML the search returns those rows from
query_table, requesting `_rowid` only when the query references it. Inside a
DELETE of `t`, prepared or not, the search must read the pinned dataset A
instead, so the DELETE removes exactly the rows the search finds in A, whether
it matches them by `_rowid` or by `id`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import lance
import pyarrow as pa


REPO_ROOT = Path(__file__).resolve().parents[2]


def fixture_rows(lo: int, hi: int) -> pa.Table:
    ids = list(range(lo, hi))
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "txt": pa.array([f"w{i % 4}" for i in ids]),
            "vec": pa.array([[float(i), 0.0] for i in ids], pa.list_(pa.float32(), 2)),
        }
    )


def arrow_stream(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "duckdb",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="lance-search-rowid-rest-") as tmp:
        dataset = str(Path(tmp) / "a.lance")
        addressed: list[pa.Array] = []

        def reset_dataset() -> None:
            # Two fragments, so a row address is not a row offset.
            lance.write_dataset(fixture_rows(0, 10), dataset, mode="overwrite")
            lance.write_dataset(fixture_rows(10, 20), dataset, mode="append")
            ds = lance.dataset(dataset)
            ds.create_scalar_index("txt", "INVERTED", replace=True)
            # Row addresses of A's ids 3 and 11, which query_table returns as
            # the `_rowid` of B's rows.
            rows = ds.to_table(columns=["id"], with_row_id=True, filter="id IN (3, 11)")
            addressed[:] = [rows["_rowid"]]

        item = {"name": "item", "nullable": True, "type": {"type": "float32"}}
        schema = {
            "fields": [
                {"name": "id", "nullable": True, "type": {"type": "int64"}},
                {"name": "txt", "nullable": True, "type": {"type": "utf8"}},
                {
                    "name": "vec",
                    "nullable": True,
                    "type": {"type": "fixed_size_list", "length": 2, "fields": [item]},
                },
            ]
        }
        requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request_body = json.loads(body) if body else {}
                path = unquote(urlparse(self.path).path)
                requests.append({"path": path, "body": request_body})

                content_type = "application/json"
                status = 200
                if self.headers.get("x-tenant") != "search":
                    status, response = 403, {"code": 403, "error": "Forbidden", "message": "wrong tenant"}
                elif path.endswith("/table/list"):
                    response: object = {"tables": ["t", "norowid"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    response = {"location": dataset, "schema": schema}
                elif path.endswith("/query") and "/table/" in path:
                    name = path.split("/")[-2].split("$")[-1]
                    metric = "_score" if request_body.get("full_text_query") else "_distance"
                    columns = [("id", pa.array([100, 101], pa.int64()))]
                    if name == "t" and request_body.get("with_row_id"):
                        # Placed mid-batch.
                        columns.append(("_rowid", addressed[0]))
                    columns += [
                        ("txt", pa.array(["b100", "b101"])),
                        ("vec", pa.array([[0.0, 0.0], [1.0, 1.0]], pa.list_(pa.float32(), 2))),
                        (metric, pa.array([0.5, 0.25], pa.float32())),
                    ]
                    response = arrow_stream(pa.table(dict(columns)))
                    content_type = "application/vnd.apache.arrow.stream"
                else:
                    status = 404
                    response = {"message": "Unhandled fixture route", "path": path}

                payload = response if isinstance(response, bytes) else json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = handle_request
            do_POST = handle_request

        fts = "lance_fts('ns.main.t', 'txt', 'w1', k = 10)"
        knn = "lance_vector_search('ns.main.t', 'vec', [0.0, 0.0]::FLOAT[2], k = 2)"
        cases = [
            {
                "name": "fts_rowid",
                "sql": f"SELECT id, _rowid FROM {fts} ORDER BY id;",
                "stdout": "100|3\n101|4294967297\n",
                "with_row_id": True,
            },
            {
                "name": "fts_star_omits_rowid",
                "sql": f"SELECT * FROM {fts} ORDER BY id;",
                "stdout": "100|b100|[0.0, 0.0]|0.5\n101|b101|[1.0, 1.0]|0.25\n",
                "with_row_id": False,
            },
            {
                "name": "knn_rowid_filter_pushed",
                "sql": f"SELECT id FROM {knn} WHERE _rowid = 3;",
                "filter": "`_rowid` = 3",
                "with_row_id": True,
            },
            {
                "name": "missing_rowid_fails",
                "sql": "SELECT _rowid FROM lance_fts('ns.main.norowid', 'txt', 'w1', k = 10);",
                "stderr": "no _rowid column",
                "with_row_id": True,
            },
            {
                "name": "delete_by_fts_rowid",
                "sql": f"DELETE FROM ns.main.t WHERE rowid IN (SELECT _rowid FROM {fts});",
                "removed": [1, 5, 9, 13, 17],
            },
            {
                "name": "delete_by_knn_rowid",
                "sql": f"DELETE FROM ns.main.t WHERE rowid IN (SELECT _rowid FROM {knn});",
                "removed": [0, 1],
            },
            {
                "name": "delete_by_fts_id",
                "sql": f"DELETE FROM ns.main.t WHERE id IN (SELECT id FROM {fts});",
                "removed": [1, 5, 9, 13, 17],
            },
            {
                "name": "prepared_delete_by_fts_rowid",
                "sql": (
                    "PREPARE delete_hits AS DELETE FROM ns.main.t WHERE rowid IN "
                    "(SELECT _rowid FROM lance_fts('ns.main.t', 'txt', $1, k = 10)); "
                    "EXECUTE delete_hits('w1'); EXECUTE delete_hits('w2');"
                ),
                "removed": [1, 2, 5, 6, 9, 10, 13, 14, 17, 18],
            },
        ]

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            prefix = (
                f"LOAD '{args.extension}'; "
                "ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}', "
                "HEADER 'X-Tenant=initial;x-tenant=search'); "
            )
            output: list[dict] = []
            for case in cases:
                reset_dataset()
                start = len(requests)
                result = subprocess.run(
                    [
                        str(args.duckdb),
                        "-unsigned",
                        "-no-agent",
                        "-noheader",
                        "-list",
                        "-c",
                        prefix + case["sql"],
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    timeout=60,
                )
                case_requests = requests[start:]
                queries = [r["body"] for r in case_requests if r["path"].endswith("/query")]
                remaining = set(lance.dataset(dataset).to_table(columns=["id"])["id"].to_pylist())
                removed = sorted(set(range(20)) - remaining)

                if "stderr" in case:
                    passed = result.returncode != 0 and case["stderr"] in result.stderr
                else:
                    passed = result.returncode == 0
                if "stdout" in case:
                    passed = passed and result.stdout == case["stdout"]
                if "filter" in case:
                    passed = passed and any(case["filter"] in (q.get("filter") or "") for q in queries)
                if "with_row_id" in case:
                    flags = [bool(q.get("with_row_id")) for q in queries]
                    passed = passed and bool(flags) and all(flag == case["with_row_id"] for flag in flags)
                if "removed" in case:
                    # The DELETE's search reads the pinned dataset, never query_table.
                    passed = passed and removed == case["removed"] and not queries
                else:
                    passed = passed and not removed
                output.append(
                    {
                        "name": case["name"],
                        "passed": passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "removed_ids": removed,
                        "requests": case_requests,
                    }
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    print(json.dumps(output, indent=2))
    if not all(case["passed"] for case in output):
        raise SystemExit(1)
    print("REST search _rowid checks passed")


if __name__ == "__main__":
    main()
