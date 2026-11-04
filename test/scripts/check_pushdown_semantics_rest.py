"""Check that predicates pushed into REST namespace query_table select DuckDB's rows.

The fixture namespace describes table `t` at a local dataset, and its
query_table answers from that dataset with the request's filter, `k` and
offset, as a Lance server would. A pushed predicate is unparsed to a filter
string there, so it must keep DuckDB's meaning after Lance parses it again;
predicates whose Lance semantics differ must stay in DuckDB.
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

ROWS = [
    (1, "straße"),
    (2, "strasse"),
    (3, "STRASSE"),
    (4, "İstanbul"),
    (5, "a\\_c"),
    (6, "a_c"),
    (7, "ΣΊΣΥΦΟΣ"),
    (8, "σίσυφος"),
    (9, "é1"),
    (10, "abc"),
]


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

    with tempfile.TemporaryDirectory(prefix="lance-pushdown-semantics-rest-") as tmp:
        dataset = str(Path(tmp) / "t.lance")
        lance.write_dataset(
            pa.table(
                {
                    "id": pa.array([r[0] for r in ROWS], pa.int64()),
                    "s": pa.array([r[1] for r in ROWS]),
                }
            ),
            dataset,
        )
        schema = {
            "fields": [
                {"name": "id", "nullable": True, "type": {"type": "int64"}},
                {"name": "s", "nullable": True, "type": {"type": "utf8"}},
            ]
        }
        filters: list[str | None] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request_body = json.loads(body) if body else {}
                path = unquote(urlparse(self.path).path)

                content_type = "application/json"
                status = 200
                if path.endswith("/table/list"):
                    response: object = {"tables": ["t"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    response = {"location": dataset, "schema": schema}
                elif path.endswith("/query") and "/table/" in path:
                    filters.append(request_body.get("filter"))
                    columns = (request_body.get("columns") or {}).get("column_names")
                    table = lance.dataset(dataset).to_table(
                        columns=columns or ["id", "s"],
                        filter=request_body.get("filter"),
                        limit=request_body.get("k") or None,
                        offset=request_body.get("offset"),
                        with_row_id=bool(request_body.get("with_row_id")),
                    )
                    response = arrow_stream(table)
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

        ids = "SELECT string_agg(id::VARCHAR, ',' ORDER BY id) FROM ns.main.t WHERE "
        cases = [
            {
                # Without ESCAPE the backslash is a literal; it is pushed doubled.
                "name": "like_backslash",
                "sql": ids + "s LIKE 'a\\_c';",
                "stdout": "5\n",
                "filter": "`s` LIKE 'a\\\\_c'",
            },
            {
                "name": "upper_not_pushed",
                "sql": ids + "upper(s) = 'STRASSE';",
                "stdout": "2,3\n",
            },
            {
                "name": "ilike_limit_not_pushed",
                "sql": "SELECT count(*) FROM (SELECT id FROM ns.main.t WHERE s ILIKE 'σίσυφος' LIMIT 1);",
                "stdout": "1\n",
            },
            {
                "name": "regex_ascii_class_not_pushed",
                "sql": ids + "NOT regexp_matches(s, '^\\w+$');",
                "stdout": "1,4,5,7,8,9\n",
            },
        ]

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            prefix = (
                f"LOAD '{args.extension}'; "
                "ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
            )
            output: list[dict] = []
            for case in cases:
                filters.clear()
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
                passed = result.returncode == 0 and result.stdout == case["stdout"] and bool(filters)
                if "filter" in case:
                    passed = passed and all(case["filter"] in (f or "") for f in filters)
                else:
                    passed = passed and not any(filters)
                output.append(
                    {
                        "name": case["name"],
                        "passed": passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "filters": list(filters),
                    }
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    print(json.dumps(output, indent=2, ensure_ascii=False))
    if not all(case["passed"] for case in output):
        raise SystemExit(1)
    print("REST pushdown semantics checks passed")


if __name__ == "__main__":
    main()
