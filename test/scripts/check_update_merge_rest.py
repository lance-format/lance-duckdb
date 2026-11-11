"""Check UPDATE and MERGE on a REST namespace table against a local fixture.

The fixture namespace describes table `t` at a local dataset and has no
`query_table` route. The DML target scan must read the dataset pinned for the
statement instead of `query_table`, and the staged transaction must commit to
that handle, so each statement changes exactly the rows its WHERE or ON clause
selects in the dataset and sends no `query_table` request.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = {
    "fields": [
        {"name": "id", "nullable": True, "type": {"type": "int64"}},
        {"name": "v", "nullable": True, "type": {"type": "int64"}},
    ]
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb", type=Path, default=REPO_ROOT / "build" / "release" / "duckdb"
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    def duckdb(sql: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(args.duckdb), "-unsigned", "-no-agent", "-noheader", "-list", "-c", sql],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )

    with tempfile.TemporaryDirectory(prefix="lance-dml-rest-") as tmp:
        dataset = str(Path(tmp) / "a.lance")
        requests: list[dict] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = unquote(urlparse(self.path).path)
                requests.append({"path": path, "body": json.loads(body) if body else {}})
                status = 200
                if path.endswith("/table/list"):
                    response: object = {"tables": ["t"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    response = {"location": dataset, "schema": SCHEMA}
                else:
                    status = 404
                    response = {"message": "Unhandled fixture route", "path": path}
                payload = json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = handle_request
            do_POST = handle_request

        cases = [
            {
                "name": "update",
                "sql": "UPDATE ns.main.t SET v = -1 WHERE id IN (2, 7);",
                "rows": "0:0,1:1,2:-1,3:3,4:4,5:5,6:6,7:-1,8:8,9:9",
            },
            {
                "name": "merge_update",
                "sql": (
                    "MERGE INTO ns.main.t USING (SELECT 3 AS id) s ON t.id = s.id "
                    "WHEN MATCHED THEN UPDATE SET v = -3;"
                ),
                "rows": "0:0,1:1,2:2,3:-3,4:4,5:5,6:6,7:7,8:8,9:9",
            },
            {
                "name": "merge_delete_insert",
                "sql": (
                    "MERGE INTO ns.main.t USING (VALUES (4), (20)) s(id) ON t.id = s.id "
                    "WHEN MATCHED THEN DELETE "
                    "WHEN NOT MATCHED THEN INSERT VALUES (s.id, s.id * 10);"
                ),
                "rows": "0:0,1:1,2:2,3:3,5:5,6:6,7:7,8:8,9:9,20:200",
            },
        ]

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        output: list[dict] = []
        try:
            for case in cases:
                shutil.rmtree(dataset, ignore_errors=True)
                setup = duckdb(
                    f"LOAD '{args.extension}'; "
                    f"COPY (SELECT i::BIGINT AS id, i::BIGINT AS v FROM range(10) r(i)) "
                    f"TO '{dataset}' (FORMAT lance);"
                )
                start = len(requests)
                prefix = (
                    f"LOAD '{args.extension}'; ATTACH 'default' AS ns "
                    f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
                )
                result = duckdb(prefix + case["sql"])
                case_requests = requests[start:]
                queries = [r for r in case_requests if r["path"].endswith("/query")]
                check = duckdb(
                    f"LOAD '{args.extension}'; "
                    f"SELECT string_agg(id::VARCHAR || ':' || v::VARCHAR, ',' ORDER BY id) "
                    f"FROM '{dataset}';"
                )
                passed = (
                    setup.returncode == 0
                    and result.returncode == 0
                    and check.stdout.strip() == case["rows"]
                    and not queries
                )
                output.append(
                    {
                        "name": case["name"],
                        "passed": passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr + setup.stderr,
                        "rows": check.stdout.strip(),
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
    print("REST UPDATE and MERGE checks passed")


if __name__ == "__main__":
    main()
