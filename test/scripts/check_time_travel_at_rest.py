"""Check that AT TIMESTAMP keeps the dataset resolved by the REST namespace.

The first two describe calls identify dataset A. A later describe would identify
dataset B, whose version 1 has a different row count. The historical read must
return A's rows without issuing that third describe.
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


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = {"fields": [{"name": "id", "nullable": True, "type": {"type": "int64"}}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=REPO_ROOT / "build" / "release" / "duckdb")
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    def duckdb(sql: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [
                str(args.duckdb),
                "-unsigned",
                "-no-agent",
                "-noheader",
                "-list",
                "-c",
                sql,
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )

    with tempfile.TemporaryDirectory(prefix="lance-at-rest-") as tmp:
        dataset_a = str(Path(tmp) / "a.lance")
        dataset_b = str(Path(tmp) / "b.lance")
        setup_a = duckdb(f"LOAD '{args.extension}'; COPY (SELECT 1::BIGINT AS id) " f"TO '{dataset_a}' (FORMAT lance);")
        setup_b = duckdb(
            f"LOAD '{args.extension}'; COPY (SELECT i::BIGINT AS id FROM range(10) r(i)) "
            f"TO '{dataset_b}' (FORMAT lance);"
        )
        describes: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = unquote(urlparse(self.path).path)
                status = 200
                if path.endswith("/table/list"):
                    response: object = {"tables": ["t"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    dataset = dataset_a if len(describes) < 2 else dataset_b
                    describes.append(dataset)
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

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = duckdb(
                f"LOAD '{args.extension}'; ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
                "SELECT count(*) FROM ns.main.t AT (TIMESTAMP => TIMESTAMP '2999-01-01');"
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    passed = (
        setup_a.returncode == 0
        and setup_b.returncode == 0
        and result.returncode == 0
        and result.stdout.strip() == "1"
        and len(describes) <= 2
    )
    print(
        json.dumps(
            {
                "passed": passed,
                "rows": result.stdout.strip(),
                "describe_locations": describes,
                "stderr": setup_a.stderr + setup_b.stderr + result.stderr,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)
    print("REST AT TIMESTAMP dataset identity check passed")


if __name__ == "__main__":
    main()
