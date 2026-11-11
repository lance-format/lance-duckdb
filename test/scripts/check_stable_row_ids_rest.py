"""Check that CREATE TABLE and CTAS create stable row id datasets through a REST namespace.

The fixture namespace reports no table, hands out a local location on
declare_table, and describes the table after that. The extension's own writer
creates the dataset there. A second fragment appended afterwards tells stable
row ids from row addresses: a stable `_rowid` counts rows, an address carries
the fragment id in its high 32 bits.
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
OPTION = "WITH (enable_stable_row_ids = {})"
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
            [str(args.duckdb), "-unsigned", "-no-agent", "-dark-mode", "-list", "-noheader", "-c", sql],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=120,
        )

    with tempfile.TemporaryDirectory(prefix="lance-stable-rest-") as tmp:
        dataset = str(Path(tmp) / "t.lance")
        declared = {"done": False}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                status = 200
                path = unquote(urlparse(self.path).path)
                if path.endswith("/table/list"):
                    response: object = {"tables": []}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/declare"):
                    declared["done"] = True
                    response = {"location": dataset}
                elif path.endswith("/describe") and declared["done"]:
                    response = {"location": dataset, "schema": SCHEMA}
                else:
                    # Code 4 is the namespace spec's TableNotFound.
                    status, response = 404, {"code": 4, "error": "table not found"}
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
        failures: list[str] = []
        try:
            prefix = (
                f"LOAD '{args.extension}'; ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
            )
            cases = []
            for stable in ("true", "false"):
                option = OPTION.format(stable)
                cases += [
                    (f"create_table_{stable}", f"CREATE TABLE ns.main.t (id BIGINT) {option};", stable),
                    (f"ctas_{stable}", f"CREATE TABLE ns.main.t {option} AS SELECT 0::BIGINT AS id;", stable),
                ]
            for name, sql, stable in cases:
                shutil.rmtree(dataset, ignore_errors=True)
                declared["done"] = False
                created = duckdb(prefix + sql)
                # CTAS writes one row, so id 2 is the third fragment; a
                # schema-only table is empty, so it is the second.
                appended = duckdb(
                    f"LOAD '{args.extension}'; "
                    f"COPY (SELECT 1::BIGINT AS id) TO '{dataset}' (FORMAT lance, mode 'append'); "
                    f"COPY (SELECT 2::BIGINT AS id) TO '{dataset}' (FORMAT lance, mode 'append'); "
                    f"SELECT _rowid >> 32 FROM '{dataset}' WHERE id = 2;"
                )
                fragment = appended.stdout.strip()
                # Stable ids count rows, so the high half is always 0.
                expected = "0" if stable == "true" else ("2" if "AS SELECT" in sql else "1")
                if created.returncode or appended.returncode or fragment != expected:
                    failures.append(
                        f"{name}: expected {expected}, got {fragment!r}: "
                        f"{created.stderr}{appended.stderr}"
                    )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    if failures:
        print("\n".join(failures))
        raise SystemExit(1)
    print("REST stable row id checks passed")


if __name__ == "__main__":
    main()
