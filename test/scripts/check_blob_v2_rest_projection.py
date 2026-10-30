"""Check REST query_table fallback against the columns actually projected.

A namespace table may advertise a Blob schema even when a statement only reads
ordinary columns. Those statements must remain on query_table without opening
the physical dataset. Blob projections must still use the native dataset path.
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


def arrow_field(
    name: str,
    kind: str,
    children: list[dict] | None = None,
    metadata: dict[str, str] | None = None,
) -> dict:
    data_type = {"type": kind}
    if children is not None:
        data_type["fields"] = children
    result = {"name": name, "nullable": True, "type": data_type}
    if metadata:
        result["metadata"] = metadata
    return result


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

    with tempfile.TemporaryDirectory(prefix="lance-blob-rest-") as tmp:
        root = Path(tmp)
        blob_path = root / "blobs.lance"
        blob_storage = pa.struct([("data", pa.large_binary()), ("uri", pa.string())])
        blob_metadata = {"ARROW:extension:name": "lance.blob.v2"}
        blob_arrow_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("payload", blob_storage, metadata=blob_metadata),
            ]
        )
        lance.write_dataset(
            pa.Table.from_arrays(
                [
                    pa.array([1, 2]),
                    pa.array(
                        [{"data": b"abc", "uri": None}, None],
                        type=blob_storage,
                    ),
                ],
                schema=blob_arrow_schema,
            ),
            blob_path,
            data_storage_version="2.2",
        )

        blob_schema = {
            "fields": [
                arrow_field("id", "int64"),
                arrow_field(
                    "payload",
                    "struct",
                    [
                        arrow_field("data", "large_binary"),
                        arrow_field("uri", "utf8"),
                    ],
                    blob_metadata,
                ),
            ]
        }
        ordinary = pa.table(
            {
                "id": pa.array([1]),
                "payload": pa.array([b"plain"], type=pa.binary()),
            }
        )
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
                if path.endswith("/table/list"):
                    response: object = {"tables": ["blobs", "ordinary"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    name = path.split("/")[-2].split("$")[-1]
                    response = {
                        "location": str(blob_path) if name == "blobs" else str(root / "must-not-open.lance"),
                        "version": 1,
                        "schema": blob_schema,
                    }
                elif path.endswith("/query") and "/ordinary/" in path:
                    columns = request_body.get("columns")
                    if isinstance(columns, dict):
                        columns = columns.get("column_names")
                    selected = ordinary.select(columns) if isinstance(columns, list) else ordinary
                    sink = pa.BufferOutputStream()
                    with pa.ipc.new_stream(sink, selected.schema) as writer:
                        writer.write_table(selected)
                    response = sink.getvalue().to_pybytes()
                    content_type = "application/vnd.apache.arrow.stream"
                else:
                    status = 404
                    response = {
                        "message": "Unhandled fixture route",
                        "path": path,
                    }

                payload = response if isinstance(response, bytes) else json.dumps(response).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = handle_request
            do_POST = handle_request

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            prefix = (
                f"LOAD '{args.extension}'; "
                "ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
            )
            cases = [
                (
                    "ordinary_projection",
                    "SELECT id FROM ns.main.ordinary;",
                    "1\n",
                ),
                (
                    "blob_projection",
                    "SELECT id,hex(payload) FROM ns.main.blobs ORDER BY id;",
                    "1|616263\n2|NULL\n",
                ),
            ]
            output: list[dict] = []
            for name, sql, expected in cases:
                start = len(requests)
                result = subprocess.run(
                    [
                        str(args.duckdb),
                        "-unsigned",
                        "-no-agent",
                        "-noheader",
                        "-list",
                        "-nullvalue",
                        "NULL",
                        "-c",
                        prefix + sql,
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                case_requests = requests[start:]
                passed = result.returncode == 0 and result.stdout == expected
                if name == "ordinary_projection":
                    passed = passed and any(
                        request["path"].endswith("/query")
                        and request["body"].get("columns", {}).get("column_names") == ["id"]
                        for request in case_requests
                    )
                output.append(
                    {
                        "name": name,
                        "passed": passed,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
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
    print("REST Blob schema projection fallback checks passed")


if __name__ == "__main__":
    main()
