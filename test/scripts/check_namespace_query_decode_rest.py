"""Exercise namespace result decoding through REST Arrow IPC file/stream responses.

Run with a release DuckDB CLI and extension. Requires pyarrow; the dataset
fallback case also uses pylance, as do the other REST fixture scripts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import lance
import pyarrow as pa


REPO_ROOT = Path(__file__).resolve().parents[2]


def field(name: str, kind: str, **kwargs: object) -> dict:
    return {"name": name, "nullable": True, "type": {"type": kind, **kwargs}}


SCHEMA = {
    "fields": [
        field("id", "int64"),
        field("txt", "utf8"),
        field("vec", "fixed_size_list", length=2, fields=[field("item", "float32")]),
        field("label", "utf8"),
        field("payload", "binary"),
        field("tags", "list", fields=[field("item", "utf8")]),
        field("day", "date32"),
        field("span", "duration"),
        field("ts", "timestamp"),
        field("half", "float16"),
        field(
            "nested",
            "struct",
            fields=[
                field("s", "timestamp"),
                field("l", "list", fields=[field("item", "timestamp")]),
                field("a", "fixed_size_list", length=2, fields=[field("item", "timestamp")]),
                field(
                    "m",
                    "map",
                    fields=[
                        {
                            **field(
                                "entries",
                                "struct",
                                fields=[{**field("key", "utf8"), "nullable": False}, field("value", "timestamp")],
                            ),
                            "nullable": False,
                        }
                    ],
                ),
            ],
        ),
        {
            **field("u", "fixed_size_binary", length=16),
            "metadata": {"ARROW:extension:name": "arrow.uuid"},
        },
    ]
}


def rows(start: int, count: int, mode: str) -> pa.Table:
    ids = list(range(start, start + count))
    unit = {
        "seconds": "s",
        "milliseconds": "ms",
        "nanos": "ns",
        "lossy": "ns",
        "infinity_ms": "ms",
        "infinity_ns": "ns",
        "overflow": "ms",
        "nested_null": "ns",
    }.get(mode, "us")
    scale = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}[unit]
    zone = "UTC" if mode in ("timezone", "timezone_ns") else None
    if mode == "timezone_ns":
        unit, scale = "ns", 1000000000
    timestamps = [(i + 1) * scale + (1 if mode == "lossy" else 0) for i in ids]
    if mode.startswith("infinity"):
        timestamps = [-(2**63) + 1 if i % 2 else 2**63 - 1 for i in ids]
    elif mode == "overflow":
        timestamps = [2**63 - 2 for _ in ids]
    elif mode == "nested_null":
        timestamps[0] = 1
    timestamp_type = pa.timestamp(unit, zone)
    nested_type = pa.struct(
        [
            ("s", timestamp_type),
            ("l", pa.list_(timestamp_type)),
            ("a", pa.list_(timestamp_type, 2)),
            ("m", pa.map_(pa.string(), timestamp_type)),
        ]
    )
    nested = pa.array([{"s": t, "l": [t, None], "a": [t, None], "m": [("x", t)]} for t in timestamps], nested_type)
    if mode == "nested_null":
        nested = pa.StructArray.from_arrays(
            [nested.field(i) for i in range(4)],
            fields=list(nested_type),
            mask=pa.array([i == 0 for i in ids]),
        )
    return pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "txt": pa.array([f"row-{i}" for i in ids], pa.large_string()),
            "vec": pa.array([[float(i), 0.0] for i in ids], pa.list_(pa.float32(), 2)),
            "label": pa.array([f"label-{i}" if i % 3 else None for i in ids]).dictionary_encode(),
            "payload": pa.array([f"blob-{i}".encode() for i in ids], pa.large_binary()),
            "tags": pa.array([[f"tag-{i}", None] for i in ids], pa.large_list(pa.large_string())),
            "day": pa.array([datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in ids], pa.date64()),
            "span": pa.array([i + 1 for i in ids], pa.duration("s")),
            "ts": pa.array(timestamps, timestamp_type),
            "half": pa.array([1.5 if i % 2 else None for i in ids], pa.float16()),
            "nested": nested,
            "u": pa.array([bytes.fromhex("00112233445566778899aabbccddeeff") for _ in ids], pa.uuid()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=REPO_ROOT / "build/release/duckdb")
    parser.add_argument(
        "--extension", type=Path, default=REPO_ROOT / "build/release/extension/lance/lance.duckdb_extension"
    )
    args = parser.parse_args()
    requests: list[dict] = []
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="lance-query-decode-") as tmp:
        dataset = str(Path(tmp) / "fallback.lance")
        lance.write_dataset(rows(0, 2, "normal"), dataset)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request = json.loads(body) if body else {}
                path = unquote(urlparse(self.path).path)
                name = path.split("/")[-2].split("$")[-1]
                content_type = "application/json"
                if path.endswith("/table/list"):
                    response: object = {"tables": ["normal"]}
                elif path.endswith("/list"):
                    response = {"namespaces": []}
                elif path.endswith("/describe"):
                    response = {"location": dataset if name == "fallback" else str(Path(tmp) / "inaccessible.lance")}
                    if name != "fallback":
                        response["schema"] = SCHEMA
                elif path.endswith("/query"):
                    requests.append({"name": name, **request})
                    search = bool(request.get("full_text_query") or request.get("vector"))
                    metric = "_score" if request.get("full_text_query") else "_distance"
                    chunks = [rows(0, 2, name)]
                    if name == "many":
                        chunks = [rows(0, 2500, name), rows(2500, 3, name)]
                    tables = []
                    for table in chunks:
                        if search:
                            table = table.append_column(metric, pa.array([0.5] * table.num_rows, pa.float32()))
                        if request.get("with_row_id"):
                            table = table.append_column("_rowid", pa.array(table["id"].to_pylist(), pa.uint64()))
                        columns = (request.get("columns") or {}).get("column_names")
                        if columns and not search:
                            table = table.select(columns)
                        if name == "missing":
                            table = table.drop(["txt"])
                        elif name == "duplicate":
                            table = table.append_column("txt", table["txt"])
                        elif name == "wrong_type":
                            table = table.set_column(
                                table.schema.get_field_index("txt"), "txt", pa.array([42] * table.num_rows, pa.int64())
                            )
                        elif name == "wrong_metric" and search:
                            table = table.set_column(
                                table.schema.get_field_index(metric),
                                metric,
                                pa.array([0.5] * table.num_rows, pa.float64()),
                            )
                        elif name == "extra":
                            table = table.append_column("unrequested", pa.nulls(table.num_rows))
                        if name == "reordered":
                            table = table.select(list(reversed(table.column_names)))
                        tables.append(table)
                    sink = pa.BufferOutputStream()
                    factory = pa.ipc.new_file if name == "file" else pa.ipc.new_stream
                    with factory(sink, tables[0].schema) as writer:
                        if name != "empty":
                            for table in tables:
                                writer.write_table(table)
                    response = sink.getvalue().to_pybytes()
                    content_type = "application/vnd.apache.arrow.stream"
                else:
                    self.send_error(404)
                    return
                payload = response if isinstance(response, bytes) else json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = handle_request
            do_POST = handle_request

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prefix = (
            f"LOAD '{args.extension}'; SET TimeZone = 'America/Los_Angeles'; "
            "ATTACH 'default' AS ns "
            f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
        )

        def check(
            name: str,
            kind: str,
            projection: str,
            expected: object = None,
            error: str = "",
            suffix: str = "",
            prepared: bool = False,
        ) -> None:
            source = f"ns.main.{name}"
            if kind == "fts":
                source = f"lance_fts('{source}', 'txt', 'row', k = 3000)"
            elif kind == "knn":
                source = f"lance_vector_search('{source}', 'vec', [0.0, 0.0]::FLOAT[2], k = 3000)"
            sql = f"SELECT {projection} FROM {source} {suffix}"
            if prepared:
                sql = f"PREPARE q AS {sql}; EXECUTE q;"
            start = len(requests)
            result = subprocess.run(
                [str(args.duckdb), "-unsigned", "-no-agent", "-json", "-c", prefix + sql],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                timeout=60,
                env={**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
            )
            if error:
                passed = result.returncode != 0 and "Invalid Input Error:" in result.stderr and error in result.stderr
            else:
                try:
                    actual = json.loads(result.stdout) if result.stdout.strip() else []
                except json.JSONDecodeError:
                    actual = result.stdout
                passed = result.returncode == 0 and actual == expected
            passed = passed and len(requests) > start
            results.append(
                {
                    "case": f"{name}/{kind}/{projection}/{prepared}",
                    "passed": passed,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

        try:
            for kind in ("scan", "fts", "knn"):
                for mode in ("normal", "file", "reordered", "extra", "fallback"):
                    check(
                        mode,
                        kind,
                        "id, txt",
                        [{"id": 0, "txt": "row-0"}, {"id": 1, "txt": "row-1"}],
                        suffix="ORDER BY id",
                    )
                check(
                    "normal",
                    kind,
                    "label, id, hex(payload) AS payload, tags, day::VARCHAR AS day, span::VARCHAR AS span, u::VARCHAR AS u",
                    [
                        {
                            "label": None,
                            "id": 0,
                            "payload": "626C6F622D30",
                            "tags": ["tag-0", None],
                            "day": "2020-01-01",
                            "span": "00:00:01",
                            "u": "00112233-4455-6677-8899-aabbccddeeff",
                        },
                        {
                            "label": "label-1",
                            "id": 1,
                            "payload": "626C6F622D31",
                            "tags": ["tag-1", None],
                            "day": "2020-01-02",
                            "span": "00:00:02",
                            "u": "00112233-4455-6677-8899-aabbccddeeff",
                        },
                    ],
                    suffix="ORDER BY id",
                )
                check("normal", kind, "txt", [{"txt": "row-0"}, {"txt": "row-1"}], suffix="ORDER BY id", prepared=True)
                check("normal", kind, "half", [{"half": None}, {"half": 1.5}], suffix="ORDER BY id")
                if kind != "scan":
                    check(
                        "reordered", kind, "_rowid::BIGINT AS rid", [{"rid": 0}, {"rid": 1}], suffix="ORDER BY _rowid"
                    )
                check("empty", kind, "txt", [])
                for mode in ("seconds", "milliseconds", "nanos", "timezone", "timezone_ns"):
                    check(
                        mode,
                        kind,
                        "epoch_us(ts) AS micros",
                        [{"micros": 1000000}, {"micros": 2000000}],
                        suffix="ORDER BY id",
                    )
                    check(
                        mode,
                        kind,
                        "epoch_us(nested.s) AS s, epoch_us(nested.l[1]) AS l, epoch_us(nested.a[1]) AS a, epoch_us(nested.m['x']) AS m",
                        [
                            {"s": 1000000, "l": 1000000, "a": 1000000, "m": 1000000},
                            {"s": 2000000, "l": 2000000, "a": 2000000, "m": 2000000},
                        ],
                        suffix="ORDER BY id",
                    )
                check(
                    "nested_null",
                    kind,
                    "epoch_us(nested.s) AS s, epoch_us(nested.l[1]) AS l, epoch_us(nested.a[1]) AS a, epoch_us(nested.m['x']) AS m",
                    [
                        {"s": None, "l": None, "a": None, "m": None},
                        {"s": 2000000, "l": 2000000, "a": 2000000, "m": 2000000},
                    ],
                    suffix="ORDER BY id",
                )
                for mode in ("infinity_ms", "infinity_ns"):
                    check(
                        mode, kind, "ts::VARCHAR AS ts", [{"ts": "infinity"}, {"ts": "-infinity"}], suffix="ORDER BY id"
                    )
                check(
                    "overflow",
                    kind,
                    "ts",
                    error='cannot represent column "ts" as TIMESTAMP: Could not convert Timestamp(MS) to Timestamp(US)',
                )
                check(
                    "lossy",
                    kind,
                    "ts",
                    error='cannot represent column "ts" as TIMESTAMP: cannot convert nanosecond timestamp without losing precision',
                )
                check("lossy", kind, "nested", error="without losing precision")
                check("missing", kind, "txt", error="missing column \"txt\"")
                check("duplicate", kind, "txt", error="duplicate column \"txt\"")
                check("wrong_type", kind, "txt", error='column "txt" has type BIGINT, expected VARCHAR')
                check(
                    "many",
                    kind,
                    "sum(length(txt))::BIGINT AS text_size, count(label) AS labels, count(*) AS n",
                    [
                        {
                            "text_size": sum(len(f"row-{i}") for i in range(2503)),
                            "labels": sum(i % 3 != 0 for i in range(2503)),
                            "n": 2503,
                        }
                    ],
                )
                if kind != "scan":
                    metric = "_score" if kind == "fts" else "_distance"
                    check("wrong_metric", kind, metric, error=f'column "{metric}" has type DOUBLE, expected FLOAT')
                check("reordered", kind, "label", [{"label": "label-1"}], suffix="WHERE sin(id::DOUBLE) > 0")
            print(json.dumps(results, indent=2))
            if not all(result["passed"] for result in results):
                raise SystemExit("NAMESPACE_DECODE_REGRESSION")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    main()
