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
        field("fixed_labels", "fixed_size_list", length=2, fields=[field("item", "utf8")]),
        field("plain_labels", "list", fields=[field("item", "utf8")]),
        field("empty_plain_labels", "list", fields=[field("item", "utf8")]),
        field("outer_fixed", "fixed_size_list", length=2, fields=[field("item", "int64")]),
        field("outer_labels", "fixed_size_list", length=2, fields=[field("item", "utf8")]),
        field("outer_struct", "struct", fields=[field("n", "int64"), field("s", "utf8")]),
        field(
            "empty_lists", "list", fields=[field("item", "fixed_size_list", length=2, fields=[field("item", "utf8")])]
        ),
        field(
            "boxed_labels",
            "struct",
            fields=[
                field("s", "fixed_size_list", length=2, fields=[field("item", "utf8")]),
                field("l", "list", fields=[field("item", "fixed_size_list", length=2, fields=[field("item", "utf8")])]),
                field(
                    "m",
                    "map",
                    fields=[
                        {
                            **field(
                                "entries",
                                "struct",
                                fields=[
                                    {**field("key", "utf8"), "nullable": False},
                                    field("value", "fixed_size_list", length=2, fields=[field("item", "utf8")]),
                                ],
                            ),
                            "nullable": False,
                        }
                    ],
                ),
            ],
        ),
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
    label_values = pa.array([value for i in ids for value in (["b", "c"] if i % 2 else ["a", None])], pa.string())
    fixed_labels = pa.FixedSizeListArray.from_arrays(
        label_values.dictionary_encode().cast(pa.dictionary(pa.int8(), pa.string())), 2
    )
    offsets = pa.array(range(count + 1), pa.int32())
    keys = pa.array([1 if i % 2 else None for i in ids], pa.int8())
    outer_fixed = pa.DictionaryArray.from_arrays(keys, pa.array([[10, 20], [30, 40]], pa.list_(pa.int64(), 2)))
    outer_labels = pa.DictionaryArray.from_arrays(keys, fixed_labels.slice(0, 2))
    outer_struct = pa.DictionaryArray.from_arrays(keys, pa.array([{"n": 7, "s": "seven"}, {"n": 8, "s": "eight"}]))
    empty_lists = pa.ListArray.from_arrays(pa.array([0] * (count + 1), pa.int32()), fixed_labels.slice(0, 0))
    boxed_labels = pa.StructArray.from_arrays(
        [
            fixed_labels,
            pa.ListArray.from_arrays(offsets, fixed_labels),
            pa.MapArray.from_arrays(offsets, pa.array(["x"] * count, pa.string()), fixed_labels),
        ],
        names=["s", "l", "m"],
    )
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
            "label": pa.array([f"label-{i}" if i % 3 else None for i in ids], pa.string()).dictionary_encode(),
            "payload": pa.array([f"blob-{i}".encode() for i in ids], pa.large_binary()),
            "tags": pa.array([[f"tag-{i}", None] for i in ids], pa.large_list(pa.large_string())),
            "day": pa.array([datetime.date(2020, 1, 1) + datetime.timedelta(days=i) for i in ids], pa.date64()),
            "span": pa.array([i + 1 for i in ids], pa.duration("s")),
            "ts": pa.array(timestamps, timestamp_type),
            "half": pa.array([1.5 if i % 2 else None for i in ids], pa.float16()),
            "fixed_labels": fixed_labels,
            "plain_labels": pa.ListArray.from_arrays(
                pa.array([i * 2 for i in range(count + 1)], pa.int32()), label_values.dictionary_encode()
            ),
            "empty_plain_labels": pa.ListArray.from_arrays(
                pa.array([0] * (count + 1), pa.int32()), label_values.dictionary_encode()
            ),
            "outer_fixed": outer_fixed,
            "outer_labels": outer_labels,
            "outer_struct": outer_struct,
            "empty_lists": empty_lists,
            "boxed_labels": boxed_labels,
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
        lance.write_dataset(
            rows(0, 2, "normal").drop(
                [
                    "fixed_labels",
                    "boxed_labels",
                    "outer_fixed",
                    "outer_labels",
                    "outer_struct",
                    "empty_lists",
                    "plain_labels",
                    "empty_plain_labels",
                ]
            ),
            dataset,
        )

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
                    mode = name.removeprefix("empty_").removeprefix("file_")
                    search = bool(request.get("full_text_query") or request.get("vector"))
                    metric = "_score" if request.get("full_text_query") else "_distance"
                    chunks = [rows(0, 2, mode)]
                    if name == "many":
                        chunks = [rows(0, 2500, name), rows(2500, 3, name)]
                    elif name == "leading_empty":
                        chunks = [rows(0, 0, "normal"), rows(0, 2, "normal")]
                    tables = []
                    for table in chunks:
                        if search:
                            table = table.append_column(metric, pa.array([0.5] * table.num_rows, pa.float32()))
                        if request.get("with_row_id"):
                            table = table.append_column("_rowid", pa.array(table["id"].to_pylist(), pa.uint64()))
                        columns = (request.get("columns") or {}).get("column_names")
                        if columns and not search:
                            table = table.select(columns)
                        if mode == "missing":
                            table = table.drop(["txt"])
                        elif mode == "duplicate":
                            table = table.append_column("txt", table["txt"])
                        elif mode == "wrong_type":
                            table = table.set_column(
                                table.schema.get_field_index("txt"), "txt", pa.array([42] * table.num_rows, pa.int64())
                            )
                        elif mode == "wrong_metric" and search:
                            table = table.set_column(
                                table.schema.get_field_index(metric),
                                metric,
                                pa.array([0.5] * table.num_rows, pa.float64()),
                            )
                        elif mode == "missing_rowid":
                            table = table.drop(["_rowid"])
                        elif mode == "wrong_rowid":
                            table = table.set_column(table.schema.get_field_index("_rowid"), "_rowid", table["id"])
                        elif mode == "extra":
                            table = table.append_column("unrequested", pa.nulls(table.num_rows))
                            values = pa.array(["unused"] * table.num_rows).dictionary_encode()
                            lists = pa.ListArray.from_arrays(pa.array(range(table.num_rows + 1), pa.int32()), values)
                            union = pa.UnionArray.from_dense(
                                pa.array([0] * table.num_rows, pa.int8()),
                                pa.array(range(table.num_rows), pa.int32()),
                                [lists],
                                field_names=["labels"],
                            )
                            table = table.append_column("unrequested_union", union)
                        if name == "reordered":
                            table = table.select(list(reversed(table.column_names)))
                        tables.append(table)
                    sink = pa.BufferOutputStream()
                    factory = pa.ipc.new_file if name == "file" or name.startswith("empty_file") else pa.ipc.new_stream
                    with factory(sink, tables[0].schema) as writer:
                        if name != "empty" and not name.startswith("empty_"):
                            for table in tables:
                                if table.num_rows == 0:
                                    writer.write_batch(
                                        pa.RecordBatch.from_arrays(
                                            [column.combine_chunks() for column in table.columns], schema=table.schema
                                        )
                                    )
                                else:
                                    writer.write_table(table)
                    response = sink.getvalue().to_pybytes()
                    if name == "leading_empty":
                        assert [batch.num_rows for batch in pa.ipc.open_stream(response)] == [0, 2]
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
            error_type: str = "Invalid Input Error:",
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
                passed = result.returncode != 0 and error_type in result.stderr and error in result.stderr
            else:
                try:
                    actual = json.loads(result.stdout) if result.stdout.strip() else []
                except json.JSONDecodeError:
                    actual = result.stdout
                passed = result.returncode == 0 and actual == expected
            passed = passed and len(requests) > start
            if kind != "scan":
                passed = passed and all(not request.get("columns") for request in requests[start:])
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
            # Direct dataset reads through REST discovery still pin a dataset
            # snapshot and can use the native late-materialization rewrite.
            endpoint = f"http://127.0.0.1:{server.server_port}"
            for source in ("ns.main.fallback", f"__lance_namespace_scan('{endpoint}', 'fallback', '$')"):
                start = len(requests)
                sql = f"SELECT id, txt FROM {source} ORDER BY id DESC LIMIT 1;"
                for explain in (False, True):
                    result = subprocess.run(
                        [
                            str(args.duckdb),
                            "-unsigned",
                            "-no-agent",
                            "-json",
                            "-c",
                            prefix
                            + "SET lance_namespace_query_table = false; "
                            + ("EXPLAIN (ANALYZE, FORMAT JSON) " if explain else "")
                            + sql,
                        ],
                        cwd=REPO_ROOT,
                        text=True,
                        capture_output=True,
                        timeout=60,
                        env={**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
                    )
                    assert result.returncode == 0, result.stderr
                    if explain:
                        assert '"Join Type": "SEMI"' in result.stdout, result.stdout
                        assert '"Lance Rows Out": "1"' in result.stdout, result.stdout
                    else:
                        assert json.loads(result.stdout) == [{"id": 1, "txt": "row-1"}], result.stdout
                result = subprocess.run(
                    [
                        str(args.duckdb),
                        "-unsigned",
                        "-no-agent",
                        "-json",
                        "-c",
                        prefix
                        + "SET lance_namespace_query_table = false; SET disabled_optimizers = 'filter_pushdown'; "
                        + f"SELECT id, txt FROM {source} WHERE rowid IN (0, 0, 1) ORDER BY id LIMIT 2;",
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    env={**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"},
                )
                assert result.returncode == 0, result.stderr
                assert json.loads(result.stdout) == [{"id": 0, "txt": "row-0"}] * 2, result.stdout
                assert len(requests) == start, "direct reads must not call query_table"
            # query_table does not pin a dataset snapshot across requests. TopN
            # must remain one remote scan, with ordering and limiting in DuckDB.
            for mode in ("normal", "file"):
                start = len(requests)
                check(
                    mode,
                    "scan",
                    "id, txt",
                    [{"id": 1, "txt": "row-1"}],
                    suffix="ORDER BY id DESC LIMIT 1",
                )
                assert len(requests) == start + 1, "TopN must use one query_table snapshot"
                assert not requests[-1].get("with_row_id"), "remote TopN must not request materialization row ids"
            for kind in ("scan", "fts", "knn"):
                for mode in ("normal", "file", "reordered", "extra", "fallback", "leading_empty"):
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
                check(
                    "normal",
                    kind,
                    "fixed_labels AS a, boxed_labels.s AS s, boxed_labels.l[1] AS l, boxed_labels.m['x'] AS m",
                    [{key: values for key in ("a", "s", "l", "m")} for values in (["a", None], ["b", "c"])],
                    suffix="ORDER BY id",
                )
                if kind != "scan":
                    check(
                        "reordered", kind, "_rowid::BIGINT AS rid", [{"rid": 0}, {"rid": 1}], suffix="ORDER BY _rowid"
                    )
                check("empty", kind, "txt", [])
                check(
                    "normal",
                    kind,
                    "plain_labels, empty_plain_labels",
                    [
                        {"plain_labels": ["a", None], "empty_plain_labels": []},
                        {"plain_labels": ["b", "c"], "empty_plain_labels": []},
                    ],
                    suffix="ORDER BY id",
                )
                check(
                    "normal",
                    kind,
                    "outer_fixed, outer_labels, outer_struct, empty_lists",
                    [
                        {"outer_fixed": None, "outer_labels": None, "outer_struct": None, "empty_lists": []},
                        {
                            "outer_fixed": [30, 40],
                            "outer_labels": ["b", "c"],
                            "outer_struct": {"n": 8, "s": "eight"},
                            "empty_lists": [],
                        },
                    ],
                    suffix="ORDER BY id",
                )
                for container in ("empty", "empty_file"):
                    check(container, kind, "plain_labels, empty_plain_labels", [])
                    check(container, kind, "outer_fixed, outer_labels, outer_struct, empty_lists", [])
                    check(container, kind, "fixed_labels, boxed_labels", [])
                    check(container, kind, "label, payload, tags, nested", [])
                    check(container + "_nanos", kind, "nested", [])
                    check(container + "_missing", kind, "txt", error='missing column "txt"')
                    check(container + "_duplicate", kind, "txt", error='duplicate column "txt"')
                    check(
                        container + "_wrong_type", kind, "txt", error='column "txt" has type BIGINT, expected VARCHAR'
                    )
                    if kind != "scan":
                        check(container, kind, "_rowid", [])
                        check(
                            container + "_missing_rowid",
                            kind,
                            "_rowid",
                            error="response has no _rowid column",
                            error_type="IO Error:",
                        )
                        check(
                            container + "_wrong_rowid",
                            kind,
                            "_rowid",
                            error="response has _rowid of type Int64, expected UInt64",
                            error_type="IO Error:",
                        )
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
