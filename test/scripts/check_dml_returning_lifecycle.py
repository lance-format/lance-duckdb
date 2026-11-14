"""Compare early result destruction with DuckDB and check bounded RETURNING failures."""

from __future__ import annotations

import argparse
import ctypes as c
import json
import tempfile
from pathlib import Path


class Result(c.Structure):
    # duckdb_result from the pinned duckdb.h; access data through the C API.
    _fields_ = [
        ("columns", c.c_uint64),
        ("rows", c.c_uint64),
        ("changed", c.c_uint64),
        ("column_data", c.c_void_p),
        ("error", c.c_void_p),
        ("internal", c.c_void_p),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    lib = c.CDLL(str(args.library.resolve()))
    ptr = c.c_void_p

    def bind(name, returns, *arguments):
        function = getattr(lib, "duckdb_" + name)
        function.restype, function.argtypes = returns, arguments
        return function

    open_db = bind("open", c.c_int, c.c_char_p, c.POINTER(ptr))
    connect = bind("connect", c.c_int, ptr, c.POINTER(ptr))
    close = bind("close", None, c.POINTER(ptr))
    disconnect = bind("disconnect", None, c.POINTER(ptr))
    query = bind("query", c.c_int, ptr, c.c_char_p, c.POINTER(Result))
    destroy = bind("destroy_result", None, c.POINTER(Result))
    error = bind("result_error", c.c_char_p, c.POINTER(Result))
    error_type = bind("result_error_type", c.c_int, c.POINTER(Result))
    value = bind("value_int64", c.c_int64, c.POINTER(Result), c.c_uint64, c.c_uint64)
    prepare = bind("prepare", c.c_int, ptr, c.c_char_p, c.POINTER(ptr))
    prepare_error = bind("prepare_error", c.c_char_p, ptr)
    destroy_prepare = bind("destroy_prepare", None, c.POINTER(ptr))
    stream = bind("execute_prepared_streaming", c.c_int, ptr, c.POINTER(Result))
    is_streaming = bind("result_is_streaming", c.c_bool, Result)
    fetch = bind("fetch_chunk", ptr, Result)
    chunk_size = bind("data_chunk_get_size", c.c_uint64, ptr)
    destroy_chunk = bind("destroy_data_chunk", None, c.POINTER(ptr))

    def execute(connection, sql, scalar=False, expected_error=None):
        result = Result()
        try:
            failed = query(connection, sql.encode(), c.byref(result))
            if expected_error is not None:
                message = (error(c.byref(result)) or b"").decode()
                assert failed and error_type(c.byref(result)) == expected_error, message
                return message
            if failed:
                raise RuntimeError(error(c.byref(result)).decode())
            return value(c.byref(result), 0, 0) if scalar else None
        finally:
            destroy(c.byref(result))

    db, connection, observer = ptr(), ptr(), ptr()
    output = []
    assert open_db(None, c.byref(db)) == 0
    try:
        assert connect(db, c.byref(connection)) == 0
        assert connect(db, c.byref(observer)) == 0
        execute(connection, "LOAD lance; SET threads=4; SET memory_limit='256MB'")
        with tempfile.TemporaryDirectory(prefix="lance-returning-lifecycle-") as directory:
            execute(connection, f"ATTACH '{directory}' AS ns (TYPE LANCE)")
            for action in ("insert", "update", "delete"):
                results = []
                for target in ("plain", "ns.main.t"):
                    execute(
                        connection, f"CREATE OR REPLACE TABLE {target} AS SELECT i AS id, i AS v FROM range(20000) r(i)"
                    )
                    statements = {
                        "insert": f"INSERT INTO {target} SELECT i, i FROM range(20000, 40000) r(i) RETURNING id, v",
                        "update": f"UPDATE {target} SET v=v+1 RETURNING id, v",
                        "delete": f"DELETE FROM {target} WHERE id < 15000 RETURNING id, v",
                    }
                    prepared, result, chunk = ptr(), Result(), ptr()
                    try:
                        rc = prepare(connection, statements[action].encode(), c.byref(prepared))
                        assert rc == 0, prepare_error(prepared)
                        rc = stream(prepared, c.byref(result))
                        assert rc == 0, error(c.byref(result))
                        streaming = bool(is_streaming(result))
                        chunk = ptr(fetch(result))
                        assert chunk and 0 < chunk_size(chunk) < 15000
                        destroy_chunk(c.byref(chunk))
                    finally:
                        if chunk:
                            destroy_chunk(c.byref(chunk))
                        destroy(c.byref(result))
                        destroy_prepare(c.byref(prepared))
                    count = execute(observer, f"SELECT count(*) FROM {target}", scalar=True)
                    total = execute(observer, f"SELECT sum(v) FROM {target}", scalar=True)
                    results.append((count, total))
                    output.append(
                        {"action": action, "target": target, "streaming": streaming, "count": count, "sum": total}
                    )
                assert results[0] == results[1], (action, results)

            execute(connection, "CREATE TABLE ns.main.bounded (id BIGINT, payload VARCHAR)")
            execute(connection, "SET memory_limit='16MB'; SET temp_directory=''")
            before = execute(connection, "SELECT sum(memory_usage_bytes) FROM duckdb_memory()", scalar=True)
            message = execute(
                connection,
                "INSERT INTO ns.main.bounded SELECT i, repeat(i::VARCHAR, 1024) FROM range(100000) r(i) RETURNING *",
                expected_error=33,  # DUCKDB_ERROR_OUT_OF_MEMORY
            )
            after = execute(connection, "SELECT sum(memory_usage_bytes) FROM duckdb_memory()", scalar=True)
            assert execute(observer, "SELECT count(*) FROM ns.main.bounded", scalar=True) == 0
            # A later small statement must succeed with the same connection and budget.
            execute(connection, "INSERT INTO ns.main.bounded VALUES (1, 'recovered') RETURNING *")
            assert execute(observer, "SELECT count(*) FROM ns.main.bounded", scalar=True) == 1
            assert after <= before + 1024 * 1024, (before, after)
            output.append({"action": "out_of_memory", "error": message, "memory_before": before, "memory_after": after})
    finally:
        if observer:
            disconnect(c.byref(observer))
        if connection:
            disconnect(c.byref(connection))
        close(c.byref(db))
    print(json.dumps(output, indent=2))
    print("DML RETURNING lifecycle checks passed")


if __name__ == "__main__":
    main()
