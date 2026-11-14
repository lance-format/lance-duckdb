"""Check pushed aggregate reservations across streaming-result lifetimes."""

from __future__ import annotations

import argparse
import ctypes as c
import tempfile
import time
from pathlib import Path


class Result(c.Structure):
    # duckdb_result from the pinned duckdb.h; do not inspect these fields.
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
    interrupt = bind("interrupt", None, ptr)

    def execute(connection, sql, scalar=False):
        result = Result()
        try:
            if query(connection, sql.encode(), c.byref(result)):
                raise RuntimeError(error(c.byref(result)).decode())
            return value(c.byref(result), 0, 0) if scalar else None
        finally:
            destroy(c.byref(result))

    db, observer = ptr(), ptr()
    assert open_db(None, c.byref(db)) == 0
    try:
        assert connect(db, c.byref(observer)) == 0
        execute(observer, "LOAD lance; SET threads=4; SET memory_limit='1GB'; SET temp_directory=''")

        def reserved():
            return execute(
                observer,
                "SELECT memory_usage_bytes FROM duckdb_memory() WHERE tag='EXTENSION'",
                scalar=True,
            )

        def wait_released():
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if reserved() == 0:
                    return
                time.sleep(0.01)
            raise AssertionError(f"Lance exec reservations leaked: {reserved()} bytes")

        with tempfile.TemporaryDirectory(prefix="lance-exec-memory-") as temp:
            dataset = (Path(temp) / "groups.lance").as_posix().replace("'", "''")
            execute(
                observer,
                f"COPY (SELECT i::BIGINT AS id, (i%100)::DOUBLE AS v FROM range(6000000) r(i)) "
                f"TO '{dataset}' (FORMAT lance, max_rows_per_file 200000)",
            )
            assert reserved() == 0
            sql = f"SELECT id, sum(v) FROM '{dataset}' GROUP BY id"
            for mode in ("eof", "early_close", "interrupt", "disconnect"):
                connection, prepared, result = ptr(), ptr(), Result()
                try:
                    assert connect(db, c.byref(connection)) == 0
                    assert prepare(connection, sql.encode(), c.byref(prepared)) == 0, prepare_error(prepared)
                    assert stream(prepared, c.byref(result)) == 0, error(c.byref(result))
                    assert is_streaming(result)
                    chunk = ptr(fetch(result))
                    assert chunk, error(c.byref(result))
                    rows = chunk_size(chunk)
                    destroy_chunk(c.byref(chunk))

                    # Prove the producer still owns tracked state before closing
                    # it; an already exhausted result cannot test this boundary.
                    held = reserved()
                    assert held > 0, "No live Lance exec reservation; aggregate pushdown was not exercised"
                    if mode == "eof":
                        while True:
                            chunk = ptr(fetch(result))
                            if not chunk:
                                break
                            rows += chunk_size(chunk)
                            destroy_chunk(c.byref(chunk))
                        assert not error(c.byref(result))
                        assert rows == 6000000, rows
                    elif mode == "interrupt":
                        interrupt(connection)
                        while True:
                            chunk = ptr(fetch(result))
                            if not chunk:
                                break
                            destroy_chunk(c.byref(chunk))
                        # DUCKDB_ERROR_INTERRUPT from the pinned duckdb.h.
                        assert error_type(c.byref(result)) == 29, error(c.byref(result))
                        # Cancellation must release the producer before the
                        # caller destroys its result handle.
                        wait_released()
                    elif mode == "disconnect":
                        destroy_prepare(c.byref(prepared))
                        disconnect(c.byref(connection))
                finally:
                    destroy(c.byref(result))
                    destroy_prepare(c.byref(prepared))
                    if connection:
                        disconnect(c.byref(connection))
                wait_released()
                print(f"{mode}: released {held} bytes")
    finally:
        if observer:
            disconnect(c.byref(observer))
        close(c.byref(db))


if __name__ == "__main__":
    main()
