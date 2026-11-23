"""Replace a directory or its table between two historical table bindings."""

import argparse
import ctypes as c
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


class Result(c.Structure):
    _fields_ = [
        ("columns", c.c_uint64),
        ("rows", c.c_uint64),
        ("changed", c.c_uint64),
        ("data", c.c_void_p),
        ("error", c.c_void_p),
        ("internal", c.c_void_p),
    ]


@contextmanager
def binding_connections(library):
    lib = c.CDLL(str(library.resolve()))
    ptr = c.c_void_p

    def bind(name, result, *args):
        function = getattr(lib, "duckdb_" + name)
        function.restype, function.argtypes = result, args
        return function

    database, connection, writer = ptr(), ptr(), ptr()
    assert bind("open", c.c_int, c.c_char_p, c.POINTER(ptr))(None, c.byref(database)) == 0
    connect = bind("connect", c.c_int, ptr, c.POINTER(ptr))
    assert connect(database, c.byref(connection)) == 0
    assert connect(database, c.byref(writer)) == 0
    query = bind("query", c.c_int, ptr, c.c_char_p, c.POINTER(Result))
    error = bind("result_error", c.c_char_p, c.POINTER(Result))
    value = bind("value_int64", c.c_int64, c.POINTER(Result), c.c_uint64, c.c_uint64)
    destroy = bind("destroy_result", None, c.POINTER(Result))

    def sql(statement, columns=0, client=connection):
        result = Result()
        try:
            if query(client, statement.encode(), c.byref(result)):
                raise RuntimeError(error(c.byref(result)).decode())
            return [value(c.byref(result), column, 0) for column in range(columns)]
        finally:
            destroy(c.byref(result))

    callbacks = []

    def register_change(name, replace_target):
        # DuckDB evaluates the second AT expression after binding the left table.
        callback_type = c.CFUNCTYPE(None, ptr, ptr, ptr)
        vector_data = bind("vector_get_data", ptr, ptr)
        set_error = bind("scalar_function_set_error", None, ptr, c.c_char_p)

        @callback_type
        def change_binding(info, _inputs, output):
            try:
                replace_target()
                c.cast(vector_data(output), c.POINTER(c.c_int64))[0] = 1
            except (OSError, RuntimeError) as exc:
                set_error(info, str(exc).encode())

        function = ptr(bind("create_scalar_function", ptr)())
        bigint = ptr(bind("create_logical_type", ptr, c.c_int)(5))  # DUCKDB_TYPE_BIGINT
        try:
            bind("scalar_function_set_name", None, ptr, c.c_char_p)(function, name.encode())
            bind("scalar_function_set_volatile", None, ptr)(function)
            bind("scalar_function_set_return_type", None, ptr, ptr)(function, bigint)
            bind("scalar_function_set_function", None, ptr, callback_type)(function, change_binding)
            assert bind("register_scalar_function", c.c_int, ptr, ptr)(connection, function) == 0
        finally:
            bind("destroy_logical_type", None, c.POINTER(ptr))(c.byref(bigint))
            bind("destroy_scalar_function", None, c.POINTER(ptr))(c.byref(function))
        callbacks.append(change_binding)  # DuckDB borrows the Python callback.

    try:
        yield sql, writer, register_change
    finally:
        disconnect = bind("disconnect", None, c.POINTER(ptr))
        disconnect(c.byref(writer))
        disconnect(c.byref(connection))
        bind("close", None, c.POINTER(ptr))(c.byref(database))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    with binding_connections(parser.parse_args().library) as (sql, writer, register_change):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            original_cwd = Path.cwd()
            current, staged = root / "current", root / "next"
            for name, marker in [("first", 11), ("second", 91)]:
                path = root / name
                path.mkdir()
                target = str(path / "t.lance").replace("'", "''")
                sql(f"COPY (SELECT {marker} AS id) TO '{target}' (FORMAT lance)")
            escaped = str(root).replace("'", "''")
            current.symlink_to(root / "first", target_is_directory=True)
            staged.symlink_to(root / "second", target_is_directory=True)
            sql(f"ATTACH '{escaped}/current' AS a (TYPE lance)")
            try:
                os.chdir(root)
                sql("ATTACH './current/' AS b (TYPE lance)")
            finally:
                os.chdir(original_cwd)

            register_change("switch_root", lambda: os.replace(staged, current))

            result = sql(
                "SELECT l.id, r.id FROM a.main.t l AT (VERSION => 1), b.main.t r AT (VERSION => switch_root())", 2
            )
            assert result == [11, 11], result
            assert sql("SELECT id FROM a.main.t AT (VERSION => 1)", 1) == [91]
            assert sql("SELECT id FROM b.main.t AT (VERSION => 1)", 1) == [91]

            sql(f"ATTACH '{escaped}/second' AS physical (TYPE lance)")
            register_change("append_row", lambda: sql("INSERT INTO physical.t VALUES (92)", client=writer))
            counts = sql(
                "SELECT (SELECT count(*) FROM physical.t AT (VERSION => 'branch:main')), "
                "(SELECT count(*) FROM b.t AT (VERSION => CASE WHEN append_row()=1 THEN 'branch:main' END))",
                2,
            )
            assert counts == [1, 1], counts
            assert sql("SELECT count(*) FROM b.t AT (VERSION => 'branch:main')", 1) == [2]

            sql("ALTER TABLE physical.t CREATE TAG 'release' AT VERSION 1", client=writer)

            def move_tag():
                sql("ALTER TABLE physical.t DROP TAG 'release'", client=writer)
                sql("ALTER TABLE physical.t CREATE TAG 'release' AT VERSION 2", client=writer)

            register_change("move_tag", move_tag)
            counts = sql(
                "SELECT (SELECT count(*) FROM b.t AT (VERSION => 'tag:release')), "
                "(SELECT count(*) FROM physical.t AT (VERSION => CASE WHEN move_tag()=1 THEN 'tag:release' END))",
                2,
            )
            assert counts == [1, 1], counts
            assert sql("SELECT count(*) FROM b.t AT (VERSION => 'tag:release')", 1) == [2]

            for next_read in (
                "query_table(CASE WHEN append_row()=1 THEN 'physical.t' END)",
                "physical.t AT (TIMESTAMP => CASE WHEN append_row()=1 THEN TIMESTAMPTZ '2100-01-01' END)",
            ):
                before = sql("SELECT count(*) FROM physical.t", 1)[0]
                counts = sql(
                    "SELECT (SELECT count(*) FROM physical.t AT (VERSION => 'branch:main')), "
                    f"(SELECT count(*) FROM {next_read})",
                    2,
                )
                assert counts == [before, before], counts
                assert sql("SELECT count(*) FROM physical.t", 1) == [before + 1]

            staged.symlink_to(root / "first", target_is_directory=True)
            pair = sql(
                "SELECT (SELECT count(*) FROM a.t AT (VERSION => 'branch:main')), "
                "(SELECT min(id) FROM b.t AT (VERSION => CASE WHEN switch_root()=1 THEN 'branch:main:1' END))",
                2,
            )
            assert pair == [4, 91], pair
            assert sql("SELECT id FROM b.t AT (VERSION => 'branch:main:1')", 1) == [11]

            # IMPORT runs both schema statements in a single DuckDB submission.
            (root / "schema.sql").write_text(
                "CREATE TEMP TABLE first_read AS SELECT count(*) n FROM physical.t AT (VERSION => 'branch:main');"
                "CREATE TEMP TABLE next_read AS SELECT count(*) n FROM "
                "query_table(CASE WHEN append_row()=1 THEN 'physical.t' END);"
            )
            (root / "load.sql").write_text("")
            sql(f"IMPORT DATABASE '{escaped}'")
            counts = sql("SELECT first_read.n, next_read.n FROM first_read, next_read", 2)
            assert counts == [4, 5], counts

        sql("ATTACH 'shared-memory://snapshots/root' AS original (TYPE lance)")
        sql("ATTACH 'shared-memory://snapshots/other/../root/.' AS alias (TYPE lance)")
        sql("CREATE TABLE original.main.t AS SELECT i FROM range(1) AS t(i)", client=writer)

        sql("CREATE TABLE original.events AS SELECT 1 AS id", client=writer)
        register_change("add_event", lambda: sql("INSERT INTO original.events VALUES (2)", client=writer))
        counts = sql(
            "SELECT (SELECT count(*) FROM original.events), "
            "(SELECT count(*) FROM alias.events AT (VERSION => CASE WHEN add_event()=1 THEN 'branch:main' END)), "
            "(SELECT count(*) FROM original.events AT (VERSION => 'branch:main'))",
            3,
        )
        assert counts == [1, 2, 2], counts

        def replace_table():
            sql("DROP TABLE original.main.t", client=writer)
            sql("CREATE TABLE original.main.t AS SELECT i FROM range(2) AS t(i)", client=writer)

        register_change("replace_table", replace_table)
        counts = sql(
            "SELECT (SELECT count(*) FROM original.main.t AT (VERSION => 1)), "
            "(SELECT count(*) FROM alias.main.t AT (VERSION => replace_table()))",
            2,
        )
        assert counts == [1, 1], counts
        assert sql("SELECT count(*) FROM original.main.t AT (VERSION => 1)", 1) == [2]

        sql("ATTACH 'shared-memory://snapshots/root' AS cold (TYPE lance)")
        register_change("drop_table", lambda: sql("DROP TABLE original.main.t", client=writer))
        counts = sql(
            "SELECT (SELECT count(*) FROM original.main.t AT (VERSION => 1)), "
            "(SELECT count(*) FROM cold.main.T AT (VERSION => drop_table()))",
            2,
        )
        assert counts == [2, 2], counts
        try:
            sql("SELECT count(*) FROM cold.main.t AT (VERSION => 1)")
        except RuntimeError as exc:
            assert "does not exist" in str(exc), exc
        else:
            raise AssertionError("The next statement still sees the dropped table")

        sql("ATTACH 'shared-memory://snapshots/root' AS fresh (TYPE lance)")
        sql("CREATE TABLE original.main.History AS SELECT 1 AS id", client=writer)
        assert sql("SELECT count(*) FROM original.main.history", 1) == [1]
        register_change("drop_history", lambda: sql("DROP TABLE original.main.History", client=writer))
        register_change(
            "new_history",
            lambda: sql("CREATE TABLE original.main.history AS SELECT i FROM range(2) t(i)", client=writer),
        )
        # The cold attachment binds the old snapshot while its name is absent.
        # A later lookup through that attachment retains the same spelling.
        counts = sql(
            "SELECT (SELECT count(*) FROM original.main.history AT (VERSION => 1)), "
            "(SELECT count(*) FROM cold.main.history AT (VERSION => drop_history())), "
            "(SELECT count(*) FROM cold.main.history AT (VERSION => new_history())), "
            "(SELECT count(*) FROM fresh.main.history AT (VERSION => 1))",
            4,
        )
        assert counts == [1, 1, 1, 2], counts
        assert sql("SELECT count(*) FROM cold.main.history AT (VERSION => 1)", 1) == [2]

        sql("CREATE TABLE original.main.Mixed AS SELECT 11 AS id", client=writer)
        assert sql("SELECT id FROM original.main.mixed", 1) == [11]

        def replace_spelling():
            sql("DROP TABLE original.main.Mixed", client=writer)
            sql("CREATE TABLE original.main.mixed AS SELECT i AS id FROM range(2) t(i)", client=writer)

        register_change("replace_spelling", replace_spelling)
        pair = sql(
            "SELECT (SELECT count(*) FROM original.mixed AT (VERSION => 'branch:main:1')), "
            "(SELECT count(*) FROM fresh.mixed AT (VERSION => CASE WHEN replace_spelling()=1 THEN 'branch:main:1' END))",
            2,
        )
        assert pair == [1, 2], pair

        base = "shared-memory://case-sensitive-history/root"
        sql(f"ATTACH '{base}' AS warm_names (TYPE lance)")
        sql(f"ATTACH '{base}' AS cold_names (TYPE lance)")
        sql(f"ATTACH '{base}' AS numeric_names (TYPE lance)")
        sql(f"COPY (SELECT 11 AS id) TO '{base}/Mixed.lance' (FORMAT lance)")
        lookup = "SELECT id FROM warm_names.mixed AT (VERSION => 'branch:main')"
        assert sql(lookup, 1) == [11]
        assert sql("SELECT id FROM numeric_names.mixed AT (VERSION => 1)", 1) == [11]
        sql(f"COPY (SELECT 22 AS id) TO '{base}/mixed.lance' (FORMAT lance)")
        assert sql(lookup, 1) == [11]
        assert sql("SELECT id FROM warm_names.mixed", 1) == [11]
        for attachment in ("cold_names", "numeric_names"):
            try:
                sql(lookup.replace("warm_names", attachment), 1)
            except RuntimeError as exc:
                assert "Ambiguous" in str(exc), exc
            else:
                raise AssertionError(f"{attachment} must reject ambiguous table names")

        # Distinct resolved storage options keep separate bindings for the same root.
        root = "shared-memory://named-options/root"
        for attachment in ("east", "west"):
            sql(f"CREATE OR REPLACE TEMP SECRET cfg (TYPE lance, SCOPE '{root}/', USER_AGENT '{attachment}')")
            sql(f"ATTACH '{root}' AS {attachment} (TYPE lance)")
        sql("CREATE TABLE east.events AS SELECT 1 AS id", client=writer)
        register_change("append_west", lambda: sql("INSERT INTO east.events VALUES (2)", client=writer))
        counts = sql(
            "SELECT (SELECT count(*) FROM east.events AT (VERSION => 'branch:main')), "
            "(SELECT count(*) FROM west.events AT (VERSION => CASE WHEN append_west()=1 THEN 'branch:main' END))",
            2,
        )
        assert counts == [1, 2], counts


if __name__ == "__main__":
    main()
