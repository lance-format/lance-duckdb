"""Check that historical reads keep the dataset resolved by the REST namespace.

The first two describe calls identify dataset A. A later describe would identify
dataset B, whose version 1 has a different row count. The historical read must
return A's rows without issuing that third describe.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from check_directory_time_travel import binding_connections

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = {"fields": [{"name": "id", "nullable": True, "type": {"type": "int64"}}]}


def manifest_time(seconds: int, nanos: int) -> bytes:
    """Encode the two varint fields of the fixture's protobuf Timestamp."""

    def varint(value: int) -> bytes:
        encoded = bytearray()
        while value >= 128:
            encoded.append(128 + value % 128)
            value //= 128
        encoded.append(value)
        return bytes(encoded)

    return b"\x08" + varint(seconds) + b"\x10" + varint(nanos)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=REPO_ROOT / "build" / "release" / "duckdb")
    parser.add_argument("--library", type=Path, help="Run the two-connection catalog binding check")
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    def duckdb(sql: str, expected: str | None = None, *, expected_error: str | None = None) -> None:
        result = subprocess.run(
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
        if expected_error is not None:
            if result.returncode == 0 or expected_error not in result.stderr:
                raise RuntimeError(f"{sql}\nexpected error {expected_error!r}, got {result.stdout!r}\n{result.stderr}")
            return
        if result.returncode or (expected is not None and result.stdout.strip() != expected):
            raise RuntimeError(f"{sql}\nexpected {expected!r}, got {result.stdout!r}\n{result.stderr}")

    with tempfile.TemporaryDirectory(prefix="lance-at-rest-") as tmp:
        dataset_a = str(Path(tmp) / "a.lance")
        dataset_b = str(Path(tmp) / "b.lance")
        shutil.copytree(REPO_ROOT / "test/data/branch_tag_fixture.lance", dataset_a)
        tags = Path(dataset_a) / "_refs/tags"
        shutil.copyfile(tags / "dev_tag.json", tags / "2.json")
        duckdb(
            f"LOAD '{args.extension}'; COPY (SELECT i::BIGINT AS id FROM range(10) r(i)) "
            f"TO '{dataset_b}' (FORMAT lance);"
        )
        describes: list[str] = []
        move_on_marker = False
        moves: list[str] = []
        tag_requests: list[str] = []
        historical = Path(tmp) / "historical.lance"
        shutil.copytree(REPO_ROOT / "test/data/branch_tag_fixture.lance", historical)
        min((historical / "_versions").glob("*.manifest")).write_bytes(b"bad\n")
        fixed_requests = []
        fixed_managed = False
        fixed_denied = False
        swapped = False
        managed_flip = False
        casepair_exists = True
        # The physical table sees v2 while the namespace can reach v3 at its
        # registered manifest path. Describes always return the same table URI.
        authority_table = Path(tmp) / "authority.lance"
        shutil.copytree(REPO_ROOT / "test/data/branch_tag_fixture.lance", authority_table)
        head = min((authority_table / "_versions").glob("*.manifest"))  # inverted version numbers
        registered = Path(tmp) / "registered-v3.manifest"
        head.rename(registered)
        registry_on, registry_off = (Path(tmp) / name for name in ("enable.csv", "disable.csv"))
        version_lookups: list[int] = []
        registry_head = {
            "version": 3,
            "manifest_path": registered.resolve().as_posix().lstrip("/"),
            "manifest_size": registered.stat().st_size,
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                nonlocal swapped, casepair_exists
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = unquote(urlparse(self.path).path)
                status = 200
                if "/default$historical/" in path and path.endswith("/describe"):
                    request = json.loads(body)
                    fixed_requests.append(request)
                    response = {"location": str(historical)}
                    if request.get("load_detailed_metadata"):
                        status, response = 500, {"code": 500, "error": "unreadable current manifest"}
                    elif request.get("version") is not None:
                        response = {
                            "location": "relative/location",
                            "table_uri": historical.resolve().as_uri(),
                            "managed_versioning": fixed_managed,
                        }
                        if fixed_denied:
                            status = 403
                            response = {"code": 403, "error": "Forbidden", "message": "version access denied"}
                elif "/default$swapping/" in path and path.endswith("/describe"):
                    response = {"location": dataset_b if swapped else dataset_a}
                elif "/default$SWAPPING/" in path and path.endswith("/describe"):
                    status, response = 404, {"code": 4, "error": "table not found"}
                elif "/default$CasePair/" in path and path.endswith("/drop"):
                    casepair_exists = False
                    response = {}
                elif "/default$CasePair/" in path and not casepair_exists:
                    status, response = 404, {"code": 4, "error": "table not found", "message": "CasePair was dropped"}
                elif "/default$CasePair/" in path and path.endswith("/describe"):
                    response = {"location": dataset_a}
                elif "/default$casepair/" in path and path.endswith("/describe"):
                    response = {"location": dataset_b}
                elif "/default$restricted/" in path and path.endswith("/describe"):
                    response = {"location": dataset_b if self.headers.get("x-api-key") == "secondary" else dataset_a}
                elif "/default$switch/" in path and path.endswith("/describe"):
                    swapped = True
                    response = {"location": dataset_a}
                elif "/default$authority/" in path and path.endswith("/version/list"):
                    response: object = {"versions": [registry_head]}
                elif "/default$authority/" in path and path.endswith("/version/describe"):
                    version = json.loads(body)["version"]
                    version_lookups.append(version)
                    if version == 2:
                        v2 = authority_table / "_versions" / f"{2**64 - 3:020d}.manifest"
                        response = {"version": {"version": 2, "manifest_path": str(v2.resolve()).lstrip("/")}}
                    elif version == 3:
                        response = {"version": registry_head}
                    else:
                        status, response = 404, {"code": 11, "error": "no such registered revision"}
                elif "/default$authority/" in path and path.endswith("/describe"):
                    response = {"location": str(authority_table.resolve()), "schema": SCHEMA}
                    if registry_on.exists():
                        response["managed_versioning"] = not registry_off.exists()
                    details = json.loads(body)
                    if managed_flip and (
                        details.get("vend_credentials") or details.get("load_detailed_metadata") is None
                    ):
                        response["managed_versioning"] = False
                    if response.get("managed_versioning") and json.loads(body).get("version") is not None:
                        status, response = 403, {"code": 403, "error": "Forbidden", "message": "use the registry"}
                elif path.endswith("/table/list"):
                    response = {"tables": ["t", "marker", "authority"] if move_on_marker else ["t", "authority"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    if move_on_marker and "marker" in path and not moves:
                        moves.append(path)
                        shutil.copyfile(tags / "main_tag.json", tags / "2.json")
                        duckdb(
                            f"LOAD '{args.extension}'; ATTACH '{tmp}' AS local_ns (TYPE LANCE); "
                            "INSERT INTO local_ns.main.a VALUES (7);"
                        )
                    dataset = dataset_a if move_on_marker or len(describes) < 2 else dataset_b
                    describes.append(dataset)
                    response = {"location": dataset, "schema": SCHEMA}
                elif "/tags/" in path:
                    tag_requests.append(path)
                    response = {}
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
            endpoint = f"http://127.0.0.1:{server.server_port}"

            def mount(alias, url=endpoint, options=""):
                return f"ATTACH 'default' AS {alias} (TYPE LANCE, ENDPOINT '{url}'{options}); "

            load = f"LOAD '{args.extension}'; "
            attach = load + mount("ns")
            duckdb(
                attach + "SELECT * FROM lance_hybrid_search('ns.main.CasePair', 'vec', "
                "[0,0]::FLOAT[2], 'text', 'orchid', version=1);",
                expected_error="Search by version requires a dataset path, URI or directory-attached Lance table",
            )
            for push_query in ("true", "false"):
                fixed_requests.clear()
                duckdb(
                    attach + f"SET lance_namespace_query_table={push_query}; "
                    "SELECT avg(id) FROM ns.main.historical AT (VERSION => 1);",
                    "1.0",
                )
                if (
                    len(fixed_requests) != 2
                    or fixed_requests[0].get("version") is not None
                    or fixed_requests[0].get("load_detailed_metadata")
                ):
                    raise RuntimeError(f"unexpected cold discovery: {fixed_requests}")
                fixed = fixed_requests[1]
                if (
                    fixed.get("version"),
                    fixed.get("with_table_uri"),
                    fixed.get("vend_credentials"),
                    fixed.get("load_detailed_metadata"),
                ) != (1, True, True, False):
                    raise RuntimeError(f"unexpected historical describe: {fixed}")
            duckdb(
                attach + "SELECT * FROM ns.main.historical AT (VERSION => 3);",
                expected_error="Failed to open version 3",
            )
            fixed_managed = True
            duckdb(
                attach + "SELECT * FROM ns.main.historical AT (VERSION => 1);",
                expected_error="changed to managed versioning",
            )
            fixed_managed = False
            fixed_denied = True
            duckdb(attach + "SELECT * FROM ns.main.historical AT (VERSION => 1);", expected_error="Forbidden")
            fixed_denied = False

            # A previously bound attachment and a cold alias must agree even
            # when binding the intervening relation moves the namespace table.
            for first, last in (("ns", "alias"), ("alias", "ns")):
                swapped = False
                duckdb(
                    load
                    + mount("ns", options=", API_KEY 'secondary', HEADER 'X-One=one;x-two=two'")
                    + mount(
                        "alias",
                        endpoint.replace("http:", "HTTP:"),
                        ", HEADER 'X-TWO=two;X-Api-Key=secondary;x-one=one'",
                    )
                    + "SET lance_namespace_query_table=false; SELECT count(*) FROM ns.main.swapping; "
                    f"SELECT a.n,b.n FROM (SELECT count(*) n FROM {first}.main.swapping AT (VERSION => 1)) a, "
                    "(SELECT count(*) FROM ns.main.switch AT (VERSION => 1)) marker, "
                    f"(SELECT count(*) n FROM {last}.main.swapping AT (VERSION => 1)) b; "
                    "SELECT count(*) FROM alias.main.swapping AT (VERSION => 1);",
                    "1\n1|1\n10",
                )
            swapped = False
            duckdb(
                attach + "SET lance_namespace_query_table=false; "
                "SELECT a.id,b.id FROM ns.main.swapping a AT (VERSION => 1), ns.main.SWAPPING b AT (VERSION => 1); "
                "SELECT id FROM ns.main.SWAPPING AT (VERSION => 1); SELECT count(*) FROM ns.main.SWAPPING;",
                "1|1\n1\n1",
            )
            duckdb(
                attach + mount("alias") + "SELECT (SELECT count(*) FROM ns.main.CasePair AT (VERSION => 1)), "
                "(SELECT count(*) FROM alias.main.casepair AT (VERSION => 1));",
                "1|10",
            )
            if args.library:
                with binding_connections(args.library) as (sql, writer, register_change):
                    sql(mount("ns"))

                    def rebind_casepair():
                        sql("DROP TABLE ns.main.CasePair", client=writer)
                        observed = sql("SELECT count(*) FROM ns.main.casepair AT (VERSION => 1)", 1, writer)
                        if observed != [10]:
                            raise RuntimeError(f"writer selected the wrong REST table: {observed}")

                    register_change("rebind_casepair", rebind_casepair)
                    assert sql(
                        "SELECT (SELECT count(*) FROM ns.main.CasePair AT (VERSION => 1)), "
                        "(SELECT count(*) FROM ns.main.casepair AT (VERSION => rebind_casepair()))",
                        2,
                    ) == [1, 1]
                    assert sql("SELECT count(*) FROM ns.main.casepair AT (VERSION => 1)", 1) == [10]
                # After DROP, the caller's spelling resolves afresh even if the
                # old exact ID has since been registered again.
                casepair_exists = True
                with binding_connections(args.library) as (sql, _, _register_change):
                    sql(mount("ns"))
                    assert sql("SELECT count(*) FROM ns.main.CasePair AT (VERSION => 1)", 1) == [1]
                    sql("DROP TABLE ns.main.CasePair")
                    casepair_exists = True
                    assert sql("SELECT count(*) FROM ns.main.casepair AT (VERSION => 1)", 1) == [10]
            duckdb(
                attach
                + mount("limited", options=", API_KEY 'secondary'")
                + "SELECT (SELECT count(*) FROM ns.main.restricted AT (VERSION => 1)), "
                "(SELECT count(*) FROM limited.main.restricted AT (VERSION => 1));",
                "1|10",
            )
            for operation in ("CREATE TAG 'new' AT VERSION 1", "DROP TAG 'dev_tag'"):
                describes.clear()
                duckdb(attach + f"ALTER TABLE ns.main.t {operation};", expected_error="directory-attached")
            if tag_requests or (tags / "new.json").exists() or not (tags / "dev_tag.json").is_file():
                raise RuntimeError("Tag DDL on a REST attachment reached the namespace or changed dataset tags")

            for query, expected in [
                ("SELECT count(*) FROM ns.main.t AT (TIMESTAMP => TIMESTAMP '2999-01-01');", "1"),
                ("SELECT sum(id) FROM ns.main.t AT (VERSION => 'tag:2');", "99"),
                ("SELECT sum(id) FROM ns.main.t AT (VERSION => 'branch:dev:3');", "99"),
                ("SELECT sum(id) FROM ns.main.t AT (VERSION => 'branch:dev');", "99"),
            ]:
                describes.clear()
                duckdb(attach + query, expected)
                if not describes or dataset_b in describes:
                    raise RuntimeError(f"Unexpected describe locations: {describes}")

            # Keep t on dataset A across the separate history and AT statements.
            move_on_marker = True
            # Replay a submicrosecond commit through AT and check DuckDB's finite
            # nanosecond boundary on a private copy of the fixture's v3 manifest.
            head_a = min((Path(dataset_a) / "_versions").glob("*.manifest"))
            original = head_a.read_bytes()
            old = manifest_time(1794979277, 289446000)
            assert original.count(old) == 1
            for seconds, nanos, expected in [
                (1794979277, 289446001, "1794979277289446001"),
                (9223372036, 854775806, "9223372036854775806"),
                (9223372036, 854775807, "NULL"),
                (9223372036, 854775808, "NULL"),
            ]:
                new = manifest_time(seconds, nanos)
                assert len(new) == len(old)
                head_a.write_bytes(original.replace(old, new))
                describes.clear()
                duckdb(
                    attach
                    + "SELECT epoch_us(timestamp), timestamp_ns IS NULL, coalesce(epoch_ns(timestamp_ns)::VARCHAR, 'NULL') "
                    "FROM lance_versions('ns.main.t') WHERE version = 3;",
                    f"{seconds * 1000000 + nanos // 1000}|{'true' if expected == 'NULL' else 'false'}|{expected}",
                )
                if nanos == 289446001:
                    for column, row in [("timestamp", "2"), ("timestamp_ns", "3")]:
                        describes.clear()
                        duckdb(
                            attach
                            + f"SET VARIABLE cutoff = (SELECT {column} FROM lance_versions('{dataset_a}') WHERE version=3); "
                            "SELECT id FROM ns.main.t AT (TIMESTAMP => getvariable('cutoff'));",
                            row,
                        )
            head_a.write_bytes(original)

            # Binding the middle table moves both the tag and the main branch head.
            duckdb(
                attach + "SELECT a.id, h.id, b.id, j.id FROM ns.main.t a AT (VERSION => 'tag:2'), "
                "ns.main.t h AT (VERSION => 'branch:main'), ns.main.marker m AT (VERSION => 1), "
                "ns.main.t b AT (VERSION => 'tag:2'), ns.main.t j AT (VERSION => 'branch:main');",
                "99|3|99|3",
            )
            if not moves:
                raise RuntimeError("Marker table did not advance the branch and move the tag")

            # Repoint a tag between EXECUTEs on a private copy of the fixture.
            shutil.copyfile(tags / "dev_tag.json", tags / "2.json")
            duckdb(
                f"LOAD '{args.extension}'; ATTACH '{tmp}' AS ns (TYPE LANCE); "
                "ALTER TABLE ns.main.a ADD COLUMN new_column BIGINT; "
                "SELECT * FROM ns.main.a AT (VERSION => 'tag:2'); "
                "PREPARE tagged AS SELECT sum(id) FROM ns.main.a AT (VERSION => 'tag:2'); "
                "EXECUTE tagged; "
                "PREPARE branched AS SELECT * FROM ns.main.a AT (VERSION => 'branch:dev:3'); "
                "EXECUTE branched; "
                "PREPARE latest AS SELECT sum(id) FROM ns.main.a AT (VERSION => 'branch:main'); "
                "EXECUTE latest; INSERT INTO ns.main.a (id) VALUES (11); EXECUTE latest; "
                f"COPY (SELECT content FROM read_text('{tags / 'main_tag.json'}')) TO '{tags / '2.json'}' "
                "(FORMAT CSV, HEADER false, QUOTE '', ESCAPE ''); "
                "EXECUTE tagged; EXECUTE branched;",
                "99\n99\n99\n10\n21\n1\n99",
            )
            # A branch selector must still read dev when no tag points to it.
            # A different Lance writer can leave a non-ASCII tag in storage.
            shutil.copyfile(tags / "main_tag.json", tags / "%C3%A9.json")
            duckdb(
                f"LOAD '{args.extension}'; ATTACH '{tmp}' AS ns (TYPE LANCE); "
                "ALTER TABLE ns.main.a DROP TAG 'é'; "
                "ALTER TABLE ns.main.a DROP TAG 'dev_tag'; "
                "SELECT tag FROM lance_tags('ns.main.a') ORDER BY tag;",
                "2\nmain_tag",
            )
            read_dev = (
                f"LOAD '{args.extension}'; ATTACH '{tmp}' AS ns (TYPE LANCE); "
                "SELECT id FROM ns.main.a AT (VERSION => '{}');"
            )
            for ref in ("branch:dev:3", "branch:dev"):
                duckdb(read_dev.format(ref), "99")
            # Lance may leave a branch directory behind after its ref is deleted.
            # A fresh statement must reject the orphaned manifests.
            (Path(dataset_a) / "_refs/branches/dev.json").unlink()
            for ref in ("branch:dev:3", "branch:dev"):
                duckdb(read_dev.format(ref), expected_error="branch dev does not exist")

            try:
                duckdb(
                    attach + "SET lance_namespace_query_table=false; "
                    "PREPARE h AS SELECT id FROM ns.main.authority AT (VERSION => 'branch:main'); "
                    "PREPARE a AS SELECT avg(id)::BIGINT FROM ns.main.authority; "
                    f"EXECUTE h; EXECUTE a; COPY (SELECT 1) TO '{registry_on}'; EXECUTE h; EXECUTE a; "
                    f"COPY (SELECT 1) TO '{registry_off}'; EXECUTE h; EXECUTE a;",
                    "2\n2\n3\n3\n2\n2",
                )
            except RuntimeError as err:
                raise RuntimeError("wrong head after changing managed versioning") from err

            # The registry's v3 is outside the directory, whose latest version is v2.
            registry_off.unlink()
            version_lookups.clear()
            duckdb(attach + "SELECT id FROM ns.main.authority AT (VERSION => 2);", "2")
            managed_flip = True
            duckdb(
                attach + "SELECT id FROM ns.main.authority AT (VERSION => 2);",
                expected_error="namespace table changed while binding a managed snapshot",
            )
            managed_flip = False
            version_lookups.clear()
            duckdb(attach + "SELECT id FROM ns.main.authority AT (TIMESTAMP => TIMESTAMPTZ 'infinity');", "3")
            if version_lookups:
                raise RuntimeError(f"Timestamp read reopened its bound head: {version_lookups}")

            fixed_query = "SELECT id FROM ns.main.authority AT (VERSION => 3)"
            version_lookups.clear()
            duckdb(attach + fixed_query, "3")
            if version_lookups != [3]:
                raise RuntimeError(f"Expected one fixed registry lookup, saw {version_lookups}")
            # A physical v3 can exist while the catalog publishes a different v3.
            local = f"LOAD '{args.extension}'; ATTACH '{tmp}' AS local_ns (TYPE LANCE); "
            duckdb(local + "UPDATE local_ns.main.authority SET id = 99")
            try:
                duckdb(attach + fixed_query, "3")
                if args.library:
                    # Local ETags use inode, length and mtime. Publish two real
                    # v3 histories under different names with the same ETag.
                    first = head.read_bytes()
                    head.unlink()
                    duckdb(local + "UPDATE local_ns.main.authority SET id = 31")
                    second = head.read_bytes()
                    # Commit timestamps have variable-length encodings. Pad
                    # after each protobuf message, before its 16-byte footer.
                    size = max(len(first), len(second)) + 1
                    first, second = (raw[:-16] + bytes(size - len(raw)) + raw[-16:] for raw in (first, second))
                    alias = Path(tmp) / "alternate.manifest"
                    alias.hardlink_to(head)
                    mtime = head.stat().st_mtime_ns
                    original = registry_head.copy()
                    try:
                        with binding_connections(args.library) as (sql, _, _):
                            sql(mount("ns"))
                            for path, data, value in ((head, first, 99), (alias, second, 31), (head, first, 99)):
                                path.write_bytes(data)
                                os.utime(path, ns=(mtime, mtime))
                                registry_head.update(
                                    manifest_path=str(path.resolve()).lstrip("/"), manifest_size=len(data)
                                )
                                assert sql(fixed_query, 1) == [value]
                    finally:
                        registry_head.update(original)
                        alias.unlink()
                registered_backup = registered.with_suffix(".backup")
                registered.rename(registered_backup)
                try:
                    duckdb(attach + fixed_query, expected_error="does not match the registration")
                finally:
                    registered_backup.rename(registered)
                registry_head["manifest_size"] += 1
                try:
                    duckdb(attach + fixed_query, expected_error="does not match the registration")
                finally:
                    registry_head["manifest_size"] -= 1
                duckdb(
                    attach + "SELECT id FROM ns.main.authority AT (VERSION => 100)",
                    expected_error="no such registered revision",
                )
                original = registry_head.copy()
                try:
                    for invalid in ("", "/", "junk.txt", "3.manifest-", "4.manifest-pending"):
                        registry_head["manifest_path"] = invalid
                        duckdb(attach + fixed_query, expected_error="invalid registered manifest path")
                    staged = head.with_suffix(".manifest-pending")
                    shutil.copyfile(head, staged)
                    registry_head.update(manifest_path=str(staged.resolve()).lstrip("/"), manifest_size=-1)
                    duckdb(attach + fixed_query, expected_error="invalid registered manifest size")
                    registry_head["manifest_size"] = staged.stat().st_size
                    duckdb(attach + fixed_query, "99")
                    assert staged.exists()
                finally:
                    registry_head.update(original)
            finally:
                head.unlink()

            # Reading the directory's v2 head does not need its older v1 manifest.
            oldest = max((authority_table / "_versions").glob("*.manifest"))
            oldest.write_bytes(b"unreadable older manifest")
            duckdb(
                f"LOAD '{args.extension}'; ATTACH '{tmp}' AS local_ns (TYPE LANCE); "
                "SELECT id FROM local_ns.main.authority AT (TIMESTAMP => TIMESTAMPTZ 'infinity');",
                "2",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    print("Historical REST reads and prepared tag and branch reads passed")


if __name__ == "__main__":
    main()
