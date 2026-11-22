"""Check that historical reads keep the dataset resolved by the REST namespace.

The first two describe calls identify dataset A. A later describe would identify
dataset B, whose version 1 has a different row count. The historical read must
return A's rows without issuing that third describe.
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
        # The physical table sees v2 while the namespace can reach v3 at its
        # registered manifest path. Describes always return the same table URI.
        authority_table = Path(tmp) / "authority.lance"
        shutil.copytree(REPO_ROOT / "test/data/branch_tag_fixture.lance", authority_table)
        head = min((authority_table / "_versions").glob("*.manifest"))  # inverted version numbers
        registered = Path(tmp) / head.name
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
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = unquote(urlparse(self.path).path)
                status = 200
                if "/default$authority/" in path and path.endswith("/version/list"):
                    response: object = {"versions": [registry_head]}
                elif "/default$authority/" in path and path.endswith("/version/describe"):
                    version_lookups.append(json.loads(body)["version"])
                    assert version_lookups[-1] == 3
                    response = {"version": registry_head}
                elif "/default$authority/" in path and path.endswith("/describe"):
                    response = {"location": str(authority_table.resolve()), "schema": SCHEMA}
                    if registry_on.exists():
                        response["managed_versioning"] = not registry_off.exists()
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
            attach = (
                f"LOAD '{args.extension}'; ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
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
            duckdb(attach + "SELECT id FROM ns.main.authority AT (TIMESTAMP => TIMESTAMPTZ 'infinity');", "3")
            if version_lookups:
                raise RuntimeError(f"Timestamp read reopened its bound head: {version_lookups}")

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
