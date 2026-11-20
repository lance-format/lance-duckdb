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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", type=Path, default=REPO_ROOT / "build" / "release" / "duckdb")
    parser.add_argument(
        "--extension",
        type=Path,
        default=REPO_ROOT / "build" / "release" / "extension" / "lance" / "lance.duckdb_extension",
    )
    args = parser.parse_args()

    def duckdb(sql: str, expected: str | None = None) -> None:
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

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: object) -> None:
                pass

            def handle_request(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                path = unquote(urlparse(self.path).path)
                status = 200
                if path.endswith("/table/list"):
                    response: object = {"tables": ["t", "marker"] if move_on_marker else ["t"]}
                elif path.endswith("/list") and "/namespace/" in path:
                    response = {"namespaces": []}
                elif path.endswith("/describe") and "/table/" in path:
                    if move_on_marker and "marker" in path and not moves:
                        shutil.copyfile(tags / "main_tag.json", tags / "2.json")
                        moves.append(path)
                    dataset = dataset_a if move_on_marker or len(describes) < 2 else dataset_b
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
            attach = (
                f"LOAD '{args.extension}'; ATTACH 'default' AS ns "
                f"(TYPE LANCE, ENDPOINT 'http://127.0.0.1:{server.server_port}'); "
            )
            for query, expected in [
                ("SELECT count(*) FROM ns.main.t AT (TIMESTAMP => TIMESTAMP '2999-01-01');", "1"),
                ("SELECT sum(id) FROM ns.main.t AT (VERSION => 'tag:2');", "99"),
                ("SELECT sum(id) FROM ns.main.t AT (VERSION => 'branch:dev:3');", "99"),
            ]:
                describes.clear()
                duckdb(attach + query, expected)
                if not describes or dataset_b in describes:
                    raise RuntimeError(f"Unexpected describe locations: {describes}")

            # Binding the middle table reassigns the tag between the two lookups.
            move_on_marker = True
            duckdb(
                attach + "SELECT a.id, b.id FROM ns.main.t a AT (VERSION => 'tag:2'), "
                "ns.main.marker m AT (VERSION => 1), ns.main.t b AT (VERSION => 'tag:2');",
                "99|99",
            )
            if not moves:
                raise RuntimeError("Marker table did not move the tag")

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
                f"COPY (SELECT content FROM read_text('{tags / 'main_tag.json'}')) TO '{tags / '2.json'}' "
                "(FORMAT CSV, HEADER false, QUOTE '', ESCAPE ''); "
                "EXECUTE tagged; EXECUTE branched;",
                "99\n99\n99\n1\n99",
            )
            # A branch selector must still read dev when no tag points to it.
            (tags / "dev_tag.json").unlink()
            duckdb(
                f"LOAD '{args.extension}'; ATTACH '{tmp}' AS ns (TYPE LANCE); "
                "SELECT id FROM ns.main.a AT (VERSION => 'branch:dev:3');",
                "99",
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    print("Historical REST reads and prepared tag reassignment passed")


if __name__ == "__main__":
    main()
