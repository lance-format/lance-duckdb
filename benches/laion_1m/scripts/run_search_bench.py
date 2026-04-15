#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_TIME_RE = re.compile(
    r"Run Time \(s\): real ([0-9.]+) user ([0-9.]+) sys ([0-9.]+)"
)


@dataclass(frozen=True)
class Backend:
    name: str
    cli: list[str]
    init_sql: Path
    workloads: list[str]


OFFICIAL_DUCKDB = os.environ.get("DUCKDB_CLI", "/opt/homebrew/bin/duckdb")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def results_dir(root: Path) -> Path:
    return root / "benches" / "laion_1m" / "data" / "results"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_checked(cmd: list[str], cwd: Path, stdout_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    ensure_parent(stdout_path) if stdout_path else None
    if stdout_path:
      with stdout_path.open("w") as out:
        return subprocess.run(
            cmd,
            cwd=cwd,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    return subprocess.run(cmd, cwd=cwd, text=True, check=True, capture_output=True)


def prepare_artifacts(root: Path) -> None:
    data_dir = root / "benches" / "laion_1m" / "data"
    source_glob = data_dir / "source" / "default" / "partial-train"
    lz4_parquet = data_dir / "laion_1m_lz4.parquet"
    indexed_db = data_dir / "laion_1m_indexed.duckdb"
    lance_ds = data_dir / "laion_1m_v22.lance"

    source_files = sorted(source_glob.glob("*.parquet"))
    if len(source_files) < 10:
        run_checked(
            ["bash", "benches/laion_1m/scripts/download_source_parquet.sh"],
            cwd=root,
        )

    if not lz4_parquet.exists():
        run_checked(
            [OFFICIAL_DUCKDB, "-c", ".read benches/laion_1m/sql/10_materialize_lz4_parquet.sql"],
            cwd=root,
        )

    if not indexed_db.exists():
        run_checked(
            [
                OFFICIAL_DUCKDB,
                str(indexed_db),
                "-c",
                ".read benches/laion_1m/sql/20_build_duckdb_indexed.sql",
            ],
            cwd=root,
        )

    if not lance_ds.exists():
        run_checked(
            [
                OFFICIAL_DUCKDB,
                "-c",
                ".read benches/laion_1m/sql/30_build_lance_v22.sql",
            ],
            cwd=root,
        )


def build_backends(root: Path) -> list[Backend]:
    workloads = ["fts", "vector_exact", "vector_indexed", "hybrid", "blob_read"]
    indexed_db = root / "benches" / "laion_1m" / "data" / "laion_1m_indexed.duckdb"
    return [
        Backend(
            name="parquet",
            cli=[OFFICIAL_DUCKDB],
            init_sql=root / "benches" / "laion_1m" / "sql" / "workloads" / "parquet" / "_init.sql",
            workloads=workloads,
        ),
        Backend(
            name="duckdb_indexed",
            cli=[
                OFFICIAL_DUCKDB,
                str(indexed_db),
            ],
            init_sql=root / "benches" / "laion_1m" / "sql" / "workloads" / "duckdb_indexed" / "_init.sql",
            workloads=workloads,
        ),
        Backend(
            name="lance",
            cli=[OFFICIAL_DUCKDB],
            init_sql=root / "benches" / "laion_1m" / "sql" / "workloads" / "lance" / "_init.sql",
            workloads=workloads,
        ),
    ]


def workload_sql(root: Path, backend: str, workload: str) -> Path:
    return root / "benches" / "laion_1m" / "sql" / "workloads" / backend / f"{workload}.sql"


def parse_run_times(log_path: Path) -> list[tuple[float, float, float]]:
    text = log_path.read_text()
    return [(float(r), float(u), float(s)) for r, u, s in RUN_TIME_RE.findall(text)]


def average_samples(samples: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    count = len(samples)
    if count == 0:
        raise RuntimeError("expected at least one sample to average")
    real = sum(sample[0] for sample in samples) / count
    user = sum(sample[1] for sample in samples) / count
    sys = sum(sample[2] for sample in samples) / count
    return real, user, sys


def run_cold(
    root: Path,
    backend: Backend,
    out_dir: Path,
    repeats: int,
) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    for workload in backend.workloads:
        samples: list[tuple[float, float, float]] = []
        for repeat in range(1, repeats + 1):
            log_path = out_dir / f"cold_{backend.name}_{workload}_{repeat:02d}.log"
            cmd = backend.cli + [
                "-c",
                ".timer on",
                "-c",
                ".output /dev/null",
                "-c",
                f".read {backend.init_sql}",
                "-c",
                f".read {workload_sql(root, backend.name, workload)}",
            ]
            run_checked(cmd, cwd=root, stdout_path=log_path)
            times = parse_run_times(log_path)
            if not times:
                raise RuntimeError(f"expected at least 1 timing entry in {log_path}, got 0")
            samples.append(times[-1])
        real_s, user_s, sys_s = average_samples(samples)
        rows.append((workload, real_s, user_s, sys_s))
    return rows


def run_warm(
    root: Path,
    backend: Backend,
    out_dir: Path,
    repeats: int,
) -> list[tuple[str, float, float, float]]:
    log_path = out_dir / f"warm_{backend.name}.log"
    cmd = backend.cli + [
        "-c",
        f".read {backend.init_sql}",
        "-c",
        ".timer off",
        "-c",
        ".output /dev/null",
    ]
    for workload in backend.workloads:
        cmd += ["-c", f".read {workload_sql(root, backend.name, workload)}"]
    cmd += ["-c", ".timer on"]
    for _ in range(repeats):
        for workload in backend.workloads:
            cmd += ["-c", f".read {workload_sql(root, backend.name, workload)}"]

    run_checked(cmd, cwd=root, stdout_path=log_path)
    times = parse_run_times(log_path)
    expected = len(backend.workloads) * repeats
    if len(times) != expected:
        raise RuntimeError(
            f"expected {expected} timing entries in {log_path}, got {len(times)}"
        )

    rows: list[tuple[str, float, float, float]] = []
    for workload_index, workload in enumerate(backend.workloads):
        samples = [
            times[(repeat * len(backend.workloads)) + workload_index]
            for repeat in range(repeats)
        ]
        real_s, user_s, sys_s = average_samples(samples)
        rows.append((workload, real_s, user_s, sys_s))
    return rows


def write_summary(
    out_path: Path,
    rows: list[tuple[str, str, float, float, float]],
) -> None:
    ensure_parent(out_path)
    with out_path.open("w") as f:
        f.write("backend\tworkload\treal_s\tuser_s\tsys_s\n")
        for backend, workload, real_s, user_s, sys_s in rows:
            f.write(f"{backend}\t{workload}\t{real_s:.6f}\t{user_s:.6f}\t{sys_s:.6f}\n")


def print_markdown(rows: list[tuple[str, str, float, float, float]], workloads: list[str]) -> None:
    by_workload = {
        (backend, workload): real_s * 1000.0
        for backend, workload, real_s, _user_s, _sys_s in rows
    }

    print("| workload | Parquet direct | DuckDB indexed | Lance native |")
    print("|---|---:|---:|---:|")
    for workload in workloads:
        parquet_ms = by_workload[("parquet", workload)]
        duckdb_ms = by_workload[("duckdb_indexed", workload)]
        lance_ms = by_workload[("lance", workload)]
        print(
            f"| {workload} | `{parquet_ms:.0f} ms` | `{duckdb_ms:.0f} ms` | `{lance_ms:.0f} ms` |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["cold", "warm", "all"],
        nargs="?",
        default="all",
        help="benchmark mode to run",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="skip download/materialize/build checks",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="number of timed repetitions to average for each workload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    out_dir = results_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_artifacts(root)

    backends = build_backends(root)
    modes = ["cold", "warm"] if args.mode == "all" else [args.mode]

    for mode in modes:
        all_rows: list[tuple[str, str, float, float, float]] = []
        for backend in backends:
            timed_rows = (
                run_cold(root, backend, out_dir, args.repeats)
                if mode == "cold"
                else run_warm(root, backend, out_dir, args.repeats)
            )
            for workload, real_s, user_s, sys_s in timed_rows:
                all_rows.append((backend.name, workload, real_s, user_s, sys_s))

        summary_path = out_dir / f"{mode}_search_summary.tsv"
        write_summary(summary_path, all_rows)
        print(f"# {mode} (avg of {args.repeats} runs)")
        print(summary_path)
        print_markdown(all_rows, backends[0].workloads)
        if mode != modes[-1]:
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
