#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
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


DUCKDB_BIN = "duckdb"
MIN_DUCKDB_VERSION = (1, 5, 2)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def results_dir(root: Path) -> Path:
    return root / "benches" / "laion_1m" / "data" / "results"


def find_duckdb_cli() -> str:
    path = shutil.which(DUCKDB_BIN)
    if path is None:
        raise RuntimeError(
            "duckdb not found in PATH; install DuckDB >= 1.5.2 and ensure `duckdb` is available"
        )
    return path


def parse_duckdb_version(version_output: str) -> tuple[int, int, int]:
    match = re.search(r"v(\d+)\.(\d+)\.(\d+)", version_output)
    if match is None:
        raise RuntimeError(f"failed to parse DuckDB version from: {version_output.strip()}")
    return tuple(int(part) for part in match.groups())


def ensure_duckdb_version(duckdb_cli: str) -> None:
    result = run_checked([duckdb_cli, "--version"], cwd=repo_root(), capture=True)
    version = parse_duckdb_version(result.stdout)
    if version < MIN_DUCKDB_VERSION:
        minimum = ".".join(str(part) for part in MIN_DUCKDB_VERSION)
        actual = ".".join(str(part) for part in version)
        raise RuntimeError(
            f"duckdb {actual} is too old; this benchmark requires duckdb >= {minimum}"
        )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def run_checked(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if stdout_path:
        ensure_parent(stdout_path)
        with stdout_path.open("w") as out:
            return subprocess.run(
                cmd,
                cwd=cwd,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )
    if capture:
        return subprocess.run(cmd, cwd=cwd, text=True, check=True, capture_output=True)
    return subprocess.run(cmd, cwd=cwd, text=True, check=True)


def prepare_artifacts(root: Path, duckdb_cli: str) -> None:
    data_dir = root / "benches" / "laion_1m" / "data"
    source_glob = data_dir / "source" / "default" / "partial-train"
    lz4_parquet = data_dir / "laion_1m_lz4.parquet"
    indexed_db = data_dir / "laion_1m_indexed.duckdb"
    lance_ds = data_dir / "laion_1m_v22.lance"

    source_files = sorted(source_glob.glob("*.parquet"))
    if len(source_files) < 10:
        log("[prepare] downloading source parquet shards from huggingface (one-time, several minutes)")
        run_checked(
            ["bash", "benches/laion_1m/scripts/download_source_parquet.sh"],
            cwd=root,
        )
    else:
        log(f"[prepare] source parquet shards present ({len(source_files)} files), skipping download")

    if not lz4_parquet.exists():
        log("[prepare] materializing lz4 parquet baseline")
        run_checked(
            [duckdb_cli, "-c", ".read benches/laion_1m/sql/10_materialize_lz4_parquet.sql"],
            cwd=root,
        )
    else:
        log("[prepare] lz4 parquet baseline present, skipping")

    if not indexed_db.exists():
        log("[prepare] building duckdb indexed database (HNSW + FTS indexes, several minutes)")
        run_checked(
            [
                duckdb_cli,
                str(indexed_db),
                "-c",
                ".read benches/laion_1m/sql/20_build_duckdb_indexed.sql",
            ],
            cwd=root,
        )
    else:
        log("[prepare] duckdb indexed database present, skipping")

    if not lance_ds.exists():
        log("[prepare] building lance dataset")
        run_checked(
            [
                duckdb_cli,
                "-c",
                ".read benches/laion_1m/sql/30_build_lance_v22.sql",
            ],
            cwd=root,
        )
    else:
        log("[prepare] lance dataset present, skipping")


def build_backends(root: Path, duckdb_cli: str) -> list[Backend]:
    workloads = ["fts", "vector_exact", "vector_indexed", "hybrid", "blob_read"]
    indexed_db = root / "benches" / "laion_1m" / "data" / "laion_1m_indexed.duckdb"
    return [
        Backend(
            name="parquet",
            cli=[duckdb_cli],
            init_sql=root / "benches" / "laion_1m" / "sql" / "workloads" / "parquet" / "_init.sql",
            workloads=workloads,
        ),
        Backend(
            name="duckdb_indexed",
            cli=[
                duckdb_cli,
                str(indexed_db),
            ],
            init_sql=root / "benches" / "laion_1m" / "sql" / "workloads" / "duckdb_indexed" / "_init.sql",
            workloads=workloads,
        ),
        Backend(
            name="lance",
            cli=[duckdb_cli],
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
            log(f"[cold] backend={backend.name} workload={workload} repeat={repeat}/{repeats}")
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
        log(f"[cold] backend={backend.name} workload={workload} avg_real_ms={real_s * 1000:.1f}")
        rows.append((workload, real_s, user_s, sys_s))
    return rows


def run_warm(
    root: Path,
    backend: Backend,
    out_dir: Path,
    repeats: int,
) -> list[tuple[str, float, float, float]]:
    log(
        f"[warm] backend={backend.name} running {repeats} repeats × "
        f"{len(backend.workloads)} workloads in one session"
    )
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
        log(f"[warm] backend={backend.name} workload={workload} avg_real_ms={real_s * 1000:.1f}")
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
    duckdb_cli = find_duckdb_cli()
    ensure_duckdb_version(duckdb_cli)
    out_dir = results_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_artifacts(root, duckdb_cli)

    backends = build_backends(root, duckdb_cli)
    modes = ["cold", "warm"] if args.mode == "all" else [args.mode]

    for mode in modes:
        log(f"[run] mode={mode} repeats={args.repeats}")
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
