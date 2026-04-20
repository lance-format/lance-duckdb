# LAION-1M Benchmark

This directory contains a reproducible benchmark for comparing three query paths
on the same LAION-derived slice:

- Parquet direct
- DuckDB indexed (a DuckDB table with FTS and VSS indexes built from the same Parquet baseline)
- Lance

The benchmark uses the public `lance-format/laion-1m` Hugging Face Parquet
export. At the moment that public export materializes `69,632` rows locally,
not the full `~1.16M` dataset. This is intentional: the goal here is a public,
repeatable benchmark source.

## What The Runner Does

The runner covers the full local lifecycle:

1. download the public Parquet shards once and cache them locally
2. materialize a local LZ4 Parquet baseline
3. build a DuckDB indexed database from that baseline
4. build a Lance dataset from that same baseline
5. run the benchmark and print a summary table

All generated files live under `benches/laion_1m/data/` and are reused across
runs. The initial download is the only networked step.

## Requirements

- `duckdb` available in `PATH`
- `duckdb >= 1.5.2`
- Python 3
- internet access for the first run
- a stable DuckDB `lance` extension release that includes indexed hybrid search
  parameters such as `use_index`, `nprobes`, and `refine_factor`

The benchmark uses the stable DuckDB `lance` extension via `INSTALL lance; LOAD lance;`.
It does not require a repository-local Lance build.

## Quick Start

Run everything:

```bash
python3 benches/laion_1m/scripts/run_search_bench.py all
```

Cold-only:

```bash
python3 benches/laion_1m/scripts/run_search_bench.py cold
```

Warm-only:

```bash
python3 benches/laion_1m/scripts/run_search_bench.py warm
```

Use a different repeat count:

```bash
python3 benches/laion_1m/scripts/run_search_bench.py all --repeats 10
```

Skip prepare steps if all artifacts already exist:

```bash
python3 benches/laion_1m/scripts/run_search_bench.py all --skip-prepare
```

## Benchmark Modes

`cold`

- each workload runs in a fresh DuckDB process
- useful for measuring process-local startup and first-query cost

`warm`

- each backend runs in a single DuckDB session
- the runner executes one silent warmup pass first
- measured runs then execute in the same session

The runner repeats each workload `5` times by default and reports the average.

## Workloads

The summary table includes:

- `fts`
- `vector_exact`
- `vector_indexed`
- `hybrid`
- `blob_read`

These workloads are aligned by purpose across backends. They are not identical
SQL implementations, but they are intended to answer the same retrieval task.

## Output

The runner prints a Markdown summary table and also writes TSV outputs to:

- `benches/laion_1m/data/results/cold_search_summary.tsv`
- `benches/laion_1m/data/results/warm_search_summary.tsv`

Detailed per-run logs are stored in the same directory.

## Manual Steps

If you need to run the steps individually instead of using the Python runner:

```bash
bash benches/laion_1m/scripts/download_source_parquet.sh
duckdb -c ".read benches/laion_1m/sql/10_materialize_lz4_parquet.sql"
duckdb benches/laion_1m/data/laion_1m_indexed.duckdb -c ".read benches/laion_1m/sql/20_build_duckdb_indexed.sql"
duckdb -c ".read benches/laion_1m/sql/30_build_lance_v22.sql"
```

You can also run the verification scripts separately:

```bash
duckdb benches/laion_1m/data/laion_1m_indexed.duckdb -c ".read benches/laion_1m/sql/50_verify_duckdb_indexed.sql"
duckdb -c ".read benches/laion_1m/sql/51_verify_lance.sql"
```
