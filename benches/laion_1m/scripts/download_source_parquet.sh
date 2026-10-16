#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source_dir="$repo_root/benches/laion_1m/data/source"
mkdir -p "$source_dir"

HF_HUB_ENABLE_HF_TRANSFER=1 uvx --from huggingface_hub hf download \
  lance-format/laion-1m \
  --repo-type dataset \
  --revision refs/convert/parquet \
  --include 'default/partial-train/*.parquet' \
  --local-dir "$source_dir" \
  --max-workers 8
