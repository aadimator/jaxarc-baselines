#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
jaxarc_dir="${JAXARC_DIR:-"$repo_dir/../JaxARC"}"
dataset_dir="$jaxarc_dir/data/MiniARC"
task_name="Most_Common_color_l6ab0lf3xztbyxsu3p.json"
task_path="$dataset_dir/data/MiniARC/$task_name"
expected_sha256="1137e986876f8f2e9e5e0e4056d5d4f95510618822277c0a678140978e53fa1c"

if [[ ! -f "$task_path" ]]; then
  if [[ -e "$dataset_dir" ]]; then
    echo "MiniARC directory exists but benchmark task is missing: $dataset_dir" >&2
    echo "Move or remove that incomplete directory, then rerun this script." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dataset_dir")"
  git clone --depth 1 \
    https://github.com/KSB21ST/MINI-ARC \
    "$dataset_dir"
fi

if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256="$(sha256sum "$task_path" | awk '{print $1}')"
else
  actual_sha256="$(shasum -a 256 "$task_path" | awk '{print $1}')"
fi

if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "Unexpected MiniARC task hash: $actual_sha256" >&2
  echo "Expected: $expected_sha256" >&2
  exit 1
fi

echo "MiniARC ready: $task_path"
