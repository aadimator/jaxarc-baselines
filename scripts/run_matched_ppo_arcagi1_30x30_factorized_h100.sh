#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_PATH="experiments/configs/matched_ppo_arcagi1_30x30_factorized_h100.yaml"
export BASE_OUTPUT_DIR="results/matched_ppo_arcagi1_30x30_factorized_h100"
export REQUIRE_EXACT_GRID=1
export MINIBATCH_SIZES="${MINIBATCH_SIZES:-512 2048 4096 8192 16384 32768 65536}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

exec bash "$script_dir/run_matched_ppo_arcagi1_30x30_h100.sh" "$@"
