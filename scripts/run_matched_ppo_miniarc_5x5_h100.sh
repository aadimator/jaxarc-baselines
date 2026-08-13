#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
bash scripts/ensure_miniarc.sh

mode="${1:-smoke}"
config="experiments/configs/matched_ppo_miniarc_5x5_h100.yaml"
output_dir="${OUTPUT_DIR:-results/matched_ppo_miniarc_5x5_h100}"

run_one() {
  local backend="$1"
  local seed="$2"
  local timesteps="$3"
  local envs="$4"
  local rollout="$5"
  local minibatch_size="$6"
  local update_batch=$((envs * rollout))

  if ((update_batch % minibatch_size != 0)); then
    echo \
      "update batch $update_batch not divisible by minibatch $minibatch_size" \
      >&2
    return 2
  fi

  local minibatches=$((update_batch / minibatch_size))
  pixi run matched-ppo train \
    --config "$config" \
    --backend "$backend" \
    --seed "$seed" \
    --total-timesteps "$timesteps" \
    --num-envs "$envs" \
    --rollout-length "$rollout" \
    --num-minibatches "$minibatches" \
    --skip-evaluation \
    --output-dir "$output_dir"
}

attempt_one() {
  if ! run_one "$@"; then
    echo \
      "FAILED backend=$1 seed=$2 envs=$4 minibatch=$6; continuing sweep" \
      >&2
  fi
}

run_pair() {
  local seed="$1"
  local timesteps="$2"
  local envs="$3"
  local rollout="$4"
  local minibatch_size="$5"
  local first="${6:-jaxarc}"

  if [[ "$first" == "jaxarc" ]]; then
    run_one jaxarc "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"
    run_one arcle "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"
  else
    run_one arcle "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"
    run_one jaxarc "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"
  fi
}

evaluate_final_262k_8192() {
  local found=0
  while IFS= read -r -d '' result_path; do
    if ! jq -e '
      .status == "complete"
      and .config.num_envs == 262144
      and .config.rollout_length == 4
      and .actual_timesteps == 10485760
      and (
        (.config.num_envs * .config.rollout_length)
        / .config.num_minibatches
      ) == 8192
      and (.seed as $seed | ([13, 42, 123] | index($seed)) != null)
      and (.backend == "jaxarc" or .backend == "arcle")
    ' "$result_path" >/dev/null; then
      continue
    fi

    found=$((found + 1))
    local seed
    seed="$(jq -r '.seed' "$result_path")"
    local run_dir
    run_dir="$(dirname "$result_path")"
    pixi run matched-ppo evaluate \
      --config "$config" \
      --backend jaxarc \
      --seed "$seed" \
      --total-timesteps 10485760 \
      --num-envs 128 \
      --rollout-length 4 \
      --num-minibatches 1 \
      --eval-episodes 300 \
      --checkpoint "$run_dir/params.msgpack" \
      --output "$run_dir/evaluation.json"
  done < <(find "$output_dir/runs" -name result.json -print0)

  if ((found == 0)); then
    echo "no completed 262K/8192 final runs found" >&2
    return 1
  fi
}

case "$mode" in
  smoke)
    run_pair 511 131072 8192 4 2048
    ;;
  minibatch-sweep)
    timesteps="${TOTAL_TIMESTEPS:-4194304}"
    for minibatch_size in 2048 8192; do
      run_pair 511 "$timesteps" 131072 4 "$minibatch_size"
    done
    ;;
  matched-scale)
    timesteps="${TOTAL_TIMESTEPS:-10485760}"
    minibatch_size="${MINIBATCH_SIZE:-2048}"
    for envs in 131072 262144; do
      attempt_one \
        jaxarc 511 "$timesteps" "$envs" 4 "$minibatch_size"
      attempt_one \
        arcle 511 "$timesteps" "$envs" 4 "$minibatch_size"
    done
    ;;
  capacity-sweep)
    minibatch_size="${MINIBATCH_SIZE:-2048}"
    updates="${CAPACITY_UPDATES:-5}"
    for envs in 65536 131072 262144 524288 1048576; do
      timesteps=$((envs * 4 * updates))
      attempt_one \
        jaxarc 511 "$timesteps" "$envs" 4 "$minibatch_size"
    done
    for envs in 65536 131072 262144; do
      timesteps=$((envs * 4 * updates))
      attempt_one \
        arcle 511 "$timesteps" "$envs" 4 "$minibatch_size"
    done
    ;;
  final)
    timesteps="${TOTAL_TIMESTEPS:-10485760}"
    envs="${NUM_ENVS:-131072}"
    minibatch_size="${MINIBATCH_SIZE:-2048}"
    run_pair 42 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    run_pair 13 "$timesteps" "$envs" 4 "$minibatch_size" arcle
    run_pair 123 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    ;;
  final-262k-8192)
    run_pair 42 10485760 262144 4 8192 jaxarc
    run_pair 13 10485760 262144 4 8192 arcle
    run_pair 123 10485760 262144 4 8192 jaxarc
    ;;
  evaluate-final-262k-8192)
    evaluate_final_262k_8192
    ;;
  analyze)
    ;;
  *)
    echo \
      "usage: $0 {smoke|minibatch-sweep|matched-scale|capacity-sweep|final|final-262k-8192|evaluate-final-262k-8192|analyze}" \
      >&2
    exit 2
    ;;
esac

pixi run matched-ppo summarize --output-dir "$output_dir"
