#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

mode="${1:-smoke}"
config="${CONFIG_PATH:-experiments/configs/matched_ppo_arcagi1_30x30_h100.yaml}"
base_output_dir="${BASE_OUTPUT_DIR:-results/matched_ppo_arcagi1_30x30_h100}"
if [[ \
  "$mode" == "validation" \
  || "$mode" == "rerun-stalled" \
  || "$mode" == "rerun-cold-seed42" \
]]; then
  output_dir="${OUTPUT_DIR:-${base_output_dir}_validation}"
elif [[ "$mode" == "screen" ]]; then
  output_dir="${OUTPUT_DIR:-${base_output_dir}_screen}"
elif [[ "$mode" == "capacity-sweep" ]]; then
  output_dir="${OUTPUT_DIR:-${base_output_dir}_capacity}"
elif [[ "$mode" == "final" || "$mode" == "evaluate-final" ]]; then
  output_dir="${OUTPUT_DIR:-${base_output_dir}_final}"
else
  output_dir="${OUTPUT_DIR:-$base_output_dir}"
fi

task_audit_args=(--config "$config" --output "$output_dir/task_shapes.md")
if [[ "${REQUIRE_EXACT_GRID:-1}" == "1" ]]; then
  task_audit_args+=(--require-exact-grid)
fi
pixi run python scripts/audit_matched_ppo_task_shapes.py "${task_audit_args[@]}"
if [[ "$mode" == "audit" ]]; then
  exit 0
fi

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

require_idle_gpu() {
  local visible_gpu="${CUDA_VISIBLE_DEVICES:-0}"
  local gpu_index="${GPU_INDEX:-${visible_gpu%%,*}}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for a monitored timing rerun" >&2
    return 1
  fi

  local compute_apps
  compute_apps="$(
    nvidia-smi \
      -i "$gpu_index" \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  )"
  if [[ -n "$compute_apps" ]]; then
    echo "GPU $gpu_index has another compute process:" >&2
    echo "$compute_apps" >&2
    return 1
  fi
}

require_available_host_memory() {
  local minimum_gib="${MIN_AVAILABLE_GIB:-16}"
  local available_kib
  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  local required_kib=$((minimum_gib * 1024 * 1024))
  if ((available_kib < required_kib)); then
    echo \
      "host has $((available_kib / 1024 / 1024)) GiB available; " \
      "wait for at least ${minimum_gib} GiB before timing ARCLE" \
      >&2
    return 1
  fi
}

run_monitored_one() {
  local backend="$1"
  local seed="$2"
  local timesteps="$3"
  local envs="$4"
  local rollout="$5"
  local minibatch_size="$6"
  local visible_gpu="${CUDA_VISIBLE_DEVICES:-0}"
  local gpu_index="${GPU_INDEX:-${visible_gpu%%,*}}"

  require_idle_gpu || return 2
  require_available_host_memory || return 2
  local telemetry_dir="$output_dir/telemetry"
  mkdir -p "$telemetry_dir"
  local label
  label="$(date -u +%Y%m%dT%H%M%SZ)_${backend}_seed${seed}"
  {
    date -u
    uptime
    free -h 2>/dev/null || true
    echo \
      "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-unset}"
    nvidia-smi -i "$gpu_index"
  } >"$telemetry_dir/${label}_preflight.txt" 2>&1

  local monitor_pids=()
  if command -v vmstat >/dev/null 2>&1; then
    vmstat 1 >"$telemetry_dir/${label}_vmstat.txt" 2>&1 &
    monitor_pids+=("$!")
  fi
  nvidia-smi dmon \
    -i "$gpu_index" \
    -s pucvmet \
    -d 1 >"$telemetry_dir/${label}_nvidia_dmon.txt" 2>&1 &
  monitor_pids+=("$!")

  set +e
  run_one \
    "$backend" "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size" \
    2>&1 | tee "$telemetry_dir/${label}_run.txt"
  local status="${PIPESTATUS[0]}"
  set -e

  local monitor_pid
  for monitor_pid in "${monitor_pids[@]}"; do
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
  done
  return "$status"
}

run_monitored_pair() {
  local seed="$1"
  local timesteps="$2"
  local envs="$3"
  local rollout="$4"
  local minibatch_size="$5"
  local first="${6:-jaxarc}"

  local first_backend="$first"
  local second_backend="arcle"
  if [[ "$first" == "arcle" ]]; then
    second_backend="jaxarc"
  fi

  local first_status=0
  local second_status=0
  if run_monitored_one \
    "$first_backend" "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"; then
    first_status=0
  else
    first_status=$?
  fi
  if run_monitored_one \
    "$second_backend" "$seed" "$timesteps" "$envs" "$rollout" "$minibatch_size"; then
    second_status=0
  else
    second_status=$?
  fi

  if ((first_status == 2 || second_status == 2)); then
    echo "screen precondition failed; stop before interpreting results" >&2
    return 2
  fi
  if ((first_status != 0 || second_status != 0)); then
    echo \
      "incomplete screen pair envs=$envs minibatch=$minibatch_size; continuing" \
      >&2
  fi
}

capacity_sweep_backend() {
  local backend="$1"
  local updates="$2"
  local minibatch_size="$3"
  local attempts_path="$4"
  local env_list="$5"
  local capacity_envs
  read -r -a capacity_envs <<<"$env_list"

  local envs
  for envs in "${capacity_envs[@]}"; do
    local timesteps=$((envs * 4 * updates))
    local started
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    local status=0
    if run_monitored_one \
      "$backend" 511 "$timesteps" "$envs" 4 "$minibatch_size"; then
      status=0
    else
      status=$?
    fi
    if ((status == 2)); then
      echo "capacity sweep precondition failed; no capacity claim recorded" >&2
      return 2
    fi
    if ((status == 0)); then
      printf '%s\t%s\t%s\t%s\t%s\t%s\tcomplete\n' \
        "$started" "$backend" "$envs" "$minibatch_size" \
        "$updates" "$timesteps" \
        >>"$attempts_path"
    else
      printf '%s\t%s\t%s\t%s\t%s\t%s\tfailed:%s\n' \
        "$started" "$backend" "$envs" "$minibatch_size" \
        "$updates" "$timesteps" "$status" \
        >>"$attempts_path"
      break
    fi
  done
}

evaluate_final() {
  local envs="${NUM_ENVS:-16384}"
  local minibatch_size="${MINIBATCH_SIZE:-8192}"
  local timesteps="${TOTAL_TIMESTEPS:-10485760}"
  local found=0

  while IFS= read -r -d '' result_path; do
    if ! jq -e \
      --argjson envs "$envs" \
      --argjson minibatch "$minibatch_size" \
      --argjson timesteps "$timesteps" '
      .status == "complete"
      and .config.num_envs == $envs
      and .config.rollout_length == 4
      and .actual_timesteps == $timesteps
      and (
        (.config.num_envs * .config.rollout_length)
        / .config.num_minibatches
      ) == $minibatch
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
      --total-timesteps "$timesteps" \
      --num-envs 32 \
      --rollout-length 4 \
      --num-minibatches 1 \
      --eval-episodes 300 \
      --checkpoint "$run_dir/params.msgpack" \
      --output "$run_dir/evaluation.json"
  done < <(find "$output_dir/runs" -name result.json -print0)

  if ((found == 0)); then
    echo \
      "no completed final runs found for envs=$envs minibatch=$minibatch_size" \
      >&2
    return 1
  fi
}

case "$mode" in
  audit)
    ;;
  smoke)
    run_pair 511 32768 256 4 512
    ;;
  matched-sweep)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    minibatch_size="${MINIBATCH_SIZE:-512}"
    for envs in 1024 2048 4096 8192 16384; do
      attempt_one jaxarc 511 "$timesteps" "$envs" 4 "$minibatch_size"
      attempt_one arcle 511 "$timesteps" "$envs" 4 "$minibatch_size"
    done
    ;;
  minibatch-sweep)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    envs="${NUM_ENVS:-16384}"
    for minibatch_size in 512 2048 8192; do
      attempt_one jaxarc 511 "$timesteps" "$envs" 4 "$minibatch_size"
      attempt_one arcle 511 "$timesteps" "$envs" 4 "$minibatch_size"
    done
    ;;
  screen)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    envs="${NUM_ENVS:-16384}"
    read -r -a minibatch_sizes \
      <<<"${MINIBATCH_SIZES:-512 2048 4096 8192 16384}"
    index=0
    for minibatch_size in "${minibatch_sizes[@]}"; do
      if ((index % 2 == 0)); then
        run_monitored_pair \
          511 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
      else
        run_monitored_pair \
          511 "$timesteps" "$envs" 4 "$minibatch_size" arcle
      fi
      index=$((index + 1))
    done
    ;;
  capacity-sweep)
    updates="${CAPACITY_UPDATES:-8}"
    jaxarc_minibatch_size="${JAXARC_MINIBATCH_SIZE:-8192}"
    arcle_minibatch_size="${ARCLE_MINIBATCH_SIZE:-2048}"
    jaxarc_capacity_envs="${JAXARC_CAPACITY_ENVS:-8192 16384 32768 65536 131072 262144}"
    arcle_capacity_envs="${ARCLE_CAPACITY_ENVS:-8192 16384 32768 65536 131072}"
    export MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-64}"
    mkdir -p "$output_dir"
    attempts_path="$output_dir/capacity_attempts_$(date -u +%Y%m%dT%H%M%SZ).tsv"
    printf 'started_utc\tbackend\tnum_envs\tminibatch_size\tupdates\ttimesteps\tstatus\n' \
      >"$attempts_path"
    capacity_sweep_backend \
      jaxarc "$updates" "$jaxarc_minibatch_size" "$attempts_path" \
      "$jaxarc_capacity_envs"
    capacity_sweep_backend \
      arcle "$updates" "$arcle_minibatch_size" "$attempts_path" \
      "$arcle_capacity_envs"
    ;;
  pilot)
    timesteps="${TOTAL_TIMESTEPS:-4194304}"
    envs="${NUM_ENVS:-16384}"
    minibatch_size="${MINIBATCH_SIZE:-512}"
    run_pair 511 "$timesteps" "$envs" 4 "$minibatch_size"
    ;;
  validation)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    envs="${NUM_ENVS:-16384}"
    minibatch_size="${MINIBATCH_SIZE:-512}"
    run_pair 42 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    run_pair 13 "$timesteps" "$envs" 4 "$minibatch_size" arcle
    run_pair 123 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    ;;
  rerun-stalled)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    envs="${NUM_ENVS:-16384}"
    minibatch_size="${MINIBATCH_SIZE:-512}"
    run_monitored_one arcle 42 "$timesteps" "$envs" 4 "$minibatch_size"
    run_monitored_one jaxarc 42 "$timesteps" "$envs" 4 "$minibatch_size"
    run_monitored_one jaxarc 123 "$timesteps" "$envs" 4 "$minibatch_size"
    run_monitored_one arcle 123 "$timesteps" "$envs" 4 "$minibatch_size"
    ;;
  rerun-cold-seed42)
    timesteps="${TOTAL_TIMESTEPS:-1048576}"
    envs="${NUM_ENVS:-16384}"
    minibatch_size="${MINIBATCH_SIZE:-512}"
    run_monitored_one arcle 42 "$timesteps" "$envs" 4 "$minibatch_size"
    ;;
  final)
    timesteps="${TOTAL_TIMESTEPS:-10485760}"
    envs="${NUM_ENVS:-16384}"
    minibatch_size="${MINIBATCH_SIZE:-8192}"
    run_pair 42 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    run_pair 13 "$timesteps" "$envs" 4 "$minibatch_size" arcle
    run_pair 123 "$timesteps" "$envs" 4 "$minibatch_size" jaxarc
    ;;
  evaluate-final)
    evaluate_final
    ;;
  analyze)
    ;;
  *)
    echo \
      "usage: $0 {audit|smoke|matched-sweep|minibatch-sweep|screen|capacity-sweep|pilot|validation|rerun-stalled|rerun-cold-seed42|final|evaluate-final|analyze}" \
      >&2
    exit 2
    ;;
esac

pixi run matched-ppo summarize --output-dir "$output_dir"
python scripts/analyze_matched_ppo_30x30_h100.py --output-dir "$output_dir"
