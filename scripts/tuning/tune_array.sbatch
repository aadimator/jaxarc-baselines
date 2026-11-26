#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# One array task runs CHUNK rows of a tuning run matrix (strategy-v4 §11c).
# Rows come from `python -m jaxarc_baselines.tuning plan|select|freeze`; submit with
#   python -m jaxarc_baselines.tuning submit DIR STAGE [--chunk 8] [--cluster quartz|kailash] [--dry-run]
# which calls: sbatch --array=0-N <cluster flags> tune_array.sbatch DIR/STAGE.jsonl CHUNK CONFIG_NAME REPO SCRATCH
# The account, partition and GPU request come from the submit command, so this script runs on
# Quartz and on Kailash.
# Runs land in DIR/STAGE/<name>/ with <name>.log (stdout) and <name>.wall (seconds,
# written only when the run exits 0). Finished runs are skipped, so resubmitting the
# same array resumes it.
# Arguments, not --export: conv-compiling jobs with --export segfaulted (v3 §12).
set -uo pipefail

matrix=${1:?usage: sbatch --array=0-N tune_array.sbatch MATRIX CHUNK CONFIG_NAME [REPO SCRATCH]}
chunk=${2:?chunk}
config_name=${3:?config name}
repo=${4:-/N/project/aadam/workspace/jaxarc-baselines}
scratch=${5:-/N/scratch/$USER/jaxtmp}
case "$(hostname)" in kailash*) unset LD_LIBRARY_PATH ;; esac  # as in the Kailash T1 jobs
stage=$(basename "$matrix" .jsonl)
out=$(dirname "$matrix")/$stage
py="$repo/.pixi/envs/default/bin/python -u"
export JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1
# Temp files and the compile cache go to a per-task scratch dir, removed on exit: a full
# node /tmp killed XLA mid-run ("Unable to write PTX contents", job 12929440).
export TMPDIR="$scratch/${SLURM_JOB_ID:-0}-${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export JAX_COMPILATION_CACHE_DIR="$TMPDIR/jaxcache"
mkdir -p "$out"
cd "$repo"

t=${SLURM_ARRAY_TASK_ID:-0}
n=$(wc -l < "$matrix" | tr -d ' ')
first=$(( t * chunk ))
last=$(( first + chunk < n ? first + chunk : n ))
echo "node $(hostname) task $t: rows $first..$((last - 1)) of $n in $matrix"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git log --oneline -1; git status --short; git -C ../JaxARC log --oneline -1

load_row() {  # index -> sets $name and the array $ov
  local lines
  mapfile -t lines < <($py -m jaxarc_baselines.tuning row "$matrix" "$1")
  name=${lines[0]}
  ov=("${lines[@]:1}")
}

run() {  # name overrides...
  local rname=$1; shift
  local t0=$SECONDS
  echo "=== START $rname"
  $py run_experiment.py --config-name "$config_name" "$@" \
    logger.base_exp_path="$out/$rname" hydra.run.dir="$out/$rname" >"$out/$rname.log" 2>&1
  local status=$?
  echo "=== END $rname exit=$status wall=$((SECONDS - t0))s"
  sed 's/\x1b\[[0-9;]*m//g' "$out/$rname.log" | grep -E "Error|Traceback|RESOURCE_EXHAUSTED|Segmentation" | tail -3
  (( status == 0 )) && echo $((SECONDS - t0)) > "$out/$rname.wall"
  return $status
}

# Preflight: the first row's config on a tiny run, no W&B, no run_meta.
load_row "$first"
pre=()
for o in "${ov[@]}"; do
  case $o in
    arch.total_timesteps=*|arch.num_evaluation=*|+run_meta.*) ;;
    *) pre+=("$o") ;;
  esac
done
if ! run "preflight-$t" "${pre[@]}" arch.total_timesteps=1048576 arch.num_evaluation=2 \
    logger.loggers.wandb.enabled=false; then
  echo "preflight failed on $(hostname)"; exit 1
fi

failed=0
for (( i = first; i < last; i++ )); do
  load_row "$i"
  if [[ -f "$out/$name.wall" ]]; then echo "=== SKIP $name (finished)"; continue; fi
  rm -rf "${out:?}/$name"
  run "$name" "${ov[@]}" || failed=$(( failed + 1 ))
done
echo "=== DONE failed=$failed"
(( failed == 0 ))
