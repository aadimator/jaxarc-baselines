# Matched native 30x30 PPO on H100

This campaign uses three ARC-AGI-1 tasks whose native inputs and outputs are
all exactly 30x30:

- `3631a71a`: four training pairs and one test pair.
- `264363fd`: three training pairs and one test pair.
- `a64e4611`: three training pairs and one test pair.

The task set has ten training-pair variants. JaxARC and ARCLE still use
identical 30x30x13 policy observations, spatial policy, action space, rewards,
rollout length, PPO epochs, and seed. ARCLE's `RawARCEnv` is instantiated with
`max_grid_size=(30, 30)`, and both adapters construct 30x30 action masks and
padded policy observations. ARCLE therefore does not receive a smaller internal
grid in this benchmark.

The lower raw SPS at 30x30 is not by itself evidence of a problem. Each
observation has 36 times as many cells as a 5x5 observation, and the shared
spatial policy has 9,001 action logits instead of 251.

## Phase 0: verify native dimensions

From the repository root:

```bash
pixi install
pixi run matched-ppo-arcagi1-30x30-h100 audit
```

The audit must end with:

```text
All native inputs and outputs are exactly the configured size: yes.
```

## Phase 1: screen minibatch size

```bash
pixi run matched-ppo-arcagi1-30x30-h100 screen
cat results/matched_ppo_arcagi1_30x30_h100_screen/analysis.md
```

This runs one matched seed at 16,384 environments for 1,048,576 steps. It
tests minibatch sizes 512, 2,048, 4,096, 8,192, and 16,384. Both backends use
every setting, and backend order alternates between settings.

Stop here and review the results before the capacity sweep. The current screen
selects minibatch 8,192 for JaxARC and 2,048 for ARCLE. JaxARC reaches 100%
active SM utilization at 8,192, and its SPS changes by less than 1% between
8,192 and 16,384. The 16,384 run has a much longer compile warmup, so it is not
worth using.

Download this directory for analysis:

```text
results/matched_ppo_arcagi1_30x30_h100_screen/
```

## Phase 1b: factorized-policy screen

The spatial policy emits 9,001 logits on a 30x30 grid. That makes policy
optimization a large part of the full PPO loop. Run the existing factorized
operation-row-column policy as a secondary controlled comparison. It emits 71
logits, represents the same point-edit actions, and remains identical across
backends.

This screen also tests factorized minibatches 32,768 and 65,536. The update
batch has 65,536 samples, so 65,536 is the largest possible minibatch at this
environment count.

```bash
pixi run matched-ppo-arcagi1-30x30-factorized-h100 audit
pixi run matched-ppo-arcagi1-30x30-factorized-h100 screen
```

Download this directory too:

```text
results/matched_ppo_arcagi1_30x30_factorized_h100_screen/
```

Keep the spatial result as the main comparison. Use the factorized result to
show how end-to-end speedup changes with policy cost. Report it only if both
backends complete cleanly.

## Phase 2: measure independent capacity

Run the independently tuned capacity sweep:

```bash
pixi run matched-ppo-arcagi1-30x30-h100 capacity-sweep
cat results/matched_ppo_arcagi1_30x30_h100_capacity/summary.md
```

JaxARC uses minibatch 8,192 and tries 8,192 through 262,144 environments.
ARCLE uses its independently selected minibatch 2,048 and tries 8,192 through
131,072 environments. Each backend stops after its first failure. Each point
uses eight complete PPO updates and seed 511. Thus the sweep measures supported
parallelism and best throughput without forcing ARCLE and JaxARC to use the
same environment count or minibatch size.
The campaign records completed result files, telemetry, and a timestamped
`capacity_attempts_*.tsv` file. It requires at least 64 GiB of available host
memory so that unrelated host pressure is not mistaken for ARCLE's capacity
limit. Each attempted run also has a `telemetry/*_run.txt` log that preserves
the failure reason.

This is a separate capability measurement. Report the largest completed
environment count and best measured SPS for each backend. Do not call the
ratio a matched speedup or a theoretical speedup. Call it the ratio between
the best measured throughput of each independently tuned backend.

Both native-30x30 wrappers set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`. The
downloaded screen telemetry showed about 61.4 GB reserved on the 80 GB H100,
consistent with a 75% pool. The larger pool leaves about 8 GB for the driver
and lets the capacity sweep test more of the available VRAM. Telemetry records
the setting for every new attempt.

## Phase 3: three-seed matched results

Validate both policy architectures with the selected shared configuration:

```bash
NUM_ENVS=16384 MINIBATCH_SIZE=8192 TOTAL_TIMESTEPS=10485760 \
  pixi run matched-ppo-arcagi1-30x30-h100 final
cat results/matched_ppo_arcagi1_30x30_h100_final/analysis.md

NUM_ENVS=16384 MINIBATCH_SIZE=8192 TOTAL_TIMESTEPS=10485760 \
  pixi run matched-ppo-arcagi1-30x30-factorized-h100 final
cat results/matched_ppo_arcagi1_30x30_factorized_h100_final/analysis.md
```

Each command runs JaxARC and ARCLE at seeds 13, 42, and 123, with alternating
backend order. Native 30x30 keeps the environment count, rollout length, and
minibatch size selected for its own compute and memory costs.

Download both directories:

```text
results/matched_ppo_arcagi1_30x30_h100_final/
results/matched_ppo_arcagi1_30x30_factorized_h100_final/
```

## Reporting boundary

- The matched three-seed result is the primary PPO comparison.
- The independent-capacity result supports only maximum parallelism and best
  achieved throughput claims.
- A failed capacity attempt counts only when the GPU was idle and the host
  memory preflight passed.
- SPS excludes the one-time compilation. Elapsed run time includes setup,
  compilation, and training.
