# Matched MiniARC 5x5 PPO on H100

This campaign measures the full PPO training loop on one H100. It uses the
same 5x5 MiniARC task, 13-channel observations, 17,836-parameter spatial policy,
rollout length 4, four PPO epochs, actions, rewards, and optimizer for JaxARC
and ARCLE.

The timing runs skip post-training evaluation. Evaluation at 131,072
environments executes at least 13.1 million extra environment steps for a
30-episode request, so including it would measure oversampling instead of
training. Training quality can be evaluated later from the saved checkpoints.

## Run order

From the repository root:

```bash
pixi install
pixi run matched-ppo-miniarc-h100 smoke
pixi run matched-ppo-miniarc-h100 minibatch-sweep
cat results/matched_ppo_miniarc_5x5_h100/summary.md
```

The campaign downloads MiniARC into the sibling `JaxARC/data/MiniARC`
directory when missing, then verifies the benchmark task hash.

The minibatch sweep compares 2,048 and 8,192 at 131,072 environments. Use
2,048 as the main setting because it is close to the submitted baseline's
1,600-sample minibatch. Keep 8,192 as a declared throughput sensitivity.

Then measure matched scaling and independent capacity:

```bash
pixi run matched-ppo-miniarc-h100 matched-scale
pixi run matched-ppo-miniarc-h100 capacity-sweep
pixi run matched-ppo-miniarc-h100 analyze
```

`matched-scale` runs both implementations at 131,072 and 262,144 environments
for 10,485,760 steps. `capacity-sweep` tries JaxARC through 1,048,576
environments and ARCLE through 262,144. Capacity rows support throughput and
maximum-batch claims only.

Run three matched repeats after checking that 131,072 fits and both backends
complete:

```bash
pixi run matched-ppo-miniarc-h100 final
```

To repeat the final comparison at 262,144 environments:

```bash
NUM_ENVS=262144 pixi run matched-ppo-miniarc-h100 final
```

The recommended follow-up fixes the stronger setting explicitly, then evaluates
all six saved policies on the same 128-environment JaxARC evaluator:

```bash
pixi run matched-ppo-miniarc-h100 final-262k-8192
pixi run matched-ppo-miniarc-h100 evaluate-final-262k-8192
cat results/matched_ppo_miniarc_5x5_h100/summary.md
```

The timing comparison remains JaxARC versus ARCLE. Common-backend evaluation
removes evaluator speed from policy-quality measurements and writes an
`evaluation.json` beside each checkpoint.

Results are written under `results/matched_ppo_miniarc_5x5_h100/`.

## Claim boundary

- Report median steady-state PPO throughput and cold training-wall speedup from
  the three matched final runs.
- The existing 131,072-environment summary contains four exact matched timing
  repetitions: final seeds 13, 42, and 123 plus pilot seed 511. Report all four
  transparently if using that aggregate.
- Report simulator-only throughput separately. It excludes policy updates.
- Do not turn the capacity sweep into a matched learning-quality comparison.
- Do not select one minibatch setting silently. Report 2,048 as primary and
  8,192 as sensitivity.
