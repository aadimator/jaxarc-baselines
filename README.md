# JaxARC Baselines

This repository contains baseline reinforcement-learning experiments for
JaxARC environments. It provides experiment configs, ARC-specific network
configs, launcher scripts, and plotting utilities for running common Stoix
baselines on MiniARC, ConceptARC, and ARC-AGI task subsets.

The current benchmark scripts cover:

- PPO
- DDQN
- PQN
- REINFORCE

## Repository Layout

```text
run_experiment.py                         # single-run Hydra entry point
experiments/configs/                      # baseline experiment configs
configs/network/                          # ARC observation network configs
configs/env/jaxarc/subsets/AGI1/          # ARC-AGI-1 task subset configs
scripts/launch_baseline_benchmarks.py     # multi-algorithm launcher
scripts/baseline_scheduler.py             # shared scheduling helpers
scripts/plot_baseline_benchmarks.py       # plotting script for benchmark runs
src/jaxarc_baselines/benchmark_log_parser.py
stoix/                                   # Stoix submodule used by the baselines
```

The main baseline configs are:

```text
experiments/configs/baseline_ff_ppo_mini_all_512k.yaml
experiments/configs/baseline_ff_ddqn_mini_all_512k.yaml
experiments/configs/baseline_ff_pqn_mini_all_512k.yaml
experiments/configs/baseline_ff_reinforce_mini_all_524k.yaml
```

Policy-gradient baselines use `configs/network/arc_shallow_cnn_hwc.yaml`.
Value-based baselines use
`configs/network/arc_shallow_cnn_hwc_q_value.yaml`.

## Setup

Clone the repository with submodules:

```bash
git clone --recurse-submodules <repo-url> jaxarc-baselines
cd jaxarc-baselines
```

If the repository was cloned without submodules, initialize them manually:

```bash
git submodule update --init --recursive
```

Install the environment with Pixi:

```bash
pixi install
```

`pixi.toml` installs JaxARC from PyPI and installs this repository plus the
`stoix` submodule as editable local packages.

## Data

JaxARC dataset configs expect data under this repository's `data/` directory:

```text
data/
  ARC-AGI-1/
  ConceptARC/
  MiniARC/
```

The benchmark launcher uses ARC-AGI-1 subset configs from
`configs/env/jaxarc/subsets/AGI1/` and ConceptARC groups discovered from
`data/ConceptARC/corpus`.

## Quick Checks

Check that Hydra can compose a baseline config:

```bash
pixi run python run_experiment.py \
  --config-name baseline_ff_ppo_mini_all_512k.yaml \
  --cfg job
```

Check the DDQN network config:

```bash
pixi run python run_experiment.py \
  --config-name baseline_ff_ddqn_mini_all_512k.yaml \
  --cfg job \
  --package network
```

Preview the benchmark launcher without starting training:

```bash
pixi run python scripts/launch_baseline_benchmarks.py \
  --datasets agi1 \
  --scenario-limit 1 \
  --num-seeds 1 \
  --algorithms ppo ddqn pqn reinforce \
  --dry-run
```

The dry run should report one job for each selected algorithm.

## Running Experiments

Run a single baseline config:

```bash
pixi run python run_experiment.py \
  --config-name baseline_ff_ppo_mini_all_512k.yaml \
  hydra.job.chdir=false
```

Run the benchmark launcher over the configured ARC-AGI-1 and ConceptARC task
sets:

```bash
pixi run python scripts/launch_baseline_benchmarks.py \
  --algorithms ppo ddqn pqn reinforce \
  --datasets all \
  --num-seeds 5 \
  --total-timesteps 10000000 \
  --total-num-envs 512
```

The launcher writes results to:

```text
results/baseline_benchmarks/subset_baselines_10m_5seed/
```

It records scheduler state in `launcher_status.json`, job logs in
`launcher_logs/`, and per-run summaries under `runs/`.

Use `--retry-failed` to resume a run after fixing failed jobs.

## Plotting

After benchmark jobs finish, generate the comparison figures with:

```bash
pixi run python scripts/plot_baseline_benchmarks.py
```

The plotter reads `success_curve.csv` files under the benchmark results
directory and writes figures to the corresponding `figures/` directory.

## Notes

- Full benchmark runs use seeds `0` through `4` by default.
- The launcher uses point actions for larger-grid datasets.
- Online Weights & Biases logging is disabled unless a W&B project is passed to
  the launcher.
- DDQN and PQN rely on the local Stoix submodule state in this repository. Keep
  the submodule initialized when running those baselines.
