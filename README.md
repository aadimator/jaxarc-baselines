# JaxARC-Baselines

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/aadimator/JaxARC-Baselines/workflows/CI/badge.svg
[actions-link]:             https://github.com/aadimator/JaxARC-Baselines/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/JaxARC-Baselines
[conda-link]:               https://github.com/conda-forge/JaxARC-Baselines-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/aadimator/JaxARC-Baselines/discussions
[pypi-link]:                https://pypi.org/project/JaxARC-Baselines/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/JaxARC-Baselines
[pypi-version]:             https://img.shields.io/pypi/v/JaxARC-Baselines
[rtd-badge]:                https://readthedocs.org/projects/JaxARC-Baselines/badge/?version=latest
[rtd-link]:                 https://JaxARC-Baselines.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

## Getting Started

Follow these steps to reproduce the baseline setup locally.

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules https://github.com/aadimator/JaxARC-Baselines.git
cd JaxARC-Baselines
```

If you already cloned the repo without the flag, pull in the `stoix` submodule manually:

```bash
git submodule update --init --recursive
```

Ensure the upstream `JaxARC` repository is available alongside this project (e.g. `../JaxARC`).

### 2. Install dependencies with pixi

```bash
pixi install
```

Enter the managed environment whenever you work on the project:

```bash
pixi shell
```

### 3. Run the default experiment

With the environment active, launch the default PPO baseline:

```bash
python run_experiment.py
```

Override any Stoix or JaxARC configuration straight from the command line, for example:

```bash
python run_experiment.py action=full system.actor_lr=0.0001
```
