"""Scores for the network ablation: held-out per-task scores, IQM and bootstrap CIs.

Each run writes per-task held-out scores to a JSON-lines file (see
``held_out.HeldOutRecorder``). A configuration's result is a (seeds, tasks)
matrix of final scores. Its point estimate is the interquartile mean over all
seed-task scores, and its confidence interval resamples seeds and tasks
independently (two-level bootstrap), so both sources of noise count.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import trim_mean

EVAL_EVENT = "evaluator"
RUN_NAME = re.compile(r"^(?P<config>.+)__lr(?P<lr>[^_]+)__s(?P<seed>\d+)$")


def read_held_out(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def curve(records: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Steps and the (evals, tasks) per-task scores of every periodic evaluation."""
    evals = sorted((r for r in records if r["event"] == EVAL_EVENT), key=lambda r: r["t"])
    steps = np.array([r["t"] for r in evals])
    return steps, np.array([r["per_task"][metric] for r in evals], dtype=float)


def final_task_scores(records: list[dict], metric: str) -> np.ndarray:
    """Per-task scores of the last periodic evaluation."""
    return curve(records, metric)[1][-1]


def iqm(scores: np.ndarray) -> float:
    """Mean of the middle 50% of all scores."""
    return float(trim_mean(np.asarray(scores, dtype=float).reshape(-1), 0.25))


def _resample(scores: np.ndarray, rng: np.random.Generator, tasks: np.ndarray) -> np.ndarray:
    seeds = rng.integers(0, scores.shape[0], scores.shape[0])
    return scores[seeds][:, tasks]


def bootstrap_ci(
    scores: np.ndarray, reps: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile CI of the IQM of a (seeds, tasks) matrix."""
    rng = np.random.default_rng(seed)
    n_tasks = scores.shape[1]
    stats = [iqm(_resample(scores, rng, rng.integers(0, n_tasks, n_tasks))) for _ in range(reps)]
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_diff(
    a: np.ndarray, b: np.ndarray, reps: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """IQM(a) - IQM(b) and its percentile CI.

    Both configurations ran on the same tasks, so one task resample is shared;
    their seeds are independent and resampled separately.
    """
    rng = np.random.default_rng(seed)
    n_tasks = a.shape[1]
    stats = []
    for _ in range(reps):
        tasks = rng.integers(0, n_tasks, n_tasks)
        stats.append(iqm(_resample(a, rng, tasks)) - iqm(_resample(b, rng, tasks)))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return iqm(a) - iqm(b), float(lo), float(hi)


def within_noise(lo: float, hi: float) -> bool:
    """The difference's CI includes zero."""
    return lo <= 0.0 <= hi


def parse_run_name(name: str) -> tuple[str, str, int]:
    """``<config>__lr<lr>__s<seed>`` -> (config, lr, seed)."""
    match = RUN_NAME.match(name)
    if match is None:
        raise ValueError(f"run name {name!r} is not <config>__lr<lr>__s<seed>")
    return match["config"], match["lr"], int(match["seed"])


@dataclass
class Runs:
    """The seeds of one (config, lr): final scores (seeds, tasks), curves (seeds, evals, tasks)."""

    seeds: list[int]
    scores: np.ndarray
    steps: np.ndarray
    curves: np.ndarray


def collect(root: str | Path, metric: str) -> dict[tuple[str, str], Runs]:
    """Group the run folders under ``root`` named ``<config>__lr<lr>__s<seed>``.

    Each run folder holds one held-out file at ``held_out/<system>/*.jsonl``.
    Folders with other names (preflight runs) are skipped.
    """
    found: dict[tuple[str, str], dict[int, list[dict]]] = {}
    for run_dir in sorted(Path(root).iterdir()):
        try:
            config, lr, seed = parse_run_name(run_dir.name)
        except ValueError:
            continue
        files = sorted(run_dir.glob("held_out/*/*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"{run_dir} has {len(files)} held-out files, expected 1")
        found.setdefault((config, lr), {})[seed] = read_held_out(files[0])

    runs = {}
    for key, by_seed in found.items():
        seeds = sorted(by_seed)
        steps, _ = curve(by_seed[seeds[0]], metric)
        curves = np.stack([curve(by_seed[s], metric)[1] for s in seeds])
        runs[key] = Runs(seeds, curves[:, -1], steps, curves)
    return runs


def best_lr(runs: dict[tuple[str, str], Runs], config: str) -> str:
    """The learning rate with the highest IQM of final scores for ``config``."""
    return max((lr for c, lr in runs if c == config), key=lambda lr: iqm(runs[(config, lr)].scores))
