"""Turn Stoix's per-episode held-out outcomes into per-task scores.

The JaxARC eval env reports each episode's outcome (see
``jaxarc.evaluation``). This module summarises them per task for logging and
appends the per-task scores of every evaluation to a JSON-lines file, which
the bootstrap over tasks and seeds reads later.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from jaxarc.evaluation import summarize

EPISODE_KEYS = ("task_idx", "pair_idx", "exact", "cell_accuracy", "submitted")


def summarize_eval_metrics(metrics: dict, k: int = 2) -> tuple[dict, dict | None]:
    """Replace per-episode outcomes with task-weighted scalars.

    Returns the metrics to log and the full summary, or the metrics unchanged
    and None when they hold no outcomes (training logs, other envs).
    """
    if not all(key in metrics for key in EPISODE_KEYS):
        return metrics, None
    summary = summarize(metrics, k=k)
    logged = {key: value for key, value in metrics.items() if key not in EPISODE_KEYS}
    for name in ("exact", f"pass@{k}", "cell_accuracy", "submit_rate"):
        logged[name] = np.asarray(summary[name])
    return logged, summary


class HeldOutRecorder:
    """Appends one JSON line per evaluation to ``<dir>/<run>_seed<seed>.jsonl``."""

    def __init__(self, directory: str | Path, seed: int, scenario: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.path = Path(directory) / f"{stamp}_seed{seed}.jsonl"
        self.seed = seed
        self.scenario = scenario

    def record(self, summary: dict, t: int, eval_step: int, event: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = {
            "event": event,
            "t": int(t),
            "eval_step": int(eval_step),
            "seed": self.seed,
            "eval_scenario": self.scenario,
            **{k: v for k, v in summary.items() if k != "per_task"},
            "per_task": {k: np.asarray(v).tolist() for k, v in summary["per_task"].items()},
        }
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")
