"""Run metadata for the paper experiments: W&B project, tags, group, job type, provenance.

A config opts in with a ``run_meta`` block; configs without one are left alone.

    run_meta:
      stage: tune-l1        # required, one of STAGES
      track: E              # E | G | M (required for tune-*, final, scale)
      question: Q2          # Q1..Q5 (required for tune-*, final, scale)
      campaign_id: E-ppo-mini   # W&B group (required for tune-*, final, scale)
      config_id: c07        # optional, the sampled configuration
      tags: []              # optional extra tags
      allow_dirty: false    # let a recorded stage run from uncommitted code

On the command line: ``+run_meta.stage=smoke``.

Opted-in runs log to the ``jaxarc-tmlr`` project. The commits and dirty flags of
jaxarc-baselines, JaxARC and Stoix are written to ``run_meta.provenance``, which
W&B stores with the rest of the config. Recorded stages refuse to run from a
dirty tree unless ``allow_dirty`` is set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PROJECT = "jaxarc-tmlr"
STAGES = (
    "smoke",
    "throughput",
    "tune-l1",
    "tune-l1-s2",
    "tune-l2",
    "tune-l2-s2",
    "final",
    "scale",
)
RECORDED = tuple(s for s in STAGES if s.startswith("tune") or s in ("final", "scale"))
TRACKS = ("E", "G", "M")
QUESTIONS = ("Q1", "Q2", "Q3", "Q4", "Q5")


def _git(path: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def provenance() -> dict:
    """Commit and dirty flag of each repository the run imports code from."""
    import jaxarc
    import stoix

    import jaxarc_baselines

    return {
        "baselines": _git(Path(jaxarc_baselines.__file__).parent),
        "jaxarc": _git(Path(jaxarc.__file__).parent),
        "stoix": _git(Path(stoix.__path__[0])),
    }


def _check(meta: DictConfig) -> None:
    stage = meta.get("stage")
    if stage not in STAGES:
        raise ValueError(f"run_meta.stage must be one of {STAGES}, got {stage!r}")
    track, question = meta.get("track"), meta.get("question")
    if track is not None and track not in TRACKS:
        raise ValueError(f"run_meta.track must be one of {TRACKS}, got {track!r}")
    if question is not None and question not in QUESTIONS:
        raise ValueError(f"run_meta.question must be one of {QUESTIONS}, got {question!r}")
    if stage in RECORDED:
        missing = [k for k in ("track", "question", "campaign_id") if not meta.get(k)]
        if missing:
            raise ValueError(f"run_meta for stage {stage!r} needs {', '.join(missing)}")


def apply_run_meta(cfg: DictConfig) -> None:
    """Fill in W&B settings and provenance from ``cfg.run_meta``, in place."""
    meta = cfg.get("run_meta")
    if meta is None:
        return
    _check(meta)

    info = provenance()
    dirty = [name for name, repo in info.items() if repo["dirty"]]
    if dirty and meta.stage in RECORDED and not meta.get("allow_dirty", False):
        raise RuntimeError(
            f"stage {meta.stage!r} must run from committed code; uncommitted changes in: "
            + ", ".join(dirty)
        )
    # Strings, not bools: Stoix's W&B config logging stringifies bools anyway.
    meta.provenance = OmegaConf.create(
        {name: {"commit": r["commit"], "dirty": "yes" if r["dirty"] else "no"} for name, r in info.items()}
    )
    meta.code_state = "dirty" if dirty else "clean"

    tags = [meta.stage, meta.get("track"), meta.get("question")]
    tags += [cfg.system.system_name, cfg.env.scenario.name, *meta.get("tags", [])]
    wandb = cfg.logger.loggers.wandb
    wandb.project = PROJECT
    wandb.tag = [str(t) for t in tags if t]
    wandb.group_tag = [meta.campaign_id] if meta.get("campaign_id") else []
    os.environ["WANDB_JOB_TYPE"] = meta.stage
