"""Two-stage random search over a published space (strategy-v4 §11c).

Stage 1 (``s1``): ``stage1.configs`` configurations sampled with a fixed seed, each on
``stage1.seeds``, run with the final budget's schedule and stopped at
``stage1.fraction`` of it. Score: mean dev exact over seeds and dev tasks at the last
eval; ties by training success.

Stage 2 (``s2``): the top ``stage2.top_k`` on fresh seeds at the full budget B, plus
optional audit configurations drawn from below the cut. Score: mean dev exact; ties
by dev cell accuracy. The winner is frozen to a file.

Final (``final``): the frozen configuration on ``final.seeds``, evaluated on the test
scenario.

This module does not import JAX, so it runs on a cluster login node.

    python -m jaxarc_baselines.tuning plan CAMPAIGN.yaml --out DIR
    python -m jaxarc_baselines.tuning submit DIR s1 [--chunk 8] [--dry-run]
    python -m jaxarc_baselines.tuning select DIR      # s1 results -> s2 matrix
    python -m jaxarc_baselines.tuning freeze DIR      # s2 results -> frozen.yaml, final matrix
    python -m jaxarc_baselines.tuning row MATRIX INDEX   # used by the sbatch script
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from jaxarc_baselines import ablation

REPO = Path(__file__).resolve().parents[2]
SPACES = REPO / "experiments" / "tuning" / "spaces"
SBATCH = REPO / "scripts" / "tuning" / "tune_array.sbatch"
DISTS = ("log_uniform", "uniform", "choice")
SCHEDULES = ("lr_horizon", "epsilon_fraction")
STAGES = ("s1", "s2", "final")
RUN_META_STAGE = {"s1": "tune-s1", "s2": "tune-s2", "final": "final"}
TIE_BREAK = {"s1": "train_success", "s2": "cell"}
COLLAPSE_LIMIT = 0.10  # share of a stage's runs that may collapse before selection stops
AUDIT_MARGIN = 0.03  # an audit config this far above the winner extends stage 2


# --- search space ----------------------------------------------------------------


@dataclass(frozen=True)
class Knob:
    name: str
    dist: str
    path: str | None = None
    low: float | None = None
    high: float | None = None
    values: tuple | None = None


@dataclass(frozen=True)
class Space:
    name: str
    schedule: str
    knobs: tuple[Knob, ...]


def load_space(path: str | Path) -> Space:
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get("schedule") not in SCHEDULES:
        raise ValueError(f"{path}: schedule must be one of {SCHEDULES}, got {raw.get('schedule')!r}")
    knobs = []
    for name, spec in raw["knobs"].items():
        dist = spec.get("dist")
        if dist not in DISTS:
            raise ValueError(f"{path}: knob {name!r} has distribution {dist!r}, not one of {DISTS}")
        if dist == "choice":
            values = tuple(spec["values"])
            if not values:
                raise ValueError(f"{path}: knob {name!r} has no values")
            if any(isinstance(v, dict) for v in values) and spec.get("path"):
                raise ValueError(f"{path}: joint knob {name!r} takes no path")
            if not any(isinstance(v, dict) for v in values) and not spec.get("path"):
                raise ValueError(f"{path}: knob {name!r} needs a path")
            knobs.append(Knob(name, dist, spec.get("path"), values=values))
        else:
            low, high = float(spec["low"]), float(spec["high"])
            if not (low < high) or (dist == "log_uniform" and low <= 0):
                raise ValueError(f"{path}: knob {name!r} has a bad range [{low}, {high}]")
            knobs.append(Knob(name, dist, spec["path"], low, high))
    return Space(raw["name"], raw["schedule"], tuple(knobs))


def _round(x: float) -> float:
    return float(f"{x:.3g}")


def sample_configs(space: Space, n: int, seed: int) -> list[dict]:
    """``n`` configurations ``{"config_id": "c00", "values": {knob: value}}``, fixed by ``seed``."""
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n):
        values = {}
        for k in space.knobs:
            if k.dist == "choice":
                values[k.name] = k.values[int(rng.integers(len(k.values)))]
            elif k.dist == "uniform":
                values[k.name] = min(max(_round(rng.uniform(k.low, k.high)), k.low), k.high)
            else:
                x = float(np.exp(rng.uniform(np.log(k.low), np.log(k.high))))
                values[k.name] = min(max(_round(x), k.low), k.high)
        configs.append({"config_id": f"c{i:02d}", "values": values})
    return configs


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def knob_overrides(space: Space, values: dict) -> list[str]:
    out = []
    for k in space.knobs:
        v = values[k.name]
        if isinstance(v, dict):
            out += [f"{path}={_fmt(x)}" for path, x in v.items()]
        else:
            out.append(f"{k.path}={_fmt(v)}")
    return out


# --- campaigns and run matrices ------------------------------------------------


def load_campaign(path: str | Path) -> dict:
    campaign = yaml.safe_load(Path(path).read_text())
    for key in ("campaign_id", "track", "question", "space", "config_name", "budget", "num_evaluation",
                "sample_seed", "stage1", "stage2", "final", "dev_scenario", "test_scenario", "fixed"):
        if key not in campaign:
            raise ValueError(f"{path}: campaign needs {key!r}")
    frac = campaign["stage1"]["fraction"]
    if not 0 < frac <= 1 or (campaign["num_evaluation"] * frac) % 1:
        raise ValueError(f"{path}: stage1.fraction {frac} must split num_evaluation evenly")
    return campaign


def _seeds(spec) -> list[int]:
    if isinstance(spec, dict):
        return list(range(spec["first"], spec["first"] + spec["count"]))
    return [int(s) for s in spec]


def stage_seeds(campaign: dict, stage: str) -> list[int]:
    return _seeds(campaign[{"s1": "stage1", "s2": "stage2", "final": "final"}[stage]]["seeds"])


def _stage_values(space: Space, values: dict, frac: float) -> dict:
    """Knob values for a run that stops at ``frac`` of the final schedule."""
    if frac == 1 or space.schedule != "epsilon_fraction":
        return values
    knob = next((k for k in space.knobs if k.path == "system.exploration_fraction"), None)
    if knob is None:
        raise ValueError(f"space {space.name}: epsilon_fraction schedule needs an exploration_fraction knob")
    stretched = values[knob.name] / frac
    if stretched > 1:
        raise ValueError(f"exploration_fraction {values[knob.name]} / {frac} exceeds 1")
    return {**values, knob.name: stretched}


def stage_matrix(campaign: dict, space: Space, configs: list[dict], stage: str,
                 seeds: list[int] | None = None, tags: tuple[str, ...] = ()) -> list[dict]:
    """One row per (config, seed): ``{"name", "config_id", "seed", "overrides"}``."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    budget, n_eval = int(campaign["budget"]), int(campaign["num_evaluation"])
    frac = campaign["stage1"]["fraction"] if stage == "s1" else 1
    steps, evals = int(budget * frac), int(n_eval * frac)
    schedule = []
    if stage == "s1" and frac < 1 and space.schedule == "lr_horizon":
        schedule = [f"system.lr_schedule_timesteps={budget}"]
    scenario = campaign["test_scenario"] if stage == "final" else campaign["dev_scenario"]
    rows = []
    for cfg in configs:
        values = _stage_values(space, cfg["values"], frac)
        for seed in seeds if seeds is not None else stage_seeds(campaign, stage):
            cid = cfg["config_id"]
            meta = [f"+run_meta.stage={RUN_META_STAGE[stage]}", f"+run_meta.track={campaign['track']}",
                    f"+run_meta.question={campaign['question']}",
                    f"+run_meta.campaign_id={campaign['campaign_id']}", f"+run_meta.config_id={cid}"]
            if tags:
                meta.append(f"+run_meta.tags=[{','.join(tags)}]")
            overrides = [*campaign["fixed"], *knob_overrides(space, values), f"arch.seed={seed}",
                         f"arch.total_timesteps={steps}", f"arch.num_evaluation={evals}", *schedule,
                         f"+env.eval_scenario={scenario}", *meta]
            rows.append({"name": f"{cid}__s{seed}", "config_id": cid, "seed": seed, "overrides": overrides})
    return rows


def write_matrix(rows: list[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def read_matrix(path: str | Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# --- results ---------------------------------------------------------------------


class RunResult(NamedTuple):
    config_id: str
    seed: int
    exact: float  # mean dev exact over tasks, last eval
    cell: float  # mean dev cell accuracy, last eval
    train_success: float  # ACTOR success rate (%), last log
    collapsed: bool  # strategy-v4 §10a
    wall_s: float | None


ANSI = re.compile(r"\x1b\[[0-9;]*m")
TRUNC = re.compile(r"ACTOR - .*?Truncation rate: ([0-9.]+)")
SUCCESS = re.compile(r"ACTOR - .*?\| Success rate: ([0-9.]+)")


def is_collapsed(submit_rates: list[float], truncation: list[float], k: int = 3) -> bool:
    """Dev submit rate 0 and training truncation >= 99.9% at each of the last k evals."""
    return (len(submit_rates) >= k and len(truncation) >= k
            and all(s == 0 for s in submit_rates[-k:]) and all(t >= 99.9 for t in truncation[-k:]))


def read_run(stage_dir: Path, row: dict) -> RunResult | None:
    """The result of one run, or None if it hasn't finished."""
    files = sorted((stage_dir / row["name"]).glob("held_out/*/*.jsonl"))
    log = stage_dir / f"{row['name']}.log"
    if len(files) != 1 or not log.exists():
        return None
    records = ablation.read_held_out(files[0])
    evals = sorted((r for r in records if r["event"] == ablation.EVAL_EVENT), key=lambda r: r["t"])
    expected = next((int(o.split("=")[1]) for o in row["overrides"] if o.startswith("arch.num_evaluation=")), None)
    if not evals or (expected is not None and len(evals) < expected):
        return None
    text = ANSI.sub("", log.read_text())
    trunc = [float(x) for x in TRUNC.findall(text)]
    success = [float(x) for x in SUCCESS.findall(text)]
    wall = stage_dir / f"{row['name']}.wall"
    last = evals[-1]["per_task"]
    return RunResult(
        config_id=row["config_id"], seed=int(row["seed"]),
        exact=float(np.mean(last["exact"])), cell=float(np.mean(last["cell_accuracy"])),
        train_success=success[-1] if success else float("nan"),
        collapsed=is_collapsed([r["submit_rate"] for r in evals], trunc),
        wall_s=float(wall.read_text()) if wall.exists() else None,
    )


def read_results(stage_dir: str | Path, rows: list[dict]) -> list[RunResult]:
    out = [read_run(Path(stage_dir), r) for r in rows]
    return [r for r in out if r is not None]


# --- selection -------------------------------------------------------------------


class CollapseAlarm(RuntimeError):
    pass


def score_configs(results: list[RunResult], tie_break: str) -> list[dict]:
    """Configs ranked by mean exact over seeds, then the tie-break metric, then id."""
    by_cfg: dict[str, list[RunResult]] = {}
    for r in results:
        by_cfg.setdefault(r.config_id, []).append(r)
    scores = []
    for cid, runs in by_cfg.items():
        scores.append({
            "config_id": cid, "n": len(runs), "seeds": sorted(r.seed for r in runs),
            "score": float(np.mean([r.exact for r in runs])),
            "tie_break": float(np.nanmean([getattr(r, tie_break) for r in runs])),
            "cell": float(np.mean([r.cell for r in runs])),
            "train_success": float(np.nanmean([r.train_success for r in runs])),
            "collapsed": int(sum(r.collapsed for r in runs)),
        })
    # Round so float noise in the means doesn't break a tie.
    scores.sort(key=lambda s: (-round(s["score"], 9), -round(s["tie_break"], 9), s["config_id"]))
    for rank, s in enumerate(scores, 1):
        s["rank"] = rank
    return scores


def _check(results: list[RunResult], expected_seeds: list[int], allow_missing: bool, force: bool,
           config_ids: list[str] | None = None) -> None:
    have: dict[str, set[int]] = {}
    for r in results:
        have.setdefault(r.config_id, set()).add(r.seed)
    ids = config_ids if config_ids is not None else sorted(have)
    missing = {c: sorted(set(expected_seeds) - have.get(c, set())) for c in ids}
    missing = {c: m for c, m in missing.items() if m}
    if missing and not allow_missing:
        raise ValueError(f"runs missing (config: seeds): {missing}")
    rate = sum(r.collapsed for r in results) / max(len(results), 1)
    if rate > COLLAPSE_LIMIT and not force:
        raise CollapseAlarm(f"{rate:.0%} of runs collapsed (limit {COLLAPSE_LIMIT:.0%}); report before selecting")


def select_top(results: list[RunResult], k: int, expected_seeds: list[int], tie_break: str,
               allow_missing: bool = False, force: bool = False,
               config_ids: list[str] | None = None) -> list[str]:
    _check(results, expected_seeds, allow_missing, force, config_ids)
    return [s["config_id"] for s in score_configs(results, tie_break)[:k]]


def audit_ids(campaign: dict, ranked: list[str]) -> list[str]:
    """Audit configs for stage 2: drawn at random (fixed seed) from below the top k."""
    audit = campaign["stage2"].get("audit")
    if not audit:
        return []
    below = ranked[campaign["stage2"]["top_k"]:]
    rng = np.random.default_rng(campaign["sample_seed"] + 1)
    pick = rng.choice(len(below), size=min(audit["configs"], len(below)), replace=False)
    return [below[i] for i in sorted(pick)]


def freeze(campaign: dict, space: Space, configs: list[dict], s2: list[RunResult],
           audit: list[str] = (), allow_missing: bool = False, force: bool = False) -> dict:
    """The stage-2 winner (audit configs excluded) as a frozen-config record."""
    audit = list(audit)
    main = [r for r in s2 if r.config_id not in audit]
    top_ids = sorted({r.config_id for r in main})
    _check(main, stage_seeds(campaign, "s2"), allow_missing, force, top_ids)
    scores = score_configs(main, TIE_BREAK["s2"])
    winner = scores[0]
    values = next(c for c in configs if c["config_id"] == winner["config_id"])["values"]
    frozen = {
        "campaign_id": campaign["campaign_id"], "space": space.name, "sample_seed": campaign["sample_seed"],
        "budget": int(campaign["budget"]), "config_id": winner["config_id"],
        "values": values, "overrides": knob_overrides(space, values),
        "stage2": [{k: s[k] for k in ("config_id", "rank", "n", "score", "cell", "collapsed")} for s in scores],
    }
    if audit:
        a_scores = score_configs([r for r in s2 if r.config_id in audit], TIE_BREAK["s2"])
        best = max((s["score"] for s in a_scores), default=float("-inf"))
        frozen["audit"] = {
            "config_ids": audit, "margin": AUDIT_MARGIN,
            "scores": [{k: s[k] for k in ("config_id", "n", "score")} for s in a_scores],
            "extend_stage2": bool(best > winner["score"] + AUDIT_MARGIN),
        }
    return frozen


def write_frozen(frozen: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(frozen, sort_keys=False))


def load_frozen(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def frozen_as_config(frozen: dict) -> dict:
    return {"config_id": "frozen", "values": frozen["values"]}


# --- campaign directory and CLI --------------------------------------------------


def _load_dir(d: Path) -> tuple[dict, Space, list[dict]]:
    campaign = load_campaign(d / "campaign.yaml")
    space = load_space(d / "space.yaml")
    configs = json.loads((d / "configs.json").read_text())
    return campaign, space, configs


def cmd_plan(args) -> None:
    campaign = load_campaign(args.campaign)
    space_path = SPACES / f"{campaign['space']}.yaml"
    space = load_space(space_path)
    out = Path(args.out)
    if (out / "s1.jsonl").exists() and not args.overwrite:
        sys.exit(f"{out} already has a plan; pass --overwrite to replace it")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.campaign, out / "campaign.yaml")
    shutil.copy(space_path, out / "space.yaml")
    configs = sample_configs(space, campaign["stage1"]["configs"], campaign["sample_seed"])
    (out / "configs.json").write_text(json.dumps(configs, indent=1))
    rows = stage_matrix(campaign, space, configs, "s1")
    write_matrix(rows, out / "s1.jsonl")
    print(f"{len(configs)} configs, {len(rows)} stage-1 runs -> {out}")


def cmd_select(args) -> None:
    d = Path(args.dir)
    campaign, space, configs = _load_dir(d)
    rows = read_matrix(d / "s1.jsonl")
    results = read_results(d / "s1", rows)
    print(f"{len(results)} of {len(rows)} stage-1 runs finished")
    ids = [c["config_id"] for c in configs]
    top = select_top(results, campaign["stage2"]["top_k"], stage_seeds(campaign, "s1"), TIE_BREAK["s1"],
                     allow_missing=args.allow_missing, force=args.force, config_ids=ids)
    scores = score_configs(results, TIE_BREAK["s1"])
    audit = audit_ids(campaign, [s["config_id"] for s in scores])
    (d / "s1_scores.json").write_text(json.dumps({"scores": scores, "top": top, "audit": audit}, indent=1))
    by_id = {c["config_id"]: c for c in configs}
    s2 = stage_matrix(campaign, space, [by_id[c] for c in top], "s2")
    if audit:
        a_seeds = stage_seeds(campaign, "s2")[: campaign["stage2"]["audit"]["seeds"]]
        s2 += stage_matrix(campaign, space, [by_id[c] for c in audit], "s2", seeds=a_seeds, tags=("audit",))
    write_matrix(s2, d / "s2.jsonl")
    print(f"top {top}, audit {audit}: {len(s2)} stage-2 runs -> {d / 's2.jsonl'}")


def cmd_freeze(args) -> None:
    d = Path(args.dir)
    campaign, space, configs = _load_dir(d)
    rows = read_matrix(d / "s2.jsonl")
    results = read_results(d / "s2", rows)
    print(f"{len(results)} of {len(rows)} stage-2 runs finished")
    audit = json.loads((d / "s1_scores.json").read_text()).get("audit", [])
    frozen = freeze(campaign, space, configs, results, audit=audit,
                    allow_missing=args.allow_missing, force=args.force)
    write_frozen(frozen, d / "frozen.yaml")
    final = stage_matrix(campaign, space, [frozen_as_config(frozen)], "final")
    write_matrix(final, d / "final.jsonl")
    print(f"frozen {frozen['config_id']} -> {d / 'frozen.yaml'}; {len(final)} final runs")
    if frozen.get("audit", {}).get("extend_stage2"):
        print("AUDIT: an audit config beat the winner by more than the margin; extend stage 2 (§11c)")


def cmd_row(args) -> None:
    row = read_matrix(args.matrix)[args.index]
    print(row["name"])
    print("\n".join(row["overrides"]))


def cmd_submit(args) -> None:
    d = Path(args.dir).resolve()
    matrix = d / f"{args.stage}.jsonl"
    campaign = load_campaign(d / "campaign.yaml")
    n = len(read_matrix(matrix))
    tasks = -(-n // args.chunk)
    cmd = ["sbatch", f"--array=0-{tasks - 1}", f"--partition={args.partition}", f"--time={args.time}",
           f"--job-name=tune-{campaign['campaign_id']}-{args.stage}", f"--output={d}/logs/%x-%A_%a.log",
           str(SBATCH), str(matrix), str(args.chunk), campaign["config_name"]]
    print(" ".join(cmd))
    if not args.dry_run:
        (d / "logs").mkdir(exist_ok=True)
        subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("campaign", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_plan)
    for name, fn in (("select", cmd_select), ("freeze", cmd_freeze)):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path)
        p.add_argument("--allow-missing", action="store_true")
        p.add_argument("--force", action="store_true", help="select despite the collapse alarm")
        p.set_defaults(fn=fn)
    p = sub.add_parser("row")
    p.add_argument("matrix", type=Path)
    p.add_argument("index", type=int)
    p.set_defaults(fn=cmd_row)
    p = sub.add_parser("submit")
    p.add_argument("dir", type=Path)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--chunk", type=int, default=8, help="runs per array task")
    p.add_argument("--partition", default="h100-single")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_submit)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
