"""Analysis of a tuning campaign directory (strategy-v4 §11c).

    python -m jaxarc_baselines.tuning_report DIR [--reps 2000]

For each stage that has results (s1, s2, final), writes to ``DIR/analysis/``:
- ``report.md`` / ``report.json``: configs ranked by the stage's selection score,
  with knob values, mean and IQM dev exact (two-level bootstrap CI over seeds and
  tasks), cell accuracy, training success, collapsed runs and GPU-hours;
- ``sensitivity_<stage>.png``: each config's score against each knob (stage-2
  points overlaid on the stage-1 plot);
- ``profile_final.png``: the performance profile of the final runs with bootstrap bands.

Scores follow rliable (Agarwal et al., 2021): IQM, stratified bootstrap CIs,
performance profiles and the probability of improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from jaxarc_baselines import ablation, tuning

METRIC_KEY = {"exact": "exact", "cell": "cell_accuracy"}


def performance_profile(scores: np.ndarray, taus: np.ndarray | None = None, reps: int = 2000,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Share of seed × task scores >= tau, with a two-level bootstrap 95% band."""
    taus = np.linspace(0, 1, 21) if taus is None else np.asarray(taus, dtype=float)
    flat = scores.reshape(-1)
    frac = np.array([(flat >= t).mean() for t in taus])
    rng = np.random.default_rng(seed)
    n_seeds, n_tasks = scores.shape
    boots = np.empty((reps, len(taus)))
    for i in range(reps):
        s = scores[rng.integers(0, n_seeds, n_seeds)][:, rng.integers(0, n_tasks, n_tasks)].reshape(-1)
        boots[i] = [(s >= t).mean() for t in taus]
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return taus, frac, lo, hi


def probability_of_improvement(a: np.ndarray, b: np.ndarray) -> float:
    """P(X_a > X_b) + 0.5 P(X_a = X_b) over seed pairs, averaged over tasks.

    ``a`` and ``b`` are (seeds, tasks) matrices on the same tasks.
    """
    per_task = []
    for t in range(a.shape[1]):
        x, y = a[:, t][:, None], b[:, t][None, :]
        per_task.append(float(((x > y) + 0.5 * (x == y)).mean()))
    return float(np.mean(per_task))


def task_scores(stage_dir: Path, rows: list[dict], metric: str) -> dict[str, np.ndarray]:
    """{config_id: (seeds, tasks)} final-eval per-task scores of the finished runs."""
    key = METRIC_KEY.get(metric, metric)
    by_cfg: dict[str, list[tuple[int, list[float]]]] = {}
    for row in rows:
        if tuning.read_run(stage_dir, row) is None:
            continue
        files = sorted((stage_dir / row["name"]).glob("held_out/*/*.jsonl"))
        final = ablation.final_task_scores(ablation.read_held_out(files[0]), key)
        by_cfg.setdefault(row["config_id"], []).append((int(row["seed"]), list(final)))
    return {c: np.array([s for _, s in sorted(v)]) for c, v in by_cfg.items()}


def _stage_summary(d: Path, stage: str, configs: dict, reps: int) -> dict | None:
    matrix = d / f"{stage}.jsonl"
    if not matrix.exists() or not (d / stage).exists():
        return None
    rows = tuning.read_matrix(matrix)
    results = tuning.read_results(d / stage, rows)
    if not results:
        return None
    tie = tuning.TIE_BREAK.get(stage, "cell")
    scores = tuning.score_configs(results, tie)
    exact = task_scores(d / stage, rows, "exact")
    audit = {r["config_id"] for r in rows if any(o.startswith("+run_meta.tags=") and "audit" in o
                                                    for o in r["overrides"])}
    out = []
    for s in scores:
        m = exact[s["config_id"]]
        lo, hi = ablation.bootstrap_ci(m, reps=reps) if m.shape[0] > 1 else (float("nan"), float("nan"))
        out.append({**s, "iqm": ablation.iqm(m), "iqm_ci": [lo, hi], "audit": s["config_id"] in audit,
                    "values": configs.get(s["config_id"], {}).get("values", {})})
    walls = [r.wall_s for r in results if r.wall_s is not None]
    return {"runs": len(results), "planned": len(rows), "collapsed": int(sum(r.collapsed for r in results)),
            "gpu_hours": float(sum(walls) / 3600), "configs": out, "_exact": exact}


def _value_str(v) -> str:
    if isinstance(v, dict):  # joint knob: "epochs=4 num_minibatches=16" -> "e4 mb16"
        short = {"epochs": "e", "num_minibatches": "mb"}
        return " ".join(f"{short.get(k.split('.')[-1], k.split('.')[-1] + '=')}{x}" for k, x in v.items())
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _sensitivity_plot(space: tuning.Space, s1: dict, s2: dict | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    knobs = space.knobs
    cols = min(3, len(knobs))
    rows = -(-len(knobs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for ax, k in zip(axes.flat, knobs):
        for summary, label, kw in ((s1, "stage 1 (B/2)", {"color": "0.45", "s": 18}),
                                   (s2, "stage 2 (B)", {"color": "C3", "s": 36, "marker": "D"})):
            if summary is None:
                continue
            pts = [c for c in summary["configs"] if c["values"]]
            if k.dist == "choice":
                labels = [_value_str(v) for v in k.values]
                x = [labels.index(_value_str(c["values"][k.name])) for c in pts]
                ax.set_xticks(range(len(labels)), labels, rotation=30, fontsize=7)
                ax.set_xlim(-0.5, len(labels) - 0.5)
            else:
                x = [c["values"][k.name] for c in pts]
                if k.dist == "log_uniform":
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_formatter(NullFormatter())
                pad = (k.high / k.low) ** 0.03 if k.dist == "log_uniform" else 0.03 * (k.high - k.low)
                ax.set_xlim(*((k.low / pad, k.high * pad) if k.dist == "log_uniform" else (k.low - pad, k.high + pad)))
            ax.scatter(x, [c["score"] for c in pts], label=label, **kw)
        ax.set_title(k.name, fontsize=9)
        ax.set_ylabel("mean dev exact", fontsize=8)
    for ax in list(axes.flat)[len(knobs):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _profile_plot(scores: np.ndarray, path: Path, reps: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    taus, frac, lo, hi = performance_profile(scores, reps=reps)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(taus, frac, color="C0")
    ax.fill_between(taus, lo, hi, color="C0", alpha=0.25)
    ax.set_xlabel("score τ (exact)")
    ax.set_ylabel("fraction of seed × task ≥ τ")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(d: str | Path, reps: int = 2000) -> Path:
    d = Path(d)
    campaign, space, configs = tuning._load_dir(d)
    by_id = {c["config_id"]: c for c in configs}
    frozen = tuning.load_frozen(d / "frozen.yaml") if (d / "frozen.yaml").exists() else None
    if frozen:
        by_id["frozen"] = tuning.frozen_as_config(frozen)
    out = d / "analysis"
    out.mkdir(exist_ok=True)
    stages = {s: _stage_summary(d, s, by_id, reps) for s in tuning.STAGES}
    lines = [f"# Tuning campaign {campaign['campaign_id']}", "",
             f"Space `{space.name}`, budget B = {int(campaign['budget']):,}, sample seed {campaign['sample_seed']}.", ""]
    knob_names = [k.name for k in space.knobs]
    for stage, summ in stages.items():
        if summ is None:
            continue
        lines += [f"## Stage {stage}", "",
                  f"{summ['runs']} of {summ['planned']} runs finished; {summ['collapsed']} collapsed; "
                  f"GPU-hours {summ['gpu_hours']:.1f} (runs only, no preflights).", "",
                  "| Rank | Config | " + " | ".join(knob_names) + " | Seeds | Mean exact | IQM exact [95% CI] | Cell | Train succ % | Collapsed |",
                  "|---:|---|" + "---|" * len(knob_names) + "---:|---:|---|---:|---:|---:|"]
        for c in summ["configs"]:
            vals = " | ".join(_value_str(c["values"].get(k, "")) for k in knob_names)
            tag = " (audit)" if c["audit"] else ""
            lines.append(f"| {c['rank']} | {c['config_id']}{tag} | {vals} | {c['n']} | {c['score']:.3f} | "
                         f"{c['iqm']:.3f} [{c['iqm_ci'][0]:.3f}, {c['iqm_ci'][1]:.3f}] | {c['cell']:.3f} | "
                         f"{c['train_success']:.1f} | {c['collapsed']} |")
        lines.append("")
    if stages["s1"] is not None:
        _sensitivity_plot(space, stages["s1"], stages["s2"], out / "sensitivity_s1.png")
        lines += ["![sensitivity](sensitivity_s1.png)", ""]
    if stages["final"] is not None and "frozen" in stages["final"]["_exact"]:
        m = stages["final"]["_exact"]["frozen"]
        _profile_plot(m, out / "profile_final.png", reps)
        lines += ["![profile](profile_final.png)", ""]
    total = sum(s["gpu_hours"] for s in stages.values() if s)
    lines += [f"Total GPU-hours (runs): {total:.1f}", ""]
    (out / "report.md").write_text("\n".join(lines))
    clean = {s: {k: v for k, v in summ.items() if k != "_exact"} for s, summ in stages.items() if summ}
    (out / "report.json").write_text(json.dumps(clean, indent=1, default=float))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path)
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args(argv)
    print(report(args.dir, args.reps) / "report.md")


if __name__ == "__main__":
    main()
