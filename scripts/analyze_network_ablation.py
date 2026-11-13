"""Tables, decision checks and learning curves for the MiniARC network ablation.

Usage:
    python scripts/analyze_network_ablation.py RUN_ROOT [--out DIR]

RUN_ROOT holds one folder per run named ``<config>__lr<lr>__s<seed>``. Scores
are the dev tasks' held-out test pairs at the last evaluation: IQM over
seed-task scores with a 95% two-level bootstrap CI (``jaxarc_baselines.ablation``).
Each config is compared at its best learning rate, chosen by cell accuracy
(the plan's selection metric).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from jaxarc_baselines import ablation  # noqa: E402

METRICS = ("cell_accuracy", "exact")
SELECT = "cell_accuracy"
# (question, candidate configs, reference configs): each side at its best (config, lr).
CHECKS = [
    ("shared vs separate torso", ["shr-cv0.25", "shr-cv0.5", "shr-cv1.0"], ["sep-f32"]),
    ("raw vs embedded colour", ["sep-raw"], ["sep-f32"]),
    ("bf16 vs f32, separate", ["sep-bf16"], ["sep-f32"]),
    ("bf16 vs f32, shared (c_v 0.5)", ["shr-bf16-cv0.5"], ["shr-cv0.5"]),
    ("spatial vs stock network", ["sep-f32"], ["stock"]),
]


def best(runs: dict, configs: list[str]) -> tuple[str, str] | None:
    keys = [(c, ablation.best_lr(runs, c)) for c in configs if any(k[0] == c for k in runs)]
    return max(keys, key=lambda k: ablation.iqm(runs[k].scores)) if keys else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=5000)
    args = parser.parse_args()
    out = args.out or args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    runs = {m: ablation.collect(args.root, m) for m in METRICS}
    report = {"rows": [], "checks": []}
    lines = [
        "| Config | lr | Seeds | Cell accuracy IQM [95% CI] | Exact IQM [95% CI] | Cell accuracy mean | Exact mean |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for key in sorted(runs[SELECT]):
        row = {"config": key[0], "lr": key[1], "seeds": runs[SELECT][key].seeds}
        cells = []
        for m in METRICS:
            scores = runs[m][key].scores
            lo, hi = ablation.bootstrap_ci(scores, reps=args.reps)
            row[m] = {"iqm": ablation.iqm(scores), "lo": lo, "hi": hi, "mean": float(scores.mean())}
            cells.append(f"{ablation.iqm(scores):.3f} [{lo:.3f}, {hi:.3f}]")
        cells += [f"{row[m]['mean']:.3f}" for m in METRICS]
        report["rows"].append(row)
        lines.append(f"| {key[0]} | {key[1]} | {len(row['seeds'])} | {' | '.join(cells)} |")

    lines += ["", "| Check | Candidate | Reference | Metric | Difference [95% CI] | Within noise |", "|---|---|---|---|---|---|"]
    for name, cand, ref in CHECKS:
        a, b = best(runs[SELECT], cand), best(runs[SELECT], ref)
        if a is None or b is None:
            continue
        for m in METRICS:
            diff, lo, hi = ablation.bootstrap_diff(runs[m][a].scores, runs[m][b].scores, reps=args.reps)
            noise = ablation.within_noise(lo, hi)
            report["checks"].append(
                {"check": name, "candidate": a, "reference": b, "metric": m, "diff": diff, "lo": lo, "hi": hi, "within_noise": noise}
            )
            lines.append(
                f"| {name} | {a[0]} @ {a[1]} | {b[0]} @ {b[1]} | {m} | {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] | {'yes' if noise else 'no'} |"
            )

    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("\n".join(lines))

    configs = sorted({c for c, _ in runs[SELECT]})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(6 * len(METRICS), 4), squeeze=False)
    for ax, m in zip(axes[0], METRICS):
        for c in configs:
            r = runs[m][(c, ablation.best_lr(runs[SELECT], c))]
            iqms = [ablation.iqm(r.curves[:, i]) for i in range(r.curves.shape[1])]
            ax.plot(np.asarray(r.steps) / 1e6, iqms, label=f"{c} (lr {ablation.best_lr(runs[SELECT], c)})")
        ax.set(xlabel="Environment steps (M)", ylabel=f"{m}, IQM over seeds and dev tasks", title=m)
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)


if __name__ == "__main__":
    main()
