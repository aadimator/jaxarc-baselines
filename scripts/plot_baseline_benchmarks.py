"""Plot ARC baseline benchmark curves for selected ARC-AGI and ConceptARC tasks.

The script reads scheduler-produced success-curve CSV files from
results/baseline_benchmarks/subset_baselines_10m_5seed/runs and writes two
figures:
  1. success_ratio: query exact-match success rate.
  2. episode_return: mean episodic return rescaled to [0, 1] across plotted data.
"""

from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pathlib import Path

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = (
    REPO_ROOT
    / "results"
    / "baseline_benchmarks"
    / "subset_baselines_10m_5seed"
    / "runs"
)

OUT_DIR = (
    REPO_ROOT
    / "results"
    / "baseline_benchmarks"
    / "subset_baselines_10m_5seed"
    / "figures"
)

# Configure tasks here. Mix tasks from either dataset; order determines panel order.
TASKS_TO_PLOT: list[str] = [
    # ARC-AGI tasks
    "agi1_entry",
    "agi1_easy",
    "agi1_medium",
    "agi1_hard",
    "agi1_multiple_solutions",
    # "agi1_single_easy",
    "agi1_tedious",
    # ConceptARC tasks
    # "concept_abovebelow",
    "concept_center",
    # "concept_cleanup",
    # "concept_completeshape",
    # "concept_copy",
    "concept_count",
    # "concept_extendtoboundary",
    "concept_extractobjects",
    "concept_fillednotfilled",
    # "concept_horizontalvertical",
    "concept_insideoutside",
    # "concept_movetoboundary",
    # "concept_order",
    # "concept_samedifferent",
    "concept_topbottom2d",
    # "concept_topbottom3d",
]

SHOW_DATASET_TITLE = True

#  dataset identity 
DATASET_CFG: dict[str, dict] = {
    "agi1": {"label": "ARC-AGI 1", "subdir": "agi1"},
    "concept": {"label": "ConceptARC", "subdir": "concept"},
}


def dataset_of(task: str) -> str:
    """Return dataset key ('agi1' or 'concept') from task name."""
    if task.startswith("agi1_"):
        return "agi1"
    if task.startswith("concept_"):
        return "concept"
    raise ValueError(f"Unknown dataset prefix for task: {task!r}")


#  pretty task labels 
TASK_LABELS: dict[str, str] = {
    # ARC-AGI
    "agi1_easy": "Easy",
    "agi1_easy_eval": "Easy (Eval)",
    "agi1_easy_train": "Easy (Train)",
    "agi1_entry": "Entry",
    "agi1_entry_eval": "Entry (Eval)",
    "agi1_entry_train": "Entry (Train)",
    "agi1_hard": "Hard",
    "agi1_hard_eval": "Hard (Eval)",
    "agi1_hard_train": "Hard (Train)",
    "agi1_medium": "Medium",
    "agi1_medium_eval": "Medium (Eval)",
    "agi1_medium_train": "Medium (Train)",
    "agi1_multiple_solutions": "Multiple Solutions",
    "agi1_multiple_solutions_eval": "Multiple Solutions (Eval)",
    "agi1_multiple_solutions_train": "Multiple Solutions (Train)",
    "agi1_single_easy": "Single Easy",
    "agi1_tedious": "Tedious",
    # ConceptARC
    "concept_abovebelow": "Above / Below",
    "concept_center": "Center",
    "concept_cleanup": "Clean Up",
    "concept_completeshape": "Complete Shape",
    "concept_copy": "Copy",
    "concept_count": "Count",
    "concept_extendtoboundary": "Extend to Boundary",
    "concept_extractobjects": "Extract Objects",
    "concept_fillednotfilled": "Filled / Not Filled",
    "concept_horizontalvertical": "Horizontal / Vertical",
    "concept_insideoutside": "Inside / Outside",
    "concept_movetoboundary": "Move to Boundary",
    "concept_order": "Order",
    "concept_samedifferent": "Same / Different",
    "concept_topbottom2d": "Top / Bottom 2D",
    "concept_topbottom3d": "Top / Bottom 3D",
}

#  algorithm config 
ALGO_CONFIGS: dict[str, dict] = {
    "ff_ppo": {"label": "PPO", "color": "#3A85C8", "marker": "o", "zorder": 4},
    "ff_ddqn": {"label": "DDQN", "color": "#D4A847", "marker": "s", "zorder": 3},
    "ff_pqn": {"label": "PQN", "color": "#E07040", "marker": "^", "zorder": 2},
    "ff_reinforce": {
        "label": "REINFORCE",
        "color": "#5CB87A",
        "marker": "D",
        "zorder": 1,
    },
}

#  data loading 


def load_algo_task_seeds(
    algo: str, task: str, metric: str
) -> dict[int, pd.Series] | None:
    """Return {seed_id: Series(timestep -> metric)}, or None if missing."""
    ds = dataset_of(task)
    subdir = DATASET_CFG[ds]["subdir"]
    task_dir = RUNS_ROOT / algo / subdir / task
    if not task_dir.exists():
        return None
    seed_series: dict[int, pd.Series] = {}
    for seed_dir in sorted(task_dir.glob("seed_*")):
        csv_path = seed_dir / "success_curve.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path).dropna(subset=["timestep", metric])
        df = df.sort_values("timestep")
        seed_id = int(seed_dir.name.split("_")[-1])
        seed_series[seed_id] = pd.Series(df[metric].values, index=df["timestep"].values)
    return seed_series if seed_series else None


def compute_band(
    seed_series: dict[int, pd.Series], lo: float = 5, hi: float = 95
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frames = [s.rename(k) for k, s in seed_series.items()]
    combined = pd.concat(frames, axis=1).sort_index().ffill().bfill()
    arr = combined.values
    return (
        combined.index.values,
        np.nanmean(arr, axis=1),
        np.nanpercentile(arr, lo, axis=1),
        np.nanpercentile(arr, hi, axis=1),
    )


#  plotting helpers 


def _fmt_millions(x, pos):
    return f"{x / 1e6:.1f}"


def style_ax(
    ax: plt.Axes,
    yticks: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
):
    """Apply base baseline panel style."""
    ax.set_facecolor("#FAFAFA")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(yticks)
    for y in yticks:
        ax.axhline(y, color="#CCCCCC", linewidth=0.6, linestyle="--", zorder=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.tick_params(axis="both", labelsize=7)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def make_figure(
    metric: str,
    ylabel: str,
    out_stem: str,
    tasks: list[str],
    n_cols: int = 4,
    clip_lo: float | None = None,
    clip_hi: float | None = None,
    loader=None,
):
    n_rows = int(np.ceil(len(tasks) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3.0, n_rows * 2.0),
        constrained_layout=False,
    )
    axes_flat: list[plt.Axes] = np.array(axes).flatten().tolist()

    legend_handles: dict[str, Line2D] = {}

    for idx, task in enumerate(tasks):
        ax = axes_flat[idx]

        # Title above the stripe
        # Build title: "Task Name (Dataset)"
        ds_label = DATASET_CFG[dataset_of(task)]["label"]
        task_label = TASK_LABELS.get(task, task)
        title = f"{task_label} ({ds_label})" if SHOW_DATASET_TITLE else task_label
        ax.set_title(title, fontsize=8.5, pad=3)
        style_ax(ax)

        _loader = loader if loader is not None else load_algo_task_seeds
        for algo, cfg in ALGO_CONFIGS.items():
            seeds = _loader(algo, task, metric)
            if not seeds:
                continue
            steps, mean, p_lo, p_hi = compute_band(seeds)

            def _clip(a):
                return np.clip(a, clip_lo, clip_hi) if clip_lo is not None else a

            mn_disp = _clip(mean)
            lo_disp = _clip(p_lo)
            hi_disp = _clip(p_hi)

            n_markers = min(10, len(steps))
            marker_idx = np.linspace(0, len(steps) - 1, n_markers, dtype=int)

            ax.plot(
                steps,
                mn_disp,
                color=cfg["color"],
                linewidth=1.5,
                zorder=cfg["zorder"],
                label=cfg["label"],
            )
            ax.plot(
                steps[marker_idx],
                mn_disp[marker_idx],
                color=cfg["color"],
                marker=cfg["marker"],
                markersize=4,
                linewidth=0,
                zorder=cfg["zorder"] + 10,
            )
            ax.fill_between(
                steps,
                lo_disp,
                hi_disp,
                color=cfg["color"],
                alpha=0.18,
                zorder=cfg["zorder"] - 1,
            )

            if cfg["label"] not in legend_handles:
                legend_handles[cfg["label"]] = Line2D(
                    [0],
                    [0],
                    color=cfg["color"],
                    marker=cfg["marker"],
                    markersize=6,
                    linewidth=1.5,
                    label=cfg["label"],
                )

        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        else:
            ax.set_yticklabels([])

        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Number of steps (x10^6)", fontsize=8)

    # hide unused axes
    for idx in range(len(tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    #  legend: algorithm row 
    algo_order = [cfg["label"] for cfg in ALGO_CONFIGS.values()]
    algo_handles = [legend_handles[l] for l in algo_order if l in legend_handles]

    fig.legend(
        handles=algo_handles,
        loc="lower center",
        ncol=len(algo_handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.001),
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.99,
        top=0.95,
        bottom=0.10,
        hspace=0.35,
        wspace=0.1,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        out_path = OUT_DIR / f"{out_stem}.{fmt}"
        fig.savefig(out_path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  saved -> {out_path.relative_to(REPO_ROOT)}")

    plt.close(fig)


#  main 


def main():
    # filter to tasks that actually have data in at least one algo
    tasks = [
        t
        for t in TASKS_TO_PLOT
        if any(
            (RUNS_ROOT / algo / DATASET_CFG[dataset_of(t)]["subdir"] / t).exists()
            for algo in ALGO_CONFIGS
        )
    ]

    print(f"Plotting {len(tasks)} tasks: {tasks}")
    if not tasks:
        print(f"No matching benchmark task data found under {RUNS_ROOT.relative_to(REPO_ROOT)}")
        return

    #  Figure 1: Success Rate 
    print("\nPlotting success_ratio ...")
    make_figure(
        metric="success_ratio",
        ylabel="Success Rate",
        out_stem="combined_success_rate",
        tasks=tasks,
        n_cols=3,
        clip_lo=0.0,
        clip_hi=1.0,
    )

    #  Figure 2: Episode Return (normalised globally) 
    print("\nNormalising episode_return globally ...")
    all_vals: list[float] = []
    for task in tasks:
        for algo in ALGO_CONFIGS:
            seeds = load_algo_task_seeds(algo, task, "episode_return_mean")
            if seeds:
                for s in seeds.values():
                    all_vals.extend(s.values.tolist())
    if not all_vals:
        print("No episode_return_mean data found; skipped episode-return figure.")
        return
    gmin, gmax = np.nanmin(all_vals), np.nanmax(all_vals)
    print(f"  episode_return range: [{gmin:.3f}, {gmax:.3f}]")

    def load_normalised(algo: str, task: str, metric: str):
        seeds = load_algo_task_seeds(algo, task, "episode_return_mean")
        if not seeds:
            return None
        return {
            sid: pd.Series((s.values - gmin) / (gmax - gmin + 1e-9), index=s.index)
            for sid, s in seeds.items()
        }

    print("Plotting episode_return (normalised) ...")
    make_figure(
        metric="episode_return_mean",
        ylabel="Episodic Return",
        out_stem="combined_episode_return",
        tasks=tasks,
        n_cols=3,
        clip_lo=0.0,
        clip_hi=1.0,
        loader=load_normalised,
    )

    print("\nDone. Figures saved to:", OUT_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
