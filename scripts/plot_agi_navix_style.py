"""
plot_agi_navix_style.py
----------------------------
Generates NAVIX-style publication plots for the ARC-AGI subset benchmark.
Reads results from:
  results/navix_style_subset_benchmark/mixed_10m_5seed_conc5/runs/
      {algorithm}/agi1/{task}/seed_{nn}/success_curve.csv

Produces two figures (saved as PDF + PNG):
  1. success_ratio   — query exact-match success rate (0–1)
  2. episode_return  — mean episodic return (rescaled to [0, 1] for display)

Style: matches NAVIX Figure 7 (multi-panel grid, 5–95 percentile shading,
markers on mean line, dashed gridlines, bottom legend).
"""

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed fonts in PDF
    "ps.fonttype": 42,
})

# ── paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "results" / "navix_style_subset_benchmark" / "mixed_10m_5seed_conc5" / "runs"
OUT_DIR   = REPO_ROOT / "results" / "navix_style_subset_benchmark" / "mixed_10m_5seed_conc5" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── algorithm config ─────────────────────────────────────────────────────────
ALGO_CONFIGS = {
    "ff_ppo":      {"label": "PPO",      "color": "#3A85C8", "marker": "o", "zorder": 4},
    "ff_dqn":      {"label": "DQN",      "color": "#D4A847", "marker": "s", "zorder": 3},
    "ff_pqn":      {"label": "PQN",      "color": "#E07040", "marker": "^", "zorder": 2},
    "ff_reinforce":{"label": "REINFORCE","color": "#5CB87A", "marker": "D", "zorder": 1},
}

# prettier task names for subplot titles
TASK_LABELS = {
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
}

# ── data loading ──────────────────────────────────────────────────────────────

def load_algo_task_seeds(algo: str, task: str, metric: str) -> dict[int, pd.Series] | None:
    """Return {seed_id: pd.Series(timestep -> metric_value)}, or None if missing."""
    task_dir = RUNS_ROOT / algo / "agi1" / task
    if not task_dir.exists():
        return None
    seed_series = {}
    for seed_dir in sorted(task_dir.glob("seed_*")):
        csv_path = seed_dir / "success_curve.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["timestep", metric])
        df = df.sort_values("timestep")
        seed_id = int(seed_dir.name.split("_")[-1])
        seed_series[seed_id] = pd.Series(
            df[metric].values,
            index=df["timestep"].values,
        )
    return seed_series if seed_series else None


def compute_band(seed_series: dict[int, pd.Series], lo: float = 5, hi: float = 95):
    """
    Align all seeds to a common timestep index (outer join, forward-fill),
    then return (common_steps, mean, p_lo, p_hi).
    """
    frames = [s.rename(k) for k, s in seed_series.items()]
    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.ffill().bfill()

    steps = combined.index.values
    arr   = combined.values  # shape (T, n_seeds)

    mean  = np.nanmean(arr, axis=1)
    p_lo  = np.nanpercentile(arr, lo,  axis=1)
    p_hi  = np.nanpercentile(arr, hi, axis=1)
    return steps, mean, p_lo, p_hi


# ── plotting helpers ──────────────────────────────────────────────────────────

def _format_millions(x, pos):
    return f"{x/1e6:.1f}"


def style_ax(ax, yticks=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)):
    ax.set_facecolor("#FAFAFA")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(yticks)
    for y in yticks:
        ax.axhline(y, color="#CCCCCC", linewidth=0.6, linestyle="--", zorder=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_millions))
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
        n_rows, n_cols,
        figsize=(n_cols * 3.6, n_rows * 2.6),
        constrained_layout=False,
    )
    # flatten even if 1-row
    axes_flat = np.array(axes).flatten()

    # track legend handles globally
    legend_handles = {}

    for idx, task in enumerate(tasks):
        ax = axes_flat[idx]
        ax.set_title(TASK_LABELS.get(task, task), fontsize=8.5, pad=3)
        style_ax(ax)

        _loader = loader if loader is not None else load_algo_task_seeds
        for algo, cfg in ALGO_CONFIGS.items():
            seeds = _loader(algo, task, metric)
            if not seeds:
                continue

            steps, mean, p_lo, p_hi = compute_band(seeds)

            # clip to [0,1] range for display
            lo_disp  = np.clip(p_lo,  clip_lo, clip_hi) if clip_lo is not None else p_lo
            hi_disp  = np.clip(p_hi,  clip_lo, clip_hi) if clip_hi is not None else p_hi
            mn_disp  = np.clip(mean,  clip_lo, clip_hi) if clip_lo is not None else mean

            # subsample marker positions so markers don't crowd each other
            n_markers = min(10, len(steps))
            marker_idx = np.linspace(0, len(steps) - 1, n_markers, dtype=int)

            line, = ax.plot(
                steps, mn_disp,
                color=cfg["color"],
                linewidth=1.5,
                zorder=cfg["zorder"],
                label=cfg["label"],
            )
            ax.plot(
                steps[marker_idx], mn_disp[marker_idx],
                color=cfg["color"],
                marker=cfg["marker"],
                markersize=4,
                linewidth=0,
                zorder=cfg["zorder"] + 10,
            )
            ax.fill_between(
                steps, lo_disp, hi_disp,
                color=cfg["color"],
                alpha=0.18,
                zorder=cfg["zorder"] - 1,
            )

            if cfg["label"] not in legend_handles:
                legend_handles[cfg["label"]] = matplotlib.lines.Line2D(
                    [0], [0],
                    color=cfg["color"],
                    marker=cfg["marker"],
                    markersize=6,
                    linewidth=1.5,
                    label=cfg["label"],
                )

        # only label y-axis on leftmost column
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        else:
            ax.set_yticklabels([])

        # only label x-axis on bottom row
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel("Number of steps (×10⁶)", fontsize=8)

    # hide unused axes
    for idx in range(len(tasks), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # ── legend (bottom centre) ──
    ordered_labels = [cfg["label"] for cfg in ALGO_CONFIGS.values()]
    handles = [legend_handles[l] for l in ordered_labels if l in legend_handles]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=2.5,
        columnspacing=1.5,
    )

    fig.subplots_adjust(
        left=0.07, right=0.99,
        top=0.97,  bottom=0.10,
        hspace=0.52, wspace=0.15,
    )

    for fmt in ("pdf", "png"):
        out_path = OUT_DIR / f"{out_stem}.{fmt}"
        fig.savefig(
            out_path,
            format=fmt,
            dpi=300,
            bbox_inches="tight",
        )
        print(f"  saved → {out_path.relative_to(REPO_ROOT)}")

    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # collect tasks that actually exist in at least one algorithm
    all_tasks = sorted(TASK_LABELS.keys())
    tasks_with_data = []
    for t in all_tasks:
        has_any = any(
            (RUNS_ROOT / algo / "agi1" / t).exists()
            for algo in ALGO_CONFIGS
        )
        if has_any:
            tasks_with_data.append(t)

    print(f"Found {len(tasks_with_data)} ARC-AGI tasks with data.")

    # ── Figure 1: Success Rate ────────────────────────────────────────────────
    print("\nPlotting success_ratio …")
    make_figure(
        metric="success_ratio",
        ylabel="Success Rate",
        out_stem="agi1_success_rate_navix_style",
        tasks=tasks_with_data,
        n_cols=4,
        clip_lo=0.0,
        clip_hi=1.0,
    )

    # ── Figure 2: Episode Return (normalised to [0,1]) ────────────────────────
    # Collect global min/max first for normalisation
    print("\nNormalising episode_return …")
    all_vals = []
    for task in tasks_with_data:
        for algo in ALGO_CONFIGS:
            seeds = load_algo_task_seeds(algo, task, "episode_return_mean")
            if seeds:
                for s in seeds.values():
                    all_vals.extend(s.values.tolist())
    global_min = np.nanmin(all_vals)
    global_max = np.nanmax(all_vals)
    print(f"  episode_return range: [{global_min:.3f}, {global_max:.3f}]")

    def load_normalised(algo, task, metric):
        seeds = load_algo_task_seeds(algo, task, "episode_return_mean")
        if not seeds:
            return None
        norm = {}
        for sid, s in seeds.items():
            v = (s.values - global_min) / (global_max - global_min + 1e-9)
            norm[sid] = pd.Series(v, index=s.index)
        return norm

    print("Plotting episode_return (normalised) …")
    make_figure(
        metric="episode_return_mean",
        ylabel="Episodic Return (norm.)",
        out_stem="agi1_episode_return_navix_style",
        tasks=tasks_with_data,
        n_cols=4,
        clip_lo=0.0,
        clip_hi=1.0,
        loader=load_normalised,
    )

    print("\nDone. Figures saved to:", OUT_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
