"""Build the final support-to-query gap analysis package for the NeurIPS ED paper.

This script aggregates task-level tracker rows from the canonical completed Phase 2
result roots and produces a paper-facing artifact package under
`results/neurips_ed_final_support_query_gap/`.

The package includes:
- a merged task-level tracker CSV,
- an aggregate summary CSV/JSON,
- a threshold-based support-to-query gap table,
- paper-ready figures,
- a running log,
- and a final summary report.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "results" / "neurips_ed_final_support_query_gap"


@dataclass(frozen=True)
class SeriesSource:
    tracker_csv: Path
    dataset: str
    subset: str
    methods: tuple[str, ...]


CANONICAL_SOURCES = (
    SeriesSource(
        tracker_csv=REPO_ROOT
        / "results"
        / "neurips_ed_phase2"
        / "EXPERIMENT_TRACKER.csv",
        dataset="AGI1",
        subset="entry_train",
        methods=("no_adaptation", "ff_ppo_adaptation_deep"),
    ),
    SeriesSource(
        tracker_csv=REPO_ROOT
        / "results"
        / "neurips_ed_phase2_fast_campaign"
        / "agi1_pretrained"
        / "EXPERIMENT_TRACKER.csv",
        dataset="AGI1",
        subset="entry_train",
        methods=("ff_ppo_pretrained_adaptation_deep",),
    ),
    SeriesSource(
        tracker_csv=REPO_ROOT
        / "results"
        / "neurips_ed_phase2_fast_campaign"
        / "agi1_recurrent"
        / "EXPERIMENT_TRACKER.csv",
        dataset="AGI1",
        subset="entry_train",
        methods=("rec_ppo_adaptation",),
    ),
    SeriesSource(
        tracker_csv=REPO_ROOT
        / "results"
        / "neurips_ed_phase2_fast_campaign"
        / "mini_all_deep"
        / "EXPERIMENT_TRACKER.csv",
        dataset="Mini",
        subset="all",
        methods=("ff_ppo_adaptation_deep",),
    ),
    SeriesSource(
        tracker_csv=REPO_ROOT
        / "results"
        / "neurips_ed_phase2_fast_campaign"
        / "concept_all_deep"
        / "EXPERIMENT_TRACKER.csv",
        dataset="Concept",
        subset="all",
        methods=("ff_ppo_adaptation_deep",),
    ),
)

NUMERIC_FIELDS = {
    "requested_budget_timesteps": int,
    "actual_budget_timesteps": int,
    "num_updates": int,
    "train_wall_clock_s": float,
    "train_steps_per_second": float,
    "support_pair_count": int,
    "query_pair_count": int,
    "support_exact_success_rate": float,
    "support_final_grid_match_rate": float,
    "support_similarity_mean": float,
    "query_exact_success_rate": float,
    "query_final_grid_match_rate": float,
    "query_similarity_mean": float,
    "submitted_support_rate": float,
    "submitted_query_rate": float,
    "seed": int,
}

TASK_TRACKER_FIELDS = [
    "source_root",
    "source_tracker",
    "phase_name",
    "dataset",
    "subset",
    "method",
    "display_label",
    "task_id",
    "requested_budget_timesteps",
    "actual_budget_timesteps",
    "num_updates",
    "train_wall_clock_s",
    "train_steps_per_second",
    "support_pair_count",
    "query_pair_count",
    "support_exact_success_rate",
    "support_final_grid_match_rate",
    "support_similarity_mean",
    "query_exact_success_rate",
    "query_final_grid_match_rate",
    "query_similarity_mean",
    "submitted_support_rate",
    "submitted_query_rate",
    "seed",
    "evaluation_mode",
]

SUMMARY_FIELDS = [
    "dataset",
    "subset",
    "method",
    "display_label",
    "requested_budget_timesteps",
    "task_count",
    "support_similarity_mean",
    "query_exact_success_rate",
    "query_final_grid_match_rate",
    "query_similarity_mean",
    "submitted_query_rate",
    "train_wall_clock_s_mean",
]

GAP_FIELDS = [
    "dataset",
    "subset",
    "method",
    "display_label",
    "selected_budget_timesteps",
    "task_count",
    "support_ge_0_8_zero_query_count",
    "support_ge_0_8_zero_query_fraction",
    "support_ge_0_9_zero_query_count",
    "support_ge_0_9_zero_query_fraction",
    "support_eq_1_0_zero_query_count",
    "support_eq_1_0_zero_query_fraction",
    "submitted_query_nonzero_and_zero_exact_count",
    "submitted_query_nonzero_and_zero_exact_fraction",
]

DISPLAY_META = {
    ("AGI1", "entry_train", "no_adaptation"): {
        "label": "AGI1 — no adapt",
        "color": "#6e6e6e",
        "marker": "o",
        "linestyle": "--",
    },
    ("AGI1", "entry_train", "ff_ppo_adaptation_deep"): {
        "label": "AGI1 — FF-PPO",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
    },
    ("AGI1", "entry_train", "ff_ppo_pretrained_adaptation_deep"): {
        "label": "AGI1 — FF-PPO pretrained",
        "color": "#2ca02c",
        "marker": "s",
        "linestyle": "-",
    },
    ("AGI1", "entry_train", "rec_ppo_adaptation"): {
        "label": "AGI1 — RecPPO",
        "color": "#d62728",
        "marker": "^",
        "linestyle": "-",
    },
    ("Mini", "all", "ff_ppo_adaptation_deep"): {
        "label": "Mini/all — FF-PPO",
        "color": "#9467bd",
        "marker": "D",
        "linestyle": "-",
    },
    ("Concept", "all", "ff_ppo_adaptation_deep"): {
        "label": "Concept/all — FF-PPO",
        "color": "#ff7f0e",
        "marker": "P",
        "linestyle": "-",
    },
}


def _parse_row(raw_row: dict[str, str], source: SeriesSource) -> dict[str, Any]:
    row: dict[str, Any] = dict(raw_row)
    for field, caster in NUMERIC_FIELDS.items():
        raw_value = row.get(field, "")
        if raw_value == "":
            row[field] = caster(0)
        else:
            row[field] = caster(raw_value)

    dataset = row.get("dataset", source.dataset)
    subset = row.get("subset", source.subset)
    method = row["method"]
    display_meta = DISPLAY_META[(dataset, subset, method)]
    row["dataset"] = dataset
    row["subset"] = subset
    row["display_label"] = display_meta["label"]
    row["source_root"] = str(source.tracker_csv.parent.relative_to(REPO_ROOT))
    row["source_tracker"] = str(source.tracker_csv.relative_to(REPO_ROOT))
    return row


def load_task_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in CANONICAL_SOURCES:
        with source.tracker_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                if raw_row.get("dataset") != source.dataset:
                    continue
                if raw_row.get("subset") != source.subset:
                    continue
                if raw_row.get("method") not in source.methods:
                    continue
                rows.append(_parse_row(raw_row, source))

    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["subset"],
            row["method"],
            int(row["requested_budget_timesteps"]),
            row["task_id"],
        )
    )
    return rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(row[key]) for row in rows) / len(rows)


def aggregate_rows(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[
            (
                row["dataset"],
                row["subset"],
                row["method"],
                int(row["requested_budget_timesteps"]),
            )
        ].append(row)

    aggregate: list[dict[str, Any]] = []
    for (dataset, subset, method, budget), rows in sorted(grouped.items()):
        aggregate.append(
            {
                "dataset": dataset,
                "subset": subset,
                "method": method,
                "display_label": DISPLAY_META[(dataset, subset, method)]["label"],
                "requested_budget_timesteps": budget,
                "task_count": len(rows),
                "support_similarity_mean": _mean(rows, "support_similarity_mean"),
                "query_exact_success_rate": _mean(rows, "query_exact_success_rate"),
                "query_final_grid_match_rate": _mean(
                    rows, "query_final_grid_match_rate"
                ),
                "query_similarity_mean": _mean(rows, "query_similarity_mean"),
                "submitted_query_rate": _mean(rows, "submitted_query_rate"),
                "train_wall_clock_s_mean": _mean(rows, "train_wall_clock_s"),
            }
        )
    return aggregate


def build_gap_thresholds(task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in task_rows:
        grouped[(row["dataset"], row["subset"], row["method"])].append(row)

    output: list[dict[str, Any]] = []
    for (dataset, subset, method), rows in sorted(grouped.items()):
        max_budget = max(int(row["requested_budget_timesteps"]) for row in rows)
        selected_rows = [
            row for row in rows if int(row["requested_budget_timesteps"]) == max_budget
        ]
        task_count = len(selected_rows)

        def threshold_count(threshold: float, *, exact_one: bool = False) -> int:
            count = 0
            for row in selected_rows:
                support = float(row["support_similarity_mean"])
                query_exact = float(row["query_exact_success_rate"])
                if exact_one:
                    support_ok = support >= 0.999999
                else:
                    support_ok = support >= threshold
                if support_ok and query_exact == 0.0:
                    count += 1
            return count

        submitted_zero_exact = sum(
            1
            for row in selected_rows
            if float(row["submitted_query_rate"]) > 0.0
            and float(row["query_exact_success_rate"]) == 0.0
        )

        support_08 = threshold_count(0.8)
        support_09 = threshold_count(0.9)
        support_10 = threshold_count(1.0, exact_one=True)
        output.append(
            {
                "dataset": dataset,
                "subset": subset,
                "method": method,
                "display_label": DISPLAY_META[(dataset, subset, method)]["label"],
                "selected_budget_timesteps": max_budget,
                "task_count": task_count,
                "support_ge_0_8_zero_query_count": support_08,
                "support_ge_0_8_zero_query_fraction": (
                    support_08 / task_count if task_count else 0.0
                ),
                "support_ge_0_9_zero_query_count": support_09,
                "support_ge_0_9_zero_query_fraction": (
                    support_09 / task_count if task_count else 0.0
                ),
                "support_eq_1_0_zero_query_count": support_10,
                "support_eq_1_0_zero_query_fraction": (
                    support_10 / task_count if task_count else 0.0
                ),
                "submitted_query_nonzero_and_zero_exact_count": submitted_zero_exact,
                "submitted_query_nonzero_and_zero_exact_fraction": (
                    submitted_zero_exact / task_count if task_count else 0.0
                ),
            }
        )
    return output


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _budget_label(budget: int) -> str:
    if budget == 0:
        return "0"
    if budget % 1000 == 0:
        return f"{budget // 1000}k"
    return str(budget)


def _prepare_matplotlib():
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    return plt


def _plot_agi1_budget_profile(
    summary_rows: list[dict[str, Any]], output_root: Path
) -> None:
    plt = _prepare_matplotlib()
    agi1_rows = [
        row
        for row in summary_rows
        if row["dataset"] == "AGI1" and row["subset"] == "entry_train"
    ]
    metrics = [
        ("support_similarity_mean", "Support similarity"),
        ("submitted_query_rate", "Submitted query rate"),
        ("query_exact_success_rate", "Query exact success"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharex=False)
    method_order = [
        "no_adaptation",
        "ff_ppo_adaptation_deep",
        "ff_ppo_pretrained_adaptation_deep",
        "rec_ppo_adaptation",
    ]
    for method in method_order:
        method_rows = [row for row in agi1_rows if row["method"] == method]
        if not method_rows:
            continue
        method_rows.sort(key=lambda row: int(row["requested_budget_timesteps"]))
        meta = DISPLAY_META[("AGI1", "entry_train", method)]
        x_values = [int(row["requested_budget_timesteps"]) for row in method_rows]
        for axis, (metric_key, ylabel) in zip(axes, metrics, strict=True):
            y_values = [float(row[metric_key]) for row in method_rows]
            axis.plot(
                x_values,
                y_values,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                linestyle=meta["linestyle"],
                linewidth=2,
                markersize=6,
            )
            axis.set_xlabel("Budget (timesteps)")
            axis.set_ylabel(ylabel)
            axis.set_title(ylabel)
            axis.set_xticks([0, 10_000, 50_000, 100_000])
            axis.set_xticklabels(["0", "10k", "50k", "100k"])
            axis.grid(True, alpha=0.35)

    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_ylim(-0.01, 0.05)
    fig.suptitle(
        "AGI1/entry_train: support improvements without hidden-query success", y=1.03
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    fig.tight_layout()
    fig.savefig(output_root / "agi1_budgeted_profile.png", bbox_inches="tight")
    fig.savefig(output_root / "agi1_budgeted_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_cross_dataset_profile(
    summary_rows: list[dict[str, Any]], output_root: Path
) -> None:
    plt = _prepare_matplotlib()
    selected_series = [
        ("AGI1", "entry_train", "ff_ppo_adaptation_deep"),
        ("Mini", "all", "ff_ppo_adaptation_deep"),
        ("Concept", "all", "ff_ppo_adaptation_deep"),
    ]
    metrics = [
        ("support_similarity_mean", "Support similarity"),
        ("submitted_query_rate", "Submitted query rate"),
        ("query_exact_success_rate", "Query exact success"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3), sharex=False)
    for dataset, subset, method in selected_series:
        series_rows = [
            row
            for row in summary_rows
            if row["dataset"] == dataset
            and row["subset"] == subset
            and row["method"] == method
        ]
        if not series_rows:
            continue
        series_rows.sort(key=lambda row: int(row["requested_budget_timesteps"]))
        meta = DISPLAY_META[(dataset, subset, method)]
        x_values = [int(row["requested_budget_timesteps"]) for row in series_rows]
        for axis, (metric_key, ylabel) in zip(axes, metrics, strict=True):
            y_values = [float(row[metric_key]) for row in series_rows]
            axis.plot(
                x_values,
                y_values,
                label=meta["label"],
                color=meta["color"],
                marker=meta["marker"],
                linestyle=meta["linestyle"],
                linewidth=2,
                markersize=6,
            )
            axis.set_xlabel("Budget (timesteps)")
            axis.set_ylabel(ylabel)
            axis.set_title(ylabel)
            axis.set_xticks([10_000, 50_000, 100_000])
            axis.set_xticklabels(["10k", "50k", "100k"])
            axis.grid(True, alpha=0.35)

    axes[0].set_ylim(0.0, 1.05)
    axes[1].set_ylim(0.0, 1.05)
    axes[2].set_ylim(-0.01, 0.05)
    fig.suptitle(
        "Cross-dataset FF-PPO profile: support improves, query success does not", y=1.03
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.15),
    )
    fig.tight_layout()
    fig.savefig(output_root / "cross_dataset_budget_profile.png", bbox_inches="tight")
    fig.savefig(output_root / "cross_dataset_budget_profile.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_gap_thresholds(gap_rows: list[dict[str, Any]], output_root: Path) -> None:
    plt = _prepare_matplotlib()
    labels = [row["display_label"] for row in gap_rows]
    positions = list(range(len(gap_rows)))
    width = 0.24
    fractions_08 = [
        float(row["support_ge_0_8_zero_query_fraction"]) for row in gap_rows
    ]
    fractions_09 = [
        float(row["support_ge_0_9_zero_query_fraction"]) for row in gap_rows
    ]
    fractions_10 = [
        float(row["support_eq_1_0_zero_query_fraction"]) for row in gap_rows
    ]

    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    bars_08 = ax.bar(
        [pos - width for pos in positions],
        fractions_08,
        width=width,
        label="Support ≥ 0.8 and query exact = 0",
        color="#4c78a8",
    )
    bars_09 = ax.bar(
        positions,
        fractions_09,
        width=width,
        label="Support ≥ 0.9 and query exact = 0",
        color="#f58518",
    )
    bars_10 = ax.bar(
        [pos + width for pos in positions],
        fractions_10,
        width=width,
        label="Support = 1.0 and query exact = 0",
        color="#54a24b",
    )

    ax.set_ylabel("Fraction of tasks")
    ax.set_xlabel("Series at strongest completed budget")
    ax.set_title("High support fit without hidden-query success")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.35)

    def annotate(bars, key_count: str, key_total: str = "task_count") -> None:
        for bar, row in zip(bars, gap_rows, strict=True):
            height = bar.get_height()
            count = int(row[key_count])
            total = int(row[key_total])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{count}/{total}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    annotate(bars_08, "support_ge_0_8_zero_query_count")
    annotate(bars_09, "support_ge_0_9_zero_query_count")
    annotate(bars_10, "support_eq_1_0_zero_query_count")

    fig.tight_layout()
    fig.savefig(output_root / "high_support_zero_query_counts.png", bbox_inches="tight")
    fig.savefig(output_root / "high_support_zero_query_counts.pdf", bbox_inches="tight")
    plt.close(fig)


def write_running_log(output_root: Path, task_rows: list[dict[str, Any]]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    unique_sources = sorted({row["source_tracker"] for row in task_rows})
    lines = [
        "# Running log",
        "",
        f"- Timestamp: {timestamp}",
        "- Step: E3 support-to-query gap analysis using completed canonical Phase 2 tracker roots.",
        "- GPU usage: none required for this step; analysis reused existing tracker artifacts and ran as a CPU-only post-processing job.",
        "",
        "## Inputs",
        "",
    ]
    lines.extend([f"- `{source}`" for source in unique_sources])
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `EXPERIMENT_TRACKER.csv` — merged task-level tracker for the canonical analysis series",
            "- `summary.csv` / `summary.json` — aggregate paper-facing summary tables",
            "- `gap_thresholds.csv` — threshold counts for high support fit with zero hidden-query success",
            "- `agi1_budgeted_profile.(png|pdf)` — AGI1 canonical benchmark figure",
            "- `cross_dataset_budget_profile.(png|pdf)` — cross-dataset FF-PPO comparison figure",
            "- `high_support_zero_query_counts.(png|pdf)` — support-to-query gap threshold figure",
            "- `FINAL_SUMMARY.md` — interpretation and paper usage notes",
            "",
            "## Notes",
            "",
            "- This step intentionally used completed tracker roots only.",
            "- No new adaptation training was launched.",
            "- This package is intended to feed the main support→query gap figures and summary table for the paper.",
        ]
    )
    (output_root / "RUNNING_LOG.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_final_summary(
    output_root: Path,
    summary_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> None:
    by_key = {
        (
            row["dataset"],
            row["subset"],
            row["method"],
            int(row["requested_budget_timesteps"]),
        ): row
        for row in summary_rows
    }

    def get(dataset: str, subset: str, method: str, budget: int) -> dict[str, Any]:
        return by_key[(dataset, subset, method, budget)]

    agi1_no = get("AGI1", "entry_train", "no_adaptation", 0)
    agi1_ff = get("AGI1", "entry_train", "ff_ppo_adaptation_deep", 100000)
    agi1_pre = get("AGI1", "entry_train", "ff_ppo_pretrained_adaptation_deep", 100000)
    agi1_rec = get("AGI1", "entry_train", "rec_ppo_adaptation", 100000)
    mini_ff = get("Mini", "all", "ff_ppo_adaptation_deep", 100000)
    concept_ff = get("Concept", "all", "ff_ppo_adaptation_deep", 100000)

    gap_by_label = {row["display_label"]: row for row in gap_rows}
    lines = [
        "# Final summary",
        "",
        "## Purpose",
        "",
        "This package finalizes the step-one support-to-query gap analysis for the NeurIPS ED paper using completed canonical Phase 2 tracker roots only.",
        "",
        "## Key findings",
        "",
        "1. Support-side adaptation is measurable across all canonical series.",
        f"   - AGI1 no-adaptation support similarity starts at `{agi1_no['support_similarity_mean']:.3f}`.",
        f"   - AGI1 FF-PPO reaches `{agi1_ff['support_similarity_mean']:.3f}` at `100k`.",
        f"   - AGI1 pretrained FF-PPO reaches `{agi1_pre['support_similarity_mean']:.3f}` at `100k`.",
        f"   - AGI1 recurrent PPO reaches `{agi1_rec['support_similarity_mean']:.3f}` at `100k`.",
        f"   - Mini/all FF-PPO reaches `{mini_ff['support_similarity_mean']:.3f}` at `100k`.",
        f"   - Concept/all FF-PPO reaches `{concept_ff['support_similarity_mean']:.3f}` at `100k`.",
        "2. Query-side correctness remains flat at zero across the canonical completed benchmark series.",
        f"   - AGI1 FF-PPO query exact at `100k`: `{agi1_ff['query_exact_success_rate']:.3f}`.",
        f"   - AGI1 pretrained FF-PPO query exact at `100k`: `{agi1_pre['query_exact_success_rate']:.3f}`.",
        f"   - AGI1 recurrent PPO query exact at `100k`: `{agi1_rec['query_exact_success_rate']:.3f}`.",
        f"   - Mini/all FF-PPO query exact at `100k`: `{mini_ff['query_exact_success_rate']:.3f}`.",
        f"   - Concept/all FF-PPO query exact at `100k`: `{concept_ff['query_exact_success_rate']:.3f}`.",
        "3. Query submission can increase without producing hidden-query success.",
        f"   - AGI1 FF-PPO submitted-query rate rises to `{agi1_ff['submitted_query_rate']:.3f}` at `100k` while query exact remains zero.",
        f"   - AGI1 pretrained FF-PPO submitted-query rate reaches `{agi1_pre['submitted_query_rate']:.3f}` at `100k` while query exact remains zero.",
        "",
        "## Strongest support-to-query gap statistics at the strongest completed budget",
        "",
    ]

    for label in [
        "AGI1 — FF-PPO",
        "AGI1 — FF-PPO pretrained",
        "AGI1 — RecPPO",
        "Mini/all — FF-PPO",
        "Concept/all — FF-PPO",
    ]:
        row = gap_by_label[label]
        lines.append(
            "- {label}: `{count08}/{total}` tasks reached support `>= 0.8` with zero query exact; "
            "`{count09}/{total}` reached support `>= 0.9` with zero query exact; "
            "`{count10}/{total}` reached support `= 1.0` with zero query exact; "
            "`{submitted}/{total}` had nonzero submitted-query rate with zero query exact.".format(
                label=label,
                count08=int(row["support_ge_0_8_zero_query_count"]),
                count09=int(row["support_ge_0_9_zero_query_count"]),
                count10=int(row["support_eq_1_0_zero_query_count"]),
                submitted=int(row["submitted_query_nonzero_and_zero_exact_count"]),
                total=int(row["task_count"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paper usage",
            "",
            "- `agi1_budgeted_profile.(png|pdf)` is the strongest candidate for the main canonical benchmark figure.",
            "- `high_support_zero_query_counts.(png|pdf)` is the strongest candidate for the support-to-query gap figure.",
            "- `cross_dataset_budget_profile.(png|pdf)` is useful either as a main-text dataset breakdown figure or as a strong appendix figure.",
            "- `summary.csv` is table-ready input for the main benchmark summary table.",
            "- `gap_thresholds.csv` is table-ready input for the support-to-query gap analysis in the main text or appendix.",
            "",
            "## Interpretation",
            "",
            "This analysis strengthens the central ED-track claim: the benchmark registers meaningful support-side adaptation and query-submission behavior, but those gains do not become hidden-query success. The gap is visible both in budgeted series summaries and in the number of tasks that achieve high support similarity while staying at zero hidden-query exact success.",
            "",
            "## Caveats",
            "",
            "- This package uses completed canonical benchmark roots only; targeted and multisample diagnostics are intentionally left for later analysis steps.",
            "- Query exact remains flat at zero in these canonical series, so the figures rely on the contrast between support fit, submission behavior, and query correctness rather than on positive query solves.",
            "- No new training was performed for this step.",
        ]
    )
    (output_root / "FINAL_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    task_rows = load_task_rows()
    summary_rows = aggregate_rows(task_rows)
    gap_rows = build_gap_thresholds(task_rows)

    _write_csv(output_root / "EXPERIMENT_TRACKER.csv", TASK_TRACKER_FIELDS, task_rows)
    _write_csv(output_root / "summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(output_root / "gap_thresholds.csv", GAP_FIELDS, gap_rows)
    _write_json(
        output_root / "summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "canonical_sources": [
                str(source.tracker_csv.relative_to(REPO_ROOT))
                for source in CANONICAL_SOURCES
            ],
            "series_count": len(
                {(row["dataset"], row["subset"], row["method"]) for row in task_rows}
            ),
            "task_row_count": len(task_rows),
            "summary_rows": summary_rows,
            "gap_rows": gap_rows,
        },
    )

    _plot_agi1_budget_profile(summary_rows, output_root)
    _plot_cross_dataset_profile(summary_rows, output_root)
    _plot_gap_thresholds(gap_rows, output_root)
    write_running_log(output_root, task_rows)
    write_final_summary(output_root, summary_rows, gap_rows)

    print(f"Wrote support-to-query gap package to {output_root}")


if __name__ == "__main__":
    main()
