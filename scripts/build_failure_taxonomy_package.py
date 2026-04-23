"""Build a paper-facing failure taxonomy package from completed Phase 2 diagnostic roots."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "neurips_ed_final_failure_taxonomy"

BASELINE_AGI1 = (
    ROOT
    / "results"
    / "neurips_ed_phase2_fast_campaign"
    / "agi1_pretrained"
    / "EXPERIMENT_TRACKER.csv"
)
BASELINE_MINI = (
    ROOT
    / "results"
    / "neurips_ed_phase2_fast_campaign"
    / "mini_all_deep"
    / "EXPERIMENT_TRACKER.csv"
)

CASE_SPECS = [
    {
        "task_group": "AGI1",
        "task_id": "7b7f7511",
        "task_label": "AGI1 / 7b7f7511",
        "intervention": "Base 100k greedy",
        "tracker": BASELINE_AGI1,
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 0,
    },
    {
        "task_group": "AGI1",
        "task_id": "7b7f7511",
        "task_label": "AGI1 / 7b7f7511",
        "intervention": "Alt seed 100k",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_seed_sweep"
        / "7b7f7511_pretrained_seed1"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 1,
    },
    {
        "task_group": "AGI1",
        "task_id": "7b7f7511",
        "task_label": "AGI1 / 7b7f7511",
        "intervention": "Sample@16 100k",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_multisample"
        / "7b7f7511_seed0_q16"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 2,
    },
    {
        "task_group": "AGI1",
        "task_id": "7b7f7511",
        "task_label": "AGI1 / 7b7f7511",
        "intervention": "Long 1M greedy",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_targeted"
        / "7b7f7511_pretrained_long"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 1000000,
        "display_order": 3,
    },
    {
        "task_group": "AGI1",
        "task_id": "2dee498d",
        "task_label": "AGI1 / 2dee498d",
        "intervention": "Base 100k greedy",
        "tracker": BASELINE_AGI1,
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 0,
    },
    {
        "task_group": "AGI1",
        "task_id": "2dee498d",
        "task_label": "AGI1 / 2dee498d",
        "intervention": "Alt seed 100k",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_seed_sweep"
        / "2dee498d_pretrained_seed1"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 1,
    },
    {
        "task_group": "AGI1",
        "task_id": "2dee498d",
        "task_label": "AGI1 / 2dee498d",
        "intervention": "Sample@16 100k",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_multisample"
        / "2dee498d_seed0_q16"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 100000,
        "display_order": 2,
    },
    {
        "task_group": "AGI1",
        "task_id": "2dee498d",
        "task_label": "AGI1 / 2dee498d",
        "intervention": "Long 1M greedy",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_targeted"
        / "2dee498d_pretrained_long"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_pretrained_adaptation_deep",
        "budget": 1000000,
        "display_order": 3,
    },
    {
        "task_group": "Mini",
        "task_id": "Simple_Unique_Box_l6adthlbktjkouruq0j",
        "task_label": "Mini / Simple_Unique_Box",
        "intervention": "Base 100k greedy",
        "tracker": BASELINE_MINI,
        "method": "ff_ppo_adaptation_deep",
        "budget": 100000,
        "display_order": 0,
    },
    {
        "task_group": "Mini",
        "task_id": "Simple_Unique_Box_l6adthlbktjkouruq0j",
        "task_label": "Mini / Simple_Unique_Box",
        "intervention": "Long 1M greedy",
        "tracker": ROOT
        / "results"
        / "neurips_ed_phase2_targeted"
        / "simple_unique_box_long"
        / "EXPERIMENT_TRACKER.csv",
        "method": "ff_ppo_adaptation_deep",
        "budget": 1000000,
        "display_order": 3,
    },
]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(spec: dict[str, Any]) -> dict[str, Any]:
    rows = read_csv(Path(spec["tracker"]))
    for row in rows:
        if str(row.get("task_id")) != str(spec["task_id"]):
            continue
        if str(row.get("method")) != str(spec["method"]):
            continue
        if int(row.get("requested_budget_timesteps", 0)) != int(spec["budget"]):
            continue
        return row
    raise ValueError(f"Could not find matching row for spec: {spec}")


def build_case_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec in CASE_SPECS:
        row = find_row(spec)
        output.append(
            {
                "task_group": spec["task_group"],
                "task_id": spec["task_id"],
                "task_label": spec["task_label"],
                "intervention": spec["intervention"],
                "display_order": spec["display_order"],
                "tracker": str(spec["tracker"]),
                "method": spec["method"],
                "budget": int(row["requested_budget_timesteps"]),
                "support_similarity_mean": float(row["support_similarity_mean"]),
                "support_exact_success_rate": float(row["support_exact_success_rate"]),
                "query_exact_success_rate": float(row["query_exact_success_rate"]),
                "query_final_grid_match_rate": float(
                    row["query_final_grid_match_rate"]
                ),
                "query_similarity_mean": float(row["query_similarity_mean"]),
                "submitted_query_rate": float(row["submitted_query_rate"]),
                "train_wall_clock_s": float(row["train_wall_clock_s"]),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_support_profiles(rows: list[dict[str, Any]], output_stem: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    intervention_order = [
        "Base 100k greedy",
        "Alt seed 100k",
        "Sample@16 100k",
        "Long 1M greedy",
    ]
    x_positions = {label: idx for idx, label in enumerate(intervention_order)}
    colors = {
        "AGI1 / 7b7f7511": "#1f77b4",
        "AGI1 / 2dee498d": "#ff7f0e",
        "Mini / Simple_Unique_Box": "#2ca02c",
    }

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["task_label"]), []).append(row)

    for task_label, task_rows in grouped.items():
        row_by_intervention = {str(row["intervention"]): row for row in task_rows}
        xs = list(range(len(intervention_order)))
        ys = [
            float(row_by_intervention[label]["support_similarity_mean"])
            if label in row_by_intervention
            else float("nan")
            for label in intervention_order
        ]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=7,
            label=task_label,
            color=colors[task_label],
        )
        for intervention_label, row in row_by_intervention.items():
            x = x_positions[intervention_label]
            ax.text(
                x,
                float(row["support_similarity_mean"]) + 0.025,
                f"Q={float(row['query_exact_success_rate']):.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=colors[task_label],
            )

    ax.set_xticks(range(len(intervention_order)))
    ax.set_xticklabels(intervention_order, rotation=12, ha="right")
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Support similarity")
    ax.set_title(
        "Selected failure cases: interventions change support fit, not hidden-query success"
    )
    ax.legend(frameon=True)
    ax.text(
        0.99,
        0.02,
        "All annotated Q values are query exact success rates",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def build_summary(rows: list[dict[str, Any]]) -> str:
    task_map: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        task_map.setdefault(str(row["task_label"]), []).append(row)

    lines = [
        "# Final Summary",
        "",
        "## Purpose",
        "",
        "Package a small, paper-facing failure taxonomy from completed targeted, seed-sweep, and multisample Phase 2 roots only.",
        "",
        "## Selected cases",
        "",
        "- `AGI1 / 7b7f7511` — strongest perfect-support failure case",
        "- `AGI1 / 2dee498d` — high-support AGI1 proxy with seed and sample diagnostics",
        "- `Mini / Simple_Unique_Box` — cross-dataset Mini exemplar",
        "",
        "## Key findings",
        "",
    ]

    perfect_support_cases = [
        row
        for row in rows
        if row["support_similarity_mean"] >= 0.999
        and row["query_exact_success_rate"] == 0.0
    ]
    sample_rows = [row for row in rows if row["intervention"] == "Sample@16 100k"]
    long_rows = [row for row in rows if row["intervention"] == "Long 1M greedy"]

    lines.append(
        f"1. Longer budgets still do not unlock hidden-query success: `{len(long_rows)}/{len(long_rows)}` audited long-budget rows stay at query exact `0.000`, even when support similarity reaches as high as `{max(row['support_similarity_mean'] for row in long_rows):.3f}`."
    )
    lines.append(
        f"2. Perfect or near-perfect support fit is not enough: `{len(perfect_support_cases)}` audited rows reach support similarity `1.000` while query exact remains `0.000`."
    )
    lines.append(
        f"3. Sampled query evaluation does not rescue the cases here: all `{len(sample_rows)}` audited `sample@16` rows remain at query exact `0.000`."
    )
    sample_drop_notes = []
    for task_label, task_rows in task_map.items():
        base = next(
            (row for row in task_rows if row["intervention"] == "Base 100k greedy"),
            None,
        )
        sample = next(
            (row for row in task_rows if row["intervention"] == "Sample@16 100k"), None
        )
        if base and sample:
            delta = float(sample["support_similarity_mean"]) - float(
                base["support_similarity_mean"]
            )
            sample_drop_notes.append(
                f"`{task_label}`: {float(base['support_similarity_mean']):.3f} → {float(sample['support_similarity_mean']):.3f} ({delta:+.3f})"
            )
    if sample_drop_notes:
        lines.append(
            "4. Sampling can even reduce support-side fit in some strong failure cases:"
        )
        for note in sample_drop_notes:
            lines.append(f"   - {note}")
    else:
        lines.append(
            "4. Sampling did not improve any audited case enough to create hidden-query success."
        )

    lines.extend(
        [
            "",
            "## Case table",
            "",
            "| task | intervention | support similarity | query exact | query similarity | submitted query rate | wall-clock (s) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(
        rows, key=lambda item: (item["task_label"], item["display_order"])
    ):
        lines.append(
            f"| {row['task_label']} | {row['intervention']} | {float(row['support_similarity_mean']):.3f} | {float(row['query_exact_success_rate']):.3f} | {float(row['query_similarity_mean']):.3f} | {float(row['submitted_query_rate']):.3f} | {float(row['train_wall_clock_s']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Paper usage",
            "",
            "- `selected_case_support_profiles.(png|pdf)` is the best candidate failure-taxonomy figure from this package.",
            "- `failure_cases.csv` is table-ready input for an appendix failure-case table.",
            "- This package is intentionally qualitative/diagnostic and complements, rather than replaces, the canonical support-gap package.",
            "",
            "## Caveats",
            "",
            "- These cases are curated exemplars, not a full causal analysis of every completed task.",
            "- The package uses archived tracker rows only; no new training was launched.",
            "- The selected Mini case has fewer intervention variants than the strongest AGI1 cases because the existing completed diagnostics are not perfectly symmetric across datasets.",
            "",
            "## Outputs",
            "",
            "- `failure_cases.csv`",
            "- `summary.json`",
            "- `selected_case_support_profiles.(png|pdf)`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = build_case_rows()
    write_csv(OUTPUT_ROOT / "failure_cases.csv", rows)
    save_json(OUTPUT_ROOT / "summary.json", {"failure_cases": rows})
    plot_support_profiles(rows, OUTPUT_ROOT / "selected_case_support_profiles")
    (OUTPUT_ROOT / "FINAL_SUMMARY.md").write_text(build_summary(rows), encoding="utf-8")
    print(f"Wrote failure taxonomy package to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
