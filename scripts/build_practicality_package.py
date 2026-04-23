"""Build the final NeurIPS ED practicality package from existing benchmark artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "neurips_ed_final_practicality"
BENCH_ROOT = ROOT / "JaxARC" / "benchmarks" / "results"

H100_BENCHMARKS = {
    "JaxARC JIT": BENCH_ROOT / "arcle_vs_jaxarc_h100_jaxarc-jit_20250915-152014.json",
    "ARCLE sync": BENCH_ROOT / "arcle_vs_jaxarc_h100_arcle-sync_20250915-152434.json",
    "ARCLE async": BENCH_ROOT / "arcle_vs_jaxarc_h100_arcle-async_20250915-153352.json",
}

PHASE2_OLD_TRACKER = ROOT / "results" / "neurips_ed_phase2" / "EXPERIMENT_TRACKER.csv"
PHASE2_FAST_TRACKER = (
    ROOT
    / "results"
    / "neurips_ed_phase2_fast_campaign"
    / "agi1_pretrained"
    / "EXPERIMENT_TRACKER.csv"
)
PHASE1_PILOT_JSON = (
    ROOT / "results" / "neurips_ed_phase1" / "arcle_wallclock_pilot.json"
)

PHASE2_SPEEDUP_TASK = "746b3537"
PHASE2_SPEEDUP_METHOD = "ff_ppo_pretrained_adaptation_deep"
PHASE2_SPEEDUP_BUDGETS = [10000, 50000, 100000]


def mean_or_none(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def std_or_none(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return float(statistics.pstdev(values))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_benchmark_rows(
    label: str, path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    rows: list[dict[str, Any]] = []
    for batch_str, entry in sorted(
        payload["throughput"].items(), key=lambda item: int(item[0])
    ):
        batch_size = int(batch_str)
        sps = [float(x) for x in entry.get("sps", [])]
        times = [float(x) for x in entry.get("times", [])]
        rows.append(
            {
                "mode": label,
                "source_file": str(path),
                "batch_size": batch_size,
                "mean_sps": mean_or_none(sps),
                "std_sps": std_or_none(sps),
                "mean_time_s": mean_or_none(times),
                "std_time_s": std_or_none(times),
                "compile_time_s": float(entry.get("compile_time", 0.0) or 0.0),
                "num_runs": len(sps),
                "valid": bool(sps),
                "error": str(entry.get("error", "") or ""),
            }
        )
    return payload["meta"], rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_tracker_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_phase2_speedup() -> list[dict[str, Any]]:
    old_rows = load_tracker_rows(PHASE2_OLD_TRACKER)
    fast_rows = load_tracker_rows(PHASE2_FAST_TRACKER)

    def collect(rows: list[dict[str, Any]]) -> dict[int, list[float]]:
        grouped: dict[int, list[float]] = {
            budget: [] for budget in PHASE2_SPEEDUP_BUDGETS
        }
        for row in rows:
            if str(row.get("method")) != PHASE2_SPEEDUP_METHOD:
                continue
            if str(row.get("task_id")) != PHASE2_SPEEDUP_TASK:
                continue
            budget = int(row["requested_budget_timesteps"])
            if budget not in grouped:
                continue
            grouped[budget].append(float(row["train_wall_clock_s"]))
        return grouped

    old_grouped = collect(old_rows)
    fast_grouped = collect(fast_rows)

    summary_rows: list[dict[str, Any]] = []
    for budget in PHASE2_SPEEDUP_BUDGETS:
        old_values = old_grouped.get(budget, [])
        new_values = fast_grouped.get(budget, [])
        old_mean = mean_or_none(old_values)
        new_mean = mean_or_none(new_values)
        speedup_x = (
            (old_mean / new_mean)
            if (old_mean is not None and new_mean not in (None, 0.0))
            else None
        )
        summary_rows.append(
            {
                "task_id": PHASE2_SPEEDUP_TASK,
                "method": PHASE2_SPEEDUP_METHOD,
                "budget": budget,
                "old_root": str(PHASE2_OLD_TRACKER),
                "new_root": str(PHASE2_FAST_TRACKER),
                "old_wall_clock_s": old_mean,
                "new_wall_clock_s": new_mean,
                "speedup_x": speedup_x,
            }
        )
    return summary_rows


def build_matched_speedup_rows(
    curve_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_mode: dict[str, dict[int, dict[str, Any]]] = {}
    for row in curve_rows:
        by_mode.setdefault(str(row["mode"]), {})[int(row["batch_size"])] = row

    jax_rows = by_mode["JaxARC JIT"]
    sync_rows = by_mode["ARCLE sync"]
    async_rows = by_mode["ARCLE async"]
    batch_sizes = sorted({*jax_rows.keys(), *sync_rows.keys(), *async_rows.keys()})

    matched_rows: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        jax_mean = jax_rows.get(batch_size, {}).get("mean_sps")
        sync_mean = sync_rows.get(batch_size, {}).get("mean_sps")
        async_mean = async_rows.get(batch_size, {}).get("mean_sps")
        speedup_sync = None
        speedup_async = None
        if jax_mean is not None and sync_mean not in (None, 0.0):
            speedup_sync = float(jax_mean) / float(sync_mean)
        if jax_mean is not None and async_mean not in (None, 0.0):
            speedup_async = float(jax_mean) / float(async_mean)
        matched_rows.append(
            {
                "batch_size": batch_size,
                "jaxarc_jit_mean_sps": jax_mean,
                "arcle_sync_mean_sps": sync_mean,
                "arcle_async_mean_sps": async_mean,
                "speedup_vs_sync": speedup_sync,
                "speedup_vs_async": speedup_async,
            }
        )
    return matched_rows


def find_peak(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get(key) is not None]
    return max(valid_rows, key=lambda row: float(row[key]))


def build_summary_rows(
    curve_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    phase2_speedup_rows: list[dict[str, Any]],
    phase1_pilot: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for mode in ["JaxARC JIT", "ARCLE sync", "ARCLE async"]:
        peak = find_peak(
            [row for row in curve_rows if row["mode"] == mode and row["valid"]],
            "mean_sps",
        )
        rows.append(
            {
                "category": "h100_peak_throughput",
                "metric": mode.lower().replace(" ", "_") + "_peak_mean_sps",
                "value": float(peak["mean_sps"]),
                "unit": "steps/s",
                "details": f"batch_size={peak['batch_size']}",
            }
        )

    valid_sync = [row for row in matched_rows if row["speedup_vs_sync"] is not None]
    valid_async = [row for row in matched_rows if row["speedup_vs_async"] is not None]
    best_sync = max(valid_sync, key=lambda row: float(row["speedup_vs_sync"]))
    best_async = max(valid_async, key=lambda row: float(row["speedup_vs_async"]))

    rows.append(
        {
            "category": "matched_speedup",
            "metric": "best_speedup_vs_arcle_sync",
            "value": float(best_sync["speedup_vs_sync"]),
            "unit": "x",
            "details": f"batch_size={best_sync['batch_size']}",
        }
    )
    rows.append(
        {
            "category": "matched_speedup",
            "metric": "best_speedup_vs_arcle_async",
            "value": float(best_async["speedup_vs_async"]),
            "unit": "x",
            "details": f"batch_size={best_async['batch_size']}",
        }
    )

    pilot = phase1_pilot["arcle_random_step_benchmark"]
    rows.append(
        {
            "category": "phase1_pilot",
            "metric": "arcle_random_steps_per_second",
            "value": float(pilot["steps_per_second"]),
            "unit": "steps/s",
            "details": f"num_steps={pilot['num_steps']}",
        }
    )

    for row in phase2_speedup_rows:
        rows.append(
            {
                "category": "phase2_fast_path",
                "metric": f"wall_clock_speedup_{int(row['budget'])}",
                "value": float(row["speedup_x"])
                if row["speedup_x"] is not None
                else None,
                "unit": "x",
                "details": f"task_id={row['task_id']}",
            }
        )

    return rows


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def plot_h100_throughput(curve_rows: list[dict[str, Any]], output_stem: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {
        "JaxARC JIT": "#1f77b4",
        "ARCLE sync": "#ff7f0e",
        "ARCLE async": "#2ca02c",
    }
    markers = {"JaxARC JIT": "o", "ARCLE sync": "s", "ARCLE async": "^"}

    for mode in ["JaxARC JIT", "ARCLE sync", "ARCLE async"]:
        rows = [row for row in curve_rows if row["mode"] == mode and row["valid"]]
        xs = [int(row["batch_size"]) for row in rows]
        ys = [float(row["mean_sps"]) for row in rows]
        ax.plot(
            xs,
            ys,
            marker=markers[mode],
            linewidth=2.2,
            markersize=6,
            label=mode,
            color=colors[mode],
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Parallel environments (batch size)")
    ax.set_ylabel("Mean throughput (steps/s)")
    ax.set_title("H100 throughput: JaxARC scales while ARCLE saturates")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_matched_speedup(matched_rows: list[dict[str, Any]], output_stem: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 6))

    sync_rows = [row for row in matched_rows if row["speedup_vs_sync"] is not None]
    async_rows = [row for row in matched_rows if row["speedup_vs_async"] is not None]

    ax.plot(
        [int(row["batch_size"]) for row in sync_rows],
        [float(row["speedup_vs_sync"]) for row in sync_rows],
        marker="s",
        linewidth=2.2,
        markersize=6,
        label="JaxARC / ARCLE sync",
        color="#ff7f0e",
    )
    ax.plot(
        [int(row["batch_size"]) for row in async_rows],
        [float(row["speedup_vs_async"]) for row in async_rows],
        marker="^",
        linewidth=2.2,
        markersize=6,
        label="JaxARC / ARCLE async",
        color="#2ca02c",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Parallel environments (batch size)")
    ax.set_ylabel("Matched-batch speedup (x)")
    ax.set_title("Matched-batch H100 speedup over ARCLE")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_phase2_speedup(rows: list[dict[str, Any]], output_stem: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    budgets = [str(int(row["budget"]) // 1000) + "k" for row in rows]
    old_times = [float(row["old_wall_clock_s"]) for row in rows]
    new_times = [float(row["new_wall_clock_s"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    old_bars = ax.bar(
        [i - width / 2 for i in x],
        old_times,
        width=width,
        label="Older Phase 2 path",
        color="#c44e52",
    )
    new_bars = ax.bar(
        [i + width / 2 for i in x],
        new_times,
        width=width,
        label="Fast cumulative path",
        color="#55a868",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(budgets)
    ax.set_xlabel("Budget")
    ax.set_ylabel("Train wall-clock (s)")
    ax.set_title("Representative Phase 2 wall-clock speedup on task 746b3537")
    ax.legend(frameon=True)

    for idx, row in enumerate(rows):
        speedup = float(row["speedup_x"])
        top = max(old_times[idx], new_times[idx])
        ax.text(
            idx,
            top + max(old_times) * 0.03,
            f"{speedup:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for container in [old_bars, new_bars]:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def build_final_summary(
    curve_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
    phase2_speedup_rows: list[dict[str, Any]],
    phase1_pilot: dict[str, Any],
    meta_by_mode: dict[str, dict[str, Any]],
) -> str:
    jax_peak = find_peak(
        [row for row in curve_rows if row["mode"] == "JaxARC JIT" and row["valid"]],
        "mean_sps",
    )
    sync_peak = find_peak(
        [row for row in curve_rows if row["mode"] == "ARCLE sync" and row["valid"]],
        "mean_sps",
    )
    async_peak = find_peak(
        [row for row in curve_rows if row["mode"] == "ARCLE async" and row["valid"]],
        "mean_sps",
    )
    best_sync = max(
        (row for row in matched_rows if row["speedup_vs_sync"] is not None),
        key=lambda row: float(row["speedup_vs_sync"]),
    )
    best_async = max(
        (row for row in matched_rows if row["speedup_vs_async"] is not None),
        key=lambda row: float(row["speedup_vs_async"]),
    )
    pilot = phase1_pilot["arcle_random_step_benchmark"]

    lines = [
        "# Final Summary",
        "",
        "## Purpose",
        "",
        "Promote the existing ARCLE-vs-JaxARC benchmark artifacts into a stable, paper-facing practicality package without launching new benchmark jobs.",
        "",
        "## Inputs",
        "",
        f"- `{H100_BENCHMARKS['JaxARC JIT']}`",
        f"- `{H100_BENCHMARKS['ARCLE sync']}`",
        f"- `{H100_BENCHMARKS['ARCLE async']}`",
        f"- `{PHASE2_OLD_TRACKER}`",
        f"- `{PHASE2_FAST_TRACKER}`",
        f"- `{PHASE1_PILOT_JSON}`",
        "",
        "## Setup",
        "",
        f"- Hardware tag in benchmark JSONs: `{meta_by_mode['JaxARC JIT']['run_tag']}`",
        f"- Fixed steps per benchmark call: `{meta_by_mode['JaxARC JIT']['fixed_steps']}`",
        f"- Repeats per batch size: `{meta_by_mode['JaxARC JIT']['runs']}`",
        f"- Matched task id: `{meta_by_mode['JaxARC JIT']['task']['task_id']}`",
        "- JaxARC throughput uses benchmark runtime excluding one-time JIT compilation; ARCLE has no compile phase.",
        "",
        "## Main results",
        "",
        f"- Peak **JaxARC JIT** mean throughput on the H100 benchmark: `{float(jax_peak['mean_sps']):,.0f}` steps/s at batch `{int(jax_peak['batch_size']):,}`.",
        f"- Peak **ARCLE sync** mean throughput on the same host benchmark: `{float(sync_peak['mean_sps']):,.0f}` steps/s at batch `{int(sync_peak['batch_size']):,}`.",
        f"- Peak **ARCLE async** mean throughput on the same host benchmark: `{float(async_peak['mean_sps']):,.0f}` steps/s at batch `{int(async_peak['batch_size']):,}`.",
        f"- Best matched-batch speedup over **ARCLE sync**: `{float(best_sync['speedup_vs_sync']):,.1f}x` at batch `{int(best_sync['batch_size']):,}`.",
        f"- Best matched-batch speedup over **ARCLE async**: `{float(best_async['speedup_vs_async']):,.1f}x` at batch `{int(best_async['batch_size']):,}`.",
        "",
        "### Phase 2 end-to-end fast-path comparison",
        "",
        "For AGI1 pretrained task `746b3537`, the cumulative fast path reduced train wall-clock from the older restart-per-budget path to:",
        "",
    ]

    for row in phase2_speedup_rows:
        lines.append(
            f"- `{int(row['budget']) // 1000}k`: `{float(row['old_wall_clock_s']):.1f} s` → `{float(row['new_wall_clock_s']):.1f} s` (`{float(row['speedup_x']):.2f}x` faster)"
        )

    lines.extend(
        [
            "",
            "### Supporting bridge artifact",
            "",
            f"- Existing Phase 1 ARCLE pilot: `{pilot['num_steps']}` random steps in `{float(pilot['wall_clock_s']):.2f} s` (`{float(pilot['steps_per_second']):.2f}` steps/s).",
            "- This pilot is useful as a caveated reference, but it is not the main matched comparison because it contrasts ARCLE random stepping with end-to-end JaxARC adaptation runs.",
            "",
            "## Paper usage",
            "",
            "- `h100_throughput_vs_batch.(png|pdf)` — best candidate for the main practicality figure.",
            "- `matched_speedup_vs_batch.(png|pdf)` — strong appendix or supplementary figure showing the matched-batch speedup profile explicitly.",
            "- `phase2_fast_path_speedup.(png|pdf)` — appendix/supporting figure for the claim that the engineering changes made the adaptation campaign materially cheaper to run.",
            "",
            "## Caveats",
            "",
            "- The H100 benchmark is a simulator-throughput comparison on a fixed ARC task, not a fully matched end-to-end adaptation comparison.",
            "- `ARCLE async` fails at larger batch sizes in the archived run due to file-descriptor limits (`Too many open files`).",
            "- The Phase 2 fast-path comparison is a within-JaxARC engineering comparison, not an ARCLE comparison.",
            "- An earlier rough note in `results/neurips_ed_phase2/RUNNING_LOG.md` quoted larger fast-path speedups; this package uses the tracker-backed wall-clock values from the archived experiment trackers and should be treated as authoritative.",
            "- The older Phase 1 pilot should be cited only as a bridge artifact, not as the main practicality claim.",
            "",
            "## Outputs",
            "",
            "- `throughput_curves.csv`",
            "- `matched_speedups.csv`",
            "- `phase2_fast_path_speedup.csv`",
            "- `summary.csv` / `summary.json`",
            "- `h100_throughput_vs_batch.(png|pdf)`",
            "- `matched_speedup_vs_batch.(png|pdf)`",
            "- `phase2_fast_path_speedup.(png|pdf)`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    meta_by_mode: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    for label, path in H100_BENCHMARKS.items():
        meta, rows = load_benchmark_rows(label, path)
        meta_by_mode[label] = meta
        curve_rows.extend(rows)

    matched_rows = build_matched_speedup_rows(curve_rows)
    phase2_speedup_rows = summarize_phase2_speedup()
    phase1_pilot = load_json(PHASE1_PILOT_JSON)
    summary_rows = build_summary_rows(
        curve_rows, matched_rows, phase2_speedup_rows, phase1_pilot
    )

    write_csv(OUTPUT_ROOT / "throughput_curves.csv", curve_rows)
    write_csv(OUTPUT_ROOT / "matched_speedups.csv", matched_rows)
    write_csv(OUTPUT_ROOT / "phase2_fast_path_speedup.csv", phase2_speedup_rows)
    write_csv(OUTPUT_ROOT / "summary.csv", summary_rows)
    save_json(
        OUTPUT_ROOT / "summary.json",
        {
            "meta_by_mode": meta_by_mode,
            "summary_rows": summary_rows,
            "phase1_pilot": phase1_pilot,
            "phase2_fast_path": phase2_speedup_rows,
        },
    )

    plot_h100_throughput(curve_rows, OUTPUT_ROOT / "h100_throughput_vs_batch")
    plot_matched_speedup(matched_rows, OUTPUT_ROOT / "matched_speedup_vs_batch")
    plot_phase2_speedup(phase2_speedup_rows, OUTPUT_ROOT / "phase2_fast_path_speedup")

    (OUTPUT_ROOT / "FINAL_SUMMARY.md").write_text(
        build_final_summary(
            curve_rows=curve_rows,
            matched_rows=matched_rows,
            phase2_speedup_rows=phase2_speedup_rows,
            phase1_pilot=phase1_pilot,
            meta_by_mode=meta_by_mode,
        ),
        encoding="utf-8",
    )

    print(f"Wrote practicality package to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
