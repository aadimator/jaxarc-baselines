"""Pilot ARCLE wall-clock benchmark for the Phase 1 results package."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from arcle.envs import RawARCEnv
from arcle.loaders import MiniARCLoader


def benchmark_arcle_random_steps(num_steps: int, seed: int) -> dict[str, Any]:
    """Benchmark ARCLE RawARCEnv with random actions on bundled MiniARC tasks."""
    env = RawARCEnv(data_loader=MiniARCLoader())
    observation, info = env.reset(seed=seed)
    del observation, info

    resets = 1
    terminated_count = 0
    truncated_count = 0
    start_time = time.perf_counter()

    for _ in range(num_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        del observation, reward, info
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if terminated or truncated:
            env.reset()
            resets += 1

    elapsed = time.perf_counter() - start_time
    env.close()
    return {
        "benchmark": "arcle_random_policy",
        "environment": "RawARCEnv(MiniARCLoader)",
        "num_steps": int(num_steps),
        "seed": int(seed),
        "wall_clock_s": float(elapsed),
        "steps_per_second": float(num_steps / elapsed) if elapsed > 0 else 0.0,
        "resets": int(resets),
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
    }


def load_tracker_rows(tracker_csv: Path) -> list[dict[str, Any]]:
    with tracker_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_tracker(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), int(row["requested_budget_timesteps"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (method, budget), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summaries.append(
            {
                "method": method,
                "budget": budget,
                "task_count": len(group),
                "mean_train_steps_per_second": statistics.mean(
                    float(row["train_steps_per_second"]) for row in group
                ),
                "mean_train_wall_clock_s": statistics.mean(
                    float(row["train_wall_clock_s"]) for row in group
                ),
                "mean_support_similarity": statistics.mean(
                    float(row["support_similarity_mean"]) for row in group
                ),
                "mean_query_exact": statistics.mean(
                    float(row["query_exact_success_rate"]) for row in group
                ),
            }
        )
    return summaries


def build_markdown(
    *,
    arcle_result: dict[str, Any],
    tracker_summaries: list[dict[str, Any]],
    tracker_csv: Path,
) -> str:
    lines = [
        "# ARCLE Wall-Clock Pilot",
        "",
        "This is a practical pilot comparison, not a perfectly matched algorithmic benchmark.",
        "It compares ARCLE random-policy environment stepping throughput on bundled MiniARC tasks",
        "against the Phase 1 JaxARC TTRL training throughputs recorded in the tracker.",
        "",
        "## ARCLE random-step benchmark",
        "",
        f"- Environment: `{arcle_result['environment']}`",
        f"- Steps: `{arcle_result['num_steps']}`",
        f"- Wall-clock: `{arcle_result['wall_clock_s']:.2f}` s",
        f"- Steps/s: `{arcle_result['steps_per_second']:.2f}`",
        f"- Resets: `{arcle_result['resets']}`",
        "",
        "## Phase 1 tracker reference",
        "",
        f"- Tracker: `{tracker_csv}`",
        "",
        "| method | budget | tasks | mean train steps/s | mean train wall-clock (s) | mean support similarity | mean query exact |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in tracker_summaries:
        lines.append(
            "| {method} | {budget} | {task_count} | {mean_train_steps_per_second:.2f} | "
            "{mean_train_wall_clock_s:.2f} | {mean_support_similarity:.3f} | {mean_query_exact:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- ARCLE here is measured as single-process random environment interaction.",
            "- JaxARC numbers are end-to-end adaptation training rates from the tracked Phase 1 PPO runs.",
            "- This is still useful as a practical pilot for the paper's speed narrative, but it is not a perfectly matched simulator-only throughput study.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracker-csv",
        type=Path,
        required=True,
        help="Phase 1 tracker CSV to summarize alongside the ARCLE benchmark.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path for JSON output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Path for Markdown output.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of random ARCLE steps to benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for the initial ARCLE reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracker_rows = load_tracker_rows(args.tracker_csv)
    tracker_summary = summarize_tracker(tracker_rows)
    arcle_result = benchmark_arcle_random_steps(args.steps, args.seed)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "arcle_random_step_benchmark": arcle_result,
                "tracker_summary": tracker_summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        build_markdown(
            arcle_result=arcle_result,
            tracker_summaries=tracker_summary,
            tracker_csv=args.tracker_csv,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()