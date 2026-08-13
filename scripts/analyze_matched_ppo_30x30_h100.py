#!/usr/bin/env python3
"""Audit and summarize the 30x30 H100 matched PPO evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


PairKey = tuple[str, int]
GroupKey = tuple[int, int, int, int]


def load_latest(
    output_dir: Path,
) -> tuple[
    dict[tuple[str, int, str], dict[str, Any]],
    int,
]:
    latest: dict[tuple[str, int, str], tuple[float, dict[str, Any]]] = {}
    attempts = 0
    for path in sorted((output_dir / "runs").glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") != "complete":
            continue
        attempts += 1
        key = (result["comparison_id"], int(result["seed"]), result["backend"])
        timestamp = path.stat().st_mtime
        previous = latest.get(key)
        if previous is None or timestamp > previous[0]:
            latest[key] = (timestamp, result)
    return {key: value[1] for key, value in latest.items()}, attempts


def minibatch_size(result: dict[str, Any]) -> int:
    config = result["config"]
    return (
        config["num_envs"]
        * config["rollout_length"]
        // config["num_minibatches"]
    )


def normalized_config(result: dict[str, Any]) -> dict[str, Any]:
    config = dict(result["config"])
    config.pop("output_dir", None)
    return config


def logged_interval_slowdown(result: dict[str, Any]) -> float:
    segments: list[float] = []
    previous_update = 0
    previous_elapsed = 0.0
    for point in result.get("curve", []):
        update = int(point["update"])
        elapsed = float(point["elapsed_seconds"])
        segments.append(
            (elapsed - previous_elapsed) / (update - previous_update)
        )
        previous_update = update
        previous_elapsed = elapsed
    positive = [segment for segment in segments if segment > 0]
    return max(positive) / min(positive) if positive else 1.0


def compile_rejection_reasons(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    compile_seconds = float(result["timing"]["training_compile_warmup_seconds"])
    if compile_seconds > 120.0:
        reasons.append(f"compile phase took {compile_seconds:.1f}s")
    return reasons


def training_rejection_reasons(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    timing = result["timing"]

    curve = result.get("curve", [])
    if curve:
        first_sps = float(curve[0]["steps_per_second"])
        final_sps = float(timing["train_steps_per_second"])
        if first_sps > 0 and final_sps < 0.5 * first_sps:
            reasons.append(
                f"final SPS was {final_sps / first_sps:.2f}x first-update SPS"
            )
    return reasons


def timing_rejection_reasons(result: dict[str, Any]) -> list[str]:
    """Detect wall-clock contamination, not ordinary run-to-run variation."""
    return compile_rejection_reasons(result) + training_rejection_reasons(result)


def matched_pairs(
    results: dict[tuple[str, int, str], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pair_keys: set[PairKey] = {(key[0], key[1]) for key in results}
    for comparison_id, seed in sorted(pair_keys):
        jaxarc = results.get((comparison_id, seed, "jaxarc"))
        arcle = results.get((comparison_id, seed, "arcle"))
        if jaxarc is None or arcle is None:
            continue
        if normalized_config(jaxarc) != normalized_config(arcle):
            raise ValueError(f"config mismatch for {comparison_id} seed {seed}")
        if jaxarc["task_hashes"] != arcle["task_hashes"]:
            raise ValueError(f"task hash mismatch for {comparison_id} seed {seed}")
        pairs.append((jaxarc, arcle))
    return pairs


def group_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[GroupKey, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[GroupKey, list[tuple[dict[str, Any], dict[str, Any]]]] = (
        defaultdict(list)
    )
    for jaxarc, arcle in pairs:
        config = jaxarc["config"]
        key = (
            int(config["num_envs"]),
            int(config["rollout_length"]),
            minibatch_size(jaxarc),
            int(jaxarc["actual_timesteps"]),
        )
        grouped[key].append((jaxarc, arcle))
    return dict(grouped)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def render(output_dir: Path) -> str:
    results, attempt_count = load_latest(output_dir)
    pairs = matched_pairs(results)
    rejected = {
        (result["backend"], int(result["seed"])): timing_rejection_reasons(result)
        for result in results.values()
        if timing_rejection_reasons(result)
    }
    steady_pairs = [
        (jaxarc, arcle)
        for jaxarc, arcle in pairs
        if not training_rejection_reasons(jaxarc)
        and not training_rejection_reasons(arcle)
    ]
    cold_pairs = [
        (jaxarc, arcle)
        for jaxarc, arcle in steady_pairs
        if not compile_rejection_reasons(jaxarc)
        and not compile_rejection_reasons(arcle)
    ]
    grouped = group_pairs(pairs)

    lines = [
        "# 30x30 H100 evidence audit",
        "",
        f"Complete attempts: {attempt_count}. Latest results: {len(results)}. "
        f"Clean steady pairs: {len(steady_pairs)} of {len(pairs)}. "
        f"Clean cold-wall pairs: {len(cold_pairs)} of {len(pairs)}.",
        "",
    ]

    if results:
        devices = sorted(
            {
                device["kind"]
                for result in results.values()
                for device in result["runtime"]["devices"]
            }
        )
        grid_sizes = sorted(
            {int(result["config"]["grid_size"]) for result in results.values()}
        )
        channels = sorted(
            {
                3 + 2 * int(result["config"]["context_pairs"])
                for result in results.values()
            }
        )
        parameter_counts = sorted(
            {int(result["policy"]["parameter_count"]) for result in results.values()}
        )
        task_hash_sets = {
            tuple(sorted(result["task_hashes"].items()))
            for result in results.values()
        }
        source_train_shapes = sorted(
            {
                tuple(shape)
                for result in results.values()
                for statistics in result["task_statistics"].values()
                for shape in statistics["train_shapes"]
            }
        )
        all_source_train_pairs_exact = all(
            shape == (grid_sizes[0], grid_sizes[0])
            for shape in source_train_shapes
        )
        lines.extend(
            [
                "## Evidence checks",
                "",
                f"- Device: {', '.join(devices)}.",
                f"- Grid size: {grid_sizes}. Observation channels: {channels}.",
                f"- Policy parameter count: {parameter_counts}.",
                f"- Distinct task-hash sets: {len(task_hash_sets)}.",
                f"- Native training shapes: {source_train_shapes}.",
                "- Every native training input and output is exactly the "
                f"configured grid size: {all_source_train_pairs_exact}.",
                "- Every reported pair has identical resolved configuration and task hashes.",
                "- Aggregates are rebuilt from the latest complete result for each "
                "backend and seed; pre-existing campaign CSVs are not used.",
                "",
                "## Matched results",
                "",
            ]
        )

    lines.extend(
        [
        "| envs | rollout | minibatch | steps | steady seeds | cold seeds | "
        "JaxARC SPS | ARCLE SPS | steady speedup | cold-wall speedup |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    summaries: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items()):
        envs, rollout, minibatch, steps = key
        steady_entries = [
            (jaxarc, arcle)
            for jaxarc, arcle in entries
            if not training_rejection_reasons(jaxarc)
            and not training_rejection_reasons(arcle)
        ]
        cold_entries = [
            (jaxarc, arcle)
            for jaxarc, arcle in steady_entries
            if not compile_rejection_reasons(jaxarc)
            and not compile_rejection_reasons(arcle)
        ]
        if not steady_entries:
            continue
        jax_sps = [
            float(jaxarc["timing"]["train_steps_per_second"])
            for jaxarc, _ in steady_entries
        ]
        arcle_sps = [
            float(arcle["timing"]["train_steps_per_second"])
            for _, arcle in steady_entries
        ]
        steady = [
            jaxarc["timing"]["train_steps_per_second"]
            / arcle["timing"]["train_steps_per_second"]
            for jaxarc, arcle in steady_entries
        ]
        cold = [
            arcle["timing"]["training_wall_seconds"]
            / jaxarc["timing"]["training_wall_seconds"]
            for jaxarc, arcle in cold_entries
        ]
        summary = {
            "envs": envs,
            "rollout": rollout,
            "minibatch": minibatch,
            "steps": steps,
            "steady_seeds": sorted(
                int(jaxarc["seed"]) for jaxarc, _ in steady_entries
            ),
            "cold_seeds": sorted(
                int(jaxarc["seed"]) for jaxarc, _ in cold_entries
            ),
            "jax_sps": median(jax_sps),
            "arcle_sps": median(arcle_sps),
            "steady": median(steady),
            "steady_min": min(steady),
            "steady_max": max(steady),
            "cold": median(cold) if cold else None,
            "cold_min": min(cold) if cold else None,
            "cold_max": max(cold) if cold else None,
        }
        summaries.append(summary)
        cold_text = (
            f"{summary['cold']:.2f}x" if summary["cold"] is not None else "n/a"
        )
        lines.append(
            f"| {envs} | {rollout} | {minibatch} | {steps} | "
            f"{','.join(map(str, summary['steady_seeds']))} | "
            f"{','.join(map(str, summary['cold_seeds'])) or 'none'} | "
            f"{summary['jax_sps']:.1f} | "
            f"{summary['arcle_sps']:.1f} | {summary['steady']:.2f}x | "
            f"{cold_text} |"
        )

    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| seed | JaxARC SPS | ARCLE SPS | steady speedup | "
            "JaxARC cold wall | ARCLE cold wall | cold-wall speedup |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for jaxarc, arcle in sorted(pairs, key=lambda pair: int(pair[0]["seed"])):
        if training_rejection_reasons(jaxarc) or training_rejection_reasons(arcle):
            continue
        jax_sps = float(jaxarc["timing"]["train_steps_per_second"])
        arcle_sps = float(arcle["timing"]["train_steps_per_second"])
        steady_speedup = jax_sps / arcle_sps
        if compile_rejection_reasons(jaxarc) or compile_rejection_reasons(arcle):
            jax_wall = arcle_wall = cold_speedup = "excluded"
        else:
            jax_wall_value = float(jaxarc["timing"]["training_wall_seconds"])
            arcle_wall_value = float(arcle["timing"]["training_wall_seconds"])
            jax_wall = f"{jax_wall_value:.2f}s"
            arcle_wall = f"{arcle_wall_value:.2f}s"
            cold_speedup = f"{arcle_wall_value / jax_wall_value:.2f}x"
        lines.append(
            f"| {jaxarc['seed']} | {jax_sps:.1f} | {arcle_sps:.1f} | "
            f"{steady_speedup:.2f}x | {jax_wall} | {arcle_wall} | "
            f"{cold_speedup} |"
        )

    lines.extend(["", "## Timing checks", ""])
    if rejected:
        lines.append(
            "The following latest attempts fail the timing checks fixed before reruns. "
            "They are excluded from aggregates:"
        )
        for (backend, seed), reasons in sorted(rejected.items()):
            lines.append(f"- {backend} seed {seed}: {'; '.join(reasons)}.")
    else:
        lines.append("- All latest attempts pass the timing checks.")

    eligible = [summary for summary in summaries if summary["steps"] >= 1_048_576]
    if len(eligible) >= 2:
        arcle_median = median([summary["arcle_sps"] for summary in eligible])
        outliers = [
            summary
            for summary in eligible
            if summary["arcle_sps"] < 0.25 * arcle_median
        ]
        if outliers:
            labels = ", ".join(str(summary["envs"]) for summary in outliers)
            lines.append(
                f"- ARCLE throughput outlier at {labels} environments. It is below "
                "25% of the sweep median. Do not use it as the headline comparison."
            )
        else:
            lines.append("- No cross-scale ARCLE throughput outlier detected.")

    worst_slowdown: tuple[float, dict[str, Any]] | None = None
    for result in results.values():
        slowdown = logged_interval_slowdown(result)
        if worst_slowdown is None or slowdown > worst_slowdown[0]:
            worst_slowdown = (slowdown, result)
    if worst_slowdown is not None:
        slowdown, result = worst_slowdown
        lines.append(
            f"- Largest logged interval slowdown: {slowdown:.1f}x, "
            f"{result['backend']} at {result['config']['num_envs']} environments, "
            f"seed {result['seed']}."
        )

    lines.extend(["", "## Decision", ""])
    eligible = [
        summary
        for summary in eligible
        if not (
            len(eligible) > 1
            and summary["arcle_sps"]
            < 0.25 * median([item["arcle_sps"] for item in eligible])
        )
    ]
    if eligible:
        peak_jax_sps = max(summary["jax_sps"] for summary in eligible)
        near_peak = [
            summary
            for summary in eligible
            if summary["jax_sps"] >= 0.8 * peak_jax_sps
        ]
        chosen = max(
            near_peak,
            key=lambda summary: (summary["envs"], summary["jax_sps"]),
        )
        lines.append(
            f"Use {chosen['envs']} environments, rollout {chosen['rollout']}, "
            f"minibatch {chosen['minibatch']}."
        )
        if (
            len(chosen["steady_seeds"]) < 3
            or len(chosen["cold_seeds"]) < 3
        ):
            rejected_seeds = sorted({seed for _backend, seed in rejected})
            if rejected_seeds:
                lines.append(
                    "Rerun only contaminated seeds "
                    f"{', '.join(map(str, rejected_seeds))} before reporting "
                    "a median or range."
                )
            else:
                lines.append(
                    "Repeat this exact configuration with seeds 13, 42, and 123 "
                    "before reporting a median or range."
                )
        elif chosen["cold"] is not None:
            lines.append(
                f"Three-seed evidence is available. Report median steady speedup "
                f"{chosen['steady']:.2f}x, range {chosen['steady_min']:.2f}x to "
                f"{chosen['steady_max']:.2f}x; median cold-wall speedup "
                f"{chosen['cold']:.2f}x, range {chosen['cold_min']:.2f}x to "
                f"{chosen['cold_max']:.2f}x."
            )
    else:
        lines.append("No complete 1,048,576-step matched pair found.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    report = render(output_dir)
    destination = output_dir / "analysis.md"
    destination.write_text(report, encoding="utf-8")
    print(report)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
