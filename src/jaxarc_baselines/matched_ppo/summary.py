"""Aggregate controlled and best-capacity PPO benchmark results."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


ResultEntry = tuple[Path, dict[str, Any]]
SelectionKey = tuple[str, int, str]


def _fallback_comparison_id(result: dict[str, Any]) -> str:
    config = dict(result["config"])
    config.pop("output_dir", None)
    payload = {
        "config": config,
        "task_hashes": result["task_hashes"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _load_results(output_dir: Path) -> list[ResultEntry]:
    loaded: list[ResultEntry] = []
    for path in sorted((output_dir / "runs").glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            continue
        evaluation_path = path.with_name("evaluation.json")
        if not payload.get("evaluation") and evaluation_path.exists():
            evaluation_payload = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            if (
                evaluation_payload.get("seed") == payload.get("seed")
                and evaluation_payload.get("task_hashes")
                == payload.get("task_hashes")
            ):
                payload["evaluation"] = evaluation_payload["evaluation"]
                payload["evaluation_backend"] = evaluation_payload.get(
                    "evaluation_backend"
                )
        payload.setdefault(
            "comparison_id",
            _fallback_comparison_id(payload),
        )
        loaded.append((path, payload))
    return loaded


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _training_wall(result: dict[str, Any]) -> float:
    timing = result["timing"]
    return timing.get(
        "training_wall_seconds",
        timing.get("cold_total_seconds"),
    )


def _compile_seconds(result: dict[str, Any]) -> float:
    timing = result["timing"]
    return timing.get(
        "training_compile_warmup_seconds",
        timing.get("compile_warmup_seconds"),
    )


def _similarity_gain(result: dict[str, Any]) -> float | None:
    evaluation = result.get("evaluation")
    if not evaluation:
        return None
    return evaluation.get("macro_mean_similarity_gain")


def _exact_success_rate(result: dict[str, Any]) -> float | None:
    evaluation = result.get("evaluation")
    if not evaluation:
        return None
    return evaluation.get("exact_success_rate")


def _select_latest(
    results: list[ResultEntry],
) -> dict[SelectionKey, ResultEntry]:
    selected: dict[SelectionKey, ResultEntry] = {}
    for path, result in results:
        key = (
            result["comparison_id"],
            int(result["seed"]),
            result["backend"],
        )
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            selected[key] = (path, result)
    return selected


def _run_row(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    config = result["config"]
    return {
        "experiment_name": result["experiment_name"],
        "comparison_id": result["comparison_id"],
        "seed": result["seed"],
        "backend": result["backend"],
        "policy": config.get("policy_architecture", "factorized"),
        "num_envs": config["num_envs"],
        "rollout_length": config["rollout_length"],
        "update_batch_size": (
            config["num_envs"] * config["rollout_length"]
        ),
        "minibatch_size": (
            config["num_envs"]
            * config["rollout_length"]
            // config["num_minibatches"]
        ),
        "epochs": config["epochs"],
        "timesteps": result.get(
            "actual_timesteps",
            config["total_timesteps"],
        ),
        "train_seconds": result["timing"]["train_seconds"],
        "train_sps": result["timing"]["train_steps_per_second"],
        "compile_seconds": _compile_seconds(result),
        "training_wall_seconds": _training_wall(result),
        "exact_success_rate": _exact_success_rate(result),
        "macro_similarity_gain": _similarity_gain(result),
        "result": str(path),
    }


def _all_run_rows(
    selected: dict[SelectionKey, ResultEntry],
) -> list[dict[str, Any]]:
    return [
        _run_row(path, result)
        for path, result in sorted(selected.values(), key=lambda item: str(item[0]))
    ]


def _paired_rows(
    selected: dict[SelectionKey, ResultEntry],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = sorted(
        {(key[0], key[1]) for key in selected}
    )
    for comparison_id, seed in comparisons:
        jaxarc_entry = selected.get((comparison_id, seed, "jaxarc"))
        arcle_entry = selected.get((comparison_id, seed, "arcle"))
        reference = jaxarc_entry or arcle_entry
        if reference is None:
            continue
        base = _run_row(*reference)
        row: dict[str, Any] = {
            key: base[key]
            for key in (
                "experiment_name",
                "comparison_id",
                "seed",
                "policy",
                "num_envs",
                "rollout_length",
                "update_batch_size",
                "minibatch_size",
                "epochs",
                "timesteps",
            )
        }
        if jaxarc_entry is None or arcle_entry is None:
            row.update(
                {
                    "matched": False,
                    "reason": "missing backend result",
                }
            )
            rows.append(row)
            continue

        jaxarc = _run_row(*jaxarc_entry)
        arcle = _run_row(*arcle_entry)
        row.update(
            {
                "matched": True,
                "reason": "",
                "jaxarc_train_sps": jaxarc["train_sps"],
                "arcle_train_sps": arcle["train_sps"],
                "train_speedup": (
                    jaxarc["train_sps"] / arcle["train_sps"]
                ),
                "jaxarc_compile_seconds": jaxarc["compile_seconds"],
                "arcle_compile_seconds": arcle["compile_seconds"],
                "jaxarc_training_wall_seconds": jaxarc[
                    "training_wall_seconds"
                ],
                "arcle_training_wall_seconds": arcle[
                    "training_wall_seconds"
                ],
                "training_wall_speedup": (
                    arcle["training_wall_seconds"]
                    / jaxarc["training_wall_seconds"]
                ),
                "jaxarc_exact_success_rate": jaxarc[
                    "exact_success_rate"
                ],
                "arcle_exact_success_rate": arcle[
                    "exact_success_rate"
                ],
                "jaxarc_macro_similarity_gain": jaxarc[
                    "macro_similarity_gain"
                ],
                "arcle_macro_similarity_gain": arcle[
                    "macro_similarity_gain"
                ],
                "jaxarc_result": jaxarc["result"],
                "arcle_result": arcle["result"],
            }
        )
        rows.append(row)
    return rows


def _comparison_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    comparison_ids = sorted({row["comparison_id"] for row in rows})
    for comparison_id in comparison_ids:
        valid = [
            row
            for row in rows
            if row["comparison_id"] == comparison_id and row["matched"]
        ]
        if not valid:
            continue
        reference = valid[0]
        summaries.append(
            {
                "experiment_name": reference["experiment_name"],
                "comparison_id": comparison_id,
                "policy": reference["policy"],
                "num_envs": reference["num_envs"],
                "rollout_length": reference["rollout_length"],
                "minibatch_size": reference["minibatch_size"],
                "epochs": reference["epochs"],
                "timesteps": reference["timesteps"],
                "matched_seeds": len(valid),
                "seeds": [row["seed"] for row in valid],
                "median_train_speedup": _median(
                    [row["train_speedup"] for row in valid]
                ),
                "median_training_wall_speedup": _median(
                    [row["training_wall_speedup"] for row in valid]
                ),
                "median_jaxarc_train_sps": _median(
                    [row["jaxarc_train_sps"] for row in valid]
                ),
                "median_arcle_train_sps": _median(
                    [row["arcle_train_sps"] for row in valid]
                ),
                "median_jaxarc_exact_success_rate": _median(
                    [
                        row["jaxarc_exact_success_rate"]
                        for row in valid
                        if row["jaxarc_exact_success_rate"] is not None
                    ]
                ),
                "median_arcle_exact_success_rate": _median(
                    [
                        row["arcle_exact_success_rate"]
                        for row in valid
                        if row["arcle_exact_success_rate"] is not None
                    ]
                ),
                "median_jaxarc_macro_similarity_gain": _median(
                    [
                        row["jaxarc_macro_similarity_gain"]
                        for row in valid
                        if row["jaxarc_macro_similarity_gain"] is not None
                    ]
                ),
                "median_arcle_macro_similarity_gain": _median(
                    [
                        row["arcle_macro_similarity_gain"]
                        for row in valid
                        if row["arcle_macro_similarity_gain"] is not None
                    ]
                ),
            }
        )
    return summaries


def _task_summaries(
    selected: dict[SelectionKey, ResultEntry],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (comparison_id, seed, backend), (_path, result) in selected.items():
        if backend not in {"jaxarc", "arcle"}:
            continue
        evaluation = result.get("evaluation")
        if not evaluation:
            continue
        counterpart = selected.get(
            (
                comparison_id,
                seed,
                "arcle" if backend == "jaxarc" else "jaxarc",
            )
        )
        if counterpart is None:
            continue
        if not counterpart[1].get("evaluation"):
            continue
        for task_id, metrics in evaluation["by_task"].items():
            grouped.setdefault((comparison_id, task_id), []).append(
                {
                    "experiment_name": result["experiment_name"],
                    "policy": result["config"].get(
                        "policy_architecture",
                        "factorized",
                    ),
                    "num_envs": result["config"]["num_envs"],
                    "timesteps": result.get(
                        "actual_timesteps",
                        result["config"]["total_timesteps"],
                    ),
                    "backend": backend,
                    **metrics,
                }
            )

    summaries: list[dict[str, Any]] = []
    for (comparison_id, task_id), rows in sorted(grouped.items()):
        jaxarc = [row for row in rows if row["backend"] == "jaxarc"]
        arcle = [row for row in rows if row["backend"] == "arcle"]
        reference = rows[0]
        summaries.append(
            {
                "experiment_name": reference["experiment_name"],
                "comparison_id": comparison_id,
                "policy": reference["policy"],
                "num_envs": reference["num_envs"],
                "timesteps": reference["timesteps"],
                "task_id": task_id,
                "matched_seeds": min(len(jaxarc), len(arcle)),
                "median_jaxarc_exact_success_rate": _median(
                    [row["exact_success_rate"] for row in jaxarc]
                ),
                "median_arcle_exact_success_rate": _median(
                    [row["exact_success_rate"] for row in arcle]
                ),
                "median_jaxarc_similarity_gain": _median(
                    [
                        row.get("mean_similarity_gain")
                        for row in jaxarc
                        if row.get("mean_similarity_gain") is not None
                    ]
                ),
                "median_arcle_similarity_gain": _median(
                    [
                        row.get("mean_similarity_gain")
                        for row in arcle
                        if row.get("mean_similarity_gain") is not None
                    ]
                ),
            }
        )
    return summaries


def _capacity_rows(
    run_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["experiment_name"],
            row["policy"],
            row["seed"],
        )
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for (
        experiment,
        policy,
        seed,
    ), candidates in sorted(grouped.items()):
        jaxarc = [
            row for row in candidates if row["backend"] == "jaxarc"
        ]
        arcle = [
            row
            for row in candidates
            if row["backend"] in {"arcle", "arcle-async"}
        ]
        if not jaxarc or not arcle:
            continue
        if (
            len({row["num_envs"] for row in jaxarc}) < 2
            or len({row["num_envs"] for row in arcle}) < 2
        ):
            continue
        best_jaxarc = max(jaxarc, key=lambda row: row["train_sps"])
        best_arcle = max(arcle, key=lambda row: row["train_sps"])
        rows.append(
            {
                "experiment_name": experiment,
                "policy": policy,
                "seed": seed,
                "jaxarc_minibatch_size": best_jaxarc["minibatch_size"],
                "arcle_minibatch_size": best_arcle["minibatch_size"],
                "jaxarc_num_envs": best_jaxarc["num_envs"],
                "arcle_num_envs": best_arcle["num_envs"],
                "arcle_backend": best_arcle["backend"],
                "jaxarc_train_sps": best_jaxarc["train_sps"],
                "arcle_train_sps": best_arcle["train_sps"],
                "best_capacity_speedup": (
                    best_jaxarc["train_sps"] / best_arcle["train_sps"]
                ),
                "jaxarc_result": best_jaxarc["result"],
                "arcle_result": best_arcle["result"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        if not fieldnames:
            return
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize_results(output_dir: Path) -> Path:
    """Write controlled-pair, capacity, and task summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _select_latest(_load_results(output_dir))
    run_rows = _all_run_rows(selected)
    paired_rows = _paired_rows(selected)
    comparisons = _comparison_summaries(paired_rows)
    task_rows = _task_summaries(selected)
    capacity_rows = _capacity_rows(run_rows)

    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparisons": comparisons,
                "capacity": capacity_rows,
                "tasks": task_rows,
                "paired_runs": paired_rows,
                "all_runs": run_rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _write_csv(output_dir / "all_runs.csv", run_rows)
    _write_csv(output_dir / "paired_runs.csv", paired_rows)
    _write_csv(output_dir / "best_capacity.csv", capacity_rows)
    _write_csv(output_dir / "per_task.csv", task_rows)

    markdown = [
        "# Matched PPO benchmark",
        "",
        "## Controlled comparisons",
        "",
        "| policy | envs | rollout | minibatch | epochs | steps | seeds | "
        "train speedup | "
        "training-wall speedup | JaxARC SPS | ARCLE SPS | "
        "JaxARC gain | ARCLE gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in comparisons:
        markdown.append(
            "| "
            + " | ".join(
                [
                    summary["policy"],
                    str(summary["num_envs"]),
                    str(summary["rollout_length"]),
                    str(summary["minibatch_size"]),
                    str(summary["epochs"]),
                    str(summary["timesteps"]),
                    str(summary["matched_seeds"]),
                    _format(summary["median_train_speedup"]),
                    _format(summary["median_training_wall_speedup"]),
                    _format(summary["median_jaxarc_train_sps"]),
                    _format(summary["median_arcle_train_sps"]),
                    _format(
                        summary["median_jaxarc_macro_similarity_gain"]
                    ),
                    _format(
                        summary["median_arcle_macro_similarity_gain"]
                    ),
                ]
            )
            + " |"
        )

    markdown.extend(
        [
            "",
            "## Best observed capacity",
            "",
            "| policy | seed | JaxARC minibatch | JaxARC envs | "
            "ARCLE mode | ARCLE minibatch | ARCLE envs | "
            "JaxARC SPS | ARCLE SPS | speedup |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_rows:
        markdown.append(
            "| "
            + " | ".join(
                [
                    row["policy"],
                    str(row["seed"]),
                    str(row["jaxarc_minibatch_size"]),
                    str(row["jaxarc_num_envs"]),
                    row["arcle_backend"],
                    str(row["arcle_minibatch_size"]),
                    str(row["arcle_num_envs"]),
                    _format(row["jaxarc_train_sps"]),
                    _format(row["arcle_train_sps"]),
                    _format(row["best_capacity_speedup"]),
                ]
            )
            + " |"
        )

    markdown.extend(
        [
            "",
            "## Per-task controlled evaluation",
            "",
            "| policy | envs | steps | task | seeds | JaxARC success | "
            "ARCLE success | JaxARC gain | ARCLE gain |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_rows:
        markdown.append(
            "| "
            + " | ".join(
                [
                    row["policy"],
                    str(row["num_envs"]),
                    str(row["timesteps"]),
                    row["task_id"],
                    str(row["matched_seeds"]),
                    _format(
                        row["median_jaxarc_exact_success_rate"]
                    ),
                    _format(
                        row["median_arcle_exact_success_rate"]
                    ),
                    _format(row["median_jaxarc_similarity_gain"]),
                    _format(row["median_arcle_similarity_gain"]),
                ]
            )
            + " |"
        )
    markdown.extend(
        [
            "",
            "Controlled rows require identical configs and task hashes. "
            "Best-capacity rows tune environment count independently and "
            "support throughput claims only.",
            "",
        ]
    )
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    print(f"wrote {markdown_path}")
    return markdown_path
