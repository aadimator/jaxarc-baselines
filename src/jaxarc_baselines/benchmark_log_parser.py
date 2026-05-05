from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TIMESTEP_RE = re.compile(r"MISC - Timestep:\s*([0-9]+)")
OOM_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"out of memory",
        r"cuda failure 'out of memory'",
        r"resource_exhausted",
        r"failed to allocate",
        r"cudnn_status_alloc_failed",
        r"xlaruntimeerror",
        r"ncclgroupend",
    ]
]


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_metric_line(line: str) -> dict[str, float | str] | None:
    clean = strip_ansi(line).strip()
    if "ACTOR -" not in clean and "EVALUATOR -" not in clean:
        return None

    section = "actor" if "ACTOR -" in clean else "evaluator"
    payload = clean.split("-", 1)[1]
    metrics: dict[str, float | str] = {"section": section}
    for part in payload.split("|"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key = key.strip().lower().replace(" ", "_")
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def load_metric_rows(log_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    current_timestep: int | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        clean = strip_ansi(line).strip()
        timestep_match = TIMESTEP_RE.search(clean)
        if timestep_match is not None:
            current_timestep = int(timestep_match.group(1))
            continue

        parsed = parse_metric_line(line)
        if parsed is None:
            continue
        if current_timestep is not None and "timestep" not in parsed:
            parsed["timestep"] = float(current_timestep)
        rows.append(parsed)

    return rows


def _row_float(row: dict[str, float | str], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (float, int)) else None


def extract_success_ratio(row: dict[str, float | str]) -> float | None:
    solved_mean = _row_float(row, "solved_mean")
    if solved_mean is not None:
        return solved_mean

    success_rate = _row_float(row, "success_rate")
    if success_rate is not None:
        return success_rate / 100.0

    solve_rate = _row_float(row, "solve_rate")
    if solve_rate is not None:
        return solve_rate / 100.0

    return None


def build_actor_success_curve(
    actor_rows: list[dict[str, float | str]],
) -> list[dict[str, float | int]]:
    curve: list[dict[str, float | int]] = []
    for checkpoint_index, row in enumerate(actor_rows):
        point: dict[str, float | int] = {"checkpoint_index": checkpoint_index}

        timestep = _row_float(row, "timestep")
        if timestep is not None:
            point["timestep"] = int(timestep)

        success_ratio = extract_success_ratio(row)
        if success_ratio is not None:
            point["success_ratio"] = success_ratio

        for metric_name in [
            "best_similarity_mean",
            "final_similarity_mean",
            "episode_return_mean",
            "steps_per_second",
        ]:
            metric_value = _row_float(row, metric_name)
            if metric_value is not None:
                point[metric_name] = metric_value

        curve.append(point)

    return curve


def summarize_actor_metrics(
    actor_rows: list[dict[str, float | str]],
) -> dict[str, float | int]:
    if not actor_rows:
        return {}

    curve = build_actor_success_curve(actor_rows)

    def latest(name: str) -> float | None:
        for row in reversed(actor_rows):
            value = _row_float(row, name)
            if value is not None:
                return value
        return None

    def best(name: str) -> float | None:
        values = [value for row in actor_rows if (value := _row_float(row, name)) is not None]
        return max(values) if values else None

    success_ratio_values = [
        float(point["success_ratio"]) for point in curve if "success_ratio" in point
    ]
    summary: dict[str, float | int] = {
        "num_actor_checkpoints": len(actor_rows),
    }

    if curve:
        first_timestep = curve[0].get("timestep")
        final_timestep = curve[-1].get("timestep")
        if isinstance(first_timestep, int):
            summary["initial_timestep"] = first_timestep
        if isinstance(final_timestep, int):
            summary["final_timestep"] = final_timestep

    if success_ratio_values:
        summary["initial_success_rate_fraction"] = success_ratio_values[0]
        summary["best_success_rate_fraction"] = max(success_ratio_values)
        summary["final_success_rate_fraction"] = success_ratio_values[-1]

    for metric_name in [
        "best_similarity_mean",
        "final_similarity_mean",
        "episode_return_mean",
        "steps_per_second",
    ]:
        latest_value = latest(metric_name)
        if latest_value is not None:
            summary[f"final_{metric_name}"] = latest_value
        best_value = best(metric_name)
        if best_value is not None:
            summary[f"best_{metric_name}"] = best_value

    return summary


def looks_like_oom(log_text: str) -> bool:
    return any(pattern.search(log_text) for pattern in OOM_PATTERNS)


def write_curve_json(curve: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(curve, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_curve_csv(curve: list[dict[str, float | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "checkpoint_index",
        "timestep",
        "success_ratio",
        "best_similarity_mean",
        "final_similarity_mean",
        "episode_return_mean",
        "steps_per_second",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in curve:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


__all__ = [
    "build_actor_success_curve",
    "extract_success_ratio",
    "load_metric_rows",
    "looks_like_oom",
    "parse_metric_line",
    "strip_ansi",
    "summarize_actor_metrics",
    "write_curve_csv",
    "write_curve_json",
]
