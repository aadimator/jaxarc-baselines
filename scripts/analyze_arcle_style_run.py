from __future__ import annotations

import argparse
import json
from pathlib import Path

from jaxarc_baselines.log_analysis import (
    build_actor_success_curve,
    load_metric_rows,
    summarize_actor_metrics,
    write_curve_csv,
    write_curve_json,
)

def build_markdown(summary: dict[str, float | int], log_path: Path, title: str) -> str:
    lines = [
        f"## {title}",
        "",
        f"- log: `{log_path}`",
        f"- actor checkpoints parsed: {int(summary.get('num_actor_checkpoints', 0))}",
        "",
        "## Key metrics",
        "",
    ]
    metric_rows = [
        ("Initial success rate (0-1)", "initial_success_rate_fraction"),
        ("Best success rate (0-1)", "best_success_rate_fraction"),
        ("Final success rate (0-1)", "final_success_rate_fraction"),
        ("Best similarity mean", "best_best_similarity_mean"),
        ("Final best-similarity mean", "final_best_similarity_mean"),
        ("Best final-similarity mean", "best_final_similarity_mean"),
        ("Final final-similarity mean", "final_final_similarity_mean"),
        ("Final episode return mean", "final_episode_return_mean"),
        ("Best steps/s", "best_steps_per_second"),
    ]
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for label, key in metric_rows:
        value = summary.get(key)
        if value is not None:
            lines.append(f"| {label} | {value:.6f} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Success ratio is derived from existing log output rather than from extra metrics added during training.",
            "- When `Solved mean` is present, it is used directly because it already lies in `[0, 1]`.",
            "- If only percentage-style `success_rate` or `solve_rate` appears, the report converts it back to a fraction by dividing by 100.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--curve-json-out", type=Path, default=None)
    parser.add_argument("--curve-csv-out", type=Path, default=None)
    parser.add_argument("--title", type=str, default="ARCLE-style run summary")
    args = parser.parse_args()

    rows = load_metric_rows(args.log_path)
    actor_rows = [row for row in rows if row.get("section") == "actor"]
    evaluator_rows = [row for row in rows if row.get("section") == "evaluator"]
    summary = summarize_actor_metrics(actor_rows)
    summary["num_evaluator_checkpoints"] = len(evaluator_rows)
    curve = build_actor_success_curve(actor_rows)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if args.md_out is not None:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(
            build_markdown(summary, args.log_path, args.title), encoding="utf-8"
        )

    if args.curve_json_out is not None:
        write_curve_json(curve, args.curve_json_out)

    if args.curve_csv_out is not None:
        write_curve_csv(curve, args.curve_csv_out)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()