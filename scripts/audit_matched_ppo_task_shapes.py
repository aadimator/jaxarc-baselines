#!/usr/bin/env python3
"""Audit native ARC grid dimensions for a matched PPO config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jaxarc_baselines.matched_ppo.config import load_config


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    if not grid or not grid[0]:
        raise ValueError("ARC grids must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("ARC grids must be rectangular")
    return len(grid), width


def render(config_path: Path, require_exact_grid: bool) -> tuple[str, bool]:
    config = load_config(config_path)
    lines = [
        "# Native task-shape audit",
        "",
        f"Configured tensor size: {config.grid_size}x{config.grid_size}.",
        "",
        "| task | split | pairs | native input shapes | native output shapes | exact grid |",
        "|---|---|---:|---|---|---:|",
    ]
    all_exact = True
    for task_id in config.task_ids:
        task_path = (
            config.task_root / config.task_split / f"{task_id}.json"
        )
        task = json.loads(task_path.read_text(encoding="utf-8"))
        for split in ("train", "test"):
            pairs = task[split]
            input_shapes = [grid_shape(pair["input"]) for pair in pairs]
            output_shapes = [grid_shape(pair["output"]) for pair in pairs]
            for shape in input_shapes + output_shapes:
                if shape[0] > config.grid_size or shape[1] > config.grid_size:
                    raise ValueError(
                        f"{task_id} {split} shape {shape} exceeds "
                        f"{config.grid_size}x{config.grid_size}"
                    )
            exact = all(
                shape == (config.grid_size, config.grid_size)
                for shape in input_shapes + output_shapes
            )
            all_exact = all_exact and exact
            input_text = ", ".join(
                f"{height}x{width}" for height, width in input_shapes
            )
            output_text = ", ".join(
                f"{height}x{width}" for height, width in output_shapes
            )
            lines.append(
                f"| {task_id} | {split} | {len(pairs)} | {input_text} | "
                f"{output_text} | {'yes' if exact else 'no'} |"
            )
    lines.extend(
        [
            "",
            "All native inputs and outputs are exactly the configured size: "
            f"{'yes' if all_exact else 'no'}.",
            "",
        ]
    )
    return "\n".join(lines), all_exact or not require_exact_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--require-exact-grid", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report, valid = render(args.config.resolve(), args.require_exact_grid)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"wrote {args.output.resolve()}")
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
