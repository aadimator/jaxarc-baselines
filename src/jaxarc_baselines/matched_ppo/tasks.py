"""Task loading shared by both environment backends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxarc.types import JaxArcTask
from jaxarc.utils.buffer import stack_task_list

from .config import BenchmarkConfig


@dataclass(frozen=True)
class PairVariant:
    """One answer-visible training pair plus its task context."""

    variant_index: int
    task_id: str
    pair_index: int
    input_grid: np.ndarray
    output_grid: np.ndarray
    context_inputs: np.ndarray
    context_outputs: np.ndarray


@dataclass(frozen=True)
class TaskBundle:
    """Canonical task material consumed by JaxARC and ARCLE."""

    variants: tuple[PairVariant, ...]
    jaxarc_buffer: Any
    context_channels: jnp.ndarray
    task_hashes: dict[str, str]
    task_statistics: dict[str, dict[str, Any]]
    variant_initial_similarities: np.ndarray


def _task_path(config: BenchmarkConfig, task_id: str) -> Path:
    return config.task_root / config.task_split / f"{task_id}.json"


def _pad_grid(
    grid: list[list[int]] | np.ndarray,
    grid_size: int,
    *,
    fill_value: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(grid, dtype=np.int32)
    height, width = array.shape
    if height > grid_size or width > grid_size:
        raise ValueError(
            f"Grid shape {array.shape} exceeds configured {grid_size}x{grid_size}"
        )
    padded = np.full((grid_size, grid_size), fill_value, dtype=np.int32)
    mask = np.zeros((grid_size, grid_size), dtype=np.bool_)
    padded[:height, :width] = array
    mask[:height, :width] = True
    return padded, mask


def _context_for_pair(
    train_pairs: list[dict[str, Any]],
    current_pair: int,
    grid_size: int,
    context_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = np.full(
        (context_pairs, grid_size, grid_size), -1, dtype=np.int32
    )
    outputs = np.full_like(inputs, -1)
    others = [
        pair for index, pair in enumerate(train_pairs) if index != current_pair
    ]
    for slot, pair in enumerate(others[:context_pairs]):
        inputs[slot], _ = _pad_grid(pair["input"], grid_size)
        outputs[slot], _ = _pad_grid(pair["output"], grid_size)
    return inputs, outputs


def _single_pair_jaxarc_task(
    variant: PairVariant,
    grid_size: int,
) -> JaxArcTask:
    input_grid, input_mask = _pad_grid(variant.input_grid, grid_size)
    output_grid, output_mask = _pad_grid(variant.output_grid, grid_size)
    empty_grid = np.full((1, grid_size, grid_size), -1, dtype=np.int32)
    empty_mask = np.zeros((1, grid_size, grid_size), dtype=np.bool_)
    return JaxArcTask(
        input_grids_examples=jnp.asarray(input_grid[None, ...]),
        input_masks_examples=jnp.asarray(input_mask[None, ...]),
        output_grids_examples=jnp.asarray(output_grid[None, ...]),
        output_masks_examples=jnp.asarray(output_mask[None, ...]),
        num_train_pairs=jnp.asarray(1, dtype=jnp.int32),
        test_input_grids=jnp.asarray(empty_grid),
        test_input_masks=jnp.asarray(empty_mask),
        true_test_output_grids=jnp.asarray(empty_grid),
        true_test_output_masks=jnp.asarray(empty_mask),
        num_test_pairs=jnp.asarray(0, dtype=jnp.int32),
        task_index=jnp.asarray(variant.variant_index, dtype=jnp.int32),
    )


def _interleave_context(variant: PairVariant) -> np.ndarray:
    channels = np.stack(
        [variant.context_inputs, variant.context_outputs], axis=1
    ).reshape(
        2 * variant.context_inputs.shape[0],
        variant.context_inputs.shape[1],
        variant.context_inputs.shape[2],
    )
    return np.transpose(channels, (1, 2, 0))


def load_task_bundle(config: BenchmarkConfig) -> TaskBundle:
    """Load exact JSON files once and build both backends from them."""
    variants: list[PairVariant] = []
    task_hashes: dict[str, str] = {}
    task_statistics: dict[str, dict[str, Any]] = {}
    for task_id in config.task_ids:
        path = _task_path(config, task_id)
        raw_bytes = path.read_bytes()
        task_hashes[task_id] = hashlib.sha256(raw_bytes).hexdigest()
        task = json.loads(raw_bytes)
        train_pairs = task["train"]
        shapes = []
        changed_pixels = []
        for pair_index, pair in enumerate(train_pairs):
            input_array = np.asarray(pair["input"], dtype=np.int32)
            output_array = np.asarray(pair["output"], dtype=np.int32)
            if input_array.shape != output_array.shape:
                raise ValueError(
                    f"{task_id} pair {pair_index} changes grid shape; "
                    "the matched point-action benchmark excludes resize."
                )
            shapes.append(list(input_array.shape))
            changed_pixels.append(int(np.count_nonzero(input_array != output_array)))
            context_inputs, context_outputs = _context_for_pair(
                train_pairs,
                pair_index,
                config.grid_size,
                config.context_pairs,
            )
            variants.append(
                PairVariant(
                    variant_index=len(variants),
                    task_id=task_id,
                    pair_index=pair_index,
                    input_grid=input_array,
                    output_grid=output_array,
                    context_inputs=context_inputs,
                    context_outputs=context_outputs,
                )
            )
        task_statistics[task_id] = {
            "train_pairs": len(train_pairs),
            "train_shapes": shapes,
            "minimum_changed_pixels": min(changed_pixels),
            "maximum_changed_pixels": max(changed_pixels),
            "mean_changed_pixels": float(np.mean(changed_pixels)),
        }

    jaxarc_tasks = [
        _single_pair_jaxarc_task(variant, config.grid_size)
        for variant in variants
    ]
    context_channels = jnp.asarray(
        np.stack([_interleave_context(variant) for variant in variants])
    )
    return TaskBundle(
        variants=tuple(variants),
        jaxarc_buffer=stack_task_list(jaxarc_tasks),
        context_channels=context_channels,
        task_hashes=task_hashes,
        task_statistics=task_statistics,
        variant_initial_similarities=np.asarray(
            [
                np.mean(variant.input_grid == variant.output_grid)
                for variant in variants
            ],
            dtype=np.float32,
        ),
    )
