"""Utilities for the Phase 0 TTRL benchmark workflow.

This module provides a custom benchmark path for ARC task-specific adaptation.
It intentionally reuses Stoix's FF-PPO learner for training while performing
support/query evaluation with an external mask-aware scorer against the true
hidden test outputs stored in JaxARC task buffers.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, ListConfig, OmegaConf
from stoa import get_final_step_metrics

from jaxarc.configs import JaxArcConfig
from jaxarc.envs.environment import Environment as JaxArcEnvironment
from jaxarc.registration import get_subset_task_ids
from jaxarc.registration import make as make_registered_env
from jaxarc.registration.subset_loader import load_all_subsets_for_dataset
from jaxarc.types import EPISODE_MODE_TEST, EPISODE_MODE_TRAIN, JaxArcTask, StepType
from jaxarc.utils.buffer import gather_task, stack_task_list
from jaxarc.utils.grid_utils import (
    compute_grid_similarity,
    get_actual_grid_shape_from_mask,
)
from jaxarc.wrappers import (
    AllowedOperationsWrapper,
    AnswerObservationWrapper,
    BboxActionWrapper,
    ContextualObservationWrapper,
    FlattenActionWrapper,
    InputGridObservationWrapper,
    PointActionWrapper,
    SelectionAwareFlattenWrapper,
)
from stoix.systems.ppo.anakin import ff_ppo
from stoix.systems.ppo.anakin import rec_ppo
from stoix.utils.checkpointing import Checkpointer
from stoix.utils import make_env as environments
from stoix.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from stoix.utils.running_statistics import normalize
from stoix.utils.total_timestep_checker import check_total_timesteps


AGGREGATE_FIELDNAMES = [
    "phase_name",
    "dataset",
    "subset",
    "method",
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

PAIR_FIELDNAMES = [
    "phase_name",
    "dataset",
    "subset",
    "method",
    "task_id",
    "split",
    "pair_index",
    "requested_budget_timesteps",
    "actual_budget_timesteps",
    "num_updates",
    "episode_return",
    "episode_length",
    "submitted",
    "truncated",
    "final_grid_match",
    "exact_success",
    "similarity",
    "matching_cells",
    "target_cells",
    "working_height",
    "working_width",
    "target_height",
    "target_width",
    "train_wall_clock_s",
    "seed",
    "evaluation_mode",
]


@dataclass(frozen=True)
class GridScore:
    """Mask-aware final-grid scoring result."""

    final_grid_match: bool
    exact_success: bool
    similarity: float
    matching_cells: int
    target_cells: int
    working_height: int
    working_width: int
    target_height: int
    target_width: int


@dataclass(frozen=True)
class RolloutMetrics:
    """Single evaluation rollout statistics."""

    episode_return: float
    episode_length: int
    submitted: bool
    truncated: bool
    final_working_grid: chex.Array
    final_working_mask: chex.Array


@dataclass(frozen=True)
class TrainingArtifacts:
    """Outputs retained after per-budget training."""

    actor_network: Any
    actor_params: FrozenDict[str, Any]
    running_statistics: Any | None
    requested_budget_timesteps: int
    actual_budget_timesteps: int
    num_updates: int
    train_wall_clock_s: float
    train_steps_per_second: float
    system_name: str
    recurrent_hidden_state_dim: int | None = None
    recurrent_cell_type: str | None = None


@dataclass(frozen=True)
class LearnerSetupArtifacts:
    """System-specific learner setup outputs."""

    learn_fn: Any
    actor_network: Any
    learner_state: Any
    recurrent_hidden_state_dim: int | None = None
    recurrent_cell_type: str | None = None


@dataclass(frozen=True)
class BenchmarkPaths:
    """Filesystem layout for a benchmark run."""

    output_root: Path
    run_dir: Path
    running_log: Path
    tracker_csv: Path
    final_summary_md: Path
    manifest_json: Path
    run_config_yaml: Path
    pair_results_jsonl: Path
    summary_csv: Path
    summary_json: Path


def _benchmark_method(config: DictConfig) -> str:
    return str(config.benchmark.get("method", "ff_ppo_adaptation"))


def _phase_name(config: DictConfig) -> str:
    return str(config.benchmark.get("phase_name", "TTRL"))


def _benchmark_dataset(config: DictConfig) -> str:
    return str(config.benchmark.get("dataset", "unknown"))


def _benchmark_subset(config: DictConfig) -> str:
    return str(config.benchmark.get("subset", "unknown"))


def _system_name(config: DictConfig) -> str:
    return str(config.system.system_name)


def _query_attempts(config: DictConfig) -> int:
    return max(1, int(config.benchmark.get("query_attempts", 1)))


def _evaluation_mode_label(config: DictConfig) -> str:
    if bool(config.arch.evaluation_greedy):
        return "greedy"
    query_attempts = _query_attempts(config)
    return "sample" if query_attempts <= 1 else f"sample@{query_attempts}"


def _to_python_scalar(value: Any) -> bool | int | float | str:
    array_value = jax.device_get(value)
    if isinstance(array_value, np.ndarray):
        if array_value.shape == ():
            return array_value.item()
        return str(array_value.tolist())
    if isinstance(array_value, np.generic):
        return array_value.item()
    return array_value


def _to_bool(value: Any) -> bool:
    return bool(_to_python_scalar(value))


def _to_int(value: Any) -> int:
    return int(_to_python_scalar(value))


def _to_float(value: Any) -> float:
    return float(_to_python_scalar(value))


def _tree_to_jsonable(tree: Any) -> Any:
    """Convert nested JAX/Numpy containers into JSON-serializable objects."""
    if isinstance(tree, (DictConfig, ListConfig)):
        return _tree_to_jsonable(OmegaConf.to_container(tree, resolve=True))
    if isinstance(tree, dict):
        return {str(key): _tree_to_jsonable(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return [_tree_to_jsonable(value) for value in tree]
    if isinstance(tree, Path):
        return str(tree)
    array_value = jax.device_get(tree)
    if isinstance(array_value, np.ndarray):
        if array_value.shape == ():
            return array_value.item()
        return array_value.tolist()
    if isinstance(array_value, np.generic):
        return array_value.item()
    return array_value


def _git_rev_parse(repo_path: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _sanitize_name(value: str) -> str:
    sanitized = []
    for char in value:
        sanitized.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(sanitized).strip("_") or "run"


def _append_markdown_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("# Running Log\n\n", encoding="utf-8")
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"- **{timestamp}** — {message}\n")


def _append_csv_row(csv_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _append_jsonl_row(jsonl_path: Path, row: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_tree_to_jsonable(row), sort_keys=True) + "\n")


def prepare_benchmark_paths(config: DictConfig, task_ids: list[str]) -> BenchmarkPaths:
    """Create the output directory layout for the current benchmark run."""
    output_root = Path(str(config.benchmark.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = _sanitize_name(str(config.benchmark.run_name))
    run_dir = output_root / "runs" / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    paths = BenchmarkPaths(
        output_root=output_root,
        run_dir=run_dir,
        running_log=output_root / "RUNNING_LOG.md",
        tracker_csv=output_root / "EXPERIMENT_TRACKER.csv",
        final_summary_md=output_root / "FINAL_SUMMARY.md",
        manifest_json=run_dir / "manifest.json",
        run_config_yaml=run_dir / "config.yaml",
        pair_results_jsonl=run_dir / "task_results.jsonl",
        summary_csv=run_dir / "summary.csv",
        summary_json=run_dir / "summary.json",
    )

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jaxarc_baselines_git_sha": _git_rev_parse(repo_root),
        "jaxarc_git_sha": _git_rev_parse(repo_root.parent / "JaxARC"),
        "phase_name": _phase_name(config),
        "dataset": _benchmark_dataset(config),
        "subset": _benchmark_subset(config),
        "method": _benchmark_method(config),
        "system_name": _system_name(config),
        "resolved_task_ids": task_ids,
        "device_count": len(jax.devices()),
        "devices": [str(device) for device in jax.devices()],
        "requested_budgets": list(config.benchmark.budgets),
        "evaluation_greedy": bool(config.arch.evaluation_greedy),
        "query_attempts": _query_attempts(config),
        "action_mode": str(config.env.action.mode),
        "observation_wrappers": _tree_to_jsonable(config.env.observation_wrappers),
        "checkpoint_load_model": bool(config.logger.checkpointing.load_model),
        "checkpoint_load_args": _tree_to_jsonable(
            config.logger.checkpointing.load_args
        ),
    }
    paths.manifest_json.write_text(
        json.dumps(_tree_to_jsonable(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    OmegaConf.save(config=config, f=str(paths.run_config_yaml))

    _append_markdown_log(
        paths.running_log,
        (
            f"Started {_phase_name(config)} benchmark run `{run_dir.name}` "
            f"using method `{_benchmark_method(config)}` on tasks {task_ids} "
            f"with budgets {list(config.benchmark.budgets)}."
        ),
    )
    return paths


def resolve_task_ids(config: DictConfig) -> list[str]:
    """Resolve the concrete task list for the benchmark run."""
    dataset = str(config.benchmark.dataset)
    load_all_subsets_for_dataset(dataset)

    explicit_task_ids = [str(task_id) for task_id in config.benchmark.task_ids]
    if explicit_task_ids:
        return explicit_task_ids

    jaxarc_config = JaxArcConfig.from_hydra(config)
    resolved = get_subset_task_ids(
        dataset,
        selector=str(config.benchmark.subset),
        config=jaxarc_config,
        auto_download=True,
    )
    limit = int(config.benchmark.num_tasks)
    if limit <= 0:
        return [str(task_id) for task_id in resolved]
    return [str(task_id) for task_id in resolved[:limit]]


def load_single_task(config: DictConfig, task_id: str) -> JaxArcTask:
    """Load a single task from the dataset registry."""
    dataset = str(config.benchmark.dataset)
    scenario_name = f"{dataset}-{task_id}"
    jaxarc_config = JaxArcConfig.from_hydra(config)
    _, params = make_registered_env(
        scenario_name,
        config=jaxarc_config,
        auto_download=True,
        episode_mode=EPISODE_MODE_TRAIN,
    )
    return gather_task(params.buffer, jnp.asarray(0, dtype=jnp.int32))


def make_single_pair_task(
    task: JaxArcTask,
    *,
    pair_index: int,
    episode_mode: int,
) -> JaxArcTask:
    """Create a single-pair view of a task for exact evaluation."""

    def zero_grid_like(source: chex.Array) -> chex.Array:
        return jnp.zeros_like(source)

    def zero_mask_like(source: chex.Array) -> chex.Array:
        return jnp.zeros_like(source, dtype=jnp.bool_)

    empty_train_grids = zero_grid_like(task.input_grids_examples)
    empty_train_masks = zero_mask_like(task.input_masks_examples)
    empty_train_outputs = zero_grid_like(task.output_grids_examples)
    empty_train_output_masks = zero_mask_like(task.output_masks_examples)
    empty_test_grids = zero_grid_like(task.test_input_grids)
    empty_test_masks = zero_mask_like(task.test_input_masks)
    empty_true_test_outputs = zero_grid_like(task.true_test_output_grids)
    empty_true_test_masks = zero_mask_like(task.true_test_output_masks)

    if episode_mode == EPISODE_MODE_TRAIN:
        input_grids_examples = empty_train_grids.at[0].set(
            task.input_grids_examples[pair_index]
        )
        input_masks_examples = empty_train_masks.at[0].set(
            task.input_masks_examples[pair_index]
        )
        output_grids_examples = empty_train_outputs.at[0].set(
            task.output_grids_examples[pair_index]
        )
        output_masks_examples = empty_train_output_masks.at[0].set(
            task.output_masks_examples[pair_index]
        )
        test_input_grids = empty_test_grids
        test_input_masks = empty_test_masks
        true_test_output_grids = empty_true_test_outputs
        true_test_output_masks = empty_true_test_masks
        num_train_pairs = 1
        num_test_pairs = 0
    else:
        input_grids_examples = empty_train_grids
        input_masks_examples = empty_train_masks
        output_grids_examples = empty_train_outputs
        output_masks_examples = empty_train_output_masks
        test_input_grids = empty_test_grids.at[0].set(task.test_input_grids[pair_index])
        test_input_masks = empty_test_masks.at[0].set(task.test_input_masks[pair_index])
        true_test_output_grids = empty_true_test_outputs.at[0].set(
            task.true_test_output_grids[pair_index]
        )
        true_test_output_masks = empty_true_test_masks.at[0].set(
            task.true_test_output_masks[pair_index]
        )
        num_train_pairs = 0
        num_test_pairs = 1

    return JaxArcTask(
        input_grids_examples=input_grids_examples,
        input_masks_examples=input_masks_examples,
        output_grids_examples=output_grids_examples,
        output_masks_examples=output_masks_examples,
        num_train_pairs=num_train_pairs,
        test_input_grids=test_input_grids,
        test_input_masks=test_input_masks,
        true_test_output_grids=true_test_output_grids,
        true_test_output_masks=true_test_output_masks,
        num_test_pairs=num_test_pairs,
        task_index=task.task_index,
    )


def _apply_stoix_compatible_wrappers(env: Any, config: DictConfig) -> Any:
    action_mode = str(config.env.action.mode)
    if action_mode == "point":
        env = PointActionWrapper(env)
    elif action_mode == "bbox":
        env = BboxActionWrapper(env)
    elif action_mode != "mask":
        raise ValueError(
            f"Unsupported action mode for benchmark evaluation: {action_mode}"
        )

    # Match make_jaxarc_env, so the policy picks from the same actions it trained on.
    if action_mode != "mask" and env.params.action.allowed_operations is not None:
        env = AllowedOperationsWrapper(env)

    if action_mode == "mask":
        env = FlattenActionWrapper(env)
    else:
        env = SelectionAwareFlattenWrapper(env)
    wrappers_cfg = config.env.get("observation_wrappers", {})
    if wrappers_cfg.get("answer_grid", False):
        env = AnswerObservationWrapper(env)
    if wrappers_cfg.get("input_grid", True):
        env = InputGridObservationWrapper(env)
    if wrappers_cfg.get("contextual", True):
        env = ContextualObservationWrapper(env)
    return env


def make_raw_eval_env(
    config: DictConfig, task: JaxArcTask, *, episode_mode: int
) -> Any:
    """Create a raw wrapped JaxARC env matching the training interfaces."""
    jaxarc_config = JaxArcConfig.from_hydra(config)
    env = JaxArcEnvironment(
        config=jaxarc_config,
        buffer=stack_task_list([task]),
        episode_mode=episode_mode,
    )
    return _apply_stoix_compatible_wrappers(env, config)


def _unwrap_base_env_state(state: Any) -> Any:
    current = state
    while hasattr(current, "base_env_state"):
        current = current.base_env_state
    return current


def score_final_grid(
    working_grid: chex.Array,
    working_mask: chex.Array,
    target_grid: chex.Array,
    target_mask: chex.Array,
    *,
    submitted: bool,
) -> GridScore:
    """Compute a mask-aware exact match and similarity score for a final grid."""
    working_height, working_width = get_actual_grid_shape_from_mask(working_mask)
    target_height, target_width = get_actual_grid_shape_from_mask(target_mask)

    rows = jnp.arange(target_grid.shape[0], dtype=jnp.int32)[:, None]
    cols = jnp.arange(target_grid.shape[1], dtype=jnp.int32)[None, :]
    target_canvas = (rows < target_height) & (cols < target_width)
    working_canvas = (rows < working_height) & (cols < working_width)

    mask_match = jnp.all(
        (working_mask & working_canvas) == (target_mask & target_canvas)
    )
    grid_match = jnp.all(
        jnp.where(target_mask, working_grid, 0)
        == jnp.where(target_mask, target_grid, 0)
    )
    shape_match = (working_height == target_height) & (working_width == target_width)
    final_grid_match = _to_bool(shape_match & mask_match & grid_match)

    matching_cells = _to_int(
        jnp.sum(
            (
                jnp.where(target_mask, working_grid, -1)
                == jnp.where(target_mask, target_grid, -2)
            )
            & target_mask
        )
    )
    similarity = _to_float(
        compute_grid_similarity(working_grid, working_mask, target_grid, target_mask)
    )
    return GridScore(
        final_grid_match=final_grid_match,
        exact_success=bool(final_grid_match and submitted),
        similarity=similarity,
        matching_cells=matching_cells,
        target_cells=_to_int(jnp.sum(target_mask)),
        working_height=_to_int(working_height),
        working_width=_to_int(working_width),
        target_height=_to_int(target_height),
        target_width=_to_int(target_width),
    )


def _select_action(
    actor_apply_fn: Any,
    actor_params: FrozenDict[str, Any],
    observation: Any,
    *,
    key: chex.PRNGKey,
    evaluation_greedy: bool,
    running_statistics: Any | None,
) -> chex.Array:
    def prepare_eval_observation(value: chex.Array) -> chex.Array:
        sample = value
        while sample.ndim > 4 and sample.shape[0] == 1:
            sample = sample[0]
        if sample.ndim == 3:
            sample = sample[jnp.newaxis, ...]
        elif sample.ndim > 4:
            flattened_leading = int(np.prod(sample.shape[:-3]))
            sample = sample.reshape((flattened_leading,) + sample.shape[-3:])
        if sample.ndim != 4:
            raise ValueError(
                f"Expected an image observation with rank 3 or 4, got shape {sample.shape}."
            )
        return sample[:1]

    if running_statistics is not None:
        observation = normalize(observation, running_statistics)
    batched_observation = jax.tree_util.tree_map(
        prepare_eval_observation,
        observation,
    )
    policy = actor_apply_fn(actor_params, batched_observation)
    action = policy.mode() if evaluation_greedy else policy.sample(seed=key)
    return jnp.asarray(action).squeeze(0)


def _prepare_recurrent_eval_observation(value: chex.Array) -> chex.Array:
    sample = value
    if sample.ndim == 3:
        sample = sample[jnp.newaxis, jnp.newaxis, ...]
    elif sample.ndim == 4:
        sample = sample[:1]
        sample = sample[jnp.newaxis, ...]
    elif sample.ndim == 5:
        sample = sample[:1, :1, ...]
    else:
        raise ValueError(
            f"Expected recurrent eval observation with rank 3-5, got shape {sample.shape}."
        )
    return sample


def _select_recurrent_action(
    actor_apply_fn: Any,
    actor_params: FrozenDict[str, Any],
    observation: Any,
    hidden_state: Any,
    *,
    key: chex.PRNGKey,
    evaluation_greedy: bool,
    reset_hidden_state: bool,
) -> tuple[Any, chex.Array]:
    batched_observation = jax.tree_util.tree_map(
        _prepare_recurrent_eval_observation,
        observation,
    )
    done = jnp.asarray([[reset_hidden_state]], dtype=jnp.bool_)
    next_hidden_state, policy = actor_apply_fn(
        actor_params,
        hidden_state,
        (batched_observation, done),
    )
    action = policy.mode() if evaluation_greedy else policy.sample(seed=key)
    return next_hidden_state, jnp.asarray(action).squeeze()


def rollout_policy_episode(
    env: Any,
    actor_apply_fn: Any,
    actor_params: FrozenDict[str, Any],
    *,
    key: chex.PRNGKey,
    evaluation_greedy: bool,
    running_statistics: Any | None,
    system_name: str = "ff_ppo",
    recurrent_hidden_state_dim: int | None = None,
    recurrent_cell_type: str | None = None,
) -> RolloutMetrics:
    """Run a single raw-environment rollout and return the final state."""
    env_state, timestep = env.reset(key)
    episode_return = 0.0
    episode_length = 0
    recurrent_hidden_state = None
    reset_hidden_state = True
    if system_name == "rec_ppo":
        if recurrent_hidden_state_dim is None or recurrent_cell_type is None:
            raise ValueError("Recurrent rollout requires hidden-state metadata.")
        recurrent_hidden_state = rec_ppo.ScannedRNN(
            hidden_state_dim=recurrent_hidden_state_dim,
            cell_type=recurrent_cell_type,
        ).initialize_carry(1)

    while not _to_bool(timestep.last()):
        key, policy_key = jax.random.split(key)
        if system_name == "rec_ppo":
            assert recurrent_hidden_state is not None
            recurrent_hidden_state, action = _select_recurrent_action(
                actor_apply_fn,
                actor_params,
                timestep.observation,
                recurrent_hidden_state,
                key=policy_key,
                evaluation_greedy=evaluation_greedy,
                reset_hidden_state=reset_hidden_state,
            )
        else:
            action = _select_action(
                actor_apply_fn,
                actor_params,
                timestep.observation,
                key=policy_key,
                evaluation_greedy=evaluation_greedy,
                running_statistics=running_statistics,
            )
        env_state, timestep = env.step(env_state, action)
        episode_return += _to_float(timestep.reward)
        episode_length += 1
        reset_hidden_state = False

    base_state = _unwrap_base_env_state(env_state)
    submitted = _to_bool(timestep.step_type == StepType.TERMINATED)
    truncated = _to_bool(timestep.step_type == StepType.TRUNCATED)
    return RolloutMetrics(
        episode_return=episode_return,
        episode_length=episode_length,
        submitted=submitted,
        truncated=truncated,
        final_working_grid=jnp.asarray(base_state.working_grid),
        final_working_mask=jnp.asarray(base_state.working_grid_mask),
    )


def _budget_config(
    base_config: DictConfig, task_id: str, budget_timesteps: int
) -> DictConfig:
    config = copy.deepcopy(base_config)
    dataset = str(config.benchmark.dataset)
    config.env.scenario.name = f"{dataset}-{task_id}"
    config.env.scenario.task_name = _sanitize_name(f"{dataset}_{task_id}")
    config.arch.total_timesteps = int(budget_timesteps)
    config.num_devices = len(jax.devices())
    return check_total_timesteps(config)


def _normalize_budget_list(budgets: list[int] | ListConfig) -> list[int]:
    return sorted({int(budget) for budget in budgets})


def _positive_budget_chunk(budgets: list[int]) -> int | None:
    positive_budgets = [budget for budget in budgets if budget > 0]
    if not positive_budgets:
        return None
    chunk = positive_budgets[0]
    for budget in positive_budgets[1:]:
        chunk = math.gcd(chunk, budget)
    return chunk if chunk > 0 else None


def _steps_per_rollout(config: DictConfig) -> int:
    return int(
        config.num_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )


def _prepare_training_session(
    base_config: DictConfig,
    *,
    task_id: str,
    budget_timesteps: int,
) -> tuple[DictConfig, LearnerSetupArtifacts]:
    config = _budget_config(base_config, task_id, budget_timesteps)
    env, _ = environments.make(config=config)
    seed = int(config.arch.seed)
    key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed), num=3)
    setup = _benchmark_learner_setup(env, (key, actor_key, critic_key), config)
    return config, setup


def _extract_training_artifacts(
    *,
    actor_network: Any,
    learner_state: Any,
    config: DictConfig,
    requested_budget_timesteps: int,
    actual_budget_timesteps: int,
    num_updates: int,
    train_wall_clock_s: float,
    train_steps_per_second: float,
    recurrent_hidden_state_dim: int | None = None,
    recurrent_cell_type: str | None = None,
) -> TrainingArtifacts:
    running_statistics = getattr(learner_state, "running_statistics", None)
    if running_statistics is not None:
        running_statistics = unreplicate_n_dims(running_statistics)

    return TrainingArtifacts(
        actor_network=actor_network,
        actor_params=unreplicate_n_dims(learner_state.params.actor_params),
        running_statistics=running_statistics,
        requested_budget_timesteps=int(requested_budget_timesteps),
        actual_budget_timesteps=int(actual_budget_timesteps),
        num_updates=int(num_updates),
        train_wall_clock_s=float(train_wall_clock_s),
        train_steps_per_second=float(train_steps_per_second),
        system_name=_system_name(config),
        recurrent_hidden_state_dim=recurrent_hidden_state_dim,
        recurrent_cell_type=recurrent_cell_type,
    )


def _maybe_restore_checkpoint_params(
    *,
    params: Any,
    config: DictConfig,
    hstates: Any | None = None,
) -> tuple[Any, Any | None]:
    if not bool(config.logger.checkpointing.load_model):
        return params, hstates

    loaded_checkpoint = Checkpointer(
        model_name=_system_name(config),
        **config.logger.checkpointing.load_args,
    )
    if hstates is None:
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        return restored_params, None

    restored_params, restored_hstates = loaded_checkpoint.restore_params(
        input_params=params,
        restore_hstates=True,
    )
    return (
        restored_params,
        restored_hstates if restored_hstates is not None else hstates,
    )


def initialize_policy_for_budget(
    base_config: DictConfig,
    *,
    task_id: str,
    budget_timesteps: int,
) -> TrainingArtifacts:
    """Initialize a policy without taking any PPO learner updates."""
    config, setup = _prepare_training_session(
        base_config,
        task_id=task_id,
        budget_timesteps=budget_timesteps,
    )
    return _extract_training_artifacts(
        actor_network=setup.actor_network,
        learner_state=setup.learner_state,
        config=config,
        requested_budget_timesteps=budget_timesteps,
        actual_budget_timesteps=0,
        num_updates=0,
        train_wall_clock_s=0.0,
        train_steps_per_second=0.0,
        recurrent_hidden_state_dim=setup.recurrent_hidden_state_dim,
        recurrent_cell_type=setup.recurrent_cell_type,
    )


def _benchmark_feedforward_learner_setup(
    env: Any,
    keys: chex.Array,
    config: DictConfig,
) -> LearnerSetupArtifacts:
    """Variant of Stoix learner setup using a real reset observation for init.

    The wrapped JaxARC observation space can report a different synthetic shape
    than the actual runtime observation tensor. Initializing from a real reset
    observation keeps Flax convolution kernels consistent with the training path.
    """
    n_devices = len(jax.devices())
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    key, actor_net_key, critic_net_key = keys

    actor_torso = ff_ppo.hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = ff_ppo.hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=num_actions,
    )
    critic_torso = ff_ppo.hydra.utils.instantiate(
        config.network.critic_network.pre_torso
    )
    critic_head = ff_ppo.hydra.utils.instantiate(
        config.network.critic_network.critic_head
    )

    actor_network = ff_ppo.Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = ff_ppo.Critic(torso=critic_torso, critic_head=critic_head)

    actor_lr = ff_ppo.make_learning_rate(
        config.system.actor_lr,
        config,
        config.system.epochs,
        config.system.num_minibatches,
    )
    critic_lr = ff_ppo.make_learning_rate(
        config.system.critic_lr,
        config,
        config.system.epochs,
        config.system.num_minibatches,
    )

    actor_optim = ff_ppo.optax.chain(
        ff_ppo.optax.clip_by_global_norm(config.system.max_grad_norm),
        ff_ppo.optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = ff_ppo.optax.chain(
        ff_ppo.optax.clip_by_global_norm(config.system.max_grad_norm),
        ff_ppo.optax.adam(critic_lr, eps=1e-5),
    )

    learn = ff_ppo.get_learner_fn(
        env,
        (actor_network.apply, critic_network.apply),
        (actor_optim.update, critic_optim.update),
        config,
    )
    learn = jax.pmap(learn, axis_name="device")

    key, *env_keys = jax.random.split(
        key,
        n_devices * config.arch.update_batch_size * config.arch.num_envs + 1,
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    def prepare_init_observation(value: chex.Array) -> chex.Array:
        sample = value
        while sample.ndim > 4 and sample.shape[0] == 1:
            sample = sample[0]
        if sample.ndim > 4:
            flattened_leading = int(np.prod(sample.shape[:-3]))
            sample = sample.reshape((flattened_leading,) + sample.shape[-3:])
        if sample.ndim != 4:
            raise ValueError(
                f"Expected a batched image observation with rank 4, got shape {sample.shape}."
            )
        return sample[:1]

    sample_observation = jax.tree_util.tree_map(
        prepare_init_observation,
        timesteps.observation,
    )

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        actor_params = actor_network.init(actor_net_key, sample_observation)
    actor_opt_state = actor_optim.init(actor_params)
    with jax.default_device(cpu_device):
        critic_params = critic_network.init(critic_net_key, sample_observation)
    critic_opt_state = critic_optim.init(critic_params)
    params = ff_ppo.ActorCriticParams(actor_params, critic_params)
    params, _ = _maybe_restore_checkpoint_params(params=params, config=config)

    reshape_states = lambda value: value.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs)
        + value.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda value: value.reshape(
        (n_devices, config.arch.update_batch_size) + value.shape[1:]
    )
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = ff_ppo.ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states)
    broadcast = lambda value: jnp.broadcast_to(
        value,
        (config.arch.update_batch_size,) + value.shape,
    )
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = ff_ppo.flax.jax_utils.replicate(
        replicate_learner,
        devices=jax.devices(),
    )
    params, opt_states = replicate_learner
    learner_state = ff_ppo.OnPolicyLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
    )

    if config.system.normalize_observations:
        dummy_obs = jax.tree.map(lambda value: value.squeeze(0), sample_observation)
        warmup_observations = ff_ppo._collect_obs_norm_rollouts(env, key, config)
        running_statistics = ff_ppo.initialize_statistics_from_data(
            dummy_obs,
            warmup_observations,
        )
        running_statistics = jax.tree.map(broadcast, running_statistics)
        running_statistics = ff_ppo.flax.jax_utils.replicate(
            running_statistics,
            devices=jax.devices(),
        )
        learner_state = ff_ppo.create_with_running_statistics(
            state=learner_state,
            running_statistics=running_statistics,
        )

    return LearnerSetupArtifacts(
        learn_fn=learn,
        actor_network=actor_network,
        learner_state=learner_state,
    )


def _prepare_recurrent_init_observation(value: chex.Array, num_envs: int) -> chex.Array:
    sample = value
    if sample.ndim == 3:
        sample = jnp.repeat(sample[jnp.newaxis, jnp.newaxis, ...], num_envs, axis=1)
    elif sample.ndim == 4:
        sample = sample[:num_envs]
        sample = sample[jnp.newaxis, ...]
    elif sample.ndim == 5:
        sample = sample[:1, :num_envs, ...]
    else:
        raise ValueError(
            f"Expected a recurrent image observation with rank 3-5, got shape {sample.shape}."
        )
    return sample


def _benchmark_recurrent_learner_setup(
    env: Any,
    keys: chex.Array,
    config: DictConfig,
) -> LearnerSetupArtifacts:
    """Variant of Stoix recurrent learner setup using a real reset observation for init."""
    n_devices = len(jax.devices())
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    key, actor_net_key, critic_net_key = keys

    actor_pre_torso = rec_ppo.hydra.utils.instantiate(
        config.network.actor_network.pre_torso
    )
    actor_post_torso = rec_ppo.hydra.utils.instantiate(
        config.network.actor_network.post_torso
    )
    actor_action_head = rec_ppo.hydra.utils.instantiate(
        config.network.actor_network.action_head,
        action_dim=num_actions,
    )
    critic_pre_torso = rec_ppo.hydra.utils.instantiate(
        config.network.critic_network.pre_torso
    )
    critic_post_torso = rec_ppo.hydra.utils.instantiate(
        config.network.critic_network.post_torso
    )
    critic_head = rec_ppo.hydra.utils.instantiate(
        config.network.critic_network.critic_head
    )

    actor_hidden_state_dim = int(
        config.network.actor_network.rnn_layer.hidden_state_dim
    )
    actor_cell_type = str(config.network.actor_network.rnn_layer.cell_type)
    critic_hidden_state_dim = int(
        config.network.critic_network.rnn_layer.hidden_state_dim
    )
    critic_cell_type = str(config.network.critic_network.rnn_layer.cell_type)

    actor_network = rec_ppo.RecurrentActor(
        pre_torso=actor_pre_torso,
        hidden_state_dim=actor_hidden_state_dim,
        cell_type=actor_cell_type,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
    )
    critic_network = rec_ppo.RecurrentCritic(
        pre_torso=critic_pre_torso,
        hidden_state_dim=critic_hidden_state_dim,
        cell_type=critic_cell_type,
        post_torso=critic_post_torso,
        critic_head=critic_head,
    )
    actor_rnn = rec_ppo.ScannedRNN(
        hidden_state_dim=actor_hidden_state_dim,
        cell_type=actor_cell_type,
    )
    critic_rnn = rec_ppo.ScannedRNN(
        hidden_state_dim=critic_hidden_state_dim,
        cell_type=critic_cell_type,
    )

    actor_lr = rec_ppo.make_learning_rate(
        config.system.actor_lr,
        config,
        config.system.epochs,
        config.system.num_minibatches,
    )
    critic_lr = rec_ppo.make_learning_rate(
        config.system.critic_lr,
        config,
        config.system.epochs,
        config.system.num_minibatches,
    )

    actor_optim = rec_ppo.optax.chain(
        rec_ppo.optax.clip_by_global_norm(config.system.max_grad_norm),
        rec_ppo.optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = rec_ppo.optax.chain(
        rec_ppo.optax.clip_by_global_norm(config.system.max_grad_norm),
        rec_ppo.optax.adam(critic_lr, eps=1e-5),
    )

    learn = rec_ppo.get_learner_fn(
        env,
        (actor_network.apply, critic_network.apply),
        (actor_optim.update, critic_optim.update),
        config,
    )
    learn = jax.pmap(learn, axis_name="device")

    key, *env_keys = jax.random.split(
        key,
        n_devices * config.arch.update_batch_size * config.arch.num_envs + 1,
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))

    sample_observation = jax.tree_util.tree_map(
        lambda value: _prepare_recurrent_init_observation(value, config.arch.num_envs),
        timesteps.observation,
    )
    init_done = jnp.zeros((1, config.arch.num_envs), dtype=bool)
    init_x = (sample_observation, init_done)

    init_policy_hstate = actor_rnn.initialize_carry(config.arch.num_envs)
    init_critic_hstate = critic_rnn.initialize_carry(config.arch.num_envs)
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_x)
    actor_opt_state = actor_optim.init(actor_params)
    with jax.default_device(cpu_device):
        critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    params = rec_ppo.ActorCriticParams(actor_params, critic_params)
    hstates = rec_ppo.ActorCriticHiddenStates(init_policy_hstate, init_critic_hstate)
    params, hstates = _maybe_restore_checkpoint_params(
        params=params,
        config=config,
        hstates=hstates,
    )

    reshape_states = lambda value: value.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs)
        + value.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    dones = jnp.zeros((config.arch.num_envs,), dtype=bool)
    truncated = jnp.zeros((config.arch.num_envs,), dtype=bool)
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda value: value.reshape(
        (n_devices, config.arch.update_batch_size) + value.shape[1:]
    )
    step_keys = reshape_keys(jnp.stack(step_keys))
    opt_states = rec_ppo.ActorCriticOptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, dones, truncated)
    broadcast = lambda value: jnp.broadcast_to(
        value,
        (config.arch.update_batch_size,) + value.shape,
    )
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)
    replicate_learner = rec_ppo.flax.jax_utils.replicate(
        replicate_learner,
        devices=jax.devices(),
    )
    params, opt_states, hstates, dones, truncated = replicate_learner
    learner_state = rec_ppo.RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        done=dones,
        truncated=truncated,
        hstates=hstates,
    )
    return LearnerSetupArtifacts(
        learn_fn=learn,
        actor_network=actor_network,
        learner_state=learner_state,
        recurrent_hidden_state_dim=actor_hidden_state_dim,
        recurrent_cell_type=actor_cell_type,
    )


def _benchmark_learner_setup(
    env: Any,
    keys: chex.Array,
    config: DictConfig,
) -> LearnerSetupArtifacts:
    if _system_name(config) == "ff_ppo":
        return _benchmark_feedforward_learner_setup(env, keys, config)
    if _system_name(config) == "rec_ppo":
        if config.system.recurrent_chunk_size is None:
            config.system.recurrent_chunk_size = config.system.rollout_length
        return _benchmark_recurrent_learner_setup(env, keys, config)
    raise ValueError(f"Unsupported TTRL benchmark system: {_system_name(config)}")


def train_for_budget(
    base_config: DictConfig,
    *,
    task_id: str,
    budget_timesteps: int,
) -> TrainingArtifacts:
    """Run FF-PPO adaptation for a single task and timestep budget."""
    if int(budget_timesteps) <= 0:
        return initialize_policy_for_budget(
            base_config,
            task_id=task_id,
            budget_timesteps=budget_timesteps,
        )

    config = _budget_config(base_config, task_id, budget_timesteps)
    if int(config.arch.num_updates) < 1:
        raise ValueError(
            "Configured benchmark budget produced zero PPO updates. "
            "Reduce total_num_envs/rollout_length or increase the budget."
        )

    _, setup = _prepare_training_session(
        base_config,
        task_id=task_id,
        budget_timesteps=budget_timesteps,
    )
    learner_state = setup.learner_state

    steps_per_rollout = _steps_per_rollout(config)

    total_train_time = 0.0
    for _eval_step in range(int(config.arch.num_evaluation)):
        start_time = time.perf_counter()
        learner_output = setup.learn_fn(learner_state)
        jax.block_until_ready(learner_output)
        total_train_time += time.perf_counter() - start_time

        # This call validates the trajectory info structure and mirrors Stoix's
        # final-episode aggregation, even though we do not rely on it for scoring.
        get_final_step_metrics(learner_output.episode_metrics)
        learner_state = learner_output.learner_state

    actual_budget_timesteps = int(steps_per_rollout * config.arch.num_evaluation)
    train_steps_per_second = (
        float(actual_budget_timesteps / total_train_time)
        if total_train_time > 0
        else 0.0
    )
    return _extract_training_artifacts(
        actor_network=setup.actor_network,
        learner_state=learner_state,
        config=config,
        requested_budget_timesteps=budget_timesteps,
        actual_budget_timesteps=actual_budget_timesteps,
        num_updates=int(config.arch.num_updates),
        train_wall_clock_s=total_train_time,
        train_steps_per_second=train_steps_per_second,
        recurrent_hidden_state_dim=setup.recurrent_hidden_state_dim,
        recurrent_cell_type=setup.recurrent_cell_type,
    )


def iter_training_for_budgets(
    base_config: DictConfig,
    *,
    task_id: str,
    budgets: list[int] | ListConfig,
) -> list[tuple[int, TrainingArtifacts]]:
    """Return ordered training artifacts for a task across requested budgets.

    When possible, this trains cumulatively in fixed budget chunks so that
    checkpoints for larger budgets reuse the same learner state rather than
    recompiling and restarting from scratch for every requested budget.
    """

    normalized_budgets = _normalize_budget_list(budgets)
    positive_budgets = [budget for budget in normalized_budgets if budget > 0]
    reuse_enabled = bool(
        base_config.benchmark.get("reuse_training_across_budgets", True)
    )

    if len(positive_budgets) <= 1 or not reuse_enabled:
        return [
            (
                budget,
                train_for_budget(base_config, task_id=task_id, budget_timesteps=budget),
            )
            for budget in normalized_budgets
        ]

    chunk_budget = _positive_budget_chunk(positive_budgets)
    if chunk_budget is None:
        return [
            (
                budget,
                train_for_budget(base_config, task_id=task_id, budget_timesteps=budget),
            )
            for budget in normalized_budgets
        ]

    chunk_config, setup = _prepare_training_session(
        base_config,
        task_id=task_id,
        budget_timesteps=chunk_budget,
    )
    chunk_num_updates = int(chunk_config.arch.num_updates)
    if chunk_num_updates < 1:
        return [
            (
                budget,
                train_for_budget(base_config, task_id=task_id, budget_timesteps=budget),
            )
            for budget in normalized_budgets
        ]

    learner_state = setup.learner_state
    results: list[tuple[int, TrainingArtifacts]] = []
    if 0 in normalized_budgets:
        results.append(
            (
                0,
                _extract_training_artifacts(
                    actor_network=setup.actor_network,
                    learner_state=learner_state,
                    config=chunk_config,
                    requested_budget_timesteps=0,
                    actual_budget_timesteps=0,
                    num_updates=0,
                    train_wall_clock_s=0.0,
                    train_steps_per_second=0.0,
                    recurrent_hidden_state_dim=setup.recurrent_hidden_state_dim,
                    recurrent_cell_type=setup.recurrent_cell_type,
                ),
            )
        )

    chunk_actual_timesteps = _steps_per_rollout(chunk_config)
    cumulative_train_time = 0.0
    cumulative_actual_timesteps = 0
    cumulative_num_updates = 0
    target_budgets = set(positive_budgets)
    max_budget = positive_budgets[-1]
    num_chunks = max_budget // chunk_budget

    for chunk_index in range(1, num_chunks + 1):
        start_time = time.perf_counter()
        learner_output = setup.learn_fn(learner_state)
        jax.block_until_ready(learner_output)
        elapsed = time.perf_counter() - start_time
        cumulative_train_time += elapsed
        cumulative_actual_timesteps += chunk_actual_timesteps
        cumulative_num_updates += chunk_num_updates

        get_final_step_metrics(learner_output.episode_metrics)
        learner_state = learner_output.learner_state

        requested_budget_timesteps = int(chunk_index * chunk_budget)
        if requested_budget_timesteps not in target_budgets:
            continue

        train_steps_per_second = (
            float(cumulative_actual_timesteps / cumulative_train_time)
            if cumulative_train_time > 0
            else 0.0
        )
        results.append(
            (
                requested_budget_timesteps,
                _extract_training_artifacts(
                    actor_network=setup.actor_network,
                    learner_state=learner_state,
                    config=chunk_config,
                    requested_budget_timesteps=requested_budget_timesteps,
                    actual_budget_timesteps=cumulative_actual_timesteps,
                    num_updates=cumulative_num_updates,
                    train_wall_clock_s=cumulative_train_time,
                    train_steps_per_second=train_steps_per_second,
                    recurrent_hidden_state_dim=setup.recurrent_hidden_state_dim,
                    recurrent_cell_type=setup.recurrent_cell_type,
                ),
            )
        )

    return results


def _pair_result_row(
    *,
    phase_name: str,
    dataset: str,
    subset: str,
    method: str,
    task_id: str,
    split: str,
    pair_index: int,
    rollout: RolloutMetrics,
    score: GridScore,
    training: TrainingArtifacts,
    seed: int,
    evaluation_mode: str,
) -> dict[str, Any]:
    row = {
        "phase_name": phase_name,
        "dataset": dataset,
        "subset": subset,
        "method": method,
        "task_id": task_id,
        "split": split,
        "pair_index": pair_index,
        "requested_budget_timesteps": training.requested_budget_timesteps,
        "actual_budget_timesteps": training.actual_budget_timesteps,
        "num_updates": training.num_updates,
        "episode_return": rollout.episode_return,
        "episode_length": rollout.episode_length,
        "submitted": rollout.submitted,
        "truncated": rollout.truncated,
        "final_grid_match": score.final_grid_match,
        "exact_success": score.exact_success,
        "similarity": score.similarity,
        "matching_cells": score.matching_cells,
        "target_cells": score.target_cells,
        "working_height": score.working_height,
        "working_width": score.working_width,
        "target_height": score.target_height,
        "target_width": score.target_width,
        "train_wall_clock_s": training.train_wall_clock_s,
        "seed": seed,
        "evaluation_mode": evaluation_mode,
    }
    return row


def _query_row_priority(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["exact_success"]),
        float(row["final_grid_match"]),
        float(row["similarity"]),
        float(row["submitted"]),
        float(row["episode_return"]),
        -float(row["episode_length"]),
    )


def evaluate_task_pairs(
    config: DictConfig,
    *,
    task_id: str,
    task: JaxArcTask,
    split: str,
    training: TrainingArtifacts,
) -> list[dict[str, Any]]:
    """Evaluate all support or query pairs for a trained task policy."""
    if split == "support":
        episode_mode = EPISODE_MODE_TRAIN
        pair_count = _to_int(task.num_train_pairs)
    elif split == "query":
        episode_mode = EPISODE_MODE_TEST
        pair_count = _to_int(task.num_test_pairs)
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'support' or 'query'.")

    seed = int(config.arch.seed)
    evaluation_mode = _evaluation_mode_label(config)
    pair_rows: list[dict[str, Any]] = []
    phase_name = _phase_name(config)
    dataset = _benchmark_dataset(config)
    subset = _benchmark_subset(config)
    method = _benchmark_method(config)
    query_attempts = _query_attempts(config) if split == "query" else 1
    for pair_index in range(pair_count):
        single_pair_task = make_single_pair_task(
            task,
            pair_index=pair_index,
            episode_mode=episode_mode,
        )
        env = make_raw_eval_env(config, single_pair_task, episode_mode=episode_mode)
        if split == "support":
            target_grid = single_pair_task.output_grids_examples[0]
            target_mask = single_pair_task.output_masks_examples[0]
        else:
            target_grid = single_pair_task.true_test_output_grids[0]
            target_mask = single_pair_task.true_test_output_masks[0]

        best_row: dict[str, Any] | None = None
        for attempt_index in range(query_attempts):
            rollout = rollout_policy_episode(
                env,
                training.actor_network.apply,
                training.actor_params,
                key=jax.random.PRNGKey(
                    seed
                    + pair_index
                    + (1000 if split == "query" else 0)
                    + (10_000 * attempt_index)
                ),
                evaluation_greedy=bool(config.arch.evaluation_greedy),
                running_statistics=training.running_statistics,
                system_name=training.system_name,
                recurrent_hidden_state_dim=training.recurrent_hidden_state_dim,
                recurrent_cell_type=training.recurrent_cell_type,
            )
            score = score_final_grid(
                rollout.final_working_grid,
                rollout.final_working_mask,
                target_grid,
                target_mask,
                submitted=rollout.submitted,
            )
            row = _pair_result_row(
                phase_name=phase_name,
                dataset=dataset,
                subset=subset,
                method=method,
                task_id=task_id,
                split=split,
                pair_index=pair_index,
                rollout=rollout,
                score=score,
                training=training,
                seed=seed,
                evaluation_mode=evaluation_mode,
            )
            if best_row is None or _query_row_priority(row) > _query_row_priority(
                best_row
            ):
                best_row = row

        assert best_row is not None
        pair_rows.append(best_row)
    return pair_rows


def _mean_from_rows(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    values = [float(row[key]) for row in rows]
    return float(sum(values) / len(values))


def aggregate_task_results(
    *,
    phase_name: str,
    dataset: str,
    subset: str,
    method: str,
    task_id: str,
    training: TrainingArtifacts,
    support_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    seed: int,
    evaluation_mode: str,
) -> dict[str, Any]:
    """Aggregate pair-level support/query metrics for a single task+budget run."""
    return {
        "phase_name": phase_name,
        "dataset": dataset,
        "subset": subset,
        "method": method,
        "task_id": task_id,
        "requested_budget_timesteps": training.requested_budget_timesteps,
        "actual_budget_timesteps": training.actual_budget_timesteps,
        "num_updates": training.num_updates,
        "train_wall_clock_s": training.train_wall_clock_s,
        "train_steps_per_second": training.train_steps_per_second,
        "support_pair_count": len(support_rows),
        "query_pair_count": len(query_rows),
        "support_exact_success_rate": _mean_from_rows(support_rows, "exact_success"),
        "support_final_grid_match_rate": _mean_from_rows(
            support_rows, "final_grid_match"
        ),
        "support_similarity_mean": _mean_from_rows(support_rows, "similarity"),
        "query_exact_success_rate": _mean_from_rows(query_rows, "exact_success"),
        "query_final_grid_match_rate": _mean_from_rows(query_rows, "final_grid_match"),
        "query_similarity_mean": _mean_from_rows(query_rows, "similarity"),
        "submitted_support_rate": _mean_from_rows(support_rows, "submitted"),
        "submitted_query_rate": _mean_from_rows(query_rows, "submitted"),
        "seed": seed,
        "evaluation_mode": evaluation_mode,
    }


def _load_tracker_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_global_summary(
    paths: BenchmarkPaths,
    *,
    config: DictConfig,
    task_ids: list[str],
) -> None:
    global_rows = _load_tracker_rows(paths.tracker_csv)
    subset_keys = sorted(
        {
            (
                row.get("phase_name", _phase_name(config)),
                row.get("dataset", _benchmark_dataset(config)),
                row.get("subset", _benchmark_subset(config)),
            )
            for row in global_rows
        }
    )

    lines = [
        f"# {_phase_name(config)} Final Summary",
        "",
        f"- Latest run directory: `{paths.run_dir}`",
        f"- Tasks: {', '.join(task_ids)}",
        f"- Evaluation mode: {_evaluation_mode_label(config)}",
        f"- Query attempts: {_query_attempts(config)}",
        f"- Requested budgets: {', '.join(str(budget) for budget in config.benchmark.budgets)}",
        "",
    ]

    for phase_name, dataset, subset in subset_keys:
        subset_rows = [
            row
            for row in global_rows
            if row.get("phase_name", _phase_name(config)) == phase_name
            and row.get("dataset", _benchmark_dataset(config)) == dataset
            and row.get("subset", _benchmark_subset(config)) == subset
        ]
        methods = sorted({row["method"] for row in subset_rows})
        budgets = sorted(
            {int(row["requested_budget_timesteps"]) for row in subset_rows}
        )
        lines.extend(
            [
                f"## {phase_name} — {dataset}/{subset}",
                "",
                f"- Methods: {', '.join(methods)}",
                "",
            ]
        )
        for method in methods:
            method_rows = [row for row in subset_rows if row["method"] == method]
            lines.extend(
                [
                    f"### {method}",
                    "",
                    "| budget | tasks | support exact | support grid match | support similarity | query exact | query grid match | query similarity | mean train wall-clock (s) |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for budget in budgets:
                rows_for_budget = [
                    row
                    for row in method_rows
                    if int(row["requested_budget_timesteps"]) == budget
                ]
                if not rows_for_budget:
                    continue
                item = {
                    "requested_budget_timesteps": budget,
                    "task_count": len(rows_for_budget),
                    "support_exact_success_rate": _mean_from_rows(
                        rows_for_budget, "support_exact_success_rate"
                    ),
                    "support_final_grid_match_rate": _mean_from_rows(
                        rows_for_budget, "support_final_grid_match_rate"
                    ),
                    "support_similarity_mean": _mean_from_rows(
                        rows_for_budget, "support_similarity_mean"
                    ),
                    "query_exact_success_rate": _mean_from_rows(
                        rows_for_budget, "query_exact_success_rate"
                    ),
                    "query_final_grid_match_rate": _mean_from_rows(
                        rows_for_budget, "query_final_grid_match_rate"
                    ),
                    "query_similarity_mean": _mean_from_rows(
                        rows_for_budget, "query_similarity_mean"
                    ),
                    "train_wall_clock_s_mean": _mean_from_rows(
                        rows_for_budget, "train_wall_clock_s"
                    ),
                }
                lines.append(
                    "| {requested_budget_timesteps} | {task_count} | "
                    "{support_exact_success_rate:.3f} | {support_final_grid_match_rate:.3f} | "
                    "{support_similarity_mean:.3f} | {query_exact_success_rate:.3f} | "
                    "{query_final_grid_match_rate:.3f} | {query_similarity_mean:.3f} | "
                    "{train_wall_clock_s_mean:.2f} |".format(**item)
                )
            lines.append("")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- `query_exact` counts only submitted episodes whose final grids exactly match the hidden target.",
            "- `query_grid_match` credits exact final grids even if the agent never submitted.",
            "- Similarity is recomputed externally against `true_test_output_*`; it does not rely on JaxARC test-mode masked targets.",
        ]
    )
    paths.final_summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_global_plots(paths: BenchmarkPaths) -> None:
    tracker_rows = _load_tracker_rows(paths.tracker_csv)
    if not tracker_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    series_keys = sorted(
        {
            (
                row.get("dataset", "unknown"),
                row.get("subset", "unknown"),
                row["method"],
            )
            for row in tracker_rows
        }
    )
    plot_specs = [
        (
            "query_exact_success_rate",
            "Query exact success",
            "query_success_vs_compute.png",
        ),
        (
            "support_similarity_mean",
            "Support similarity",
            "support_similarity_vs_compute.png",
        ),
        ("train_wall_clock_s", "Train wall-clock (s)", "wallclock_vs_compute.png"),
    ]

    for metric_key, ylabel, filename in plot_specs:
        plt.figure(figsize=(7, 4.5))
        for dataset, subset, method in series_keys:
            method_rows = [
                row
                for row in tracker_rows
                if row.get("dataset", "unknown") == dataset
                and row.get("subset", "unknown") == subset
                and row["method"] == method
            ]
            budgets = sorted(
                {int(row["requested_budget_timesteps"]) for row in method_rows}
            )
            x_values: list[float] = []
            y_values: list[float] = []
            for budget in budgets:
                rows_for_budget = [
                    row
                    for row in method_rows
                    if int(row["requested_budget_timesteps"]) == budget
                ]
                x_values.append(float(budget))
                y_values.append(_mean_from_rows(rows_for_budget, metric_key))
            if x_values:
                plt.plot(
                    x_values,
                    y_values,
                    marker="o",
                    label=f"{dataset}/{subset}:{method}",
                )

        plt.xscale("symlog", linthresh=1.0)
        plt.xlabel("Adaptation budget (timesteps)")
        plt.ylabel(ylabel)
        plt.title(ylabel)
        plt.grid(True, alpha=0.3)
        if series_keys:
            plt.legend()
        plt.tight_layout()
        plt.savefig(paths.output_root / filename, dpi=200)
        plt.close()


def write_summary(
    paths: BenchmarkPaths,
    *,
    config: DictConfig,
    aggregate_rows: list[dict[str, Any]],
    task_ids: list[str],
) -> None:
    """Write per-run summaries and refresh global summary/plot artifacts."""
    paths.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with paths.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDNAMES)
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(
                {field: row.get(field, "") for field in AGGREGATE_FIELDNAMES}
            )

    budgets = sorted({int(row["requested_budget_timesteps"]) for row in aggregate_rows})
    by_budget: list[dict[str, Any]] = []
    for budget in budgets:
        rows_for_budget = [
            row
            for row in aggregate_rows
            if int(row["requested_budget_timesteps"]) == budget
        ]
        by_budget.append(
            {
                "requested_budget_timesteps": budget,
                "task_count": len(rows_for_budget),
                "support_exact_success_rate": _mean_from_rows(
                    rows_for_budget, "support_exact_success_rate"
                ),
                "support_final_grid_match_rate": _mean_from_rows(
                    rows_for_budget, "support_final_grid_match_rate"
                ),
                "support_similarity_mean": _mean_from_rows(
                    rows_for_budget, "support_similarity_mean"
                ),
                "query_exact_success_rate": _mean_from_rows(
                    rows_for_budget, "query_exact_success_rate"
                ),
                "query_final_grid_match_rate": _mean_from_rows(
                    rows_for_budget, "query_final_grid_match_rate"
                ),
                "query_similarity_mean": _mean_from_rows(
                    rows_for_budget, "query_similarity_mean"
                ),
                "train_wall_clock_s_mean": _mean_from_rows(
                    rows_for_budget, "train_wall_clock_s"
                ),
            }
        )

    summary_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase_name": _phase_name(config),
        "dataset": _benchmark_dataset(config),
        "subset": _benchmark_subset(config),
        "method": _benchmark_method(config),
        "system_name": _system_name(config),
        "task_ids": task_ids,
        "budgets": list(config.benchmark.budgets),
        "aggregate_rows": aggregate_rows,
        "by_budget": by_budget,
    }
    paths.summary_json.write_text(
        json.dumps(_tree_to_jsonable(summary_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        f"# {_phase_name(config)} Run Summary",
        "",
        f"- Run directory: `{paths.run_dir}`",
        f"- Method: `{_benchmark_method(config)}`",
        f"- System: `{_system_name(config)}`",
        f"- Dataset/subset: `{_benchmark_dataset(config)}` / `{_benchmark_subset(config)}`",
        f"- Tasks: {', '.join(task_ids)}",
        f"- Evaluation mode: {_evaluation_mode_label(config)}",
        f"- Query attempts: {_query_attempts(config)}",
        f"- Requested budgets: {', '.join(str(budget) for budget in config.benchmark.budgets)}",
        "",
        "## Budget summary",
        "",
        "| budget | tasks | support exact | support grid match | support similarity | query exact | query grid match | query similarity | mean train wall-clock (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in by_budget:
        lines.append(
            "| {requested_budget_timesteps} | {task_count} | "
            "{support_exact_success_rate:.3f} | {support_final_grid_match_rate:.3f} | "
            "{support_similarity_mean:.3f} | {query_exact_success_rate:.3f} | "
            "{query_final_grid_match_rate:.3f} | {query_similarity_mean:.3f} | "
            "{train_wall_clock_s_mean:.2f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `query_exact` counts only submitted episodes whose final grids exactly match the hidden target.",
            "- `query_grid_match` credits exact final grids even if the agent never submitted.",
            "- Similarity is recomputed externally against `true_test_output_*`; it does not rely on JaxARC test-mode masked targets.",
        ]
    )
    paths.final_summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_global_summary(paths, config=config, task_ids=task_ids)
    if bool(config.benchmark.get("generate_plots", True)):
        _write_global_plots(paths)


def run_ttrl_benchmark(config: DictConfig) -> dict[str, Any]:
    """Execute the full Phase 0 benchmark sweep."""
    task_ids = resolve_task_ids(config)
    paths = prepare_benchmark_paths(config, task_ids)
    aggregate_rows: list[dict[str, Any]] = []
    evaluation_mode = _evaluation_mode_label(config)
    seed = int(config.arch.seed)
    phase_name = _phase_name(config)
    dataset = _benchmark_dataset(config)
    subset = _benchmark_subset(config)
    method = _benchmark_method(config)

    for task_id in task_ids:
        task = load_single_task(config, task_id)
        budget_artifacts = iter_training_for_budgets(
            config,
            task_id=task_id,
            budgets=config.benchmark.budgets,
        )
        for budget, training in budget_artifacts:
            _append_markdown_log(
                paths.running_log,
                f"Starting method `{method}` task `{task_id}` at budget `{budget}` timesteps.",
            )
            support_rows = evaluate_task_pairs(
                config,
                task_id=task_id,
                task=task,
                split="support",
                training=training,
            )
            query_rows = evaluate_task_pairs(
                config,
                task_id=task_id,
                task=task,
                split="query",
                training=training,
            )

            for row in [*support_rows, *query_rows]:
                _append_jsonl_row(paths.pair_results_jsonl, row)

            aggregate_row = aggregate_task_results(
                phase_name=phase_name,
                dataset=dataset,
                subset=subset,
                method=method,
                task_id=task_id,
                training=training,
                support_rows=support_rows,
                query_rows=query_rows,
                seed=seed,
                evaluation_mode=evaluation_mode,
            )
            aggregate_rows.append(aggregate_row)
            _append_csv_row(paths.tracker_csv, AGGREGATE_FIELDNAMES, aggregate_row)
            write_summary(
                paths, config=config, aggregate_rows=aggregate_rows, task_ids=task_ids
            )
            _append_markdown_log(
                paths.running_log,
                (
                    f"Finished method `{method}` task `{task_id}` budget `{budget}` with support exact "
                    f"{aggregate_row['support_exact_success_rate']:.3f} and query exact "
                    f"{aggregate_row['query_exact_success_rate']:.3f}."
                ),
            )

    summary = {
        "run_dir": str(paths.run_dir),
        "task_ids": task_ids,
        "aggregate_rows": aggregate_rows,
    }
    _append_markdown_log(
        paths.running_log,
        f"Completed {_phase_name(config)} benchmark run `{paths.run_dir.name}`.",
    )
    return summary


__all__ = [
    "AGGREGATE_FIELDNAMES",
    "PAIR_FIELDNAMES",
    "BenchmarkPaths",
    "GridScore",
    "RolloutMetrics",
    "TrainingArtifacts",
    "aggregate_task_results",
    "evaluate_task_pairs",
    "load_single_task",
    "make_raw_eval_env",
    "make_single_pair_task",
    "prepare_benchmark_paths",
    "resolve_task_ids",
    "rollout_policy_episode",
    "run_ttrl_benchmark",
    "score_final_grid",
    "train_for_budget",
    "write_summary",
]
