"""Training and evaluation orchestration for the matched PPO benchmark."""

from __future__ import annotations

import json
import hashlib
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.training.train_state import TrainState

from .backends import (
    ARCLEAsyncBackend,
    ARCLEBackend,
    BatchState,
    JaxArcBackend,
    make_backend,
)
from .config import BenchmarkConfig, REPO_ROOT
from .policy import (
    PolicyAction,
    PolicyModel,
    make_model,
    policy_description,
    select_action,
)
from .ppo import (
    LossMetrics,
    RolloutBatch,
    create_train_state,
    make_rollout_update,
)
from .tasks import TaskBundle, load_task_bundle


Collector = Callable[
    [Any, BatchState, jax.Array],
    tuple[BatchState, jax.Array, RolloutBatch],
]
EnvironmentBackend = JaxArcBackend | ARCLEBackend | ARCLEAsyncBackend


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _run_directory(config: BenchmarkConfig, backend: str, seed: int) -> Path:
    run_name = (
        f"{_utc_timestamp()}_{config.experiment_name}_{backend}_seed{seed}"
    )
    run_dir = config.output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "editable-or-unavailable"


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _runtime_metadata() -> dict[str, Any]:
    devices = [
        {
            "platform": device.platform,
            "kind": getattr(device, "device_kind", "unknown"),
            "id": device.id,
        }
        for device in jax.devices()
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "flax": _package_version("flax"),
        "arcle": _package_version("arcle"),
        "jaxarc": _package_version("JaxARC"),
        "devices": devices,
        "git": _git_metadata(),
    }


def _parameter_count(params: Any) -> int:
    return sum(int(value.size) for value in jax.tree_util.tree_leaves(params))


def _comparison_id(
    config: BenchmarkConfig,
    tasks: TaskBundle,
) -> str:
    resolved_config = config.as_dict()
    resolved_config.pop("output_dir", None)
    payload = {
        "config": resolved_config,
        "task_hashes": tasks.task_hashes,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _stack_rollout(items: list[RolloutBatch]) -> RolloutBatch:
    return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *items)


def _make_jaxarc_collector(
    backend: JaxArcBackend,
    model: PolicyModel,
    rollout_length: int,
    *,
    deterministic: bool,
) -> Collector:
    """Compile environment rollout and inference together."""

    @jax.jit
    def collect(
        params: Any,
        initial_state: BatchState,
        initial_key: jax.Array,
    ) -> tuple[BatchState, jax.Array, RolloutBatch]:
        def step(carry, _unused):
            state, key = carry
            key, action_key, reset_key = jax.random.split(key, 3)
            action = select_action(
                model,
                params,
                state.observation,
                state.valid_rows,
                state.valid_cols,
                action_key,
                deterministic=deterministic,
            )
            reset_keys = jax.random.split(
                reset_key, state.observation.shape[0]
            )
            transition = backend.step(
                state,
                action.operation,
                action.row,
                action.col,
                reset_keys,
            )
            record = RolloutBatch(
                observation=state.observation,
                valid_rows=state.valid_rows,
                valid_cols=state.valid_cols,
                variant_index=state.variant_index,
                operation=action.operation,
                row=action.row,
                col=action.col,
                old_log_probability=action.log_probability,
                old_value=action.value,
                reward=transition.reward,
                done=transition.done,
                success=transition.success,
                terminal_similarity=transition.terminal_similarity,
            )
            return (transition.state, key), record

        (final_state, final_key), rollout = jax.lax.scan(
            step,
            (initial_state, initial_key),
            xs=None,
            length=rollout_length,
        )
        return final_state, final_key, rollout

    return collect


def _make_arcle_collector(
    backend: ARCLEBackend,
    model: PolicyModel,
    rollout_length: int,
    *,
    deterministic: bool,
) -> Collector:
    """Run ARCLE transitions on host while keeping policy inference jitted."""

    @jax.jit
    def choose(
        params: Any,
        state: BatchState,
        key: jax.Array,
    ) -> PolicyAction:
        return select_action(
            model,
            params,
            state.observation,
            state.valid_rows,
            state.valid_cols,
            key,
            deterministic=deterministic,
        )

    def collect(
        params: Any,
        initial_state: BatchState,
        initial_key: jax.Array,
    ) -> tuple[BatchState, jax.Array, RolloutBatch]:
        state = initial_state
        key = initial_key
        records: list[RolloutBatch] = []
        for _ in range(rollout_length):
            key, action_key, reset_key = jax.random.split(key, 3)
            action = choose(params, state, action_key)
            transition = backend.step(
                state,
                action.operation,
                action.row,
                action.col,
                jax.random.split(reset_key, state.observation.shape[0]),
            )
            records.append(
                RolloutBatch(
                    observation=state.observation,
                    valid_rows=state.valid_rows,
                    valid_cols=state.valid_cols,
                    variant_index=state.variant_index,
                    operation=action.operation,
                    row=action.row,
                    col=action.col,
                    old_log_probability=action.log_probability,
                    old_value=action.value,
                    reward=transition.reward,
                    done=transition.done,
                    success=transition.success,
                    terminal_similarity=transition.terminal_similarity,
                )
            )
            state = transition.state
        return state, key, _stack_rollout(records)

    return collect


def _make_collector(
    backend: EnvironmentBackend,
    model: PolicyModel,
    rollout_length: int,
    *,
    deterministic: bool,
) -> Collector:
    if isinstance(backend, JaxArcBackend):
        return _make_jaxarc_collector(
            backend,
            model,
            rollout_length,
            deterministic=deterministic,
        )
    return _make_arcle_collector(
        backend,
        model,
        rollout_length,
        deterministic=deterministic,
    )


def _make_value_function(model: PolicyModel):
    @jax.jit
    def value(params: Any, observation: jax.Array) -> jax.Array:
        return model.apply(params, observation).value

    return value


def _loss_metrics_dict(metrics: LossMetrics) -> dict[str, float]:
    return {
        field: float(np.asarray(getattr(metrics, field)))
        for field in LossMetrics._fields
    }


def _make_training_step(
    backend: EnvironmentBackend,
    model: PolicyModel,
    config: BenchmarkConfig,
) -> Callable[
    [TrainState, BatchState, jax.Array],
    tuple[
        TrainState,
        BatchState,
        jax.Array,
        RolloutBatch,
        LossMetrics,
    ],
]:
    """Build one complete rollout-plus-PPO update.

    JaxARC compiles this whole function. ARCLE must return to Python for
    environment transitions, then calls the same compiled PPO update.
    """
    collector = _make_collector(
        backend,
        model,
        config.rollout_length,
        deterministic=False,
    )
    value_function = _make_value_function(model)
    update_rollout = make_rollout_update(model, config)

    def step(
        train_state: TrainState,
        state: BatchState,
        key: jax.Array,
    ) -> tuple[
        TrainState,
        BatchState,
        jax.Array,
        RolloutBatch,
        LossMetrics,
    ]:
        state, key, rollout = collector(train_state.params, state, key)
        last_value = value_function(train_state.params, state.observation)
        train_state, key, metrics = update_rollout(
            train_state,
            rollout,
            last_value,
            key,
        )
        return train_state, state, key, rollout, metrics

    return jax.jit(step) if isinstance(backend, JaxArcBackend) else step


def _rollout_statistics(rollout: RolloutBatch) -> dict[str, float | int]:
    rewards = np.asarray(jax.device_get(rollout.reward))
    dones = np.asarray(jax.device_get(rollout.done), dtype=np.bool_)
    successes = np.asarray(jax.device_get(rollout.success), dtype=np.bool_)
    similarities = np.asarray(
        jax.device_get(rollout.terminal_similarity)
    )
    completed = int(dones.sum())
    exact = int(successes.sum())
    terminal_values = similarities[dones]
    return {
        "mean_step_reward": float(rewards.mean()),
        "completed_episodes": completed,
        "exact_successes": exact,
        "exact_success_rate": exact / completed if completed else 0.0,
        "mean_terminal_similarity": (
            float(terminal_values.mean()) if completed else 0.0
        ),
    }


def _evaluate(
    backend: EnvironmentBackend,
    model: PolicyModel,
    params: Any,
    config: BenchmarkConfig,
    seed: int,
) -> dict[str, Any]:
    variants_by_task: dict[str, list[int]] = {
        task_id: [] for task_id in config.task_ids
    }
    for variant in backend.tasks.variants:
        variants_by_task[variant.task_id].append(variant.variant_index)

    task_base, task_remainder = divmod(
        config.eval_episodes,
        len(config.task_ids),
    )
    variant_quotas: dict[int, int] = {}
    for task_position, task_id in enumerate(config.task_ids):
        task_quota = task_base + int(task_position < task_remainder)
        variant_indices = variants_by_task[task_id]
        variant_base, variant_remainder = divmod(
            task_quota,
            len(variant_indices),
        )
        for variant_position, variant_index in enumerate(variant_indices):
            variant_quotas[variant_index] = (
                variant_base + int(variant_position < variant_remainder)
            )

    collector = _make_collector(
        backend,
        model,
        config.rollout_length,
        deterministic=True,
    )
    warmup_started = time.perf_counter()
    warmup_state = backend.reset(seed)
    warmup_key = jax.random.PRNGKey(seed)
    warmup_state, _, _ = collector(params, warmup_state, warmup_key)
    jax.block_until_ready(warmup_state.observation)
    warmup_seconds = time.perf_counter() - warmup_started

    state = backend.reset(seed)
    key = jax.random.PRNGKey(seed)
    per_variant: dict[int, dict[str, Any]] = {
        variant.variant_index: {
            "episodes": 0,
            "exact_successes": 0,
            "terminal_similarities": [],
            "similarity_gains": [],
        }
        for variant in backend.tasks.variants
    }
    environment_steps = 0
    started = time.perf_counter()
    while any(
        per_variant[index]["episodes"] < quota
        for index, quota in variant_quotas.items()
    ):
        state, key, rollout = collector(params, state, key)
        environment_steps += int(rollout.reward.size)
        dones = np.asarray(jax.device_get(rollout.done), dtype=np.bool_)
        successes = np.asarray(
            jax.device_get(rollout.success), dtype=np.bool_
        )
        similarities = np.asarray(
            jax.device_get(rollout.terminal_similarity)
        )
        variant_indices = np.asarray(
            jax.device_get(rollout.variant_index)
        )
        for done, success, similarity, variant_index in zip(
            dones.reshape(-1),
            successes.reshape(-1),
            similarities.reshape(-1),
            variant_indices.reshape(-1),
            strict=True,
        ):
            if not done:
                continue
            index = int(variant_index)
            variant_metrics = per_variant[index]
            if variant_metrics["episodes"] >= variant_quotas[index]:
                continue
            baseline = float(
                backend.tasks.variant_initial_similarities[index]
            )
            variant_metrics["episodes"] += 1
            variant_metrics["exact_successes"] += int(success)
            variant_metrics["terminal_similarities"].append(
                float(similarity)
            )
            variant_metrics["similarity_gains"].append(
                float(similarity) - baseline
            )
    jax.block_until_ready(state.observation)
    seconds = time.perf_counter() - started

    by_variant: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    all_similarities: list[float] = []
    all_gains: list[float] = []
    exact = 0
    for variant in backend.tasks.variants:
        metrics = per_variant[variant.variant_index]
        similarities = np.asarray(
            metrics["terminal_similarities"],
            dtype=np.float64,
        )
        gains = np.asarray(
            metrics["similarity_gains"],
            dtype=np.float64,
        )
        exact += metrics["exact_successes"]
        all_similarities.extend(similarities.tolist())
        all_gains.extend(gains.tolist())
        by_variant[
            f"{variant.task_id}:pair{variant.pair_index}"
        ] = {
            "task_id": variant.task_id,
            "pair_index": variant.pair_index,
            "episodes": metrics["episodes"],
            "exact_successes": metrics["exact_successes"],
            "exact_success_rate": (
                metrics["exact_successes"] / metrics["episodes"]
            ),
            "initial_similarity": float(
                backend.tasks.variant_initial_similarities[
                    variant.variant_index
                ]
            ),
            "mean_terminal_similarity": float(similarities.mean()),
            "mean_similarity_gain": float(gains.mean()),
        }

    for task_id in config.task_ids:
        task_variants = [
            metrics
            for metrics in by_variant.values()
            if metrics["task_id"] == task_id
        ]
        task_episodes = sum(item["episodes"] for item in task_variants)
        task_exact = sum(
            item["exact_successes"] for item in task_variants
        )
        by_task[task_id] = {
            "episodes": task_episodes,
            "exact_successes": task_exact,
            "exact_success_rate": task_exact / task_episodes,
            "mean_initial_similarity": float(
                np.mean(
                    [
                        item["initial_similarity"]
                        for item in task_variants
                    ]
                )
            ),
            "mean_terminal_similarity": float(
                np.mean(
                    [
                        item["mean_terminal_similarity"]
                        for item in task_variants
                    ]
                )
            ),
            "mean_similarity_gain": float(
                np.mean(
                    [
                        item["mean_similarity_gain"]
                        for item in task_variants
                    ]
                )
            ),
        }
    terminal_array = np.asarray(all_similarities, dtype=np.float64)
    gain_array = np.asarray(all_gains, dtype=np.float64)
    macro_success = float(
        np.mean(
            [metrics["exact_success_rate"] for metrics in by_task.values()]
        )
    )
    macro_similarity = float(
        np.mean(
            [
                metrics["mean_terminal_similarity"]
                for metrics in by_task.values()
            ]
        )
    )
    macro_gain = float(
        np.mean(
            [metrics["mean_similarity_gain"] for metrics in by_task.values()]
        )
    )
    return {
        "episodes": config.eval_episodes,
        "environment_steps": environment_steps,
        "compile_warmup_seconds": warmup_seconds,
        "seconds": seconds,
        "steps_per_second": environment_steps / seconds,
        "exact_successes": exact,
        "exact_success_rate": exact / config.eval_episodes,
        "mean_terminal_similarity": float(terminal_array.mean()),
        "median_terminal_similarity": float(np.median(terminal_array)),
        "mean_similarity_gain": float(gain_array.mean()),
        "median_similarity_gain": float(np.median(gain_array)),
        "macro_exact_success_rate": macro_success,
        "macro_mean_terminal_similarity": macro_similarity,
        "macro_mean_similarity_gain": macro_gain,
        "by_task": by_task,
        "by_variant": by_variant,
    }


def _save_checkpoint(params: Any, path: Path) -> None:
    path.write_bytes(flax.serialization.to_bytes(params))


def _restore_checkpoint(
    template_params: Any,
    path: Path,
) -> Any:
    return flax.serialization.from_bytes(template_params, path.read_bytes())


def run_training(
    config: BenchmarkConfig,
    backend_name: str,
    seed: int,
) -> Path:
    """Train one backend/seed run and optionally evaluate it."""
    config.validate()
    cold_started = time.perf_counter()
    run_dir = _run_directory(config, backend_name, seed)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.as_dict(), sort_keys=False),
        encoding="utf-8",
    )

    setup_started = time.perf_counter()
    tasks: TaskBundle = load_task_bundle(config)
    backend = make_backend(backend_name, config, tasks)
    model = make_model(config.policy_architecture, config.grid_size)
    initialization_key, training_key = jax.random.split(
        jax.random.PRNGKey(seed)
    )
    initial_train_state = create_train_state(model, config, initialization_key)
    training_step = _make_training_step(
        backend,
        model,
        config,
    )
    setup_seconds = time.perf_counter() - setup_started

    compile_started = time.perf_counter()
    warm_state = backend.reset(seed)
    (
        warm_train_state,
        warm_state,
        _,
        _,
        _,
    ) = training_step(
        initial_train_state,
        warm_state,
        training_key,
    )
    jax.block_until_ready(
        (warm_train_state.step, warm_state.observation)
    )
    compile_seconds = time.perf_counter() - compile_started

    state = backend.reset(seed)
    train_state = initial_train_state
    key = training_key
    curve: list[dict[str, Any]] = []
    train_started = time.perf_counter()
    for update_index in range(1, config.num_updates + 1):
        (
            train_state,
            state,
            key,
            rollout,
            loss_metrics,
        ) = training_step(
            train_state,
            state,
            key,
        )
        if (
            update_index == 1
            or update_index % config.log_interval_updates == 0
            or update_index == config.num_updates
        ):
            elapsed = time.perf_counter() - train_started
            timesteps = update_index * config.steps_per_update
            record = {
                "update": update_index,
                "timesteps": timesteps,
                "elapsed_seconds": elapsed,
                "steps_per_second": timesteps / elapsed,
                **_rollout_statistics(rollout),
                **_loss_metrics_dict(loss_metrics),
            }
            curve.append(record)
            print(
                f"{backend_name} seed={seed} "
                f"{timesteps}/{config.actual_timesteps} "
                f"{record['steps_per_second']:.0f} SPS "
                f"episodes={record['completed_episodes']} "
                f"success={record['exact_success_rate']:.3f}",
                flush=True,
            )
    jax.block_until_ready(train_state.step)
    train_seconds = time.perf_counter() - train_started
    training_wall_seconds = time.perf_counter() - cold_started

    evaluation = None
    if config.evaluate_after_training:
        evaluation = _evaluate(
            backend,
            model,
            train_state.params,
            config,
            seed + 10_000,
        )
    checkpoint_path = run_dir / "params.msgpack"
    _save_checkpoint(train_state.params, checkpoint_path)
    runtime = _runtime_metadata()
    full_run_seconds = time.perf_counter() - cold_started
    result = {
        "status": "complete",
        "experiment_name": config.experiment_name,
        "backend": backend_name,
        "seed": seed,
        "comparison_id": _comparison_id(config, tasks),
        "config": config.as_dict(),
        "task_hashes": tasks.task_hashes,
        "task_statistics": tasks.task_statistics,
        "num_pair_variants": len(tasks.variants),
        "actual_timesteps": config.actual_timesteps,
        "policy": {
            "architecture": policy_description(model),
            "architecture_key": config.policy_architecture,
            "parameter_count": _parameter_count(train_state.params),
            "action_space": (
                "joint color-cell categorical plus submit"
                if config.policy_architecture == "spatial"
                else "factorized operation-row-column plus submit"
            ),
        },
        "runtime": runtime,
        "timing": {
            "setup_seconds": setup_seconds,
            "training_compile_warmup_seconds": compile_seconds,
            "train_seconds": train_seconds,
            "training_wall_seconds": training_wall_seconds,
            "full_run_seconds": full_run_seconds,
            "train_steps_per_second": (
                config.actual_timesteps / train_seconds
            ),
        },
        "evaluation": evaluation,
        "curve": curve,
        "checkpoint": str(checkpoint_path),
    }
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if isinstance(backend, ARCLEAsyncBackend):
        backend.close()
    print(f"wrote {result_path}", flush=True)
    return result_path


def evaluate_checkpoint(
    config: BenchmarkConfig,
    backend_name: str,
    seed: int,
    checkpoint_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one saved policy on either matched backend."""
    config.validate()
    tasks = load_task_bundle(config)
    backend = make_backend(backend_name, config, tasks)
    model = make_model(config.policy_architecture, config.grid_size)
    template = create_train_state(
        model,
        config,
        jax.random.PRNGKey(seed),
    ).params
    params = _restore_checkpoint(template, checkpoint_path)
    evaluation = _evaluate(
        backend,
        model,
        params,
        config,
        seed + 10_000,
    )
    if isinstance(backend, ARCLEAsyncBackend):
        backend.close()
    if output_path is None:
        print(json.dumps(evaluation, indent=2, sort_keys=True))
    else:
        payload = {
            "checkpoint": str(checkpoint_path),
            "evaluation_backend": backend_name,
            "seed": seed,
            "config": config.as_dict(),
            "task_hashes": tasks.task_hashes,
            "evaluation": evaluation,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"wrote {output_path}")
    return evaluation
