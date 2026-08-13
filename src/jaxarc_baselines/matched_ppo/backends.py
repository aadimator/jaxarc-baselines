"""Matched JaxARC and ARCLE environment backends."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from jaxarc.configs import (
    ActionConfig,
    DatasetConfig,
    EnvironmentConfig,
    GridInitializationConfig,
    JaxArcConfig,
    RewardConfig as JaxRewardConfig,
)
from jaxarc.envs.actions import Action
from jaxarc.envs.environment import Environment as JaxArcEnvironment
from jaxarc.types import StepType

from .config import BenchmarkConfig
from .tasks import PairVariant, TaskBundle


SUBMIT_ACTION = 10
JAXARC_SUBMIT_OPERATION = 34
ARCLE_SUBMIT_OPERATION = 11
NUM_COMMON_OPERATIONS = 11


class BatchState(NamedTuple):
    """Backend-independent policy inputs and native environment state."""

    native: Any
    observation: jax.Array
    valid_rows: jax.Array
    valid_cols: jax.Array
    variant_index: jax.Array


class BatchStep(NamedTuple):
    """One batched transition, with already-reset next state."""

    state: BatchState
    reward: jax.Array
    done: jax.Array
    success: jax.Array
    terminal_similarity: jax.Array


def _expand_done(done: jax.Array, value: jax.Array) -> jax.Array:
    extra_dims = value.ndim - done.ndim
    return done.reshape(done.shape + (1,) * extra_dims)


class JaxArcBackend:
    """Compiled, vectorized JaxARC backend."""

    def __init__(self, config: BenchmarkConfig, tasks: TaskBundle):
        jaxarc_config = JaxArcConfig(
            environment=EnvironmentConfig(
                max_episode_steps=config.max_episode_steps,
                debug_level="off",
                render_mode="rgb_array",
            ),
            dataset=DatasetConfig(
                dataset_name="matched-ppo",
                dataset_path=str(config.task_root),
                max_grid_height=config.grid_size,
                max_grid_width=config.grid_size,
                min_grid_height=1,
                min_grid_width=1,
                max_colors=10,
                background_color=-1,
                max_train_pairs=1,
                max_test_pairs=1,
                task_split="train",
                shuffle_tasks=False,
            ),
            action=ActionConfig(
                max_operations=35,
                allowed_operations=tuple(range(10)) + (JAXARC_SUBMIT_OPERATION,),
                validate_actions=True,
                allow_invalid_actions=False,
                dynamic_action_filtering=False,
            ),
            reward=JaxRewardConfig(
                step_penalty=config.reward.step_penalty,
                similarity_weight=config.reward.similarity_weight,
                success_bonus=config.reward.success_bonus,
                unsolved_submission_penalty=config.reward.failed_submit_penalty,
            ),
            grid_initialization=GridInitializationConfig(
                demo_weight=1.0,
                permutation_weight=0.0,
                empty_weight=0.0,
                random_weight=0.0,
            ),
        )
        self.config = config
        self.tasks = tasks
        self.env = JaxArcEnvironment(
            config=jaxarc_config,
            buffer=tasks.jaxarc_buffer,
            episode_mode=0,
        )
        self.context_channels = tasks.context_channels

    def _observation(self, state: Any) -> jax.Array:
        context = self.context_channels[state.task_idx]
        return jnp.concatenate(
            [
                state.working_grid[..., None],
                state.target_grid[..., None],
                state.input_grid[..., None],
                context,
            ],
            axis=-1,
        ).astype(jnp.float32)

    def _policy_state(self, native_state: Any) -> BatchState:
        observations = jax.vmap(self._observation)(native_state)
        valid_rows = jnp.any(native_state.working_grid_mask, axis=2)
        valid_cols = jnp.any(native_state.working_grid_mask, axis=1)
        return BatchState(
            native_state,
            observations,
            valid_rows,
            valid_cols,
            native_state.task_idx,
        )

    def reset(self, seed: int) -> BatchState:
        keys = jax.random.split(jax.random.PRNGKey(seed), self.config.num_envs)
        native_state, _ = jax.vmap(lambda key: self.env.reset(key))(keys)
        return self._policy_state(native_state)

    def step(
        self,
        state: BatchState,
        operation: jax.Array,
        row: jax.Array,
        col: jax.Array,
        reset_keys: jax.Array,
    ) -> BatchStep:
        """Step all environments and reset terminal lanes inside JAX."""

        def step_one(native_state, common_op, action_row, action_col):
            native_op = jnp.where(
                common_op == SUBMIT_ACTION,
                JAXARC_SUBMIT_OPERATION,
                common_op,
            ).astype(jnp.int32)
            selection = jnp.zeros(
                (self.config.grid_size, self.config.grid_size), dtype=jnp.bool_
            )
            selection = selection.at[action_row, action_col].set(True)
            next_state, timestep = self.env.step(
                native_state,
                Action(operation=native_op, selection=selection),
            )
            done = timestep.step_type != StepType.MID
            success = (
                (common_op == SUBMIT_ACTION)
                & (next_state.similarity_score >= 1.0)
            )
            return (
                next_state,
                timestep.reward,
                done,
                success,
                next_state.similarity_score,
            )

        next_native, reward, done, success, similarity = jax.vmap(step_one)(
            state.native, operation, row, col
        )
        reset_native, _ = jax.vmap(lambda key: self.env.reset(key))(reset_keys)
        next_native = jax.tree_util.tree_map(
            lambda reset_value, next_value: jnp.where(
                _expand_done(done, next_value),
                reset_value,
                next_value,
            ),
            reset_native,
            next_native,
        )
        return BatchStep(
            state=self._policy_state(next_native),
            reward=reward.astype(jnp.float32),
            done=done,
            success=success,
            terminal_similarity=similarity.astype(jnp.float32),
        )


class _PairLoader:
    """Minimal ARCLE loader over the exact pair variants used by JaxARC."""

    def __init__(self, variants: tuple[PairVariant, ...]):
        self.data = [
            (
                [variant.input_grid.astype(np.int8)],
                [variant.output_grid.astype(np.int8)],
                [],
                [],
                {
                    "id": variant.task_id,
                    "pair_index": variant.pair_index,
                    "variant_index": variant.variant_index,
                },
            )
            for variant in variants
        ]

    def pick(self, data_index: int | None = None, **_kwargs):
        if data_index is None:
            raise ValueError("The matched ARCLE backend always supplies data_index")
        return self.data[data_index]


@dataclass
class _ARCLELane:
    env: Any
    variant_index: int = 0


class ARCLEBackend:
    """Host-side ARCLE backend using the same task pairs and policy inputs."""

    def __init__(self, config: BenchmarkConfig, tasks: TaskBundle):
        try:
            from arcle.envs import RawARCEnv
        except ImportError as exc:
            raise RuntimeError(
                "ARCLE is not installed. Run `pixi install` before this benchmark."
            ) from exc

        self.config = config
        self.tasks = tasks
        self.loader = _PairLoader(tasks.variants)
        self.lanes = [
            _ARCLELane(
                RawARCEnv(
                    data_loader=self.loader,
                    max_grid_size=(config.grid_size, config.grid_size),
                    max_trial=1,
                    render_mode=None,
                )
            )
            for _ in range(config.num_envs)
        ]
        self.rng = np.random.default_rng(0)

    def _padded(self, grid: np.ndarray) -> np.ndarray:
        padded = np.full(
            (self.config.grid_size, self.config.grid_size), -1, dtype=np.float32
        )
        height, width = grid.shape
        padded[:height, :width] = grid
        return padded

    def _observation(self, lane: _ARCLELane) -> np.ndarray:
        variant = self.tasks.variants[lane.variant_index]
        state = lane.env.current_state
        height, width = (int(value) for value in state["grid_dim"])
        working = np.full(
            (self.config.grid_size, self.config.grid_size), -1, dtype=np.float32
        )
        working[:height, :width] = state["grid"][:height, :width]
        answer = self._padded(variant.output_grid)
        input_grid = self._padded(variant.input_grid)
        context = np.transpose(
            np.stack(
                [variant.context_inputs, variant.context_outputs], axis=1
            ).reshape(
                2 * self.config.context_pairs,
                self.config.grid_size,
                self.config.grid_size,
            ),
            (1, 2, 0),
        ).astype(np.float32)
        return np.concatenate(
            [
                working[..., None],
                answer[..., None],
                input_grid[..., None],
                context,
            ],
            axis=-1,
        )

    def _valid_axes(self, lane: _ARCLELane) -> tuple[np.ndarray, np.ndarray]:
        height, width = (int(value) for value in lane.env.current_state["grid_dim"])
        valid_rows = np.arange(self.config.grid_size) < height
        valid_cols = np.arange(self.config.grid_size) < width
        return valid_rows, valid_cols

    def _reset_lane(self, lane: _ARCLELane) -> None:
        lane.variant_index = int(self.rng.integers(0, len(self.tasks.variants)))
        lane.env.reset(
            seed=int(self.rng.integers(0, np.iinfo(np.int32).max)),
            options={
                "prob_index": lane.variant_index,
                "subprob_index": 0,
                "adaptation": True,
                "reset_on_submit": False,
            },
        )

    def _batch_state(self) -> BatchState:
        observations = np.stack([self._observation(lane) for lane in self.lanes])
        axes = [self._valid_axes(lane) for lane in self.lanes]
        return BatchState(
            native=None,
            observation=jnp.asarray(observations),
            valid_rows=jnp.asarray(np.stack([item[0] for item in axes])),
            valid_cols=jnp.asarray(np.stack([item[1] for item in axes])),
            variant_index=jnp.asarray(
                [lane.variant_index for lane in self.lanes],
                dtype=jnp.int32,
            ),
        )

    def reset(self, seed: int) -> BatchState:
        self.rng = np.random.default_rng(seed)
        for lane in self.lanes:
            self._reset_lane(lane)
        return self._batch_state()

    @staticmethod
    def _similarity(lane: _ARCLELane) -> float:
        state = lane.env.current_state
        height, width = (int(value) for value in state["grid_dim"])
        answer = lane.env.answer
        answer_height, answer_width = answer.shape
        if (height, width) != (answer_height, answer_width):
            active = np.zeros_like(answer, dtype=np.int8)
            copy_height = min(height, answer_height)
            copy_width = min(width, answer_width)
            active[:copy_height, :copy_width] = state["grid"][
                :copy_height, :copy_width
            ]
            return float(np.mean(active == answer))
        return float(
            np.mean(state["grid"][:answer_height, :answer_width] == answer)
        )

    def step(
        self,
        state: BatchState,
        operation: jax.Array,
        row: jax.Array,
        col: jax.Array,
        reset_keys: jax.Array,
    ) -> BatchStep:
        del state, reset_keys
        operations = np.asarray(jax.device_get(operation), dtype=np.int32)
        rows = np.asarray(jax.device_get(row), dtype=np.int32)
        cols = np.asarray(jax.device_get(col), dtype=np.int32)
        rewards = np.zeros(self.config.num_envs, dtype=np.float32)
        dones = np.zeros(self.config.num_envs, dtype=np.bool_)
        successes = np.zeros(self.config.num_envs, dtype=np.bool_)
        similarities = np.zeros(self.config.num_envs, dtype=np.float32)

        for index, lane in enumerate(self.lanes):
            before = self._similarity(lane)
            common_op = int(operations[index])
            native_op = (
                ARCLE_SUBMIT_OPERATION
                if common_op == SUBMIT_ACTION
                else common_op
            )
            selection = np.zeros(
                (self.config.grid_size, self.config.grid_size), dtype=np.int8
            )
            selection[int(rows[index]), int(cols[index])] = 1
            _, _, terminated, _, _ = lane.env.step(
                {"selection": selection, "operation": native_op}
            )
            after = self._similarity(lane)
            submitted = common_op == SUBMIT_ACTION
            success = submitted and after >= 1.0
            reward = (
                self.config.reward.similarity_weight * (after - before)
                + self.config.reward.step_penalty
                + (self.config.reward.success_bonus if success else 0.0)
                + (
                    self.config.reward.failed_submit_penalty
                    if submitted and not success
                    else 0.0
                )
            )
            truncated = lane.env.action_steps >= self.config.max_episode_steps
            done = bool(terminated or truncated)

            rewards[index] = reward
            dones[index] = done
            successes[index] = success
            similarities[index] = after
            if done:
                self._reset_lane(lane)

        return BatchStep(
            state=self._batch_state(),
            reward=jnp.asarray(rewards),
            done=jnp.asarray(dones),
            success=jnp.asarray(successes),
            terminal_similarity=jnp.asarray(similarities),
        )


class _MatchedARCLEEnv(gym.Env):
    """Single matched ARCLE lane suitable for Gym vector environments."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: BenchmarkConfig,
        variants: tuple[PairVariant, ...],
    ):
        from arcle.envs import RawARCEnv

        self.config = config
        self.variants = variants
        self.loader = _PairLoader(variants)
        self.env = RawARCEnv(
            data_loader=self.loader,
            max_grid_size=(config.grid_size, config.grid_size),
            max_trial=1,
            render_mode=None,
        )
        self.variant_index = 0
        self.action_space = gym.spaces.MultiDiscrete(
            [NUM_COMMON_OPERATIONS, config.grid_size, config.grid_size]
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=-1.0,
                    high=9.0,
                    shape=(
                        config.grid_size,
                        config.grid_size,
                        config.observation_channels,
                    ),
                    dtype=np.float32,
                ),
                "valid_rows": gym.spaces.MultiBinary(config.grid_size),
                "valid_cols": gym.spaces.MultiBinary(config.grid_size),
                "variant_index": gym.spaces.Discrete(len(variants)),
            }
        )

    def _padded(self, grid: np.ndarray) -> np.ndarray:
        padded = np.full(
            (self.config.grid_size, self.config.grid_size),
            -1,
            dtype=np.float32,
        )
        height, width = grid.shape
        padded[:height, :width] = grid
        return padded

    def _observation(self) -> dict[str, np.ndarray | np.int32]:
        variant = self.variants[self.variant_index]
        state = self.env.current_state
        height, width = (int(value) for value in state["grid_dim"])
        working = np.full(
            (self.config.grid_size, self.config.grid_size),
            -1,
            dtype=np.float32,
        )
        working[:height, :width] = state["grid"][:height, :width]
        context = np.transpose(
            np.stack(
                [variant.context_inputs, variant.context_outputs],
                axis=1,
            ).reshape(
                2 * self.config.context_pairs,
                self.config.grid_size,
                self.config.grid_size,
            ),
            (1, 2, 0),
        ).astype(np.float32)
        observation = np.concatenate(
            [
                working[..., None],
                self._padded(variant.output_grid)[..., None],
                self._padded(variant.input_grid)[..., None],
                context,
            ],
            axis=-1,
        )
        return {
            "observation": observation,
            "valid_rows": (
                np.arange(self.config.grid_size) < height
            ).astype(np.int8),
            "valid_cols": (
                np.arange(self.config.grid_size) < width
            ).astype(np.int8),
            "variant_index": np.int32(self.variant_index),
        }

    def _similarity(self) -> float:
        state = self.env.current_state
        height, width = (int(value) for value in state["grid_dim"])
        answer = self.env.answer
        answer_height, answer_width = answer.shape
        if (height, width) != (answer_height, answer_width):
            active = np.zeros_like(answer, dtype=np.int8)
            copy_height = min(height, answer_height)
            copy_width = min(width, answer_width)
            active[:copy_height, :copy_width] = state["grid"][
                :copy_height, :copy_width
            ]
            return float(np.mean(active == answer))
        return float(
            np.mean(
                state["grid"][:answer_height, :answer_width] == answer
            )
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.variant_index = int(
            self.np_random.integers(0, len(self.variants))
        )
        self.env.reset(
            seed=int(
                self.np_random.integers(0, np.iinfo(np.int32).max)
            ),
            options={
                "prob_index": self.variant_index,
                "subprob_index": 0,
                "adaptation": True,
                "reset_on_submit": False,
            },
        )
        return self._observation(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        common_op, row, col = (int(value) for value in action)
        before = self._similarity()
        native_op = (
            ARCLE_SUBMIT_OPERATION
            if common_op == SUBMIT_ACTION
            else common_op
        )
        selection = np.zeros(
            (self.config.grid_size, self.config.grid_size),
            dtype=np.int8,
        )
        selection[row, col] = 1
        _, _, terminated, _, _ = self.env.step(
            {"selection": selection, "operation": native_op}
        )
        after = self._similarity()
        submitted = common_op == SUBMIT_ACTION
        success = submitted and after >= 1.0
        reward = (
            self.config.reward.similarity_weight * (after - before)
            + self.config.reward.step_penalty
            + (self.config.reward.success_bonus if success else 0.0)
            + (
                self.config.reward.failed_submit_penalty
                if submitted and not success
                else 0.0
            )
        )
        truncated = (
            self.env.action_steps >= self.config.max_episode_steps
        )
        return (
            self._observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            {
                "success": bool(success),
                "terminal_similarity": float(after),
            },
        )


class ARCLEAsyncBackend:
    """Multiprocess ARCLE backend for best-capacity PPO measurements."""

    def __init__(self, config: BenchmarkConfig, tasks: TaskBundle):
        self.config = config
        self.tasks = tasks
        env_fns = [
            partial(_MatchedARCLEEnv, config, tasks.variants)
            for _ in range(config.num_envs)
        ]
        self.env = gym.vector.AsyncVectorEnv(
            env_fns,
            shared_memory=True,
            context="spawn",
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )

    @staticmethod
    def _policy_state(observation: dict[str, np.ndarray]) -> BatchState:
        return BatchState(
            native=None,
            observation=jnp.asarray(observation["observation"]),
            valid_rows=jnp.asarray(
                observation["valid_rows"],
                dtype=jnp.bool_,
            ),
            valid_cols=jnp.asarray(
                observation["valid_cols"],
                dtype=jnp.bool_,
            ),
            variant_index=jnp.asarray(
                observation["variant_index"],
                dtype=jnp.int32,
            ),
        )

    def reset(self, seed: int) -> BatchState:
        observation, _ = self.env.reset(seed=seed)
        return self._policy_state(observation)

    def step(
        self,
        state: BatchState,
        operation: jax.Array,
        row: jax.Array,
        col: jax.Array,
        reset_keys: jax.Array,
    ) -> BatchStep:
        del state, reset_keys
        actions = np.stack(
            [
                np.asarray(jax.device_get(operation), dtype=np.int32),
                np.asarray(jax.device_get(row), dtype=np.int32),
                np.asarray(jax.device_get(col), dtype=np.int32),
            ],
            axis=-1,
        )
        observation, rewards, terminated, truncated, infos = self.env.step(
            actions
        )
        done = np.logical_or(terminated, truncated)
        successes = np.asarray(
            infos.get(
                "success",
                np.zeros(self.config.num_envs, dtype=np.bool_),
            ),
            dtype=np.bool_,
        )
        similarities = np.asarray(
            infos.get(
                "terminal_similarity",
                np.zeros(self.config.num_envs, dtype=np.float32),
            ),
            dtype=np.float32,
        )
        final_infos = infos.get("final_info")
        if final_infos is not None:
            for index in np.flatnonzero(done):
                if isinstance(final_infos, dict):
                    successes[index] = bool(
                        final_infos["success"][index]
                    )
                    similarities[index] = float(
                        final_infos["terminal_similarity"][index]
                    )
                else:
                    final_info = final_infos[index]
                    successes[index] = bool(final_info["success"])
                    similarities[index] = float(
                        final_info["terminal_similarity"]
                    )
        return BatchStep(
            state=self._policy_state(observation),
            reward=jnp.asarray(rewards, dtype=jnp.float32),
            done=jnp.asarray(done),
            success=jnp.asarray(successes),
            terminal_similarity=jnp.asarray(similarities),
        )

    def close(self) -> None:
        self.env.close()


def make_backend(
    name: str,
    config: BenchmarkConfig,
    tasks: TaskBundle,
) -> JaxArcBackend | ARCLEBackend | ARCLEAsyncBackend:
    """Create one of the two matched environment implementations."""
    if name == "jaxarc":
        return JaxArcBackend(config, tasks)
    if name == "arcle":
        return ARCLEBackend(config, tasks)
    if name == "arcle-async":
        return ARCLEAsyncBackend(config, tasks)
    raise ValueError(f"Unknown backend: {name}")
