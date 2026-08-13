"""Configuration for the matched PPO benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RewardConfig:
    """Reward shared by the JaxARC and ARCLE backends."""

    step_penalty: float = -0.005
    similarity_weight: float = 1.0
    success_bonus: float = 10.0
    failed_submit_penalty: float = -1.0


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete matched PPO run configuration."""

    experiment_name: str
    task_root: Path
    task_split: str
    task_ids: tuple[str, ...]
    grid_size: int
    context_pairs: int
    max_episode_steps: int
    policy_architecture: str
    num_envs: int
    rollout_length: int
    total_timesteps: int
    eval_episodes: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    max_grad_norm: float
    epochs: int
    num_minibatches: int
    log_interval_updates: int
    output_dir: Path
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluate_after_training: bool = True

    @property
    def observation_channels(self) -> int:
        """Working, answer, input, then five input-output context pairs."""
        return 3 + 2 * self.context_pairs

    @property
    def steps_per_update(self) -> int:
        return self.num_envs * self.rollout_length

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.steps_per_update

    @property
    def actual_timesteps(self) -> int:
        return self.num_updates * self.steps_per_update

    @property
    def minibatch_size(self) -> int:
        return self.steps_per_update // self.num_minibatches

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        payload = asdict(self)
        payload["task_root"] = str(self.task_root)
        payload["output_dir"] = str(self.output_dir)
        payload["task_ids"] = list(self.task_ids)
        return payload

    def with_overrides(
        self,
        *,
        total_timesteps: int | None = None,
        num_envs: int | None = None,
        rollout_length: int | None = None,
        eval_episodes: int | None = None,
        policy_architecture: str | None = None,
        num_minibatches: int | None = None,
        evaluate_after_training: bool | None = None,
        output_dir: Path | None = None,
    ) -> BenchmarkConfig:
        """Apply CLI overrides without mutating the loaded config."""
        return replace(
            self,
            total_timesteps=(
                self.total_timesteps if total_timesteps is None else total_timesteps
            ),
            num_envs=self.num_envs if num_envs is None else num_envs,
            rollout_length=(
                self.rollout_length if rollout_length is None else rollout_length
            ),
            eval_episodes=(
                self.eval_episodes if eval_episodes is None else eval_episodes
            ),
            policy_architecture=(
                self.policy_architecture
                if policy_architecture is None
                else policy_architecture
            ),
            num_minibatches=(
                self.num_minibatches
                if num_minibatches is None
                else num_minibatches
            ),
            evaluate_after_training=(
                self.evaluate_after_training
                if evaluate_after_training is None
                else evaluate_after_training
            ),
            output_dir=self.output_dir if output_dir is None else output_dir,
        )

    def validate(self) -> None:
        """Reject configurations that would invalidate or break the benchmark."""
        if not self.task_ids:
            raise ValueError("task_ids must not be empty")
        if self.grid_size < 1:
            raise ValueError("grid_size must be positive")
        if self.policy_architecture not in {"factorized", "spatial"}:
            raise ValueError(
                "policy_architecture must be 'factorized' or 'spatial'"
            )
        if self.observation_channels != 13:
            raise ValueError(
                "The matched PPO benchmark requires 13 channels. Set context_pairs to 5."
            )
        if self.num_envs < 1 or self.rollout_length < 1:
            raise ValueError("num_envs and rollout_length must be positive")
        if (
            self.evaluate_after_training
            and self.eval_episodes < len(self.task_ids)
        ):
            raise ValueError(
                "eval_episodes must cover every configured task"
            )
        if self.num_updates < 1:
            raise ValueError("total_timesteps is smaller than one rollout")
        batch_size = self.steps_per_update
        if batch_size % self.num_minibatches:
            raise ValueError(
                "num_envs * rollout_length must be divisible by num_minibatches"
            )


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path) -> BenchmarkConfig:
    """Load and resolve a YAML benchmark configuration."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    reward = RewardConfig(**raw.pop("reward", {}))
    raw["task_root"] = _resolve_repo_path(raw["task_root"])
    raw["output_dir"] = _resolve_repo_path(raw["output_dir"])
    raw["task_ids"] = tuple(str(task_id) for task_id in raw["task_ids"])
    config = BenchmarkConfig(reward=reward, **raw)
    config.validate()
    return config
