"""Shared PPO update implementation."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .config import BenchmarkConfig
from .policy import PolicyModel, action_statistics


class RolloutBatch(NamedTuple):
    observation: jax.Array
    valid_rows: jax.Array
    valid_cols: jax.Array
    variant_index: jax.Array
    operation: jax.Array
    row: jax.Array
    col: jax.Array
    old_log_probability: jax.Array
    old_value: jax.Array
    reward: jax.Array
    done: jax.Array
    success: jax.Array
    terminal_similarity: jax.Array


class TrainingBatch(NamedTuple):
    observation: jax.Array
    valid_rows: jax.Array
    valid_cols: jax.Array
    operation: jax.Array
    row: jax.Array
    col: jax.Array
    old_log_probability: jax.Array
    old_value: jax.Array
    advantage: jax.Array
    return_value: jax.Array


class LossMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approximate_kl: jax.Array
    clip_fraction: jax.Array


def create_train_state(
    model: PolicyModel,
    config: BenchmarkConfig,
    key: jax.Array,
) -> TrainState:
    """Initialize the common CNN and Adam optimizer."""
    dummy_observation = jnp.zeros(
        (
            1,
            config.grid_size,
            config.grid_size,
            config.observation_channels,
        ),
        dtype=jnp.float32,
    )
    params = model.init(key, dummy_observation)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate, eps=1.0e-5),
    )
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )


@jax.jit
def calculate_gae(
    rollout: RolloutBatch,
    last_value: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculate generalized advantage estimates over a rollout."""

    def scan_step(carry, transition):
        reward, done, value, next_value = transition
        nonterminal = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * nonterminal * next_value - value
        advantage = (
            delta + gamma * gae_lambda * nonterminal * carry
        )
        return advantage, advantage

    next_values = jnp.concatenate(
        [rollout.old_value[1:], last_value[None, ...]], axis=0
    )
    _, reverse_advantages = jax.lax.scan(
        scan_step,
        jnp.zeros_like(last_value),
        (
            rollout.reward[::-1],
            rollout.done[::-1],
            rollout.old_value[::-1],
            next_values[::-1],
        ),
    )
    advantages = reverse_advantages[::-1]
    return advantages, advantages + rollout.old_value


def flatten_training_batch(
    rollout: RolloutBatch,
    advantages: jax.Array,
    returns: jax.Array,
) -> TrainingBatch:
    """Flatten time and environment axes for PPO minibatches."""
    batch_size = rollout.reward.shape[0] * rollout.reward.shape[1]

    def flatten(value):
        return value.reshape((batch_size,) + value.shape[2:])

    flat_advantages = flatten(advantages)
    flat_advantages = (
        flat_advantages - jnp.mean(flat_advantages)
    ) / (jnp.std(flat_advantages) + 1.0e-8)
    return TrainingBatch(
        observation=flatten(rollout.observation),
        valid_rows=flatten(rollout.valid_rows),
        valid_cols=flatten(rollout.valid_cols),
        operation=flatten(rollout.operation),
        row=flatten(rollout.row),
        col=flatten(rollout.col),
        old_log_probability=flatten(rollout.old_log_probability),
        old_value=flatten(rollout.old_value),
        advantage=flat_advantages,
        return_value=flatten(returns),
    )


def make_minibatch_update(
    model: PolicyModel,
    config: BenchmarkConfig,
):
    """Build the jitted update shared by both environment backends."""

    @jax.jit
    def update(
        train_state: TrainState,
        batch: TrainingBatch,
    ) -> tuple[TrainState, LossMetrics]:
        def loss_fn(params):
            log_probability, entropy, value = action_statistics(
                model,
                params,
                batch.observation,
                batch.valid_rows,
                batch.valid_cols,
                batch.operation,
                batch.row,
                batch.col,
            )
            log_ratio = log_probability - batch.old_log_probability
            ratio = jnp.exp(log_ratio)
            unclipped = ratio * batch.advantage
            clipped = (
                jnp.clip(
                    ratio,
                    1.0 - config.clip_epsilon,
                    1.0 + config.clip_epsilon,
                )
                * batch.advantage
            )
            policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))

            clipped_value = batch.old_value + jnp.clip(
                value - batch.old_value,
                -config.clip_epsilon,
                config.clip_epsilon,
            )
            value_loss = 0.5 * jnp.mean(
                jnp.maximum(
                    jnp.square(value - batch.return_value),
                    jnp.square(clipped_value - batch.return_value),
                )
            )
            mean_entropy = jnp.mean(entropy)
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * mean_entropy
            )
            approximate_kl = jnp.mean((ratio - 1.0) - log_ratio)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > config.clip_epsilon).astype(jnp.float32)
            )
            metrics = LossMetrics(
                loss=loss,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=mean_entropy,
                approximate_kl=approximate_kl,
                clip_fraction=clip_fraction,
            )
            return loss, metrics

        (_, metrics), gradients = jax.value_and_grad(
            loss_fn, has_aux=True
        )(train_state.params)
        return train_state.apply_gradients(grads=gradients), metrics

    return update


def make_rollout_update(
    model: PolicyModel,
    config: BenchmarkConfig,
):
    """Compile GAE, shuffling, PPO epochs, and all minibatch updates."""
    update_minibatch = make_minibatch_update(model, config)
    batch_size = config.steps_per_update
    minibatch_size = config.minibatch_size

    @jax.jit
    def update(
        train_state: TrainState,
        rollout: RolloutBatch,
        last_value: jax.Array,
        key: jax.Array,
    ) -> tuple[TrainState, jax.Array, LossMetrics]:
        advantages, returns = calculate_gae(
            rollout,
            last_value,
            config.gamma,
            config.gae_lambda,
        )
        batch = flatten_training_batch(rollout, advantages, returns)

        def update_epoch(carry, _unused):
            epoch_state, epoch_key = carry
            epoch_key, permutation_key = jax.random.split(epoch_key)
            permutation = jax.random.permutation(
                permutation_key,
                batch_size,
            ).reshape((config.num_minibatches, minibatch_size))

            def update_one_minibatch(state, indices):
                return update_minibatch(
                    state,
                    select_minibatch(batch, indices),
                )

            epoch_state, epoch_metrics = jax.lax.scan(
                update_one_minibatch,
                epoch_state,
                permutation,
            )
            return (epoch_state, epoch_key), epoch_metrics

        (train_state, key), metrics = jax.lax.scan(
            update_epoch,
            (train_state, key),
            xs=None,
            length=config.epochs,
        )
        mean_metrics = jax.tree_util.tree_map(
            lambda value: jnp.mean(value, axis=(0, 1)),
            metrics,
        )
        return train_state, key, mean_metrics

    return update


def select_minibatch(
    batch: TrainingBatch,
    indices: jax.Array,
) -> TrainingBatch:
    """Index every field of a training batch."""
    return jax.tree_util.tree_map(lambda value: value[indices], batch)
