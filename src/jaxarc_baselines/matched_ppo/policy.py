"""Shared Flax actor-critic used with both backends."""

from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from .backends import NUM_COMMON_OPERATIONS, SUBMIT_ACTION


class PolicyOutput(NamedTuple):
    operation_logits: jax.Array
    row_logits: jax.Array
    col_logits: jax.Array
    value: jax.Array


class SpatialPolicyOutput(NamedTuple):
    action_logits: jax.Array
    value: jax.Array


class PolicyAction(NamedTuple):
    operation: jax.Array
    row: jax.Array
    col: jax.Array
    log_probability: jax.Array
    value: jax.Array


class ActorCritic(nn.Module):
    """Factorized operation-row-column reference policy."""

    grid_size: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, observation: jax.Array) -> PolicyOutput:
        x = (observation.astype(jnp.float32) + 1.0) / 10.0
        x = nn.relu(
            nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        )
        x = nn.relu(
            nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
        )
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(nn.Dense(self.hidden_size)(x))
        return PolicyOutput(
            operation_logits=nn.Dense(NUM_COMMON_OPERATIONS)(x),
            row_logits=nn.Dense(self.grid_size)(x),
            col_logits=nn.Dense(self.grid_size)(x),
            value=nn.Dense(1)(x).squeeze(-1),
        )


class SpatialActorCritic(nn.Module):
    """Grid-aligned joint color-cell policy plus submit."""

    grid_size: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, observation: jax.Array) -> SpatialPolicyOutput:
        x = (observation.astype(jnp.float32) + 1.0) / 10.0
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3))(x))
        edit_logits = nn.Conv(features=10, kernel_size=(1, 1))(x)
        pooled = jnp.mean(x, axis=(1, 2))
        value_features = nn.relu(nn.Dense(self.hidden_size)(pooled))
        submit_logit = nn.Dense(1)(value_features)
        return SpatialPolicyOutput(
            action_logits=jnp.concatenate(
                [edit_logits.reshape((x.shape[0], -1)), submit_logit],
                axis=-1,
            ),
            value=nn.Dense(1)(value_features).squeeze(-1),
        )


PolicyModel = ActorCritic | SpatialActorCritic


def make_model(architecture: str, grid_size: int) -> PolicyModel:
    """Build one of the controlled policy architectures."""
    if architecture == "factorized":
        return ActorCritic(grid_size=grid_size)
    if architecture == "spatial":
        return SpatialActorCritic(grid_size=grid_size)
    raise ValueError(f"Unknown policy architecture: {architecture}")


def policy_description(model: PolicyModel) -> str:
    if isinstance(model, SpatialActorCritic):
        return "conv32-conv32-joint-color-cell"
    return "conv32-s2-conv32-s2-dense128-factorized-point"


def _mask_logits(logits: jax.Array, valid: jax.Array) -> jax.Array:
    return jnp.where(valid, logits, jnp.asarray(-1.0e9, dtype=logits.dtype))


def _categorical_log_probability(
    logits: jax.Array, action: jax.Array
) -> jax.Array:
    return jnp.take_along_axis(
        jax.nn.log_softmax(logits),
        action[..., None],
        axis=-1,
    ).squeeze(-1)


def _categorical_entropy(logits: jax.Array) -> jax.Array:
    probabilities = jax.nn.softmax(logits)
    log_probabilities = jax.nn.log_softmax(logits)
    return -jnp.sum(probabilities * log_probabilities, axis=-1)


def _spatial_logits(
    output: SpatialPolicyOutput,
    valid_rows: jax.Array,
    valid_cols: jax.Array,
) -> jax.Array:
    valid_cells = valid_rows[:, :, None] & valid_cols[:, None, :]
    valid_edits = jnp.broadcast_to(
        valid_cells[..., None],
        valid_cells.shape + (10,),
    ).reshape((valid_cells.shape[0], -1))
    valid_actions = jnp.concatenate(
        [
            valid_edits,
            jnp.ones((valid_edits.shape[0], 1), dtype=jnp.bool_),
        ],
        axis=-1,
    )
    return _mask_logits(output.action_logits, valid_actions)


def _decode_spatial_action(
    action_index: jax.Array,
    grid_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    submit_index = grid_size * grid_size * 10
    submitted = action_index == submit_index
    edit_index = jnp.minimum(action_index, submit_index - 1)
    operation = edit_index % 10
    cell = edit_index // 10
    row = cell // grid_size
    col = cell % grid_size
    return (
        jnp.where(submitted, SUBMIT_ACTION, operation).astype(jnp.int32),
        jnp.where(submitted, 0, row).astype(jnp.int32),
        jnp.where(submitted, 0, col).astype(jnp.int32),
    )


def _encode_spatial_action(
    operation: jax.Array,
    row: jax.Array,
    col: jax.Array,
    grid_size: int,
) -> jax.Array:
    submit_index = grid_size * grid_size * 10
    edit_index = ((row * grid_size + col) * 10) + operation
    return jnp.where(operation == SUBMIT_ACTION, submit_index, edit_index)


def select_action(
    model: PolicyModel,
    params,
    observation: jax.Array,
    valid_rows: jax.Array,
    valid_cols: jax.Array,
    key: jax.Array,
    *,
    deterministic: bool,
) -> PolicyAction:
    """Select a compact operation-row-column action."""
    output = model.apply(params, observation)
    if isinstance(model, SpatialActorCritic):
        logits = _spatial_logits(output, valid_rows, valid_cols)
        if deterministic:
            action_index = jnp.argmax(logits, axis=-1)
        else:
            action_index = jax.random.categorical(key, logits)
        operation, row, col = _decode_spatial_action(
            action_index,
            model.grid_size,
        )
        return PolicyAction(
            operation=operation,
            row=row,
            col=col,
            log_probability=_categorical_log_probability(
                logits,
                action_index,
            ),
            value=output.value,
        )

    row_logits = _mask_logits(output.row_logits, valid_rows)
    col_logits = _mask_logits(output.col_logits, valid_cols)
    operation_key, row_key, col_key = jax.random.split(key, 3)
    if deterministic:
        operation = jnp.argmax(output.operation_logits, axis=-1)
        row = jnp.argmax(row_logits, axis=-1)
        col = jnp.argmax(col_logits, axis=-1)
    else:
        operation = jax.random.categorical(
            operation_key, output.operation_logits
        )
        row = jax.random.categorical(row_key, row_logits)
        col = jax.random.categorical(col_key, col_logits)

    operation_log_probability = _categorical_log_probability(
        output.operation_logits, operation
    )
    edit_action = operation != SUBMIT_ACTION
    coordinate_log_probability = _categorical_log_probability(
        row_logits, row
    ) + _categorical_log_probability(col_logits, col)
    log_probability = operation_log_probability + jnp.where(
        edit_action, coordinate_log_probability, 0.0
    )
    return PolicyAction(
        operation=operation.astype(jnp.int32),
        row=row.astype(jnp.int32),
        col=col.astype(jnp.int32),
        log_probability=log_probability,
        value=output.value,
    )


def action_statistics(
    model: PolicyModel,
    params,
    observation: jax.Array,
    valid_rows: jax.Array,
    valid_cols: jax.Array,
    operation: jax.Array,
    row: jax.Array,
    col: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return log probability, entropy, and value for recorded actions."""
    output = model.apply(params, observation)
    if isinstance(model, SpatialActorCritic):
        logits = _spatial_logits(output, valid_rows, valid_cols)
        action_index = _encode_spatial_action(
            operation,
            row,
            col,
            model.grid_size,
        )
        return (
            _categorical_log_probability(logits, action_index),
            _categorical_entropy(logits),
            output.value,
        )

    row_logits = _mask_logits(output.row_logits, valid_rows)
    col_logits = _mask_logits(output.col_logits, valid_cols)
    operation_log_probability = _categorical_log_probability(
        output.operation_logits, operation
    )
    edit_action = operation != SUBMIT_ACTION
    coordinate_log_probability = _categorical_log_probability(
        row_logits, row
    ) + _categorical_log_probability(col_logits, col)
    log_probability = operation_log_probability + jnp.where(
        edit_action, coordinate_log_probability, 0.0
    )

    operation_entropy = _categorical_entropy(output.operation_logits)
    edit_probability = 1.0 - jax.nn.softmax(output.operation_logits)[
        ..., SUBMIT_ACTION
    ]
    coordinate_entropy = _categorical_entropy(
        row_logits
    ) + _categorical_entropy(col_logits)
    entropy = operation_entropy + edit_probability * coordinate_entropy
    return log_probability, entropy, output.value
