"""Shared spatial network for the Stoix baselines: one torso, policy, value and Q heads.

The torso is fully convolutional, so the same params work at any grid size. The
action heads match JaxARC's flat point actions (`SelectionAwareFlattenWrapper`):
index op_slot*H*W + row*W + col for each selection-dependent allowed operation,
then one action per selection-independent allowed operation (Copy input, Submit),
both groups in allowed-operations order.
"""

from __future__ import annotations

from collections.abc import Sequence

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import Initializer, orthogonal
from jaxarc.constants import NUM_OPERATIONS
from jaxarc.envs.grid_operations import SELECTION_INDEPENDENT_OPERATIONS
from tensorflow_probability.substrates.jax.distributions import Categorical

NUM_COLOURS = 10
# Padding (-1 in observations) gets its own category after colours 0-9.
PADDING_CATEGORY = NUM_COLOURS
DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}


def colour_categories(observation: jax.Array) -> jax.Array:
    """Colours 0-9 as themselves, padding (any negative value) as category 10."""
    categories = jnp.where(observation < 0, PADDING_CATEGORY, observation)
    return jnp.clip(categories, 0, PADDING_CATEGORY).astype(jnp.int32)


def split_operations(
    allowed_operations: Sequence[int] | None,
) -> tuple[list[int], list[int]]:
    """Selection-dependent and selection-independent operations, in allowed order.

    None means every operation, as when the env applies no operation filter.
    """
    operations = (
        list(range(NUM_OPERATIONS))
        if allowed_operations is None
        else [int(op) for op in allowed_operations]
    )
    dependent = [op for op in operations if op not in SELECTION_INDEPENDENT_OPERATIONS]
    independent = [op for op in operations if op in SELECTION_INDEPENDENT_OPERATIONS]
    return dependent, independent


def global_pool(x: jax.Array) -> jax.Array:
    """Mean and max over the grid, concatenated: (..., H, W, C) -> (..., 2C)."""
    return jnp.concatenate([x.mean(axis=(-3, -2)), x.max(axis=(-3, -2))], axis=-1)


class ColourEmbedding(nn.Module):
    """A learned embedding per observation channel: (..., C) -> (..., C * embed_dim).

    Channel c reads row 11c + category of one (11C, embed_dim) table. It is a
    one-hot matmul, not a gather: a gather's gradient is a scatter-add onto those
    few rows, which is very slow in bfloat16 on GPU.
    """

    embed_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        num_channels, categories = observation.shape[-1], PADDING_CATEGORY + 1
        table = self.param(
            "embedding",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal", out_axis=0),
            (num_channels * categories, self.embed_dim),
        )
        one_hot = jax.nn.one_hot(
            colour_categories(observation), categories, dtype=self.dtype
        )
        table = table.reshape(num_channels, categories, self.embed_dim).astype(
            self.dtype
        )
        x = jnp.einsum("...ck,cke->...ce", one_hot, table)
        return x.reshape(x.shape[:-2] + (num_channels * self.embed_dim,))


class SpatialTorso(nn.Module):
    """Residual conv blocks with pool-and-broadcast global context.

    Returns a (..., H, W, channels) feature map in the compute dtype. `dtype`
    sets the compute precision only; params stay float32. The defaults (64
    channels, 2 blocks, bfloat16) train about 2x faster than the stock CNN on an
    H100 at 30x30; 64 x 4 was slower than stock, and a shared 32 x 2 bfloat16
    torso did not learn on MiniARC.
    """

    channels: int = 64
    num_blocks: int = 2
    global_every: int = 1  # add the pooled summary back after every this many blocks
    colour_input: str = (
        "embed"  # "embed": learned per-channel embedding; "raw": values as-is
    )
    embed_dim: int = 8
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        if self.dtype not in DTYPES:
            raise ValueError(
                f"Unknown dtype {self.dtype!r}; use one of {sorted(DTYPES)}"
            )
        dtype = DTYPES[self.dtype]

        if self.colour_input == "embed":
            x = ColourEmbedding(self.embed_dim, dtype=dtype, name="colour_embedding")(
                observation
            )
        elif self.colour_input == "raw":
            x = observation.astype(dtype)
        else:
            raise ValueError(
                f"Unknown colour_input {self.colour_input!r}; use 'embed' or 'raw'"
            )

        conv = lambda: nn.Conv(self.channels, (3, 3), dtype=dtype)
        x = conv()(x)
        for block in range(self.num_blocks):
            y = conv()(nn.relu(x))
            x = x + conv()(nn.relu(y))
            if (block + 1) % self.global_every == 0:
                context = nn.Dense(self.channels, dtype=dtype)(global_pool(x))
                x = x + context[..., None, None, :]
        return nn.relu(x)


class SpatialActionScores(nn.Module):
    """One score per flat point action: a 1x1 conv per cell, a pooled MLP for the rest."""

    num_cell_ops: int
    num_global_ops: int
    hidden_size: int
    kernel_init: Initializer

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        # Heads run in float32 whatever the torso dtype: they are cheap, and the
        # logits and Q-values are what the losses read.
        features, dtype = features.astype(jnp.float32), jnp.float32
        cell = nn.Conv(
            self.num_cell_ops,
            (1, 1),
            kernel_init=self.kernel_init,
            dtype=dtype,
            name="cell",
        )(features)
        # (..., H, W, ops) -> (..., ops, H, W) -> op-major, then row-major cells.
        cell = jnp.moveaxis(cell, -1, -3).reshape(features.shape[:-3] + (-1,))

        hidden = nn.relu(
            nn.Dense(self.hidden_size, dtype=dtype, name="global_hidden")(
                global_pool(features)
            )
        )
        global_scores = nn.Dense(
            self.num_global_ops,
            kernel_init=self.kernel_init,
            dtype=dtype,
            name="global_out",
        )(hidden)
        return jnp.concatenate([cell, global_scores], axis=-1)


def _action_scores(head, features: jax.Array, kernel_init: Initializer) -> jax.Array:
    dependent, independent = split_operations(head.allowed_operations)
    height, width = features.shape[-3:-1]
    expected = len(dependent) * height * width + len(independent)
    if expected != head.action_dim:
        raise ValueError(
            f"The spatial head gives {expected} actions for {len(dependent)} selection-"
            f"dependent and {len(independent)} other operations on a {height}x{width} grid, "
            f"but the env has {head.action_dim}. It needs point actions and the env's "
            "allowed_operations."
        )
    return SpatialActionScores(
        len(dependent), len(independent), head.hidden_size, kernel_init, name="scores"
    )(features)


class SpatialPolicyHead(nn.Module):
    """One categorical over every flat point action; computed in float32."""

    action_dim: int
    allowed_operations: Sequence[int] | None = None
    hidden_size: int = 64
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, features: jax.Array) -> Categorical:
        return Categorical(logits=_action_scores(self, features, self.kernel_init))


class SpatialQHead(nn.Module):
    """One Q-value per flat point action, in the policy head's layout; float32."""

    action_dim: int
    epsilon: float = 0.1
    allowed_operations: Sequence[int] | None = None
    hidden_size: int = 64
    kernel_init: Initializer = orthogonal(1.0)

    @nn.compact
    def __call__(
        self, features: jax.Array, epsilon: float | None = None
    ) -> distrax.EpsilonGreedy:
        q_values = _action_scores(self, features, self.kernel_init)
        return distrax.EpsilonGreedy(
            preferences=q_values, epsilon=self.epsilon if epsilon is None else epsilon
        )


class SpatialValueHead(nn.Module):
    """Mean and max pooling, then a small MLP; computed in float32.

    The last layer starts near zero, so every seed starts with V(s) close to 0.
    With orthogonal(1.0) the initial value varied by about +-0.5 across seeds.
    Under pbrs a wrong submit pays minus the current similarity, so a seed with
    an optimistic critic punished early submits and PPO stopped submitting.
    """

    hidden_size: int = 64
    kernel_init: Initializer = orthogonal(0.01)

    @nn.compact
    def __call__(self, features: jax.Array) -> jax.Array:
        features = features.astype(jnp.float32)  # float32 head, as for the action scores
        hidden = nn.relu(nn.Dense(self.hidden_size)(global_pool(features)))
        return nn.Dense(1, kernel_init=self.kernel_init)(hidden).squeeze(-1)
