"""Tests for the compute precision of the matched PPO policy (shared by both backends)."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxarc_baselines.matched_ppo.__main__ import build_parser
from jaxarc_baselines.matched_ppo.config import load_config
from jaxarc_baselines.matched_ppo.policy import make_model

CONFIG = Path(__file__).resolve().parents[1] / "experiments/configs/matched_ppo_arcagi1_30x30_h100.yaml"
GRID = 6


def observation():
    key = jax.random.PRNGKey(0)
    return jax.random.randint(key, (4, GRID, GRID, 13), -1, 10).astype(jnp.float32)


def test_precision_defaults_to_float32():
    assert load_config(CONFIG).precision == "float32"


def test_precision_override_changes_the_comparison_config():
    config = load_config(CONFIG).with_overrides(precision="bfloat16")

    assert config.precision == "bfloat16"
    assert config.as_dict()["precision"] == "bfloat16"


def test_unknown_precision_is_rejected():
    config = load_config(CONFIG).with_overrides(precision="float16")

    with pytest.raises(ValueError, match="precision"):
        config.validate()


def test_cli_accepts_precision():
    args = build_parser().parse_args(
        ["train", "--config", str(CONFIG), "--backend", "jaxarc", "--precision", "bfloat16"]
    )
    assert args.precision == "bfloat16"


@pytest.mark.parametrize("architecture", ["spatial", "factorized"])
def test_bfloat16_computes_in_bf16_but_keeps_float32_params_and_outputs(architecture):
    obs = observation()
    f32 = make_model(architecture, GRID)
    bf16 = make_model(architecture, GRID, precision="bfloat16")
    params = f32.init(jax.random.PRNGKey(1), obs)

    assert all(p.dtype == jnp.float32 for p in jax.tree.leaves(bf16.init(jax.random.PRNGKey(1), obs)))
    out32, out16 = f32.apply(params, obs), bf16.apply(params, obs)
    for a, b in zip(jax.tree.leaves(out32), jax.tree.leaves(out16)):
        assert b.dtype == jnp.float32
        assert jnp.allclose(a, b, atol=5e-2), float(jnp.abs(a - b).max())
    # bf16 really ran: the outputs are not bit-identical.
    assert any(not jnp.array_equal(a, b) for a, b in zip(jax.tree.leaves(out32), jax.tree.leaves(out16)))


def test_unknown_precision_in_make_model_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        make_model("spatial", GRID, precision="float16")
