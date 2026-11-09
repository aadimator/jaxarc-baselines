"""Checks on the network and subset configs."""

from __future__ import annotations

from pathlib import Path

import hydra.utils
import jax
import jax.numpy as jnp
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = sorted((ROOT / "configs" / "network").glob("*.yaml"))
AGI1_SUBSETS = ROOT / "configs" / "env" / "jaxarc" / "subsets" / "AGI1"
# JaxARC observations are height x width x channels.
OBSERVATION = jnp.zeros((1, 5, 5, 12), dtype=jnp.float32)


def first_conv_kernel(params):
    """The kernel of the first convolution, found by name."""
    convs = sorted(
        (path, leaf)
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
        if "Conv_0" in jax.tree_util.keystr(path) and leaf.ndim == 4
    )
    return convs[0][1]


@pytest.mark.parametrize("path", NETWORKS, ids=lambda p: p.stem)
def test_networks_read_observation_channels_as_channels(path):
    config = yaml.safe_load(path.read_text())
    networks = [n for n in config.values() if isinstance(n, dict) and "pre_torso" in n]
    assert networks
    for network in networks:
        torso = hydra.utils.instantiate(network["pre_torso"])
        params = torso.init(jax.random.PRNGKey(0), OBSERVATION)

        assert first_conv_kernel(params).shape[2] == OBSERVATION.shape[-1]


def task_ids(name):
    return yaml.safe_load((AGI1_SUBSETS / f"{name}.yaml").read_text())["task_ids"]


@pytest.mark.parametrize(
    "level", ["entry", "easy", "medium", "hard", "tedious", "multiple_solutions"]
)
def test_each_level_is_its_train_and_eval_tasks(level):
    train, evaluation = set(task_ids(f"{level}_train")), set(task_ids(f"{level}_eval"))

    assert not train & evaluation
    assert set(task_ids(level)) == train | evaluation
    assert len(task_ids(level)) == len(train) + len(evaluation)
