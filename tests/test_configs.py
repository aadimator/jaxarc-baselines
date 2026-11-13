"""Checks on the network and subset configs."""

from __future__ import annotations

from pathlib import Path

import hydra.utils
import jax
import jax.numpy as jnp
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig

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


def input_channels_per_observation_channel(torso_config):
    """Conv inputs per observation channel: 1, or the embedding width."""
    if torso_config.get("colour_input") == "embed":
        return torso_config.embed_dim
    return 1


@pytest.mark.parametrize("path", NETWORKS, ids=lambda p: p.stem)
def test_networks_read_observation_channels_as_channels(path):
    # Network configs may interpolate from `network.*` and `action.*`.
    config = OmegaConf.create(
        {
            "network": yaml.safe_load(path.read_text()),
            "action": {"allowed_operations": None},
        }
    ).network
    networks = [n for n in config.values() if OmegaConf.is_dict(n) and "pre_torso" in n]
    assert networks
    for network in networks:
        torso = hydra.utils.instantiate(network.pre_torso)
        params = torso.init(jax.random.PRNGKey(0), OBSERVATION)
        per_channel = input_channels_per_observation_channel(network.pre_torso)

        assert first_conv_kernel(params).shape[2] == OBSERVATION.shape[-1] * per_channel


def compose_experiment(*overrides):
    with initialize_config_dir(
        str(ROOT / "experiments" / "configs"), version_base="1.2"
    ):
        config = compose("default_ppo_jaxarc", overrides=list(overrides))
    OmegaConf.set_struct(config, False)
    return config


def test_pbrs_preset_is_potential_shaping_with_no_penalties():
    reward = JaxArcConfig.from_hydra(compose_experiment("reward=pbrs")).reward

    assert reward.shaping == "potential"
    assert reward.step_penalty == 0.0
    assert reward.unsolved_submission_penalty == 0.0
    assert reward.similarity_weight == 1.0
    assert reward.success_bonus == 10.0


@pytest.mark.parametrize("gamma", [0.99, 0.95])
def test_pbrs_reward_gamma_follows_the_agent_gamma(gamma):
    config = compose_experiment("reward=pbrs", f"system.gamma={gamma}")

    assert JaxArcConfig.from_hydra(config).reward.gamma == gamma


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
