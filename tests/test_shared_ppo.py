"""The shared-torso PPO variant: one torso, a policy and a value head, one optimizer."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig
from jaxarc.registration import available_task_ids
from jaxarc.registration.subset_loader import load_all_subsets_for_dataset
from jaxarc_baselines.shared_ppo import learner_setup, shared_ppo_loss
from jaxarc_baselines.spatial_network import (
    SpatialPolicyHead,
    SpatialTorso,
    SpatialValueHead,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_OPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 33, 34]


def compose_config(*overrides):
    with initialize_config_dir(
        str(ROOT / "experiments" / "configs"), version_base="1.2"
    ):
        config = compose(
            "default_ppo_jaxarc",
            overrides=[
                "system=ppo/ff_ppo_shared",
                "network=arc_spatial_shared",
                *overrides,
            ],
        )
    OmegaConf.set_struct(config, False)
    return config


def test_shared_system_config_has_one_learning_rate():
    system = compose_config().system

    assert system.system_name == "ff_ppo_shared"
    assert "lr" in system and "actor_lr" not in system and "critic_lr" not in system
    assert system.vf_coef == 0.5


def fixed_batch(network, params, size=5, batch=64):
    from stoix.systems.ppo.ppo_types import PPOTransition

    keys = jax.random.split(jax.random.PRNGKey(1), 4)
    obs = jax.random.randint(keys[0], (batch, size, size, 12), -1, 10).astype(
        jnp.float32
    )
    policy, value = network.apply(params, obs)
    action = policy.sample(seed=keys[1])
    zeros = jnp.zeros((batch,))
    transition = PPOTransition(
        done=zeros,
        truncated=zeros,
        action=action,
        value=value,
        reward=zeros,
        bootstrap_value=zeros,
        log_prob=policy.log_prob(action),
        obs=obs,
        info={},
    )
    advantages = jax.random.normal(keys[2], (batch,))
    targets = jax.random.normal(keys[3], (batch,))
    return transition, advantages, targets


def test_loss_decreases_on_a_fixed_batch():
    from stoix.networks.base import FeedForwardActorCritic

    network = FeedForwardActorCritic(
        torso=SpatialTorso(channels=16, num_blocks=2),
        action_head=SpatialPolicyHead(action_dim=276, allowed_operations=RAW_OPS),
        critic_head=SpatialValueHead(),
    )
    system = compose_config().system
    obs = jnp.zeros((1, 5, 5, 12))
    params = network.init(jax.random.PRNGKey(0), obs)
    batch = fixed_batch(network, params)
    loss = jax.jit(lambda p: shared_ppo_loss(p, network.apply, *batch, system))
    optimiser = optax.adam(1e-3)
    opt_state = optimiser.init(params)

    first, info = loss(params)
    for _ in range(30):
        grads, _ = jax.grad(lambda p: loss(p), has_aux=True)(params)
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
    last, _ = loss(params)

    assert set(info) >= {"actor_loss", "value_loss", "entropy", "total_loss"}
    assert float(last) < float(first)


def test_total_loss_is_policy_plus_weighted_value_minus_weighted_entropy():
    from stoix.networks.base import FeedForwardActorCritic

    network = FeedForwardActorCritic(
        torso=SpatialTorso(channels=8, num_blocks=1),
        action_head=SpatialPolicyHead(action_dim=276, allowed_operations=RAW_OPS),
        critic_head=SpatialValueHead(),
    )
    system = compose_config("system.vf_coef=0.25", "system.ent_coef=0.1").system
    params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, 5, 5, 12)))

    total, info = shared_ppo_loss(
        params, network.apply, *fixed_batch(network, params), system
    )

    expected = info["actor_loss"] + 0.25 * info["value_loss"] - 0.1 * info["entropy"]
    assert float(total) == pytest.approx(float(expected), rel=1e-5)


def test_learner_runs_a_few_updates_on_cpu():
    from stoix.utils import make_env as environments
    from stoix.utils.total_timestep_checker import check_total_timesteps

    if not available_task_ids("Mini", config=JaxArcConfig()):
        pytest.skip("No Mini tasks available")
    load_all_subsets_for_dataset("Mini")
    config = compose_config(
        "env.scenario.name=Mini-all",
        "arch.total_num_envs=4",
        "arch.total_timesteps=64",
        "arch.num_evaluation=1",
        "system.rollout_length=4",
        "system.epochs=2",
        "system.num_minibatches=2",
        "network.torso.channels=8",
        "network.torso.num_blocks=1",
    )
    config.num_devices = len(jax.devices())
    config = check_total_timesteps(config)
    env, _ = environments.make(config=config)

    learn, _, state = learner_setup(
        env, jax.random.split(jax.random.PRNGKey(0), 2), config
    )
    output = learn(state)
    jax.block_until_ready(output)

    before, after = jax.tree.leaves(state.params), jax.tree.leaves(
        output.learner_state.params
    )
    assert any(not jnp.allclose(a, b) for a, b in zip(before, after))
    for name in ("actor_loss", "value_loss", "entropy", "total_loss"):
        assert jnp.isfinite(output.train_metrics[name]).all(), name


def test_run_experiment_registers_the_shared_system():
    import ast
    import importlib

    tree = ast.parse((ROOT / "run_experiment.py").read_text())
    runners = next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and getattr(node.targets[0], "id", "") == "RUNNER_MODULES"
    )

    assert callable(importlib.import_module(runners["ff_ppo_shared"]).run_experiment)
