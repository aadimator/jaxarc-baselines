"""Tests that TTRL evaluates with the same actions it trained on."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jaxarc import JaxArcConfig
from jaxarc.registration import available_task_ids
from jaxarc.registration.subset_loader import load_all_subsets_for_dataset
from jaxarc.stoix_adapter import make_jaxarc_env
from jaxarc.types import StepType
from jaxarc_baselines.ttrl import (
    load_single_task,
    make_raw_eval_env,
    make_single_pair_task,
    resolve_task_ids,
)

CONFIGS = Path(__file__).resolve().parents[1] / "experiments" / "configs"
RESIZE_CHOICE, SUBMIT_CHOICE = 10, 11
CELLS = 25  # Mini grids are 5x5.


@pytest.fixture(scope="module", params=["raw", "full"])
def envs(request):
    if not available_task_ids("Mini", config=JaxArcConfig()):
        pytest.skip("No Mini tasks available")
    load_all_subsets_for_dataset("Mini")
    with initialize_config_dir(str(CONFIGS), version_base="1.2"):
        config = compose(
            "ttrl_benchmark", overrides=["logger=logger", f"action={request.param}"]
        )
    OmegaConf.set_struct(config, False)

    task_id = resolve_task_ids(config)[0]
    config.env.scenario.name = f"Mini-{task_id}"
    train_env, _ = make_jaxarc_env(config)
    task = make_single_pair_task(
        load_single_task(config, task_id), pair_index=0, episode_mode=0
    )
    return train_env, make_raw_eval_env(config, task, episode_mode=0), request.param


def test_eval_env_has_the_training_action_count(envs):
    train_env, eval_env, _ = envs

    assert int(eval_env.action_space().num_values) == int(
        train_env.action_space().num_values
    )


def test_eval_env_can_resize_and_submit(envs):
    _, eval_env, profile = envs
    if profile != "raw":
        pytest.skip("The choice indices are for the raw operations")
    state, _ = eval_env.reset(jax.random.PRNGKey(0))

    resized, _ = eval_env.step(state, jnp.asarray(RESIZE_CHOICE * CELLS))
    assert int(resized.working_grid_mask.sum()) == 1

    _, timestep = eval_env.step(state, jnp.asarray(SUBMIT_CHOICE * CELLS))
    assert int(timestep.step_type) == int(StepType.TERMINATED)


def test_every_eval_action_matches_the_training_action(envs):
    train_env, eval_env, _ = envs
    state, _ = eval_env.reset(jax.random.PRNGKey(0))
    actions = jnp.arange(int(train_env.action_space().num_values))

    def chosen(env):
        return jax.vmap(lambda a: env.step(state, a)[1].extras["canonical_action"])

    with jax.disable_jit(False):
        trained = jax.jit(chosen(train_env))(actions)
        evaluated = jax.jit(chosen(eval_env))(actions)

    assert jnp.array_equal(trained["operation"], evaluated["operation"])
    assert jnp.array_equal(trained["selection"], evaluated["selection"])
