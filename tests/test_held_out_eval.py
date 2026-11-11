"""Held-out evaluation through Stoix's own evaluator, with a random policy."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from stoix.evaluator import evaluator_setup
from stoix.utils.make_env import make

from jaxarc import JaxArcConfig
from jaxarc.evaluation import summarize
from jaxarc.registration import available_task_ids
from jaxarc.registration.subset_loader import load_all_subsets_for_dataset
from jaxarc_baselines.held_out import HeldOutRecorder, summarize_eval_metrics

CONFIGS = Path(__file__).resolve().parents[1] / "experiments" / "configs"
ATTEMPTS = 2


@pytest.fixture(scope="module")
def evaluation():
    if not available_task_ids("Mini", config=JaxArcConfig()):
        pytest.skip("No Mini tasks available")
    load_all_subsets_for_dataset("Mini")
    with initialize_config_dir(str(CONFIGS), version_base="1.2"):
        config = compose(
            "default_ppo_jaxarc",
            overrides=["logger=logger", "env.scenario.name=Mini-all"],
        )
    OmegaConf.set_struct(config, False)
    config.environment.max_episode_steps = 20
    config.env.eval_reset_fn = {"_target_": "jaxarc.evaluation.make_eval_reset_fn"}

    _, eval_env = make(config)
    params = eval_env.unwrapped.params
    tasks, pairs = params.all_pairs()
    config.arch.num_eval_episodes = ATTEMPTS * int(tasks.shape[0])
    num_actions = eval_env.action_space().num_values

    def random_act_fn(_, observation, key):
        return jax.random.randint(key, (observation.shape[0],), 0, num_actions)

    evaluator, _, (trained_params, eval_keys) = evaluator_setup(
        eval_env,
        jax.random.PRNGKey(0),
        random_act_fn,
        {"w": jnp.zeros((1, 1))},
        config,
    )
    output = evaluator(trained_params, eval_keys)
    return output.episode_metrics, tasks, pairs


def test_every_held_out_pair_is_run_once_per_attempt(evaluation):
    metrics, tasks, pairs = evaluation
    ran = np.stack([metrics["task_idx"].reshape(-1), metrics["pair_idx"].reshape(-1)], 1)
    expected = np.tile(np.stack([tasks, pairs], 1), (ATTEMPTS, 1))
    assert np.array_equal(ran, expected)


def test_outcomes_agree_with_the_episode_ends(evaluation):
    metrics, _, _ = evaluation
    exact = metrics["exact"].reshape(-1)
    submitted = metrics["submitted"].reshape(-1)
    accuracy = metrics["cell_accuracy"].reshape(-1)
    length = metrics["episode_length"].reshape(-1)

    assert not np.any(exact & ~submitted)
    assert np.all(accuracy[~submitted] == 0.0)
    # An episode that never submits runs to the step limit.
    assert np.all(length[~submitted] == 20)
    assert np.all((accuracy >= 0.0) & (accuracy <= 1.0))


def test_summary_covers_every_held_out_task(evaluation):
    metrics, tasks, _ = evaluation
    summary = summarize(metrics, k=ATTEMPTS)

    assert summary["num_tasks"] == len(np.unique(np.asarray(tasks)))
    assert summary["min_attempts"] == ATTEMPTS
    assert 0.0 <= summary["exact"] <= summary["pass@2"] <= 1.0


def test_logged_metrics_hold_task_scores_not_episode_arrays(evaluation):
    metrics, _, _ = evaluation
    logged, summary = summarize_eval_metrics(dict(metrics))

    assert not {"task_idx", "pair_idx", "submitted"} & set(logged)
    assert float(logged["exact"]) == summary["exact"]
    assert float(logged["pass@2"]) == summary["pass@2"]
    assert "episode_return" in logged


def test_training_metrics_pass_through_unchanged():
    metrics = {"episode_return": np.zeros(4)}
    logged, summary = summarize_eval_metrics(metrics)
    assert logged is metrics
    assert summary is None


def test_recorder_appends_one_line_per_evaluation(evaluation, tmp_path):
    metrics, _, _ = evaluation
    _, summary = summarize_eval_metrics(dict(metrics))
    recorder = HeldOutRecorder(tmp_path, seed=3, scenario="Mini-all")
    recorder.record(summary, t=100, eval_step=0, event="evaluator")
    recorder.record(summary, t=200, eval_step=1, event="evaluator")

    lines = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [line["t"] for line in lines] == [100, 200]
    assert lines[0]["seed"] == 3
    assert len(lines[0]["per_task"]["exact"]) == summary["num_tasks"]
