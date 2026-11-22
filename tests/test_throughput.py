"""The simulator throughput benchmark (strategy-v4 §14)."""

from __future__ import annotations

import pytest

from jaxarc import JaxArcConfig
from jaxarc.registration import available_task_ids
from jaxarc_baselines import throughput


def mini_available() -> bool:
    return bool(available_task_ids("Mini", config=JaxArcConfig()))


needs_mini = pytest.mark.skipif(not mini_available(), reason="No Mini tasks available")


@needs_mini
@pytest.mark.parametrize("stack", ["bare", "training"])
@pytest.mark.parametrize(("interface", "actions"), [("point", 276), ("bbox", 25 * 25 * 11 + 1)])
def test_jaxarc_point_runs_every_stack_and_counts_its_steps(stack, interface, actions):
    result = throughput.measure_jaxarc(
        dataset="mini", interface=interface, stack=stack, num_envs=4, steps=8, repeats=2
    )

    assert result["num_actions"] == actions
    assert result["steps_timed"] == 4 * 8
    assert len(result["times_s"]) == 2
    assert result["sps_median"] > 0
    assert result["compile_s"] > 0
    assert result["devices"] == 1


@needs_mini
def test_training_stack_is_the_stoix_wrapper_chain():
    env, num_actions = throughput.jaxarc_env("mini", "point", "training", num_envs=4)

    # Stoix's chain ends in a vmapped, auto-resetting env over num_envs.
    assert type(env).__name__ != "PointActionWrapper"
    assert num_actions == 276


def test_multi_process_sps_counts_every_step_over_the_slowest_process():
    total = throughput.combine_processes(
        steps_per_process=[1000, 1000], seconds_per_process=[1.0, 2.0]
    )

    assert total == pytest.approx(2000 / 2.0)


@needs_mini
def test_arcle_runs_bbox_on_the_same_canvas():
    pytest.importorskip("arcle")
    result = throughput.measure_arcle(
        dataset="mini", num_envs=2, steps=4, repeats=1, mode="sync"
    )

    assert result["backend"] == "arcle"
    assert result["canvas"] == 5
    assert result["steps_timed"] == 2 * 4
    assert result["sps_median"] > 0
