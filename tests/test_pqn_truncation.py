"""PQN's Q(lambda) targets on a hand-computed 3-step rollout.

One env, rewards 1, 2, 3, and max Q at each true next observation of 10, 20,
30. gamma = lambda = 0.5, so G_t = r + 0.5 * (0.5 * v + 0.5 * G_{t+1}), and the
last step bootstraps fully: G_2 = 3 + 0.5 * 30 = 18.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from stoix.systems.q_learning.ff_pqn import compute_q_lambda_targets

GAMMA = LAMBDA = 0.5


def targets(terminated, truncated):
    rewards = jnp.array([[1.0, 2.0, 3.0]])
    # Two actions; the max is the value that matters.
    q_next = jnp.array([[[10.0, 0.0], [0.0, 20.0], [30.0, 5.0]]])
    return compute_q_lambda_targets(
        rewards,
        q_next,
        jnp.array([terminated], dtype=jnp.float32),
        jnp.array([truncated], dtype=jnp.float32),
        GAMMA,
        LAMBDA,
    )[0].tolist()


def test_no_ending_is_the_plain_lambda_return():
    # G_1 = 2 + 0.5 * (10 + 9) = 11.5, G_0 = 1 + 0.5 * (5 + 5.75) = 6.375
    assert targets([0, 0, 0], [0, 0, 0]) == pytest.approx([6.375, 11.5, 18.0])


def test_termination_stops_the_bootstrap():
    # G_1 = 2, G_0 = 1 + 0.5 * (5 + 1) = 4
    assert targets([0, 1, 0], [0, 0, 0]) == pytest.approx([4.0, 2.0, 18.0])


def test_truncation_bootstraps_from_the_final_observation():
    # G_1 = 2 + 0.5 * 20 = 12, with no trace from the next episode.
    # G_0 = 1 + 0.5 * (5 + 6) = 6.5
    assert targets([0, 0, 0], [0, 1, 0]) == pytest.approx([6.5, 12.0, 18.0])


def test_truncation_on_the_last_step_bootstraps_as_usual():
    assert targets([0, 0, 0], [0, 0, 1]) == pytest.approx([6.375, 11.5, 18.0])


def test_each_env_gets_its_own_targets():
    rewards = jnp.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    q_next = jnp.array([[[10.0], [20.0], [30.0]], [[10.0], [20.0], [30.0]]])
    out = compute_q_lambda_targets(
        rewards,
        q_next,
        jnp.zeros((2, 3)),
        jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        GAMMA,
        LAMBDA,
    )
    assert out[0].tolist() == pytest.approx([6.5, 12.0, 18.0])
    assert out[1].tolist() == pytest.approx([6.375, 11.5, 18.0])
