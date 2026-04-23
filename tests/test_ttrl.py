from __future__ import annotations

import unittest

import jax.numpy as jnp

from jaxarc.types import EPISODE_MODE_TEST, EPISODE_MODE_TRAIN, JaxArcTask
from jaxarc_baselines.ttrl import (
    _normalize_budget_list,
    _positive_budget_chunk,
    _query_row_priority,
    make_single_pair_task,
    score_final_grid,
)


def _dummy_task() -> JaxArcTask:
    grid_shape = (2, 3, 4)
    mask_shape = (2, 3, 4)
    train_inputs = jnp.arange(24, dtype=jnp.int32).reshape(grid_shape)
    train_outputs = (train_inputs + 1).astype(jnp.int32)
    test_inputs = (train_inputs + 10).astype(jnp.int32)
    test_outputs = (train_inputs + 20).astype(jnp.int32)
    train_masks = jnp.ones(mask_shape, dtype=jnp.bool_)
    test_masks = jnp.ones(mask_shape, dtype=jnp.bool_)
    return JaxArcTask(
        input_grids_examples=train_inputs,
        input_masks_examples=train_masks,
        output_grids_examples=train_outputs,
        output_masks_examples=train_masks,
        num_train_pairs=2,
        test_input_grids=test_inputs,
        test_input_masks=test_masks,
        true_test_output_grids=test_outputs,
        true_test_output_masks=test_masks,
        num_test_pairs=2,
        task_index=jnp.asarray(7, dtype=jnp.int32),
    )


class HiddenQueryScoringTests(unittest.TestCase):
    def test_budget_helpers_normalize_and_chunk(self) -> None:
        budgets = [50000, 0, 10000, 100000, 50000]
        self.assertEqual(_normalize_budget_list(budgets), [0, 10000, 50000, 100000])
        self.assertEqual(_positive_budget_chunk(_normalize_budget_list(budgets)), 10000)

    def test_query_row_priority_prefers_exact_then_similarity(self) -> None:
        lower_similarity = {
            "exact_success": 0.0,
            "final_grid_match": 0.0,
            "similarity": 0.5,
            "submitted": 1.0,
            "episode_return": 3.0,
            "episode_length": 10.0,
        }
        higher_similarity = {
            "exact_success": 0.0,
            "final_grid_match": 0.0,
            "similarity": 0.8,
            "submitted": 0.0,
            "episode_return": 1.0,
            "episode_length": 20.0,
        }
        exact_match = {
            "exact_success": 1.0,
            "final_grid_match": 1.0,
            "similarity": 1.0,
            "submitted": 1.0,
            "episode_return": 0.0,
            "episode_length": 50.0,
        }
        self.assertGreater(
            _query_row_priority(higher_similarity),
            _query_row_priority(lower_similarity),
        )
        self.assertGreater(
            _query_row_priority(exact_match), _query_row_priority(higher_similarity)
        )

    def test_score_final_grid_requires_submit_for_exact_success(self) -> None:
        target_grid = jnp.array([[1, 2], [3, 4]], dtype=jnp.int32)
        target_mask = jnp.array([[True, True], [True, True]], dtype=jnp.bool_)
        score = score_final_grid(
            working_grid=target_grid,
            working_mask=target_mask,
            target_grid=target_grid,
            target_mask=target_mask,
            submitted=False,
        )
        self.assertTrue(score.final_grid_match)
        self.assertFalse(score.exact_success)
        self.assertEqual(score.similarity, 1.0)

    def test_make_single_pair_task_preserves_selected_hidden_pair(self) -> None:
        task = _dummy_task()
        single_query_task = make_single_pair_task(
            task,
            pair_index=1,
            episode_mode=EPISODE_MODE_TEST,
        )
        self.assertEqual(int(single_query_task.num_train_pairs), 0)
        self.assertEqual(int(single_query_task.num_test_pairs), 1)
        self.assertTrue(
            bool(
                jnp.all(
                    single_query_task.test_input_grids[0] == task.test_input_grids[1]
                )
            )
        )
        self.assertTrue(
            bool(
                jnp.all(
                    single_query_task.true_test_output_grids[0]
                    == task.true_test_output_grids[1]
                )
            )
        )

    def test_make_single_pair_task_preserves_selected_support_pair(self) -> None:
        task = _dummy_task()
        single_support_task = make_single_pair_task(
            task,
            pair_index=0,
            episode_mode=EPISODE_MODE_TRAIN,
        )
        self.assertEqual(int(single_support_task.num_train_pairs), 1)
        self.assertEqual(int(single_support_task.num_test_pairs), 0)
        self.assertTrue(
            bool(
                jnp.all(
                    single_support_task.output_grids_examples[0]
                    == task.output_grids_examples[0]
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
