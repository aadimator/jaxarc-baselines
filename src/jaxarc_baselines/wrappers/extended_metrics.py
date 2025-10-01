"""Extended metrics wrapper for collecting additional episode statistics.

This wrapp        # Merge extended metrics into episode_metrics (not as a separate dict)
        episode_metrics = timestep.extras.get("episode_metrics", {})
        updated_episode_metrics = {
            **episode_metrics,
            "best_similarity": jnp.array(0.0, dtype=jnp.float32),
            "solved": jnp.array(False, dtype=bool),
            "steps_to_solve": jnp.array(0, dtype=jnp.int32),
            "final_similarity": jnp.array(0.0, dtype=jnp.float32),
            "was_truncated": jnp.array(False, dtype=bool),
        }

        # Update extras with merged episode_metrics
        new_extras = {**timestep.extras, "episode_metrics": updated_episode_metrics}
        timestep = timestep.replace(extras=new_extras)  # type: ignore[call-arg]the basic RecordEpisodeMetrics wrapper by tracking:
- Success rate (solved episodes)
- Best similarity achieved during episode
- Final similarity at episode end
- Steps to solve (for successful episodes)
- Truncation information
"""

from __future__ import annotations

import jax.numpy as jnp
from chex import Numeric, PRNGKey
from stoa.core_wrappers.wrapper import Wrapper, WrapperState, wrapper_state_replace
from stoa.env_types import Action, EnvParams, TimeStep
from stoa.stoa_struct import dataclass


@dataclass(custom_replace_fn=wrapper_state_replace)
class ExtendedMetricsState(WrapperState):
    """State for tracking extended episode metrics.

    Tracks additional metrics beyond basic return and length:
    - Best similarity achieved during the episode
    - Whether the episode was solved (similarity >= 1.0)
    - Steps taken when solved (0 if not solved)
    - Whether episode was truncated
    """

    # Running metrics during episode
    best_similarity: Numeric
    solved: Numeric  # Boolean: did we solve this episode?
    steps_to_solve: Numeric  # Steps when first solved (0 if not solved)

    # Final metrics (recorded on terminal/truncated step)
    final_similarity: Numeric
    was_truncated: Numeric  # Boolean: was episode truncated?


class ExtendedMetrics(Wrapper[ExtendedMetricsState]):
    """
    A wrapper that records extended episode metrics.

    This wrapper tracks additional metrics beyond episode return and length:
    - Success rate: Percentage of episodes where similarity >= 1.0
    - Best similarity: Highest similarity score achieved during episode
    - Final similarity: Similarity score at episode end
    - Steps to solve: Number of steps taken when first solving (0 if not solved)
    - Truncation info: Whether episode ended due to step limit

    Metrics are stored in the `extras` field of the `TimeStep` object under the
    key "extended_metrics" and are updated at each environment step.

    The metrics are reset at the beginning of each episode.
    """

    def reset(
        self, key: PRNGKey, env_params: EnvParams | None = None
    ) -> tuple[ExtendedMetricsState, TimeStep]:
        """
        Resets the environment and initializes extended metrics.

        Args:
            key: The random key for resetting the environment.
            env_params: Optional environment parameters.

        Returns:
            A tuple containing the initial state with initialized extended metrics
            and the initial TimeStep.
        """
        base_env_state, timestep = self._env.reset(key, env_params)

        # Initialize all extended metrics to zero/false
        state = ExtendedMetricsState(
            base_env_state=base_env_state,
            best_similarity=jnp.array(0.0, dtype=jnp.float32),
            solved=jnp.array(False, dtype=bool),
            steps_to_solve=jnp.array(0, dtype=jnp.int32),
            final_similarity=jnp.array(0.0, dtype=jnp.float32),
            was_truncated=jnp.array(False, dtype=bool),
        )

        # Initialize extended metrics in episode_metrics to maintain consistent pytree structure
        episode_metrics = timestep.extras.get("episode_metrics", {})
        episode_metrics = {
            **episode_metrics,
            "best_similarity": jnp.array(0.0, dtype=jnp.float32),
            "solved": jnp.array(False, dtype=bool),
            "steps_to_solve": jnp.array(0, dtype=jnp.int32),
            "final_similarity": jnp.array(0.0, dtype=jnp.float32),
            "was_truncated": jnp.array(False, dtype=bool),
        }

        # Update timestep extras
        new_extras = {**timestep.extras, "episode_metrics": episode_metrics}
        timestep = timestep.replace(extras=new_extras)  # type: ignore[call-arg]

        return state, timestep

    def step(
        self,
        state: ExtendedMetricsState,
        action: Action,
        env_params: EnvParams | None = None,
    ) -> tuple[ExtendedMetricsState, TimeStep]:
        """
        Steps the environment and updates extended metrics.

        Args:
            state: The current state, including extended metrics.
            action: The action to take in the environment.
            env_params: Optional environment parameters.

        Returns:
            A tuple containing the updated state and the new TimeStep,
            with updated extended metrics.
        """
        base_env_state, timestep = self._env.step(
            state.base_env_state, action, env_params
        )

        # Get current similarity from base state if available
        # Assuming base_env_state has similarity_score attribute (JaxARC specific)
        current_similarity = getattr(
            base_env_state, "similarity_score", jnp.array(0.0, dtype=jnp.float32)
        )

        # Check if episode is done
        done = timestep.done()

        # Check if episode was truncated (done but discount != 0)
        was_truncated = done & (timestep.discount != 0.0)

        # Track best similarity achieved so far
        new_best_similarity = jnp.maximum(state.best_similarity, current_similarity)

        # Check if solved (similarity >= 1.0)
        is_solved_now = current_similarity >= 1.0
        newly_solved = is_solved_now & ~state.solved

        # Get current step count from base state if available
        current_step = getattr(
            base_env_state, "step_count", jnp.array(0, dtype=jnp.int32)
        )

        # Update steps_to_solve only when first solved
        new_steps_to_solve = jnp.where(newly_solved, current_step, state.steps_to_solve)

        # Update solved flag
        new_solved = state.solved | is_solved_now

        # Record final values when episode ends
        final_similarity = jnp.where(done, current_similarity, state.final_similarity)
        final_was_truncated = jnp.where(done, was_truncated, state.was_truncated)

        # Reset metrics when episode ends, otherwise keep tracking
        # Use jnp.where to maintain type consistency (avoids bool->int32 conversion)
        reset_best_similarity = jnp.where(
            done, jnp.array(0.0, dtype=jnp.float32), new_best_similarity
        )
        reset_solved = jnp.where(done, jnp.array(False, dtype=bool), new_solved)
        reset_steps_to_solve = jnp.where(
            done, jnp.array(0, dtype=jnp.int32), new_steps_to_solve
        )

        # For logging: use final values on done, current tracking values otherwise
        log_best_similarity = jnp.where(
            done, new_best_similarity, state.best_similarity
        )
        log_solved = jnp.where(done, new_solved, state.solved)
        log_steps_to_solve = jnp.where(done, new_steps_to_solve, state.steps_to_solve)

        # Merge extended metrics into episode_metrics (not as a separate dict)
        # This ensures they're logged by Stoix's logging system
        episode_metrics = timestep.extras.get("episode_metrics", {})
        updated_episode_metrics = {
            **episode_metrics,
            "best_similarity": log_best_similarity,
            "solved": log_solved,
            "steps_to_solve": log_steps_to_solve,
            "final_similarity": final_similarity,
            "was_truncated": final_was_truncated,
        }

        # Update extras with merged episode_metrics
        new_extras = {**timestep.extras, "episode_metrics": updated_episode_metrics}
        timestep = timestep.replace(extras=new_extras)  # type: ignore[call-arg]

        # Create new state
        new_state = ExtendedMetricsState(
            base_env_state=base_env_state,
            best_similarity=reset_best_similarity,
            solved=reset_solved,
            steps_to_solve=reset_steps_to_solve,
            final_similarity=final_similarity,
            was_truncated=final_was_truncated,
        )

        return new_state, timestep


__all__ = ["ExtendedMetrics", "ExtendedMetricsState"]
