"""Custom metrics functions for JaxARC experiments.

This module provides custom metric functions that process raw episode metrics
into derived statistics (e.g., success_rate, avg_steps_to_solve).

These functions follow the same pattern as Stoix's solve_rate_custom_metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Metrics = dict[str, Any]


def jaxarc_extended_metrics(metrics: Metrics) -> Metrics:
    """Calculate extended metrics from JaxARC episodes.

    This function processes the extended metrics tracked by ExtendedMetrics wrapper:
    - Computes success_rate from 'solved' field
    - Computes average steps to solve for successful episodes
    - Computes truncation rate
    - Computes average best/final similarity

    Note: This only works for ACTOR training metrics because it relies on
    is_terminal_step to filter completed episodes. Stoix's evaluator doesn't
    extract wrapper metrics from timestep.extras, so extended metrics won't
    appear in EVALUATOR logs.

    Args:
        metrics: Dictionary of metrics containing extended metrics fields.

    Returns:
        Dictionary with processed extended metrics and original fields removed.
    """
    # Only process if we have extended metrics
    if "solved" not in metrics:
        return metrics

    # Extended metrics arrays contain data that's already been accumulated per-episode
    # by RecordEpisodeMetrics wrapper. We don't need to filter by is_terminal_step.
    # The arrays may be nested due to batching, so we flatten them before computing statistics.
    solved = np.asarray(metrics.get("solved", np.array([]))).flatten()
    steps_to_solve = np.asarray(metrics.get("steps_to_solve", np.array([]))).flatten()
    best_similarity = np.asarray(metrics.get("best_similarity", np.array([]))).flatten()
    final_similarity = np.asarray(
        metrics.get("final_similarity", np.array([]))
    ).flatten()
    was_truncated = np.asarray(metrics.get("was_truncated", np.array([]))).flatten()

    # Get number of episodes from the data
    n_episodes = len(solved)
    if n_episodes == 0:
        return metrics

    # Compute aggregate metrics (scalars) for binary/discrete values.
    # For these metrics (e.g., success_rate, truncation_rate), summary statistics like mean, std, min, and max are not meaningful,
    # so we only compute and report the single aggregate value rather than passing them to describe().
    n_solved = int(np.sum(solved))
    success_rate = float((n_solved / n_episodes) * 100)  # As percentage
    metrics["success_rate"] = success_rate

    truncation_rate = float((np.sum(was_truncated) / n_episodes) * 100)  # As percentage
    metrics["truncation_rate"] = truncation_rate

    # Average steps to solve (only for solved episodes, to avoid 0s skewing the mean)
    if n_solved > 0:
        solved_mask = solved.astype(bool)
        avg_steps_to_solve = float(np.mean(steps_to_solve[solved_mask]))
    else:
        avg_steps_to_solve = 0.0
    metrics["avg_steps_to_solve"] = avg_steps_to_solve

    # Keep similarity arrays so downstream processing (e.g., logging or analysis functions such as describe())
    # can compute full statistics (mean/std/min/max). These are continuous values where all statistics are meaningful.
    metrics["best_similarity"] = best_similarity
    metrics["final_similarity"] = final_similarity

    # Remove the old fields that we've processed
    metrics.pop("solved", None)
    metrics.pop("was_truncated", None)
    metrics.pop("steps_to_solve", None)

    return metrics


def combined_custom_metrics(metrics: Metrics) -> Metrics:
    """Combined custom metrics function that applies both solve_rate and extended metrics.

    This function processes both solve_rate (if solved_episode exists) and
    extended metrics from the ExtendedMetrics wrapper.

    Note: Extended metrics only apply to ACTOR (training) logs, not EVALUATOR logs,
    because Stoix's evaluator doesn't extract metrics from timestep.extras.

    Use this as the custom_metrics_fn in StoixLogger initialization:
        logger = StoixLogger(config, custom_metrics_fn=combined_custom_metrics)

    Args:
        metrics: Dictionary of metrics.

    Returns:
        Dictionary with all custom metrics computed.
    """
    # Apply solve_rate logic (inline to avoid import issues)
    if "solved_episode" in metrics:
        is_terminal_steps = np.asarray(metrics.get("is_terminal_step", np.array([])))
        n_episodes = int(np.sum(is_terminal_steps))
        if n_episodes > 0:
            n_solved_episodes = int(np.sum(np.asarray(metrics["solved_episode"])))
            solve_rate = float((n_solved_episodes / n_episodes) * 100)
            metrics["solve_rate"] = solve_rate
            metrics.pop("solved_episode")

    # Apply our extended metrics (only works for ACTOR training rollouts)
    return jaxarc_extended_metrics(metrics)


__all__ = ["combined_custom_metrics", "jaxarc_extended_metrics"]
