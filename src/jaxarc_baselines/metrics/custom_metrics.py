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
    
    # Get terminal step mask to filter for completed episodes (ACTOR training only)
    is_terminal_steps = metrics.get("is_terminal_step", np.array([]))
    is_terminal_steps = np.asarray(is_terminal_steps)
    
    n_episodes = int(np.sum(is_terminal_steps))
    if n_episodes == 0:
        return metrics
    
    # Extract extended metrics (only for terminal steps)
    solved = np.asarray(metrics.get("solved", np.array([])))[is_terminal_steps]
    steps_to_solve = np.asarray(metrics.get("steps_to_solve", np.array([])))[is_terminal_steps]
    best_similarity = np.asarray(metrics.get("best_similarity", np.array([])))[is_terminal_steps]
    final_similarity = np.asarray(metrics.get("final_similarity", np.array([])))[is_terminal_steps]
    was_truncated = np.asarray(metrics.get("was_truncated", np.array([])))[is_terminal_steps]
    
    # Calculate success rate (percentage of episodes solved)
    n_solved: int = int(np.sum(solved))
    success_rate: float = float((n_solved / n_episodes) * 100)
    
    # Calculate average steps to solve (only for solved episodes)
    if n_solved > 0:
        solved_mask = solved.astype(bool)
        avg_steps_to_solve: float = np.mean(steps_to_solve[solved_mask])
    else:
        avg_steps_to_solve: float = 0.0
    
    # Calculate truncation rate (percentage of episodes truncated)
    n_truncated: int = int(np.sum(was_truncated))
    truncation_rate: float = float((n_truncated / n_episodes) * 100)
    
    # Calculate average best and final similarity
    avg_best_similarity: float = float(np.mean(best_similarity))
    avg_final_similarity: float = float(np.mean(final_similarity))
    
    # Add derived metrics
    metrics["success_rate"] = success_rate
    metrics["avg_steps_to_solve"] = avg_steps_to_solve
    metrics["truncation_rate"] = truncation_rate
    metrics["avg_best_similarity"] = avg_best_similarity
    metrics["avg_final_similarity"] = avg_final_similarity
    
    # Remove raw extended metrics (they're already processed)
    # This keeps logs cleaner and focuses on the aggregate statistics
    metrics.pop("solved", None)
    metrics.pop("steps_to_solve", None)
    metrics.pop("was_truncated", None)
    # Remove best/final_similarity too since we have avg_ versions
    metrics.pop("best_similarity", None)
    metrics.pop("final_similarity", None)
    
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
