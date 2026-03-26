"""Custom metrics functions for JaxARC experiments."""

from __future__ import annotations

import numpy as np
from jaxarc.stoix_adapter import jaxarc_custom_metrics
from stoix.base_types import Metrics


def combined_custom_metrics(metrics: Metrics) -> Metrics:
    """Combined custom metrics function.

    This function processes both the standard Stoix solve_rate (if solved_episode
    exists in metrics) and the JaxARC extended metrics from the ExtendedMetrics
    wrapper (via JaxARC's official stoix_adapter).

    Note: Extended metrics only apply to ACTOR (training) logs, not EVALUATOR logs,
    because Stoix's evaluator doesn't currently extract metrics from timestep.extras.

    Args:
        metrics: Dictionary of metrics.

    Returns:
        Dictionary with all custom metrics computed.
    """
    # 1. Apply Stoix's standard solve_rate logic inline.
    if "solved_episode" in metrics:
        is_terminal_steps = np.asarray(metrics.get("is_terminal_step", np.array([])))
        n_episodes = int(np.sum(is_terminal_steps))
        if n_episodes > 0:
            n_solved_episodes = int(np.sum(np.asarray(metrics["solved_episode"])))
            solve_rate = float((n_solved_episodes / n_episodes) * 100)
            metrics["solve_rate"] = solve_rate
            metrics.pop("solved_episode")

    # 2. Add extra return distribution analysis (percentiles, high-return %)
    # This helps monitor training stability for complex tasks.
    episode_returns = np.asarray(metrics.get("episode_return", np.array([]))).flatten()
    if len(episode_returns) > 0:
        high_return_mask = episode_returns > 5.0
        metrics["pct_high_return_episodes"] = float(np.mean(high_return_mask) * 100)
        metrics["return_p25"] = float(np.percentile(episode_returns, 25))
        metrics["return_p50"] = float(np.percentile(episode_returns, 50))
        metrics["return_p75"] = float(np.percentile(episode_returns, 75))
        metrics["return_p95"] = float(np.percentile(episode_returns, 95))

    # 3. Apply JaxARC's official extended metrics processing.
    # It safely returns the dict unmodified if extended metrics aren't present.
    # Note: jaxarc_custom_metrics is JAX-based, so convert back/forth if needed,
    # though it currently handles numpy arrays fine.
    return jaxarc_custom_metrics(metrics)


__all__ = ["combined_custom_metrics"]
