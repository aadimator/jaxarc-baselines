"""
Copyright (c) 2025 Aadam. All rights reserved.

JaxARC-Baselines: Baselines for the JaxARC environment.

A collection of baseline implementations and utilities for training RL agents
on JaxARC environments using the Stoix framework.
"""

from __future__ import annotations

# Metrics
from jaxarc_baselines.metrics import combined_custom_metrics
from jaxarc_baselines.ttrl import make_single_pair_task, run_ttrl_benchmark, score_final_grid

__version__ = "0.1.0"

__all__ = [
    # Package metadata
    "__version__",
    # Metrics
    "combined_custom_metrics",
    # TTRL benchmark helpers
    "make_single_pair_task",
    "run_ttrl_benchmark",
    "score_final_grid",
]
