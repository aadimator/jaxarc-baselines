"""
Copyright (c) 2025 Aadam. All rights reserved.

JaxARC-Baselines: Baselines for the JaxARC environment.

A collection of baseline implementations and utilities for training RL agents
on JaxARC environments using the Stoix framework.
"""

from __future__ import annotations

# Metrics
from jaxarc_baselines.metrics import (
    combined_custom_metrics,
    jaxarc_extended_metrics,
)

# Utils
from jaxarc_baselines.utils import get_custom_make_fn

# Wrappers
from jaxarc_baselines.wrappers import ExtendedMetrics, ExtendedMetricsState

__version__ = "0.1.0"

__all__ = [
    # Wrappers
    "ExtendedMetrics",
    "ExtendedMetricsState",
    # Package metadata
    "__version__",
    # Metrics
    "combined_custom_metrics",
    # Utils
    "get_custom_make_fn",
    "jaxarc_extended_metrics",
]
