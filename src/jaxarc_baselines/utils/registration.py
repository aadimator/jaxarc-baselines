"""
Enhanced JaxARC registration with lazy subset loading.

This module provides a wrapper around jaxarc.registration.make that adds
lazy loading of task subsets from YAML files, without modifying JaxARC itself.

Usage:
    # Import the enhanced make function instead of the original
    from jaxarc_baselines.registration import make

    # Use exactly like the original, but with automatic subset loading
    env, params = make("Mini-easy", auto_download=True)
"""

from __future__ import annotations

from typing import Any

from jaxarc.registration import (
    available_named_subsets,
    register_subset,
    subset_task_ids,
)
from jaxarc.registration import make as _jaxarc_make
from loguru import logger

from jaxarc_baselines.utils.subset_loader import load_subset_if_needed

# Re-export for convenience
__all__ = [
    "available_named_subsets",
    "make",
    "register_subset",
    "subset_task_ids",
]


def make(id: str, **kwargs: Any) -> tuple[Any, Any]:
    """
    Enhanced version of jaxarc.registration.make with lazy subset loading.

    This wrapper automatically attempts to load subsets from YAML files if they
    are not already registered. This allows you to define subsets in:
        configs/env/jaxarc/subsets/{Dataset}/{subset_name}.yaml

    and use them without manual registration:
        env, params = make("Mini-easy")

    Args:
        id: Environment ID in format "{Dataset}-{selector}" or just "{Dataset}"
        **kwargs: Additional arguments passed to jaxarc.registration.make

    Returns:
        (env, params) tuple from jaxarc.registration.make

    Examples:
        >>> # Load a custom subset (lazy loads from YAML if needed)
        >>> env, params = make("Mini-easy")

        >>> # Load all tasks (no YAML needed)
        >>> env, params = make("Mini-all")

        >>> # Load a single task (no YAML needed)
        >>> env, params = make("Mini-task_id_123")
    """
    # Parse the ID to extract dataset and selector
    dataset_key, selector = _parse_id(id)

    # If there's a selector, try to load it from YAML if not already registered
    if selector:
        # Check if it's already registered
        available = available_named_subsets(dataset_key)
        if selector.lower() not in available:
            # Try to load from YAML
            logger.debug(f"Attempting lazy load of subset '{dataset_key}-{selector}'")
            loaded = load_subset_if_needed(dataset_key, selector)
            if loaded:
                logger.debug(
                    f"Successfully lazy loaded subset '{dataset_key}-{selector}'"
                )

    # Call the original make function
    return _jaxarc_make(id, **kwargs)


def _parse_id(id: str) -> tuple[str, str]:
    """
    Parse environment ID and extract dataset key and selector.

    Args:
        id: Environment ID (e.g., "Mini-easy", "AGI1-train", "Concept")

    Returns:
        (dataset_key, selector) tuple
    """
    tokens = id.split("-", 1)
    dataset_key = tokens[0]
    selector = tokens[1] if len(tokens) > 1 else ""
    return dataset_key, selector
