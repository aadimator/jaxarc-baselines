"""
Lazy subset loader for JaxARC tasks.

This module provides on-demand loading of task subsets from YAML configuration files.
Subsets are only loaded when requested, avoiding unnecessary overhead.

YAML file format (in configs/env/jaxarc/subsets/{Dataset}/{subset_name}.yaml):
    task_ids:
      - "task_id_1"
      - "task_id_2"
      - ...

Usage:
    from jaxarc_baselines.utils.subset_loader import load_subset_if_needed

    # This will check if 'easy' is a registered subset, and if not,
    # attempt to load it from configs/env/jaxarc/subsets/Mini/easy.yaml
    load_subset_if_needed("Mini", "easy")
"""

from __future__ import annotations

from pathlib import Path

import yaml
from jaxarc.registration import available_named_subsets, register_subset
from loguru import logger
from pyprojroot import here


def find_config_root() -> Path | None:
    """
    Find the configs directory using pyprojroot to locate project root.

    Returns:
        Path to configs directory, or None if not found
    """
    try:
        # Use pyprojroot to find project root (looks for pyproject.toml, .git, etc.)
        project_root = here()
        configs_dir = project_root / "configs"

        if configs_dir.exists() and configs_dir.is_dir():
            return configs_dir

        logger.debug(f"Project root found at {project_root}, but no configs directory")
        return None

    except Exception as e:
        logger.debug(f"Could not find project root using pyprojroot: {e}")
        return None


def load_subset_from_yaml(
    dataset_key: str, subset_name: str, config_root: Path | None = None
) -> bool:
    """
    Load a single subset from a YAML file and register it.

    Args:
        dataset_key: Dataset name (e.g., "Mini", "AGI1")
        subset_name: Subset name (e.g., "easy", "hard")
        config_root: Optional path to configs directory. If None, will search for it.

    Returns:
        True if subset was loaded successfully, False otherwise
    """
    if config_root is None:
        config_root = find_config_root()
        if config_root is None:
            logger.debug("Could not find configs directory for subset loading")
            return False

    # Construct path to subset YAML file
    subset_file = (
        config_root / "env" / "jaxarc" / "subsets" / dataset_key / f"{subset_name}.yaml"
    )

    if not subset_file.exists():
        logger.debug(f"Subset file not found: {subset_file}")
        return False

    try:
        with subset_file.open() as f:
            data = yaml.safe_load(f)

        # Skip if no task_ids or empty list
        if not data or "task_ids" not in data or not data["task_ids"]:
            logger.debug(
                f"Skipping empty subset {dataset_key}/{subset_name} from {subset_file}"
            )
            return False

        task_ids = data["task_ids"]

        # Register the subset
        register_subset(dataset_key, subset_name, task_ids)
        logger.info(
            f"Loaded subset '{dataset_key}-{subset_name}' with {len(task_ids)} tasks"
        )
        return True

    except Exception as e:
        logger.warning(f"Failed to load subset {dataset_key}/{subset_name}: {e}")
        return False


def load_subset_if_needed(
    dataset_key: str, subset_name: str, config_root: Path | None = None
) -> bool:
    """
    Check if a subset is already registered, and if not, try to load it from YAML.

    Args:
        dataset_key: Dataset name (e.g., "Mini", "AGI1")
        subset_name: Subset name (e.g., "easy", "hard")
        config_root: Optional path to configs directory. If None, will search for it.

    Returns:
        True if subset is available (either already registered or successfully loaded)
    """
    # Check if already registered
    available = available_named_subsets(dataset_key)
    if subset_name.lower() in available:
        logger.debug(f"Subset '{dataset_key}-{subset_name}' already registered")
        return True

    # Try to load from YAML
    logger.debug(f"Attempting to load subset '{dataset_key}-{subset_name}' from YAML")
    return load_subset_from_yaml(dataset_key, subset_name, config_root)


def load_all_subsets_for_dataset(
    dataset_key: str, config_root: Path | None = None
) -> int:
    """
    Load all subsets for a specific dataset.

    Args:
        dataset_key: Dataset name (e.g., "Mini", "AGI1")
        config_root: Optional path to configs directory. If None, will search for it.

    Returns:
        Number of subsets loaded
    """
    if config_root is None:
        config_root = find_config_root()
        if config_root is None:
            logger.debug("Could not find configs directory for subset loading")
            return 0

    dataset_dir = config_root / "env" / "jaxarc" / "subsets" / dataset_key

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        logger.debug(f"No subsets directory found for {dataset_key}: {dataset_dir}")
        return 0

    count = 0
    for yaml_file in dataset_dir.glob("*.yaml"):
        subset_name = yaml_file.stem
        if load_subset_from_yaml(dataset_key, subset_name, config_root):
            count += 1

    return count
