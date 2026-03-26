"""
Main entry point for running JaxARC experiments with Stoix.

This script relies on Stoix's official JaxARC support natively via `jaxarc.stoix_adapter.make_jaxarc_env`.

The only monkey-patch remaining is to inject custom metrics into the StoixLogger,
as Stoix's FF PPO hardcodes `StoixLogger(config)` without accepting custom_metrics_fn
via configuration.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

# 1. Import our custom metrics function and monkey-patch StoixLogger
from stoix.utils import logger as stoix_logger_module

from jaxarc_baselines.metrics import combined_custom_metrics

# Save original StoixLogger class
_OriginalStoixLogger = stoix_logger_module.StoixLogger


class JaxARCStoixLogger(_OriginalStoixLogger):
    """Wrapper around StoixLogger that uses our custom metrics by default."""

    def __init__(self, config: DictConfig, custom_metrics_fn=None) -> None:
        # Use our custom metrics function if none is provided
        if custom_metrics_fn is None:
            custom_metrics_fn = combined_custom_metrics
        super().__init__(config, custom_metrics_fn)


# Monkey-patch the StoixLogger class
stoix_logger_module.StoixLogger = JaxARCStoixLogger


# 2. Bootstrap JaxARC registry with all local subset configurations
from jaxarc.registration.subset_loader import load_all_subsets_for_dataset  # noqa: E402

for ds in ["Mini", "Concept", "AGI1", "AGI2"]:
    load_all_subsets_for_dataset(ds)

# 3. Define the main experiment entry point using Hydra.
# By making the system configurable, you can switch between PPO, DPO, etc.
# Defaulting to ff_ppo here for backwards compatibility.
@hydra.main(
    config_path="experiments/configs",
    config_name="default_ppo_jaxarc.yaml",
    version_base="1.2",
)
def run(cfg: DictConfig) -> float:
    """
    Runs the experiment using the composed Hydra configuration.
    """
    # Allow dynamic attributes to be added to the config, matching stoix's behavior.
    OmegaConf.set_struct(cfg, False)

    # Lazily import the requested system (defaults to ff_ppo)
    system = cfg.system.system_name
    if system in {"ff_ppo", "ppo"}:
        from stoix.systems.ppo.anakin import ff_ppo as runner
    elif system == "ff_dpo":
        from stoix.systems.ppo.anakin import ff_dpo_continuous as runner
    else:
        # Fallback to ff_ppo if the system is not well mapped here
        from stoix.systems.ppo.anakin import ff_ppo as runner

    # The runner will now use our custom StoixLogger with extended metrics processing
    return runner.run_experiment(cfg)


if __name__ == "__main__":
    run()
