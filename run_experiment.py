"""
Main entry point for running JaxARC experiments with Stoix.
This script acts as a bridge, initializing a custom environment factory
and then calling the desired Stoix system's main training function.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

# 1. Import the original module and save its make function.
from stoix.utils import make_env as stoix_make_env_module

# 2. Import our factory and create the custom make function.
from jaxarc_baselines.utils import get_custom_make_fn

# 3. Monkey-patch the `make` function in the stoix module.
stoix_make_env_module.make = get_custom_make_fn(stoix_make_env_module.make)

# 4. Import our custom metrics function and monkey-patch StoixLogger
# NOTE: These imports must come after make_env patch but before ff_ppo import
from stoix.utils import logger as stoix_logger_module  # noqa: E402

from jaxarc_baselines.metrics import combined_custom_metrics  # noqa: E402

# Save original StoixLogger class
_OriginalStoixLogger = stoix_logger_module.StoixLogger


# Create a wrapper class that injects our custom metrics function
class JaxARCStoixLogger(_OriginalStoixLogger):
    """Wrapper around StoixLogger that uses our custom metrics by default."""

    def __init__(self, config: DictConfig, custom_metrics_fn=None) -> None:
        # Use our custom metrics function if none is provided
        if custom_metrics_fn is None:
            custom_metrics_fn = combined_custom_metrics
        super().__init__(config, custom_metrics_fn)


# Monkey-patch the StoixLogger class
stoix_logger_module.StoixLogger = JaxARCStoixLogger

# 5. Now that both patches are in place, we can import the Stoix system.
# Any call to `stoix.utils.make_env.make` inside ff_ppo will now call our custom version.
# Any instantiation of StoixLogger will use our custom metrics function.
from stoix.systems.ppo.anakin import ff_ppo  # noqa: E402


# 6. Define the main experiment entry point using Hydra.
@hydra.main(
    config_path="experiments/configs",
    config_name="default_ppo_jaxarc.yaml",
    version_base="1.2",
)
def run(cfg: DictConfig) -> float:
    """
    Runs the PPO experiment using the composed Hydra configuration.
    """
    # Allow dynamic attributes to be added to the config, matching stoix's behavior.
    OmegaConf.set_struct(cfg, False)

    # The `ff_ppo.run_experiment` function will now use:
    # - Our custom `make_env` factory for JaxARC environments
    # - Our custom StoixLogger with extended metrics processing
    return ff_ppo.run_experiment(cfg)


if __name__ == "__main__":
    run()
