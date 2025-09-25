"""
Main entry point for running JaxARC experiments with Stoix.
This script acts as a bridge, initializing a custom environment factory
and then calling the desired Stoix system's main training function.
"""
import hydra
from omegaconf import DictConfig, OmegaConf

# 1. Import the original module and save its make function.
from stoix.utils import make_env as stoix_make_env_module

# 2. Import our factory and create the custom make function.
from utils.make_env import get_custom_make_fn

# 3. Monkey-patch the `make` function in the stoix module.
stoix_make_env_module.make = get_custom_make_fn(stoix_make_env_module.make)


# 4. Now that the patch is in place, we can import the Stoix system.
# Any call to `stoix.utils.make_env.make` inside ff_ppo will now call our custom version.
from stoix.systems.ppo.anakin import ff_ppo  # noqa: E402


# 5. Define the main experiment entry point using Hydra.
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

    # The `ff_ppo.run_experiment` function will now use our custom `make_env`
    # factory, which knows how to create JaxARC environments.
    return ff_ppo.run_experiment(cfg)


if __name__ == "__main__":
    run()
