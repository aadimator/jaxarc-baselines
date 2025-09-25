from typing import Callable, Tuple

from omegaconf import DictConfig
from stoa.environment import Environment


def get_custom_make_fn(
    original_make_fn: Callable[[DictConfig], Tuple[Environment, Environment]]
) -> Callable[[DictConfig], Tuple[Environment, Environment]]:
    """
    Factory function that takes the original Stoix make_env function and
    returns a new, patched version that can handle the 'jaxarc' environment.
    """

    def custom_make(config: DictConfig) -> Tuple[Environment, Environment]:
        """
        Creates training and evaluation environments.

        This function checks if the requested environment is 'jaxarc'.
        If so, it uses the JaxARC registration and configuration system, and then
        applies the necessary wrappers to make it compatible with Stoix.
        """
        if config.env.env_name == "jaxarc":
            from jaxarc.configs import JaxArcConfig
            from jaxarc.envs import (
                BboxActionWrapper,
                PointActionWrapper,
                FlattenActionWrapper,
                AnswerObservationWrapper
            )
            from jaxarc.registration import make as make_jaxarc
            from stoa.core_wrappers.auto_reset import AutoResetWrapper
            from stoa.core_wrappers.episode_metrics import RecordEpisodeMetrics
            from stoa.core_wrappers.wrapper import AddRNGKey
            from stoix.utils.make_env import apply_core_wrappers

            jaxarc_config = JaxArcConfig.from_hydra(config)

            # 1. Create the base environment.
            env, _ = make_jaxarc(config.env.scenario.name, config=jaxarc_config)
            eval_env, _ = make_jaxarc(config.env.scenario.name, config=jaxarc_config)

            # 3. Apply the appropriate JaxARC action wrapper based on the config.
            action_mode = config.env.action.mode
            if action_mode == "point":
                env = PointActionWrapper(env)
                eval_env = PointActionWrapper(eval_env)
            elif action_mode == "bbox":
                env = BboxActionWrapper(env)
                eval_env = BboxActionWrapper(eval_env)
            elif action_mode != "mask":
                raise ValueError(f"Unknown action mode: {action_mode}")

            # 4. Flatten the resulting DictSpace into a single DiscreteSpace for Stoix.
            env = FlattenActionWrapper(env)
            eval_env = FlattenActionWrapper(eval_env)

            # 4.5. Add the answer to the observation for training.
            env = AnswerObservationWrapper(env)
            eval_env = AnswerObservationWrapper(eval_env)

            # 5. Apply the core wrappers from Stoix to the training environment.
            # This includes the VmapWrapper which is crucial for handling batches of keys.
            env = apply_core_wrappers(env, config)

            # 6. Apply non-vectorized core wrappers to the evaluation environment.
            # The evaluator handles its own vectorization.
            eval_env = AddRNGKey(eval_env)
            eval_env = RecordEpisodeMetrics(eval_env)
            eval_env = AutoResetWrapper(eval_env, next_obs_in_extras=True)

            return env, eval_env

        # If not 'jaxarc', delegate to the original Stoix make_env function.
        else:
            return original_make_fn(config=config)

    return custom_make
