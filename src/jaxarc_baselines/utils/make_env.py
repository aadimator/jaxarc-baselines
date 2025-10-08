from __future__ import annotations

from collections.abc import Callable

from omegaconf import DictConfig
from stoa.environment import Environment


def get_custom_make_fn(
    original_make_fn: Callable[[DictConfig], tuple[Environment, Environment]],
) -> Callable[[DictConfig], tuple[Environment, Environment]]:
    """
    Factory function that takes the original Stoix make_env function and
    returns a new, patched version that can handle the 'jaxarc' environment.
    """

    def custom_make(config: DictConfig) -> tuple[Environment, Environment]:
        """
        Creates training and evaluation environments.

        This function checks if the requested environment is 'jaxarc'.
        If so, it uses the JaxARC registration and configuration system, and then
        applies the necessary wrappers to make it compatible with Stoix.
        """
        if config.env.env_name == "jaxarc":
            from jaxarc.configs import JaxArcConfig
            from jaxarc.envs import (
                AnswerObservationWrapper,
                BboxActionWrapper,
                ContextualObservationWrapper,
                FlattenActionWrapper,
                InputGridObservationWrapper,
                PointActionWrapper,
            )
            from stoa.core_wrappers.auto_reset import AutoResetWrapper
            from stoa.core_wrappers.episode_metrics import RecordEpisodeMetrics
            from stoa.core_wrappers.vmap import VmapWrapper
            from stoa.core_wrappers.wrapper import AddRNGKey

            from jaxarc_baselines import ExtendedMetrics
            from jaxarc_baselines.utils.registration import make as make_jaxarc

            jaxarc_config = JaxArcConfig.from_hydra(config)

            # 1. Create the base environment.
            env, _ = make_jaxarc(
                config.env.scenario.name, config=jaxarc_config, auto_download=True
            )
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
                msg = f"Unknown action mode: {action_mode}"
                raise ValueError(msg)

            # 4. Flatten the resulting DictSpace into a single DiscreteSpace for Stoix.
            env = FlattenActionWrapper(env)
            eval_env = FlattenActionWrapper(eval_env)

            # 4.5. Apply observation wrappers based on config
            # These add additional channels to the observation (multi-channel input)
            obs_wrappers = config.env.get("observation_wrappers", {})

            if obs_wrappers.get("answer_grid", True):
                env = AnswerObservationWrapper(env)
                eval_env = AnswerObservationWrapper(eval_env)

            if obs_wrappers.get("input_grid", True):
                env = InputGridObservationWrapper(env)
                eval_env = InputGridObservationWrapper(eval_env)

            if obs_wrappers.get("contextual", True):
                env = ContextualObservationWrapper(env)
                eval_env = ContextualObservationWrapper(eval_env)

            # 4.6. Apply extended metrics BEFORE vectorization
            # This wrapper needs to be applied before VmapWrapper
            env = AddRNGKey(env)
            env = RecordEpisodeMetrics(env)
            env = ExtendedMetrics(env)

            # 5. Apply auto-reset and vectorization wrappers
            # Note: VmapWrapper without num_envs expects pre-split keys from caller
            env = AutoResetWrapper(env, next_obs_in_extras=True)
            env = VmapWrapper(env)

            # 6. Apply non-vectorized core wrappers to the evaluation environment.
            # The evaluator handles its own vectorization.
            eval_env = AddRNGKey(eval_env)
            eval_env = RecordEpisodeMetrics(eval_env)
            eval_env = ExtendedMetrics(eval_env)
            eval_env = AutoResetWrapper(eval_env, next_obs_in_extras=True)

            return env, eval_env

        # If not 'jaxarc', delegate to the original Stoix make_env function.
        return original_make_fn(config=config)

    return custom_make
