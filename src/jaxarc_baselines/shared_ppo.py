"""PPO with one torso shared by the policy and value heads (anakin, feed-forward).

A copy of Stoix's `stoix.systems.ppo.anakin.ff_ppo` with one network and one
optimizer in place of separate actor and critic networks. The loss is
policy loss + vf_coef * value loss - ent_coef * entropy. Rollouts, GAE (with its
truncation bootstrap), minibatching, observation normalisation, evaluation and
logging are unchanged. Run it through run_experiment.py with
system=ppo/ff_ppo_shared and a network config with a `shared_network` torso.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from omegaconf import DictConfig, OmegaConf
from stoa import Environment, get_final_step_metrics

from stoix.base_types import AnakinExperimentOutput, LearnerFn, OnPolicyLearnerState
from stoix.evaluator import evaluator_setup, get_distribution_act_fn
from stoix.networks.base import FeedForwardActorCritic
from stoix.systems.ppo.ppo_types import PPOTransition
from stoix.utils import make_env as environments
from stoix.utils.checkpointing import Checkpointer
from stoix.utils.jax_utils import (
    merge_leading_dims,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from stoix.utils.logger import LogEvent, StoixLogger
from stoix.utils.loss import clipped_value_loss, ppo_clip_loss
from stoix.utils.multistep import batch_truncated_generalized_advantage_estimation
from stoix.utils.running_statistics import (
    create_with_running_statistics,
    initialize_statistics_from_data,
    normalize,
    update_statistics,
)
from stoix.utils.total_timestep_checker import check_total_timesteps
from stoix.utils.training import make_learning_rate_schedule


def shared_ppo_loss(
    params: FrozenDict,
    apply_fn: Any,
    traj_batch: PPOTransition,
    gae: chex.Array,
    targets: chex.Array,
    system: DictConfig,
) -> Tuple:
    """Policy loss + vf_coef * value loss - ent_coef * entropy, on one forward pass."""
    # RERUN NETWORK
    actor_policy, value = apply_fn(params, traj_batch.obs)
    log_prob = actor_policy.log_prob(traj_batch.action)

    # CALCULATE ACTOR LOSS
    loss_actor = ppo_clip_loss(log_prob, traj_batch.log_prob, gae, system.clip_eps)
    entropy = actor_policy.entropy().mean()

    # CALCULATE VALUE LOSS
    value_loss = clipped_value_loss(value, traj_batch.value, targets, system.clip_eps)

    total_loss = loss_actor + system.vf_coef * value_loss - system.ent_coef * entropy
    loss_info = {
        "actor_loss": loss_actor,
        "entropy": entropy,
        "advantages": gae,
        "value_loss": value_loss,
        "pred_value": value,
        "target_value": targets,
        "total_loss": total_loss,
    }
    return total_loss, loss_info


def get_learner_fn(
    env: Environment,
    apply_fn: Any,
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> LearnerFn[OnPolicyLearnerState]:
    """Get the learner function."""

    def _update_step(
        learner_state: OnPolicyLearnerState, _: Any
    ) -> Tuple[OnPolicyLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            learner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            learner_state: OnPolicyLearnerState, _: Any
        ) -> Tuple[OnPolicyLearnerState, PPOTransition]:
            """Step the environment."""
            params, opt_states, key, env_state, last_timestep = learner_state

            # GET OBSERVATION
            observation = last_timestep.observation

            # Get running statistics if normalizing observations.
            # It would not be present if observation normalization is not implemented/setup.
            running_statistics = getattr(learner_state, "running_statistics", None)
            if running_statistics is not None:
                observation = normalize(observation, running_statistics)

            # SELECT ACTION
            key, policy_key = jax.random.split(key)
            actor_policy, value = apply_fn(params, observation)
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = env.step(env_state, action)

            # LOG EPISODE METRICS
            done = (timestep.discount == 0.0).reshape(-1)
            truncated = (timestep.last() & (timestep.discount != 0.0)).reshape(-1)
            info = timestep.extras["episode_metrics"]
            # Save bootstrap value for the next step.
            # Due to auto-resetting and truncation, we have to specifically save the bootstrap value
            # for the next (potentially final) observation.
            bootstrap_obs = timestep.extras["next_obs"]
            if running_statistics is not None:
                bootstrap_obs = normalize(bootstrap_obs, running_statistics)
            _, bootstrap_value = apply_fn(params, bootstrap_obs)

            transition = PPOTransition(
                done,
                truncated,
                action,
                value,
                timestep.reward,
                bootstrap_value,
                log_prob,
                last_timestep.observation,
                info,
            )
            # Replace the learner state with the new environment state and timestep.
            learner_state = learner_state._replace(
                key=key,
                env_state=env_state,
                timestep=timestep,
            )
            return learner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        learner_state, traj_batch = jax.lax.scan(
            _env_step, learner_state, None, config.system.rollout_length
        )

        # UNPACK LEARNER STATE
        params, opt_states, key, _, _ = learner_state

        # IF NORMALIZING OBSERVATIONS, UPDATE RUNNING STATISTICS
        running_statistics = getattr(learner_state, "running_statistics", None)
        if running_statistics is not None:
            # We get the raw observations from the trajectory batch.
            raw_obs = traj_batch.obs
            # We normalize with pre-updated statistics
            normalized_obs = normalize(traj_batch.obs, running_statistics)
            # Update running statistics
            running_statistics = update_statistics(
                running_statistics,
                raw_obs,
                pmap_axes=["device", "batch"],
                std_min_value=5e-4,
                std_max_value=5e4,
            )
            learner_state = learner_state._replace(running_statistics=running_statistics)  # type: ignore
            # We then replace the observations in the trajectory batch with the normalized ones.
            traj_batch = traj_batch._replace(obs=normalized_obs)

        # CALCULATE ADVANTAGE
        v_tm1 = traj_batch.value
        r_t = traj_batch.reward * config.system.reward_scale
        v_t = traj_batch.bootstrap_value
        d_t = 1.0 - traj_batch.done.astype(jnp.float32)
        d_t = (d_t * config.system.gamma).astype(jnp.float32)
        advantages, targets = batch_truncated_generalized_advantage_estimation(
            r_t,
            d_t,
            config.system.gae_lambda,
            v_tm1=v_tm1,
            v_t=v_t,
            time_major=True,
            standardize_advantages=config.system.standardize_advantages,
            truncation_t=traj_batch.truncated,
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # UNPACK TRAIN STATE AND BATCH INFO
                params, opt_states = train_state
                traj_batch, advantages, targets = batch_info

                # CALCULATE LOSS
                grad_fn = jax.grad(shared_ppo_loss, has_aux=True)
                grads, loss_info = grad_fn(
                    params, apply_fn, traj_batch, advantages, targets, config.system
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # This pmean could be a regular mean as the batch axis is on the same device.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="batch")
                # pmean over devices.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="device")

                # UPDATE PARAMS AND OPTIMISER STATE
                updates, new_opt_state = update_fn(grads, opt_states)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), loss_info

            (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            ) = update_state
            key, shuffle_key = jax.random.split(key)

            # SHUFFLE MINIBATCHES
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [config.system.num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_states), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = learner_state._replace(params=params, opt_states=opt_states, key=key)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: OnPolicyLearnerState,
    ) -> AnakinExperimentOutput[OnPolicyLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (WrapperState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, None, config.arch.num_updates_per_eval
        )
        return AnakinExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def schedule_updates(config: DictConfig) -> int:
    """Updates the linear lr decay spans: the run's own, or ``system.lr_schedule_timesteps``.

    A horizon longer than the run lets stage 1 of the tuning search run the final
    budget's schedule and stop early, so a stage-1 run is the start of a full run.
    """
    horizon = config.system.get("lr_schedule_timesteps")
    if horizon is None:
        return int(config.arch.num_updates)
    per_update = (
        config.num_devices
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )
    updates = int(horizon) // per_update
    if updates < config.arch.num_updates:
        raise ValueError(
            f"system.lr_schedule_timesteps={horizon} gives {updates} updates, fewer than the "
            f"run's {config.arch.num_updates}; the lr would go negative"
        )
    return updates


def make_lr(config: DictConfig):
    """The optimizer's learning rate: constant, or linear decay over ``schedule_updates``."""
    if not config.system.decay_learning_rates:
        return config.system.lr
    return make_learning_rate_schedule(
        config.system.lr,
        schedule_updates(config),
        config.system.epochs,
        config.system.num_minibatches,
    )


def _collect_obs_norm_rollouts(
    env: Environment,
    key: chex.PRNGKey,
    config: DictConfig,
) -> chex.ArrayTree:
    """Collect observations for observation normalization.
    This function collects observations from the environment by taking random actions
    for a specified number of warmup steps."""
    num_warmup_obs = config.system.obs_norm_warmup_steps * config.arch.total_num_envs
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}Initializing observation normalization with "
        f"{num_warmup_obs} observations..."
        f"Be aware, we do not count this in the timestep budget.{Style.RESET_ALL}"
    )

    @jax.jit
    def _warmup_step(env_state: Any, step_key: chex.PRNGKey) -> Tuple[Any, chex.ArrayTree]:
        """Single warmup step: random action -> environment step -> collect observation."""
        action = jax.random.randint(
            step_key, (config.arch.total_num_envs,), 0, config.system.action_dim
        )
        env_state, timestep = env.step(env_state, action)
        return env_state, timestep.observation

    # Initialize environments for data collection
    key, *env_keys = jax.random.split(key, config.arch.total_num_envs + 1)
    env_states, initial_timesteps = env.reset(jnp.stack(env_keys))

    # Collect warmup observations through random actions
    step_keys = jax.random.split(key, config.system.obs_norm_warmup_steps)
    _, warmup_observations = jax.lax.scan(_warmup_step, env_states, step_keys)

    # Combine initial reset observations with warmup observations
    all_observations = jax.tree.map(
        lambda reset_obs, warmup_obs: jnp.concatenate(
            [
                reset_obs[
                    jnp.newaxis,
                ],
                warmup_obs,
            ],
            axis=0,
        ),
        initial_timesteps.observation,
        warmup_observations,
    )
    return all_observations


def learner_setup(
    env: Environment, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[OnPolicyLearnerState], FeedForwardActorCritic, OnPolicyLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number/dimension of actions.
    num_actions = int(env.action_space().num_values)
    config.system.action_dim = num_actions

    # PRNG keys.
    key, net_key = keys

    # Define network and optimiser.
    torso = hydra.utils.instantiate(config.network.shared_network.pre_torso)
    action_head = hydra.utils.instantiate(
        config.network.actor_network.action_head, action_dim=num_actions
    )
    critic_head = hydra.utils.instantiate(config.network.critic_network.critic_head)
    network = FeedForwardActorCritic(
        torso=torso, action_head=action_head, critic_head=critic_head
    )

    lr = make_lr(config)
    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    # Initialise observation
    init_x = env.observation_space().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # Initialise params and optimiser state.
    params = network.init(net_key, init_x)
    opt_state = optim.init(params)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, network.apply, optim.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.arch.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = env.reset(jnp.stack(env_keys))
    reshape_states = lambda x: x.reshape(
        (n_devices, config.arch.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.system.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Define params to be replicated across devices and batches.
    key, step_key = jax.random.split(key)
    step_keys = jax.random.split(step_key, n_devices * config.arch.update_batch_size)
    reshape_keys = lambda x: x.reshape((n_devices, config.arch.update_batch_size) + x.shape[1:])
    step_keys = reshape_keys(jnp.stack(step_keys))
    replicate_learner = (params, opt_state)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.arch.update_batch_size,) + x.shape)
    replicate_learner = jax.tree_util.tree_map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states = replicate_learner
    init_learner_state = OnPolicyLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
    )

    # If normalizing observations, initialize running statistics.
    # This will add running statistics to the learner state.
    if config.system.normalize_observations:
        dummy_obs = jax.tree.map(lambda x: x.squeeze(0), init_x)
        warmup_observations = _collect_obs_norm_rollouts(env, key, config)
        running_statistics = initialize_statistics_from_data(dummy_obs, warmup_observations)
        running_statistics = jax.tree.map(broadcast, running_statistics)
        running_statistics = flax.jax_utils.replicate(running_statistics, devices=jax.devices())
        init_learner_state = create_with_running_statistics(
            state=init_learner_state, running_statistics=running_statistics
        )

    return learn, network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config.num_devices = n_devices
    config = check_total_timesteps(config)
    assert (
        config.arch.num_updates >= config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Create the environments for train and eval.
    env, eval_env = environments.make(config=config)

    # PRNG keys.
    key, key_e, net_key = jax.random.split(jax.random.PRNGKey(config.arch.seed), num=3)

    # Setup learner.
    learn, network, learner_state = learner_setup(env, (key, net_key), config)

    # Setup evaluator. It acts with the policy head; the value head is unused.
    evaluator, absolute_metric_evaluator, (trained_params, eval_keys) = evaluator_setup(
        eval_env=eval_env,
        key_e=key_e,
        eval_act_fn=get_distribution_act_fn(config, lambda p, o: network.apply(p, o)[0]),
        params=learner_state.params,
        config=config,
    )

    # Calculate environment steps per evaluation.
    steps_per_rollout = (
        n_devices
        * config.arch.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = StoixLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))
    print(f"{Fore.YELLOW}{Style.BRIGHT}JAX Global Devices {jax.devices()}{Style.RESET_ALL}")

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.system.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_learner_state = unreplicate_batch_dim(learner_state)
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        # Calculate the number of optimiser steps per second. Since gradients are aggregated
        # across the device and batch axis, we don't consider updates per device/batch as part of
        # the SPS for the learner.
        opt_steps_per_eval = config.arch.num_updates_per_eval * (
            config.system.epochs * config.system.num_minibatches
        )
        train_metrics["steps_per_second"] = opt_steps_per_eval / elapsed_time
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        start_time = time.time()
        trained_params = unreplicate_batch_dim(learner_output.learner_state.params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        # Evaluate.
        running_statistics = getattr(learner_output.learner_state, "running_statistics", None)
        if running_statistics is not None:
            running_statistics = unreplicate_batch_dim(running_statistics)
        evaluator_output = evaluator(trained_params, eval_keys, running_statistics)
        jax.block_until_ready(evaluator_output)

        # Log the results of the evaluation.
        elapsed_time = time.time() - start_time
        episode_return = jnp.mean(evaluator_output.episode_metrics["episode_return"])

        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.EVAL)

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=int(steps_per_rollout * (eval_step + 1)),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_learner_state = copy.deepcopy(unreplicate_batch_dim(learner_state))
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Measure absolute metric.
    if config.arch.absolute_metric:
        start_time = time.time()

        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)

        best_params = best_learner_state.params
        best_running_statistics = getattr(best_learner_state, "running_statistics", None)
        evaluator_output = absolute_metric_evaluator(
            best_params, eval_keys, best_running_statistics
        )
        jax.block_until_ready(evaluator_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        steps_per_eval = int(jnp.sum(evaluator_output.episode_metrics["episode_length"]))
        evaluator_output.episode_metrics["steps_per_second"] = steps_per_eval / elapsed_time
        logger.log(evaluator_output.episode_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()
    # Record the performance for the final evaluation run. If the absolute metric is not
    # calculated, this will be the final evaluation run.
    eval_performance = float(jnp.mean(evaluator_output.episode_metrics[config.env.eval_metric]))
    return eval_performance
