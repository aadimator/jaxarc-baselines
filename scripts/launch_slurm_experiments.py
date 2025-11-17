"""
Enhanced SLURM launcher for JaxARC experiments.

This launcher supports:
- Multiple network architectures
- Multiple environments/tasks
- Multiple seeds
- Hyperparameter sweeps
- Intelligent WandB run naming and grouping
"""

from __future__ import annotations

import itertools
import subprocess
from typing import Any

import hydra
import submitit
from omegaconf import DictConfig


def generate_run_name(
    network: str,
    task: str,
    seed: int,
    lr: float | None = None,
    template: str = "{network}_{task}_s{seed}",
) -> str:
    """Generate a descriptive run name for WandB."""
    # Extract short names
    network_short = network.replace("arc_", "").replace("_", "-")
    task_short = task.split("/")[-1] if "/" in task else task

    name_vars = {
        "network": network_short,
        "task": task_short,
        "seed": seed,
        "lr": f"lr{lr}" if lr else "",
    }

    return template.format(**name_vars).replace("__", "_").strip("_")


def generate_group_name(
    experiment_group: str,
    task: str,
    template: str = "{experiment_group}_{task}",
) -> str:
    """Generate a group name for organizing related runs in WandB."""
    task_short = task.split("/")[-1] if "/" in task else task
    return template.format(experiment_group=experiment_group, task=task_short)


def run_experiment(
    algorithm_exec_file: str,
    environment: str,
    network: str,
    seed: int,
    wandb_config: dict[str, Any],
    learning_rate: float | None = None,
    task_subset: str | None = None,
    obs_config: dict[str, bool] | None = None,
    additional_args: dict[str, Any] | None = None,
) -> None:
    """
    Runs a single JaxARC experiment via subprocess.

    Args:
        algorithm_exec_file: Path to the experiment runner script
        environment: Environment config (e.g. 'jaxarc/mini_arc')
        network: Network config (e.g. 'arc_visual_resnet')
        seed: Random seed for reproducibility
        wandb_config: WandB configuration dict
        learning_rate: Optional learning rate override
        task_subset: Optional task subset config
        obs_config: Optional observation wrapper config
        additional_args: Optional additional command-line arguments
    """
    # Build base command with pixi run to ensure correct environment
    cmd_parts = [
        "pixi run python",
        algorithm_exec_file,
        f"env={environment}",
        f"network={network}",
        f"arch.seed={seed}",
    ]

    # Add learning rate if specified
    if learning_rate is not None:
        cmd_parts.append(f"system.actor_lr={learning_rate}")
        cmd_parts.append(f"system.critic_lr={learning_rate}")

    # Add task subset if specified
    # This overrides env.scenario.name to load different task subsets
    # Examples: "Mini-easy", "AGI1-easy-eval", "AGI1-medium", "Concept-all"
    if task_subset and task_subset != "default":
        cmd_parts.append(f"env.scenario.name={task_subset}")

    # Add observation wrapper configuration if specified
    if obs_config:
        for wrapper_name, enabled in obs_config.items():
            cmd_parts.append(f"env.observation_wrappers.{wrapper_name}={enabled}")

    # Log network name as a custom config field for WandB grouping
    # This creates an "architecture" column in WandB that you can group by
    cmd_parts.append(f"+architecture={network}")

    # Log observation config as a custom field for WandB grouping
    # Creates an "obs_config" column (e.g., "ans_inp_contx", "ans_contx", "default")
    if obs_config:
        # Simple mapping: answer_grid->ans, input_grid->inp, contextual->contx
        parts = []
        if obs_config.get("input_grid"):
            parts.append("inp")
        if obs_config.get("contextual"):
            parts.append("contx")
        if obs_config.get("answer_grid"):
            parts.append("ans")
        obs_config_str = "_".join(parts) if parts else "none"
    else:
        obs_config_str = "default"
    cmd_parts.append(f"+obs_config={obs_config_str}")

    # Add WandB configuration
    if wandb_config.get("enabled", True):
        cmd_parts.append("logger.loggers.wandb.enabled=True")

        if wandb_config.get("project"):
            cmd_parts.append(f"logger.loggers.wandb.project={wandb_config['project']}")

        if wandb_config.get("entity"):
            cmd_parts.append(f"logger.loggers.wandb.entity={wandb_config['entity']}")

        # Build meaningful tags for filtering and grouping in WandB
        # Tags are the primary way to filter/group runs in WandB UI
        tags = list(wandb_config.get("tags", []))

        # Add network type as tag (for filtering by architecture)
        tags.append(f"arch:{network}")

        # Add task/environment as tag (for filtering by task)
        task_short = environment.split("/")[-1] if "/" in environment else environment
        tags.append(f"task:{task_short}")

        # Add seed as tag (for filtering by seed)
        tags.append(f"seed:{seed}")

        # Add learning rate tag if it's a custom value
        if learning_rate is not None:
            tags.append(f"lr:{learning_rate}")

        # Add task subset tag if specified (for curriculum learning experiments)
        if task_subset and task_subset != "default":
            tags.append(f"subset:{task_subset}")

        # Add observation wrapper tags if specified
        if obs_config:
            for wrapper_name, enabled in obs_config.items():
                if enabled:
                    tags.append(f"obs:{wrapper_name}")

        # Add experiment group tag for high-level organization
        exp_group = wandb_config.get("experiment_group", "default")
        tags.append(f"group:{exp_group}")

        tags_str = ",".join(tags)
        cmd_parts.append(f"logger.loggers.wandb.tag=[{tags_str}]")

        # Use group_tag for WandB grouping (will be joined with _ in logger)
        # This creates the "Group" column in WandB which is useful for comparing runs
        group_name = generate_group_name(
            experiment_group=wandb_config.get("experiment_group", "default"),
            task=environment,
            template=wandb_config.get("group_template", "{experiment_group}_{task}"),
        )
        if group_name:
            cmd_parts.append(f"logger.loggers.wandb.group_tag=[{group_name}]")

    # Add any additional arguments
    if additional_args:
        for key, value in additional_args.items():
            cmd_parts.append(f"{key}={value}")

    cmd = " ".join(cmd_parts)

    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def filter_none_values(d: dict) -> dict:
    """Remove None values from dictionary."""
    return {key: value for key, value in d.items() if value is not None}


@hydra.main(
    version_base="1.2",
    config_path="../configs/launcher",
    config_name="jaxarc_experiments",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for launching multiple JaxARC experiments on SLURM.

    This launcher creates a Cartesian product of:
    - Networks
    - Environments
    - Seeds
    - Learning rates (optional)
    - Task subsets (optional)

    Each combination is submitted as a separate SLURM job.
    """
    print("=" * 80)
    print("JaxARC SLURM Experiment Launcher")
    print("=" * 80)
    print(f"Experiment Group: {cfg.experiment_group}")
    print(f"Networks: {cfg.experiment.networks}")
    print(f"Environments: {cfg.experiment.environments}")
    print(f"Seeds: {cfg.experiment.seeds}")
    print(f"Learning Rates: {cfg.experiment.get('learning_rates', ['default'])}")
    print(f"Max Parallel Jobs: {cfg.slurm.get('max_parallel_jobs', 4)}")
    print("=" * 80)

    # Create the submitit executor for SLURM
    executor = submitit.AutoExecutor(folder=cfg.slurm.folder)

    # Build update_parameters arguments
    # Note: submitit's update_parameters() uses clean names (mem_gb, time, etc.)
    # and handles the slurm_ prefix and conversions internally
    update_params = {
        "nodes": cfg.slurm.nodes,
        "gpus_per_node": cfg.slurm.gpus_per_node,
        "cpus_per_task": cfg.slurm.cpus_per_task,
        "timeout_min": cfg.slurm.get("timeout_min", 60),  # Maps to --time
        "name": cfg.experiment_group,  # Maps to --job-name
    }

    # Add memory constraint if specified (prevents oversubscription)
    # submitit's _convert_mem() handles conversion to proper format
    if cfg.slurm.get("mem_gb"):
        update_params["mem_gb"] = cfg.slurm.mem_gb

    # Add exclusive node access if specified
    if cfg.slurm.get("exclusive", False):
        update_params["exclusive"] = True

    # Build additional_parameters dict for slurm-specific options
    # These bypass submitit's equivalence dict and go directly to sbatch
    additional_params = {}

    if cfg.slurm.account:
        additional_params["account"] = cfg.slurm.account

    if cfg.slurm.qos:
        additional_params["qos"] = cfg.slurm.qos

    if cfg.slurm.partition:
        additional_params["partition"] = cfg.slurm.partition

    if cfg.slurm.get("constraint"):
        additional_params["constraint"] = cfg.slurm.constraint

    if cfg.slurm.get("gres"):
        additional_params["gres"] = cfg.slurm.gres

    if additional_params:
        update_params["additional_parameters"] = additional_params

    # Remove None values
    update_params = filter_none_values(update_params)

    # Limit concurrent jobs to avoid blocking other users
    # This tells the scheduler to only run at most 4 jobs at once
    max_parallel_jobs = cfg.slurm.get("max_parallel_jobs", 4)
    update_params["slurm_array_parallelism"] = max_parallel_jobs

    # Update the executor with SLURM parameters
    executor.update_parameters(**update_params)

    # Prepare WandB config
    wandb_config = {
        "enabled": True,
        "project": cfg.wandb.get("project", "jaxarc-baselines"),
        "entity": cfg.wandb.get("entity"),
        "tags": cfg.wandb.get("tags", []),
        "name_template": cfg.wandb.get("name_template", "{network}_{task}_s{seed}"),
        "group_template": cfg.wandb.get("group_template", "{experiment_group}_{task}"),
        "experiment_group": cfg.experiment_group,
    }

    # Get learning rates (None means use default from config)
    learning_rates = list(cfg.experiment.get("learning_rates", [None]))

    # Get task subsets (default means use default from config)
    task_subsets = list(cfg.experiment.get("task_subsets", ["default"]))

    # Get observation wrapper configurations
    obs_wrapper_configs = cfg.experiment.get("observation_wrapper_configs", [None])
    if obs_wrapper_configs is None or len(obs_wrapper_configs) == 0:
        obs_wrapper_configs = [None]

    # Get algorithm exec file
    algorithm_exec = cfg.experiment.algorithm_exec_files[0]

    # Prepare the Cartesian product of all experiment parameters
    jobs = []
    job_count = 0

    with executor.batch():
        for network, env, seed, lr, subset, obs_config in itertools.product(
            cfg.experiment.networks,
            cfg.experiment.environments,
            cfg.experiment.seeds,
            learning_rates,
            task_subsets,
            obs_wrapper_configs,
        ):
            job_count += 1

            # Generate descriptive name for logging
            run_name = generate_run_name(
                network=network,
                task=env,
                seed=seed,
                lr=lr,
                template=wandb_config["name_template"],
            )

            # Add observation config suffix to run name if specified
            if obs_config:
                obs_suffix = "_".join(
                    [f"{k[:3]}{int(v)}" for k, v in obs_config.items()]
                )
                run_name = f"{run_name}_{obs_suffix}"

            print(f"[{job_count}] Submitting: {run_name}")

            job = executor.submit(
                run_experiment,
                algorithm_exec,
                env,
                network,
                seed,
                wandb_config,
                lr,
                subset if subset != "default" else None,
                obs_config,
            )
            jobs.append((run_name, job))

    print("=" * 80)
    print(f"Successfully submitted {job_count} jobs to SLURM!")
    print(f"Logs will be saved to: {cfg.slurm.folder}")
    print(f"WandB project: {wandb_config['project']}")
    print("=" * 80)

    # Optionally wait for jobs and report status
    # for run_name, job in jobs:
    #     try:
    #         result = job.result()
    #         print(f"✓ {run_name} completed successfully")
    #     except Exception as e:
    #         print(f"✗ {run_name} failed: {e}")


if __name__ == "__main__":
    main()
