"""
Enhanced SLURM launcher for JaxARC experiments.

This launcher supports:
- Multiple network architectures
- Multiple environments/tasks
- Multiple seeds
- Hyperparameter sweeps
- Intelligent WandB run naming and grouping
"""

import itertools
import os
import subprocess
from typing import Any, Dict, List, Optional

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf


def generate_run_name(
    network: str,
    task: str,
    seed: int,
    lr: Optional[float] = None,
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
    wandb_config: Dict[str, Any],
    learning_rate: Optional[float] = None,
    task_subset: Optional[str] = None,
    additional_args: Optional[Dict[str, Any]] = None,
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
        additional_args: Optional additional command-line arguments
    """
    # Build base command
    cmd_parts = [
        f"python {algorithm_exec_file}",
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

    # Generate WandB run name and group
    run_name = generate_run_name(
        network=network,
        task=environment,
        seed=seed,
        lr=learning_rate,
        template=wandb_config.get("name_template", "{network}_{task}_s{seed}"),
    )

    group_name = generate_group_name(
        experiment_group=wandb_config.get("experiment_group", "default"),
        task=environment,
        template=wandb_config.get("group_template", "{experiment_group}_{task}"),
    )

    # Add WandB configuration
    if wandb_config.get("enabled", True):
        cmd_parts.append(f"logger.loggers.wandb.enabled=True")
        cmd_parts.append(f"+logger.loggers.wandb.name={run_name}")
        cmd_parts.append(f"+logger.loggers.wandb.group={group_name}")

        if wandb_config.get("project"):
            cmd_parts.append(f"logger.loggers.wandb.project={wandb_config['project']}")

        if wandb_config.get("entity"):
            cmd_parts.append(f"+logger.loggers.wandb.entity={wandb_config['entity']}")

        if wandb_config.get("tags"):
            tags_str = ",".join(wandb_config["tags"])
            cmd_parts.append(f"+logger.loggers.wandb.tags=[{tags_str}]")

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
    print("=" * 80)

    # Create the submitit executor for SLURM
    executor = submitit.AutoExecutor(folder=cfg.slurm.folder)

    # Build SLURM parameter dictionary
    slurm_params = {
        "nodes": cfg.slurm.nodes,
        "gpus_per_node": cfg.slurm.gpus_per_node,
        "cpus_per_task": cfg.slurm.cpus_per_task,
        "time": cfg.slurm.time,
        "chdir": os.getcwd(),
        "slurm_account": cfg.slurm.account,
        "slurm_qos": cfg.slurm.qos,
        "slurm_partition": cfg.slurm.partition,
    }
    slurm_params = filter_none_values(slurm_params)

    # Update the executor with SLURM parameters
    executor.update_parameters(
        slurm_job_name=cfg.experiment_group,
        slurm_additional_parameters=slurm_params,
    )

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

    # Get algorithm exec file
    algorithm_exec = cfg.experiment.algorithm_exec_files[0]

    # Prepare the Cartesian product of all experiment parameters
    jobs = []
    job_count = 0

    with executor.batch():
        for network, env, seed, lr, subset in itertools.product(
            cfg.experiment.networks,
            cfg.experiment.environments,
            cfg.experiment.seeds,
            learning_rates,
            task_subsets,
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
