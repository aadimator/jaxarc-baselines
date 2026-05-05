from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import baseline_scheduler as base

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = base.DEFAULT_RESULTS_ROOT / "subset_baselines_10m_5seed"
DEFAULT_OUTPUTS_ROOT = base.DEFAULT_OUTPUTS_ROOT / "subset_baselines_10m_5seed"


@dataclass(frozen=True)
class AlgorithmSpec:
    key: str
    config_name: str
    system_config: str
    algorithm_tag: str
    network: str | None
    min_free_mem_mib: int
    exclusive_gpu: bool = False
    command_overrides: tuple[str, ...] = ()


ALGORITHM_SPECS: dict[str, AlgorithmSpec] = {
    "ppo": AlgorithmSpec(
        key="ppo",
        config_name="baseline_ff_ppo_mini_all_512k.yaml",
        system_config="ppo/ff_ppo",
        algorithm_tag="ff_ppo",
        network="arc_shallow_cnn_hwc",
        min_free_mem_mib=20_000,
    ),
    "ddqn": AlgorithmSpec(
        key="ddqn",
        config_name="baseline_ff_ddqn_mini_all_512k.yaml",
        system_config="q_learning/ff_ddqn",
        algorithm_tag="ff_ddqn",
        network="arc_shallow_cnn_hwc_q_value",
        min_free_mem_mib=40_000,
        exclusive_gpu=True,
        command_overrides=(
            "system.total_buffer_size=65536",
            "system.total_batch_size=256",
        ),
    ),
    "pqn": AlgorithmSpec(
        key="pqn",
        config_name="baseline_ff_pqn_mini_all_512k.yaml",
        system_config="q_learning/ff_pqn",
        algorithm_tag="ff_pqn",
        network="arc_shallow_cnn_hwc_q_value",
        min_free_mem_mib=20_000,
    ),
    "reinforce": AlgorithmSpec(
        key="reinforce",
        config_name="baseline_ff_reinforce_mini_all_524k.yaml",
        system_config="vpg/ff_reinforce",
        algorithm_tag="ff_reinforce",
        network="arc_shallow_cnn_hwc",
        min_free_mem_mib=20_000,
    ),
}
ALGORITHM_SPECS_BY_TAG = {spec.algorithm_tag: spec for spec in ALGORITHM_SPECS.values()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ppo", "ddqn", "pqn", "reinforce"],
        choices=sorted(ALGORITHM_SPECS),
    )
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", "concept", "agi1"])
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--scenario-limit", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--total-num-envs", type=int, default=512)
    parser.add_argument("--num-evaluation", type=int, default=20)
    parser.add_argument("--num-eval-episodes", type=int, default=32)
    parser.add_argument("--max-concurrent-jobs", type=int, default=5)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-mem-mib", type=int, default=20_000)
    parser.add_argument("--max-oom-retries", type=int, default=2)
    parser.add_argument("--max-generic-retries", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--action-mode", default="point")
    parser.add_argument("--answer-grid", dest="answer_grid", action="store_true")
    parser.add_argument("--no-answer-grid", dest="answer_grid", action="store_false")
    parser.add_argument("--input-grid", dest="input_grid", action="store_true")
    parser.add_argument("--no-input-grid", dest="input_grid", action="store_false")
    parser.add_argument("--contextual", dest="contextual", action="store_true")
    parser.add_argument("--no-contextual", dest="contextual", action="store_false")
    parser.set_defaults(answer_grid=True, input_grid=True, contextual=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_command(
    *,
    spec: AlgorithmSpec,
    args: argparse.Namespace,
    job_name: str,
    scenario: str,
    dataset_config: str,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    base._validate_action_mode(dataset_config, args.action_mode)
    wandb_enabled = args.wandb_project is not None
    command = [
        "pixi",
        "run",
        "python",
        "run_experiment.py",
        "--config-name",
        spec.config_name,
        "hydra.job.chdir=false",
        f"system={spec.system_config}",
        f"dataset={dataset_config}",
        f"arch.seed={seed}",
        f"arch.total_timesteps={args.total_timesteps}",
        f"env.scenario.name={scenario}",
        f"env.scenario.task_name={base._slugify(job_name)}",
        f"storage.base_output_dir={output_dir}",
        f"storage.run_name={job_name}",
        f"logger.base_exp_path={run_dir / 'stoix_logs'}",
        "logger.loggers.console.enabled=true",
        "logger.loggers.json.enabled=true",
        f"wandb.enabled={'true' if wandb_enabled else 'false'}",
        f"logger.loggers.wandb.enabled={'true' if wandb_enabled else 'false'}",
    ]

    if args.total_num_envs is not None:
        command.append(f"arch.total_num_envs={args.total_num_envs}")
    if args.num_evaluation is not None:
        command.append(f"arch.num_evaluation={args.num_evaluation}")
    if args.num_eval_episodes is not None:
        command.append(f"arch.num_eval_episodes={args.num_eval_episodes}")
    if spec.network is not None:
        command.append(f"network={spec.network}")
    if args.action_mode is not None:
        command.append(f"env.action.mode={args.action_mode}")
    if args.answer_grid is not None:
        command.append(
            f"env.observation_wrappers.answer_grid={'true' if args.answer_grid else 'false'}"
        )
    if args.input_grid is not None:
        command.append(
            f"env.observation_wrappers.input_grid={'true' if args.input_grid else 'false'}"
        )
    if args.contextual is not None:
        command.append(
            f"env.observation_wrappers.contextual={'true' if args.contextual else 'false'}"
        )
    command.extend(spec.command_overrides)
    if args.wandb_project is not None:
        command.append(f"logger.loggers.wandb.project={args.wandb_project}")
        if args.wandb_entity is not None:
            command.append(f"logger.loggers.wandb.entity={args.wandb_entity}")

    return command


def _build_jobs(args: argparse.Namespace, results_root: Path, outputs_root: Path) -> list[base.Job]:
    jobs: list[base.Job] = []
    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))
    algorithm_specs = [ALGORITHM_SPECS[key] for key in args.algorithms]

    for dataset, dataset_config, scenarios in base._dataset_targets(args):
        selected_scenarios = scenarios[: args.scenario_limit] if args.scenario_limit else scenarios
        for scenario in selected_scenarios:
            scenario_slug = base._slugify(scenario)
            for seed in seeds:
                seed_slug = f"seed_{seed:02d}"
                for spec in algorithm_specs:
                    algorithm_slug = base._slugify(spec.algorithm_tag)
                    job_name = f"{algorithm_slug}__{scenario_slug}__s{seed:02d}"
                    run_dir = results_root / "runs" / algorithm_slug / dataset.lower() / scenario_slug / seed_slug
                    output_dir = outputs_root / algorithm_slug / dataset.lower() / scenario_slug / seed_slug
                    command = _build_command(
                        spec=spec,
                        args=args,
                        job_name=job_name,
                        scenario=scenario,
                        dataset_config=dataset_config,
                        seed=seed,
                        run_dir=run_dir,
                        output_dir=output_dir,
                    )
                    jobs.append(
                        base.Job(
                            name=job_name,
                            algorithm=spec.algorithm_tag,
                            dataset=dataset,
                            dataset_config=dataset_config,
                            scenario=scenario,
                            seed=seed,
                            command=command,
                            run_dir=str(run_dir),
                            output_dir=str(output_dir),
                        )
                    )

    return jobs


def _running_jobs_by_gpu(jobs: list[base.Job]) -> dict[int, list[base.Job]]:
    running: dict[int, list[base.Job]] = {}
    for job in jobs:
        if job.status != "running" or job.gpu is None:
            continue
        running.setdefault(job.gpu, []).append(job)
    return running


def _eligible_gpus_for_job(
    jobs: list[base.Job],
    *,
    spec: AlgorithmSpec,
    min_free_mem_mib: int,
    avoid: set[int],
    max_jobs_per_gpu: int,
) -> list[dict[str, int]]:
    running_jobs = _running_jobs_by_gpu(jobs)
    effective_min_free_mem_mib = max(min_free_mem_mib, spec.min_free_mem_mib)
    eligible: list[dict[str, int]] = []

    for gpu in base._query_gpus():
        gpu_index = gpu["index"]
        gpu_running_jobs = running_jobs.get(gpu_index, [])
        if gpu["free_mem_mib"] < effective_min_free_mem_mib:
            continue
        if gpu_index in avoid:
            continue
        if len(gpu_running_jobs) >= max_jobs_per_gpu:
            continue
        if spec.exclusive_gpu and gpu_running_jobs:
            continue

        running_specs = [ALGORITHM_SPECS_BY_TAG[job.algorithm] for job in gpu_running_jobs]
        if any(running_spec.exclusive_gpu for running_spec in running_specs):
            continue

        eligible.append(gpu)

    eligible.sort(
        key=lambda gpu: (
            -gpu["free_mem_mib"],
            len(running_jobs.get(gpu["index"], [])),
            gpu["utilization"],
            gpu["index"],
        )
    )
    return eligible


def main() -> int:
    args = _parse_args()
    if args.max_jobs_per_gpu < 1:
        raise ValueError("--max-jobs-per-gpu must be at least 1.")

    results_root = args.results_root.resolve()
    outputs_root = args.outputs_root.resolve()
    launcher_log_root = results_root / "launcher_logs"
    running_log = results_root / "RUNNING_LOG.md"
    status_path = results_root / "launcher_status.json"

    jobs = base._merge_existing_status(
        _build_jobs(args, results_root, outputs_root),
        status_path,
        args.retry_failed,
    )

    base._append_log(
        results_root,
        running_log,
        "Mixed scheduler started/resumed with "
        f"{len(jobs)} jobs across algorithms={args.algorithms}, "
        f"max_concurrent_jobs={args.max_concurrent_jobs}, "
        f"max_jobs_per_gpu={args.max_jobs_per_gpu}, "
        f"default_min_free_mem_mib={args.min_free_mem_mib}. "
        "DDQN uses exclusive GPUs plus safer replay-buffer overrides.",
    )
    base._write_status(
        status_path,
        jobs,
        poll_seconds=args.poll_seconds,
        max_concurrent_jobs=args.max_concurrent_jobs,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        min_free_mem_mib=args.min_free_mem_mib,
        max_oom_retries=args.max_oom_retries,
        max_generic_retries=args.max_generic_retries,
    )

    if args.dry_run:
        preview = [job.name for job in jobs[: min(16, len(jobs))]]
        print(json.dumps({"num_jobs": len(jobs), "preview": preview}, indent=2))
        return 0

    active_processes: dict[str, subprocess.Popen[str]] = {}
    active_handles: dict[str, Any] = {}

    while True:
        for job_name, process in list(active_processes.items()):
            job = next((candidate for candidate in jobs if candidate.name == job_name), None)
            if job is None:
                continue
            if process.poll() is None:
                continue
            base._handle_finished_job(
                job=job,
                process=process,
                log_handle=active_handles[job_name],
                results_root=results_root,
                running_log=running_log,
                max_oom_retries=args.max_oom_retries,
                max_generic_retries=args.max_generic_retries,
            )
            active_processes.pop(job_name, None)
            active_handles.pop(job_name, None)
            base._write_status(
                status_path,
                jobs,
                poll_seconds=args.poll_seconds,
                max_concurrent_jobs=args.max_concurrent_jobs,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
                min_free_mem_mib=args.min_free_mem_mib,
                max_oom_retries=args.max_oom_retries,
                max_generic_retries=args.max_generic_retries,
            )

        while len(active_processes) < args.max_concurrent_jobs:
            next_job = next((job for job in jobs if job.status == "pending"), None)
            if next_job is None:
                break
            spec = ALGORITHM_SPECS_BY_TAG[next_job.algorithm]
            avoid = set(next_job.tried_gpus) if next_job.oom_attempts > 0 else set()
            eligible_gpus = _eligible_gpus_for_job(
                jobs,
                spec=spec,
                min_free_mem_mib=args.min_free_mem_mib,
                avoid=avoid,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
            )
            if not eligible_gpus and avoid:
                eligible_gpus = _eligible_gpus_for_job(
                    jobs,
                    spec=spec,
                    min_free_mem_mib=args.min_free_mem_mib,
                    avoid=set(),
                    max_jobs_per_gpu=args.max_jobs_per_gpu,
                )
            if not eligible_gpus:
                break

            process, handle = base._start_job(next_job, eligible_gpus[0], launcher_log_root)
            active_processes[next_job.name] = process
            active_handles[next_job.name] = handle
            gpu_job_count = sum(
                1 for job in jobs if job.status == "running" and job.gpu == eligible_gpus[0]["index"]
            )
            base._append_log(
                results_root,
                running_log,
                f"Launched `{next_job.name}` on GPU {eligible_gpus[0]['index']} with "
                f"{eligible_gpus[0]['free_mem_mib']} MiB free; occupancy now "
                f"{gpu_job_count}/{args.max_jobs_per_gpu} jobs on that GPU. "
                f"Algorithm policy: exclusive_gpu={spec.exclusive_gpu}, "
                f"min_free_mem_mib={max(args.min_free_mem_mib, spec.min_free_mem_mib)}.",
            )
            base._write_status(
                status_path,
                jobs,
                poll_seconds=args.poll_seconds,
                max_concurrent_jobs=args.max_concurrent_jobs,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
                min_free_mem_mib=args.min_free_mem_mib,
                max_oom_retries=args.max_oom_retries,
                max_generic_retries=args.max_generic_retries,
            )

        terminal_statuses = {"completed", "failed"}
        if all(job.status in terminal_statuses for job in jobs):
            failures = [job for job in jobs if job.status == "failed"]
            if failures:
                base._append_log(
                    results_root,
                    running_log,
                    f"Mixed scheduler finished with {len(failures)} failed jobs recorded for debugging.",
                )
                base._write_status(
                    status_path,
                    jobs,
                    poll_seconds=args.poll_seconds,
                    max_concurrent_jobs=args.max_concurrent_jobs,
                    max_jobs_per_gpu=args.max_jobs_per_gpu,
                    min_free_mem_mib=args.min_free_mem_mib,
                    max_oom_retries=args.max_oom_retries,
                    max_generic_retries=args.max_generic_retries,
                )
                return 1

            base._append_log(results_root, running_log, "Mixed scheduler finished all queued jobs.")
            base._write_status(
                status_path,
                jobs,
                poll_seconds=args.poll_seconds,
                max_concurrent_jobs=args.max_concurrent_jobs,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
                min_free_mem_mib=args.min_free_mem_mib,
                max_oom_retries=args.max_oom_retries,
                max_generic_retries=args.max_generic_retries,
            )
            return 0

        base._write_status(
            status_path,
            jobs,
            poll_seconds=args.poll_seconds,
            max_concurrent_jobs=args.max_concurrent_jobs,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            min_free_mem_mib=args.min_free_mem_mib,
            max_oom_retries=args.max_oom_retries,
            max_generic_retries=args.max_generic_retries,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
