"""Shared helpers for baseline subset experiment schedulers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jaxarc_baselines.benchmark_log_parser import (
    build_actor_success_curve,
    load_metric_rows,
    looks_like_oom,
    summarize_actor_metrics,
    write_curve_csv,
    write_curve_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "baseline_benchmarks"
DEFAULT_OUTPUTS_ROOT = REPO_ROOT / "outputs" / "research" / "baseline_benchmarks"
JAXARC_DATASET_CONFIG_ROOT = REPO_ROOT / "JaxARC" / "src" / "jaxarc" / "conf" / "dataset"
SAFE_FLAT_ACTION_LIMIT = 1_000_000
KNOWN_DATASET_GRID_SHAPES: dict[str, tuple[int, int]] = {
    "mini_arc": (5, 5),
    "concept_arc": (30, 30),
    "arc_agi_1": (30, 30),
    "arc_agi_2": (30, 30),
}


@dataclass
class Job:
    name: str
    algorithm: str
    dataset: str
    dataset_config: str
    scenario: str
    seed: int
    command: list[str]
    run_dir: str
    output_dir: str
    status: str = "pending"
    gpu: int | None = None
    pid: int | None = None
    attempts: int = 0
    oom_attempts: int = 0
    generic_attempts: int = 0
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    returncode: int | None = None
    log_path: str | None = None
    summary_path: str | None = None
    curve_json_path: str | None = None
    curve_csv_path: str | None = None
    tried_gpus: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _slugify(text: str) -> str:
    return text.lower().replace("/", "_").replace("-", "_")


def _append_log(results_root: Path, running_log: Path, message: str) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    if not running_log.exists():
        running_log.write_text("# baseline subset scheduler log\n\n", encoding="utf-8")
    with running_log.open("a", encoding="utf-8") as handle:
        handle.write(f"- **{_utc_now()}** - {message}\n")


def _load_status(status_path: Path) -> dict[str, Any]:
    if not status_path.exists():
        return {}
    return json.loads(status_path.read_text(encoding="utf-8"))


def _write_status(
    status_path: Path,
    jobs: list[Job],
    *,
    poll_seconds: int,
    max_concurrent_jobs: int,
    max_jobs_per_gpu: int,
    min_free_mem_mib: int,
    max_oom_retries: int,
    max_generic_retries: int,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at_utc": _utc_now(),
        "poll_seconds": poll_seconds,
        "max_concurrent_jobs": max_concurrent_jobs,
        "max_jobs_per_gpu": max_jobs_per_gpu,
        "min_free_mem_mib": min_free_mem_mib,
        "max_oom_retries": max_oom_retries,
        "max_generic_retries": max_generic_retries,
        "gpu_policy": "highest_free_memory_packed_up_to_max_jobs_per_gpu",
        "jobs": [asdict(job) for job in jobs],
    }
    status_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _query_gpus() -> list[dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus: list[dict[str, int]] = []
    for line in completed.stdout.strip().splitlines():
        index_str, free_mem_str, util_str = [part.strip() for part in line.split(",")]
        gpus.append(
            {
                "index": int(index_str),
                "free_mem_mib": int(free_mem_str),
                "utilization": int(util_str),
            }
        )
    return gpus


def _load_dataset_grid_shape(dataset_config: str) -> tuple[int, int]:
    if dataset_config in KNOWN_DATASET_GRID_SHAPES:
        return KNOWN_DATASET_GRID_SHAPES[dataset_config]

    config_path = JAXARC_DATASET_CONFIG_ROOT / f"{dataset_config}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found for {dataset_config}: {config_path}")

    max_height: int | None = None
    max_width: int | None = None
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("max_grid_height:"):
            max_height = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("max_grid_width:"):
            max_width = int(stripped.split(":", 1)[1].strip())

    if max_height is None or max_width is None:
        raise ValueError(
            f"Could not parse max_grid_height/max_grid_width from dataset config: {config_path}"
        )
    return max_height, max_width


def _estimated_flat_action_dim(dataset_config: str, action_mode: str) -> int:
    height, width = _load_dataset_grid_shape(dataset_config)
    if action_mode == "point":
        return 35 * height * width
    if action_mode == "bbox":
        return 35 * height * width * height * width
    if action_mode == "mask":
        return 35 * (2 ** (height * width))
    raise ValueError(f"Unsupported action mode for estimation: {action_mode}")


def _validate_action_mode(dataset_config: str, action_mode: str) -> None:
    estimated_dim = _estimated_flat_action_dim(dataset_config, action_mode)
    if estimated_dim > SAFE_FLAT_ACTION_LIMIT:
        raise ValueError(
            "Requested action mode is incompatible with the current flattened categorical policy head. "
            f"dataset={dataset_config}, action_mode={action_mode}, estimated_flat_action_dim={estimated_dim:,}. "
            "Use point actions for large-grid datasets, or implement a factorized/multidiscrete policy head before using bbox/mask here."
        )


def _available_gpus(
    jobs: list[Job],
    *,
    min_free_mem_mib: int,
    avoid: set[int],
    max_jobs_per_gpu: int,
) -> list[dict[str, int]]:
    running_jobs_per_gpu: dict[int, int] = {}
    for job in jobs:
        if job.status != "running" or job.gpu is None:
            continue
        running_jobs_per_gpu[job.gpu] = running_jobs_per_gpu.get(job.gpu, 0) + 1

    eligible = [
        gpu
        for gpu in _query_gpus()
        if gpu["free_mem_mib"] >= min_free_mem_mib
        and running_jobs_per_gpu.get(gpu["index"], 0) < max_jobs_per_gpu
        and gpu["index"] not in avoid
    ]
    eligible.sort(
        key=lambda gpu: (
            -gpu["free_mem_mib"],
            running_jobs_per_gpu.get(gpu["index"], 0),
            gpu["utilization"],
            gpu["index"],
        )
    )
    return eligible


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _concept_scenarios() -> list[str]:
    concept_root = REPO_ROOT / "data" / "ConceptARC" / "corpus"
    if not concept_root.exists():
        raise FileNotFoundError(f"ConceptARC corpus directory not found: {concept_root}")
    return [f"Concept-{path.name}" for path in sorted(concept_root.iterdir()) if path.is_dir()]


def _agi1_subset_scenarios() -> list[str]:
    subset_root = REPO_ROOT / "configs" / "env" / "jaxarc" / "subsets" / "AGI1"
    if not subset_root.exists():
        raise FileNotFoundError(f"AGI1 subset directory not found: {subset_root}")
    return [f"AGI1-{path.stem}" for path in sorted(subset_root.glob("*.yaml"))]


def _dataset_targets(args: argparse.Namespace) -> list[tuple[str, str, list[str]]]:
    targets: list[tuple[str, str, list[str]]] = []
    requested = {dataset.lower() for dataset in args.datasets}
    if "all" in requested or "concept" in requested:
        concept_scenarios = _concept_scenarios()
        targets.append(("Concept", "concept_arc", concept_scenarios))
    if "all" in requested or "agi1" in requested:
        agi1_scenarios = _agi1_subset_scenarios()
        targets.append(("AGI1", "arc_agi_1", agi1_scenarios))
    return targets


def _build_command(
    *,
    args: argparse.Namespace,
    job_name: str,
    scenario: str,
    dataset_config: str,
    seed: int,
    run_dir: Path,
    output_dir: Path,
) -> list[str]:
    _validate_action_mode(dataset_config, args.action_mode)
    wandb_enabled = args.wandb_project is not None
    command = [
        "pixi",
        "run",
        "python",
        "run_experiment.py",
        "--config-name",
        args.config_name,
        "hydra.job.chdir=false",
        f"system={args.system_config}",
        f"dataset={dataset_config}",
        f"arch.seed={seed}",
        f"arch.total_timesteps={args.total_timesteps}",
        f"env.scenario.name={scenario}",
        f"env.scenario.task_name={_slugify(job_name)}",
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
    if args.network is not None:
        command.append(f"network={args.network}")
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
    if args.wandb_project is not None:
        command.append(f"logger.loggers.wandb.project={args.wandb_project}")
        if args.wandb_entity is not None:
            command.append(f"logger.loggers.wandb.entity={args.wandb_entity}")

    return command


def _build_jobs(args: argparse.Namespace, results_root: Path, outputs_root: Path) -> list[Job]:
    jobs: list[Job] = []
    seeds = list(range(args.seed_offset, args.seed_offset + args.num_seeds))

    for dataset, dataset_config, scenarios in _dataset_targets(args):
        selected_scenarios = scenarios[: args.scenario_limit] if args.scenario_limit else scenarios
        for scenario in selected_scenarios:
            scenario_slug = _slugify(scenario)
            for seed in seeds:
                seed_slug = f"seed_{seed:02d}"
                algorithm_slug = _slugify(args.algorithm_tag)
                job_name = f"{algorithm_slug}__{scenario_slug}__s{seed:02d}"
                run_dir = results_root / "runs" / algorithm_slug / dataset.lower() / scenario_slug / seed_slug
                output_dir = outputs_root / algorithm_slug / dataset.lower() / scenario_slug / seed_slug
                command = _build_command(
                    args=args,
                    job_name=job_name,
                    scenario=scenario,
                    dataset_config=dataset_config,
                    seed=seed,
                    run_dir=run_dir,
                    output_dir=output_dir,
                )
                jobs.append(
                    Job(
                        name=job_name,
                        algorithm=args.algorithm_tag,
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


def _merge_existing_status(jobs: list[Job], status_path: Path, retry_failed: bool) -> list[Job]:
    payload = _load_status(status_path)
    previous_jobs = {job["name"]: job for job in payload.get("jobs", []) if "name" in job}
    for job in jobs:
        previous = previous_jobs.get(job.name)
        if previous is None:
            continue
        for field_name in (
            "status",
            "gpu",
            "pid",
            "attempts",
            "oom_attempts",
            "generic_attempts",
            "started_at_utc",
            "finished_at_utc",
            "returncode",
            "log_path",
            "summary_path",
            "curve_json_path",
            "curve_csv_path",
            "tried_gpus",
            "notes",
        ):
            if field_name in previous:
                setattr(job, field_name, previous[field_name])
        if job.status == "running" and not _pid_is_alive(job.pid):
            job.notes.append(
                f"Recovered stale running state on {_utc_now()}; resetting to pending."
            )
            job.status = "pending"
            job.gpu = None
            job.pid = None
            job.started_at_utc = None
            job.finished_at_utc = None
            job.returncode = None
        elif retry_failed and job.status == "failed":
            job.notes.append(f"Reset failed job to pending on {_utc_now()} for retry.")
            job.status = "pending"
            job.gpu = None
            job.pid = None
            job.started_at_utc = None
            job.finished_at_utc = None
            job.returncode = None
    return jobs


def _start_job(job: Job, gpu: dict[str, int], launcher_log_root: Path) -> tuple[subprocess.Popen[str], Any]:
    launcher_log_root.mkdir(parents=True, exist_ok=True)
    log_path = launcher_log_root / f"{job.name}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_autotune_level=0",
        }
    )
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        job.command,
        cwd=REPO_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    job.status = "running"
    job.gpu = gpu["index"]
    job.pid = process.pid
    job.attempts += 1
    job.started_at_utc = _utc_now()
    job.finished_at_utc = None
    job.returncode = None
    job.log_path = str(log_path)
    if gpu["index"] not in job.tried_gpus:
        job.tried_gpus.append(gpu["index"])
    return process, handle


def _write_run_artifacts(job: Job) -> None:
    if job.log_path is None:
        return
    log_path = Path(job.log_path)
    run_dir = Path(job.run_dir)
    rows = load_metric_rows(log_path)
    actor_rows = [row for row in rows if row.get("section") == "actor"]
    evaluator_rows = [row for row in rows if row.get("section") == "evaluator"]
    summary = summarize_actor_metrics(actor_rows)
    summary["num_evaluator_checkpoints"] = len(evaluator_rows)
    summary["algorithm"] = job.algorithm
    summary["dataset"] = job.dataset
    summary["scenario"] = job.scenario
    summary["seed"] = job.seed
    curve = build_actor_success_curve(actor_rows)

    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    curve_json_path = run_dir / "success_curve.json"
    curve_csv_path = run_dir / "success_curve.csv"

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_curve_json(curve, curve_json_path)
    write_curve_csv(curve, curve_csv_path)

    job.summary_path = str(summary_path)
    job.curve_json_path = str(curve_json_path)
    job.curve_csv_path = str(curve_csv_path)


def _handle_finished_job(
    *,
    job: Job,
    process: subprocess.Popen[str],
    log_handle: Any,
    results_root: Path,
    running_log: Path,
    max_oom_retries: int,
    max_generic_retries: int,
) -> None:
    returncode = process.poll()
    if returncode is None:
        return

    log_handle.close()
    job.returncode = returncode
    job.finished_at_utc = _utc_now()

    log_text = ""
    if job.log_path is not None and Path(job.log_path).exists():
        log_text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")

    if returncode == 0:
        job.status = "completed"
        _write_run_artifacts(job)
        _append_log(
            results_root,
            running_log,
            f"Completed `{job.name}` successfully on GPU {job.gpu}.",
        )
        return

    is_oom = looks_like_oom(log_text)
    if is_oom:
        job.oom_attempts += 1
        if job.oom_attempts <= max_oom_retries:
            job.status = "pending"
            job.notes.append(
                f"OOM-like failure on GPU {job.gpu} at {_utc_now()}; retrying on another GPU (attempt {job.oom_attempts}/{max_oom_retries})."
            )
            _append_log(
                results_root,
                running_log,
                f"`{job.name}` hit an OOM-like failure on GPU {job.gpu}; re-queueing it.",
            )
        else:
            job.status = "failed"
            job.notes.append(
                f"Exceeded OOM retry budget ({max_oom_retries}) at {_utc_now()}."
            )
            _append_log(
                results_root,
                running_log,
                f"`{job.name}` exceeded OOM retries and is marked failed. See `{job.log_path}`.",
            )
    else:
        job.generic_attempts += 1
        if job.generic_attempts <= max_generic_retries:
            job.status = "pending"
            job.notes.append(
                f"Non-OOM failure at {_utc_now()}; retrying once more (attempt {job.generic_attempts}/{max_generic_retries})."
            )
            _append_log(
                results_root,
                running_log,
                f"`{job.name}` failed for a non-OOM reason; re-queueing it once more.",
            )
        else:
            job.status = "failed"
            job.notes.append(
                f"Exceeded generic retry budget ({max_generic_retries}) at {_utc_now()}."
            )
            _append_log(
                results_root,
                running_log,
                f"`{job.name}` failed and is recorded for manual debugging. See `{job.log_path}`.",
            )

    job.gpu = None
    job.pid = None
    if job.status == "pending":
        job.started_at_utc = None
        job.finished_at_utc = None
        job.returncode = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="subset_benchmark_arc_style.yaml")
    parser.add_argument("--system-config", default="ppo/ff_ppo")
    parser.add_argument("--algorithm-tag", default="ff_ppo")
    parser.add_argument("--datasets", nargs="+", default=["all"], choices=["all", "concept", "agi1"])
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--scenario-limit", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--total-num-envs", type=int, default=512)
    parser.add_argument("--num-evaluation", type=int, default=20)
    parser.add_argument("--num-eval-episodes", type=int, default=32)
    parser.add_argument("--max-concurrent-jobs", type=int, default=3)
    parser.add_argument("--max-jobs-per-gpu", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--min-free-mem-mib", type=int, default=20_000)
    parser.add_argument("--max-oom-retries", type=int, default=2)
    parser.add_argument("--max-generic-retries", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--network", default=None)
    parser.add_argument("--action-mode", default="point")
    parser.add_argument("--answer-grid", dest="answer_grid", action="store_true")
    parser.add_argument("--no-answer-grid", dest="answer_grid", action="store_false")
    parser.add_argument("--input-grid", dest="input_grid", action="store_true")
    parser.add_argument("--no-input-grid", dest="input_grid", action="store_false")
    parser.add_argument("--contextual", dest="contextual", action="store_true")
    parser.add_argument("--no-contextual", dest="contextual", action="store_false")
    parser.set_defaults(answer_grid=True, input_grid=False, contextual=False)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT / "ff_ppo_5m")
    parser.add_argument("--outputs-root", type=Path, default=DEFAULT_OUTPUTS_ROOT / "ff_ppo_5m")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_jobs_per_gpu < 1:
        raise ValueError("--max-jobs-per-gpu must be at least 1.")
    results_root = args.results_root.resolve()
    outputs_root = args.outputs_root.resolve()
    launcher_log_root = results_root / "launcher_logs"
    running_log = results_root / "RUNNING_LOG.md"
    status_path = results_root / "launcher_status.json"

    jobs = _merge_existing_status(
        _build_jobs(args, results_root, outputs_root),
        status_path,
        args.retry_failed,
    )

    _append_log(
        results_root,
        running_log,
        "Scheduler started/resumed with "
        f"{len(jobs)} jobs, max_concurrent_jobs={args.max_concurrent_jobs}, "
        f"max_jobs_per_gpu={args.max_jobs_per_gpu}, min_free_mem_mib={args.min_free_mem_mib}.",
    )
    _write_status(
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
        preview = [job.name for job in jobs[: min(10, len(jobs))]]
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
            _handle_finished_job(
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
            _write_status(
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
            avoid = set(next_job.tried_gpus) if next_job.oom_attempts > 0 else set()
            eligible_gpus = _available_gpus(
                jobs,
                min_free_mem_mib=args.min_free_mem_mib,
                avoid=avoid,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
            )
            if not eligible_gpus and avoid:
                eligible_gpus = _available_gpus(
                    jobs,
                    min_free_mem_mib=args.min_free_mem_mib,
                    avoid=set(),
                    max_jobs_per_gpu=args.max_jobs_per_gpu,
                )
            if not eligible_gpus:
                break

            process, handle = _start_job(next_job, eligible_gpus[0], launcher_log_root)
            active_processes[next_job.name] = process
            active_handles[next_job.name] = handle
            gpu_job_count = sum(
                1
                for job in jobs
                if job.status == "running" and job.gpu == eligible_gpus[0]["index"]
            )
            _append_log(
                results_root,
                running_log,
                f"Launched `{next_job.name}` on GPU {eligible_gpus[0]['index']} with "
                f"{eligible_gpus[0]['free_mem_mib']} MiB free; occupancy now "
                f"{gpu_job_count}/{args.max_jobs_per_gpu} jobs on that GPU.",
            )
            _write_status(
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
                _append_log(
                    results_root,
                    running_log,
                    f"Scheduler finished with {len(failures)} failed jobs recorded for debugging.",
                )
                _write_status(
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

            _append_log(results_root, running_log, "Scheduler finished all queued jobs.")
            _write_status(
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

        _write_status(
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
