from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "neurips_ed_final_program"
LOG_ROOT = RESULTS_ROOT / "launcher_logs"
STATUS_PATH = RESULTS_ROOT / "launcher_status.json"
RUNNING_LOG = RESULTS_ROOT / "RUNNING_LOG.md"

POLL_SECONDS = int(os.getenv("JAXARC_SCHEDULER_POLL_SECONDS", "60"))
MAX_CONCURRENT_JOBS = int(os.getenv("JAXARC_MAX_PARALLEL_JOBS", "5"))
DEFAULT_MIN_FREE_MEM_MIB = int(os.getenv("JAXARC_MIN_FREE_MEM_MIB", "20000"))
RETRY_FAILED_JOBS = os.getenv("JAXARC_RETRY_FAILED_JOBS", "true").lower() == "true"


@dataclass
class Job:
    name: str
    command: list[str]
    paper_destination: str
    result_root: str
    min_free_mem_mib: int = DEFAULT_MIN_FREE_MEM_MIB
    depends_on: str | None = None
    status: str = "pending"
    gpu: int | None = None
    pid: int | None = None
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    returncode: int | None = None
    log_path: str | None = None
    notes: list[str] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _append_log(message: str) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if not RUNNING_LOG.exists():
        RUNNING_LOG.write_text("# Final program running log\n\n", encoding="utf-8")
    with RUNNING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- **{_utc_now()}** — {message}\n")


def _write_status(jobs: list[Job]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "updated_at_utc": _utc_now(),
        "poll_seconds": POLL_SECONDS,
        "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
        "gpu_policy": "free_memory_only_ignore_utilization",
        "jobs": [asdict(job) for job in jobs],
    }
    STATUS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


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


def _available_gpus(jobs: list[Job], *, min_free_mem_mib: int) -> list[dict[str, int]]:
    active_gpu_ids = {
        job.gpu for job in jobs if job.status == "running" and job.gpu is not None
    }
    eligible = [
        gpu
        for gpu in _query_gpus()
        if gpu["free_mem_mib"] >= min_free_mem_mib and gpu["index"] not in active_gpu_ids
    ]
    eligible.sort(key=lambda gpu: (-gpu["free_mem_mib"], gpu["index"]))
    return eligible


def _build_jobs() -> list[Job]:
    return [
        Job(
            name="r0_ff_dqn_mini_easy_smoke",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "feasibility_r0_ff_dqn_mini_easy.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="R0 feasibility gate for cross-family baselines",
            result_root="results/neurips_ed_final_program/r0_feasibility",
        ),
        Job(
            name="r0_ff_reinforce_mini_easy_smoke",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "feasibility_r0_ff_reinforce_mini_easy.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="R0 feasibility gate for cross-family baselines",
            result_root="results/neurips_ed_final_program/r0_feasibility",
        ),
        Job(
            name="r0_ff_pqn_mini_easy_smoke",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "feasibility_r0_ff_pqn_mini_easy.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="R0 feasibility gate for cross-family baselines",
            result_root="results/neurips_ed_final_program/r0_feasibility",
        ),
        Job(
            name="phasea_ff_ppo_mini_all_512k",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phasea_ff_ppo_mini_all_512k.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-A PPO baseline",
            result_root="results/neurips_ed_final_program/phasea_mini_all",
        ),
        Job(
            name="phasea_ff_reinforce_mini_all_524k",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phasea_ff_reinforce_mini_all_524k.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-A REINFORCE baseline",
            result_root="results/neurips_ed_final_program/phasea_mini_all",
            depends_on="r0_ff_reinforce_mini_easy_smoke",
        ),
        Job(
            name="phasea_ff_pqn_mini_all_512k",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phasea_ff_pqn_mini_all_512k.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-A PQN baseline",
            result_root="results/neurips_ed_final_program/phasea_mini_all",
            depends_on="r0_ff_pqn_mini_easy_smoke",
        ),
        Job(
            name="phasea_ff_dqn_mini_all_512k",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phasea_ff_dqn_mini_all_512k.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-A DQN baseline",
            result_root="results/neurips_ed_final_program/phasea_mini_all",
            depends_on="r0_ff_dqn_mini_easy_smoke",
        ),
        Job(
            name="phaseb_ff_ppo_mini_all_100m",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phaseb_ff_ppo_mini_all_100m.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-B PPO long-budget baseline",
            result_root="results/neurips_ed_final_program/phaseb_mini_all",
            depends_on="phasea_ff_ppo_mini_all_512k",
        ),
        Job(
            name="phaseb_ff_reinforce_mini_all_100m",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "phaseb_ff_reinforce_mini_all_100m.yaml",
                "hydra.job.chdir=false",
            ],
            paper_destination="Mini-all phase-B REINFORCE long-budget baseline",
            result_root="results/neurips_ed_final_program/phaseb_mini_all",
            depends_on="phasea_ff_reinforce_mini_all_524k",
        ),
    ]


def _load_existing_status() -> dict[str, dict[str, Any]]:
    if not STATUS_PATH.exists():
        return {}
    payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    jobs = payload.get("jobs", [])
    return {job["name"]: job for job in jobs if "name" in job}


def _pid_is_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _merge_existing_status(jobs: list[Job]) -> list[Job]:
    previous = _load_existing_status()
    for job in jobs:
        prior = previous.get(job.name)
        if prior is None:
            continue
        for field_name in (
            "status",
            "gpu",
            "pid",
            "started_at_utc",
            "finished_at_utc",
            "returncode",
            "log_path",
            "notes",
        ):
            if field_name in prior:
                setattr(job, field_name, prior[field_name])
        if RETRY_FAILED_JOBS and job.status in {"failed", "blocked"}:
            job.notes.append(
                f"Reset terminal status `{job.status}` to pending on {_utc_now()} for retry."
            )
            job.status = "pending"
            job.gpu = None
            job.pid = None
            job.started_at_utc = None
            job.finished_at_utc = None
            job.returncode = None
        if job.status == "running" and not _pid_is_alive(job.pid):
            job.notes.append(
                f"Recovered stale running state on {_utc_now()}; resetting job to pending."
            )
            job.status = "pending"
            job.gpu = None
            job.pid = None
            job.started_at_utc = None
            job.finished_at_utc = None
            job.returncode = None
    return jobs


def _job_by_name(jobs: list[Job], name: str | None) -> Job | None:
    if name is None:
        return None
    for job in jobs:
        if job.name == name:
            return job
    return None


def _dependency_satisfied(jobs: list[Job], job: Job) -> bool:
    dep = _job_by_name(jobs, job.depends_on)
    if dep is None:
        return True
    return dep.status == "completed"


def _dependency_failed(jobs: list[Job], job: Job) -> bool:
    dep = _job_by_name(jobs, job.depends_on)
    if dep is None:
        return False
    return dep.status in {"failed", "blocked"}


def _start_job(job: Job, gpu: dict[str, int]) -> subprocess.Popen[str]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{job.name}.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu["index"]),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
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
    job.started_at_utc = _utc_now()
    job.log_path = str(log_path)
    _append_log(
        f"Launched `{job.name}` on GPU {gpu['index']} with {gpu['free_mem_mib']} MiB free "
        f"(utilization {gpu['utilization']}%). Policy ignores utilization and gates on free memory only."
    )
    return process


def main() -> int:
    jobs = _merge_existing_status(_build_jobs())
    _append_log(
        "Final-program scheduler started/resumed. It launches up to "
        f"{MAX_CONCURRENT_JOBS} jobs in parallel, one per visible GPU, and treats GPU "
        "utilization as informational rather than blocking."
    )
    _write_status(jobs)

    active_processes: dict[str, subprocess.Popen[str]] = {}

    while True:
        for job in jobs:
            if job.status == "pending" and _dependency_failed(jobs, job):
                job.status = "blocked"
                job.finished_at_utc = _utc_now()
                job.notes.append(
                    f"Blocked because dependency `{job.depends_on}` did not complete successfully."
                )
                _append_log(
                    f"Blocked `{job.name}` because dependency `{job.depends_on}` failed or was blocked."
                )

        for name, process in list(active_processes.items()):
            job = _job_by_name(jobs, name)
            if job is None:
                active_processes.pop(name, None)
                continue
            returncode = process.poll()
            if returncode is None:
                continue
            job.returncode = returncode
            job.finished_at_utc = _utc_now()
            if returncode == 0:
                job.status = "completed"
                _append_log(f"Completed `{job.name}` successfully.")
            else:
                job.status = "failed"
                _append_log(
                    f"`{job.name}` exited with return code {returncode}. See `{job.log_path}`."
                )
            active_processes.pop(name, None)

        while len(active_processes) < MAX_CONCURRENT_JOBS:
            next_job = next(
                (
                    job
                    for job in jobs
                    if job.status == "pending" and _dependency_satisfied(jobs, job)
                ),
                None,
            )
            if next_job is None:
                break

            eligible_gpus = _available_gpus(jobs, min_free_mem_mib=next_job.min_free_mem_mib)
            if not eligible_gpus:
                break

            process = _start_job(next_job, eligible_gpus[0])
            active_processes[next_job.name] = process
            _write_status(jobs)

        terminal_statuses = {"completed", "failed", "blocked"}
        if all(job.status in terminal_statuses for job in jobs):
            if any(job.status in {"failed", "blocked"} for job in jobs):
                _append_log(
                    "Final-program scheduler reached a terminal state with some failed or blocked jobs."
                )
                _write_status(jobs)
                return 1
            _append_log("Final-program scheduler finished all queued jobs.")
            _write_status(jobs)
            return 0

        _write_status(jobs)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())