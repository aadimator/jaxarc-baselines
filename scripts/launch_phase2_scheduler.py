from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "neurips_ed_phase2"
LOG_ROOT = RESULTS_ROOT / "launcher_logs"
STATUS_PATH = RESULTS_ROOT / "launcher_status.json"
RUNNING_LOG = RESULTS_ROOT / "RUNNING_LOG.md"

POLL_SECONDS = 60
MIN_FREE_MEM_MIB = 40_000


@dataclass
class Job:
    name: str
    command: list[str]
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
        RUNNING_LOG.write_text("# Phase 2 Running Log\n\n", encoding="utf-8")
    with RUNNING_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"- **{_utc_now()}** — {message}\n")


def _write_status(jobs: list[Job]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "updated_at_utc": _utc_now(),
        "poll_seconds": POLL_SECONDS,
        "min_free_mem_mib": MIN_FREE_MEM_MIB,
        "gpu_policy": "memory_only",
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


def _pick_gpu() -> dict[str, int] | None:
    eligible = [gpu for gpu in _query_gpus() if gpu["free_mem_mib"] >= MIN_FREE_MEM_MIB]
    if not eligible:
        return None
    eligible.sort(
        key=lambda gpu: (gpu["utilization"], -gpu["free_mem_mib"], gpu["index"])
    )
    return eligible[0]


def _build_jobs() -> list[Job]:
    return [
        Job(
            name="phase2_agi1_entry_no_adapt",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_agi1_entry_no_adapt.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_agi1_entry_deep",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_agi1_entry_deep.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_agi1_entry_pretrain",
            command=[
                "pixi",
                "run",
                "python",
                "run_experiment.py",
                "--config-name",
                "pretrain_phase2_ffppo_agi1_entry.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_agi1_entry_pretrained",
            depends_on="phase2_agi1_entry_pretrain",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_agi1_entry_pretrained.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_agi1_entry_recurrent",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_agi1_entry_recurrent.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_mini_all_deep",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_mini_all_deep.yaml",
                "hydra.job.chdir=false",
            ],
        ),
        Job(
            name="phase2_concept_all_deep",
            command=[
                "pixi",
                "run",
                "python",
                "scripts/run_ttrl_benchmark.py",
                "--config-name",
                "ttrl_phase2_concept_all_deep.yaml",
                "hydra.job.chdir=false",
            ],
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


def _start_job(job: Job, gpu: dict[str, int]) -> subprocess.Popen[str]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{job.name}.log"
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
    job.started_at_utc = _utc_now()
    job.log_path = str(log_path)
    _append_log(
        f"Launched `{job.name}` on GPU {gpu['index']} with command: {' '.join(job.command)}"
    )
    return process


def main() -> int:
    jobs = _merge_existing_status(_build_jobs())
    _append_log(
        "Phase 2 scheduler started/resumed. It will launch jobs serially when a GPU has "
        f">= {MIN_FREE_MEM_MIB} MiB free memory, regardless of current utilization."
    )
    _write_status(jobs)

    active_process: subprocess.Popen[str] | None = None
    active_job: Job | None = None

    while True:
        if active_process is not None and active_job is not None:
            returncode = active_process.poll()
            if returncode is not None:
                active_job.returncode = returncode
                active_job.finished_at_utc = _utc_now()
                if returncode == 0:
                    active_job.status = "completed"
                    _append_log(f"Completed `{active_job.name}` successfully.")
                else:
                    active_job.status = "failed"
                    _append_log(
                        f"`{active_job.name}` exited with return code {returncode}. See `{active_job.log_path}`."
                    )
                    _write_status(jobs)
                    return returncode
                active_process = None
                active_job = None
                _write_status(jobs)
                continue

        if all(job.status == "completed" for job in jobs):
            _append_log("Phase 2 scheduler finished all queued jobs.")
            _write_status(jobs)
            return 0

        if active_process is None:
            next_job = next(
                (
                    job
                    for job in jobs
                    if job.status == "pending" and _dependency_satisfied(jobs, job)
                ),
                None,
            )
            if next_job is not None:
                gpu = _pick_gpu()
                if gpu is not None:
                    active_process = _start_job(next_job, gpu)
                    active_job = next_job
                    _write_status(jobs)
                    continue

        _write_status(jobs)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
