"""Analysis of a tuning campaign: profiles, probability of improvement, tables and plots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxarc_baselines import tuning, tuning_report

ROOT = Path(__file__).resolve().parents[1]


def test_performance_profile_is_the_share_of_scores_at_or_above_tau():
    scores = np.array([[0.0, 0.5, 1.0, 1.0], [0.0, 0.25, 0.75, 1.0]])

    taus, frac, lo, hi = tuning_report.performance_profile(scores, taus=np.array([0.0, 0.5, 1.0]), reps=200)

    assert frac.tolist() == [1.0, 5 / 8, 3 / 8]
    assert np.all(lo <= frac + 1e-12) and np.all(frac <= hi + 1e-12)


def test_probability_of_improvement_counts_ties_as_half():
    a = np.array([[1.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0]])

    # Task 0: a always wins (1.0); task 1: always ties (0.5). Mean over tasks.
    assert tuning_report.probability_of_improvement(a, b) == pytest.approx(0.75)
    assert tuning_report.probability_of_improvement(b, a) == pytest.approx(0.25)
    assert tuning_report.probability_of_improvement(a, a) == pytest.approx(0.5)


def make_campaign(tmp_path, n_configs=4, s1_seeds=(1, 2)):
    from test_tuning import write_run  # noqa: PLC0415  (pytest puts tests/ on sys.path)

    campaign = tuning.load_campaign(ROOT / "experiments" / "tuning" / "campaigns" / "E-ppo-mini.yaml")
    space = tuning.load_space(ROOT / "experiments" / "tuning" / "spaces" / "ppo_shared.yaml")
    d = tmp_path / "camp"
    d.mkdir()
    (d / "campaign.yaml").write_text((ROOT / "experiments" / "tuning" / "campaigns" / "E-ppo-mini.yaml").read_text())
    (d / "space.yaml").write_text((ROOT / "experiments" / "tuning" / "spaces" / "ppo_shared.yaml").read_text())
    configs = tuning.sample_configs(space, n_configs, campaign["sample_seed"])
    (d / "configs.json").write_text(json.dumps(configs))
    rows = tuning.stage_matrix(campaign, space, configs, "s1", seeds=list(s1_seeds))
    for r in rows:
        r["overrides"] = [o for o in r["overrides"] if not o.startswith("arch.num_evaluation")]
    tuning.write_matrix(rows, d / "s1.jsonl")
    (d / "s1").mkdir()
    for i, r in enumerate(rows):
        e = 0.1 * int(r["config_id"][1:])
        write_run(d / "s1", r["name"], exact=[0, e, e], submit=[0.3] * 3, trunc=[80] * 3, success=[1.0, 2.0, 3.0])
    return d, configs


def test_task_scores_are_seed_by_task_matrices(tmp_path):
    d, configs = make_campaign(tmp_path)

    scores = tuning_report.task_scores(d / "s1", tuning.read_matrix(d / "s1.jsonl"), "exact")

    assert set(scores) == {c["config_id"] for c in configs}
    assert scores["c02"].shape == (2, 2)
    assert scores["c02"][:, 0] == pytest.approx([0.2, 0.2])


def test_report_writes_tables_and_sensitivity_plots(tmp_path):
    d, _ = make_campaign(tmp_path)

    out = tuning_report.report(d, reps=50)

    text = (out / "report.md").read_text()
    assert "Stage s1" in text and "c03" in text and "GPU-hours" in text
    assert (out / "sensitivity_s1.png").exists()
    summary = json.loads((out / "report.json").read_text())
    assert summary["s1"]["configs"][0]["config_id"] == "c03"
    assert summary["s1"]["gpu_hours"] == pytest.approx(8 * 123 / 3600)
