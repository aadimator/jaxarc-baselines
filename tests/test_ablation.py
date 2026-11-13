"""Tests for the network-ablation analysis: held-out scores, IQM and bootstrap CIs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxarc_baselines import ablation


def record(t: int, exact: list[float], event: str = "evaluator") -> dict:
    return {
        "event": event,
        "t": t,
        "eval_step": t // 10,
        "seed": 1,
        "exact": float(np.mean(exact)),
        "per_task": {
            "task_idx": list(range(len(exact))),
            "exact": exact,
            "cell_accuracy": [0.5 + e / 2 for e in exact],
        },
    }


def test_read_held_out_reads_json_lines(tmp_path):
    path = tmp_path / "run_seed1.jsonl"
    path.write_text("\n".join(json.dumps(record(t, [0.0, 1.0])) for t in (10, 20)) + "\n")

    records = ablation.read_held_out(path)

    assert [r["t"] for r in records] == [10, 20]


def test_final_scores_come_from_the_last_evaluation():
    records = [record(20, [1.0, 1.0]), record(10, [0.0, 0.0]), record(30, [9.0, 9.0], "absolute")]

    scores = ablation.final_task_scores(records, "exact")

    np.testing.assert_array_equal(scores, [1.0, 1.0])
    np.testing.assert_array_equal(ablation.final_task_scores(records, "cell_accuracy"), [1.0, 1.0])


def test_curve_is_sorted_by_step():
    records = [record(20, [1.0, 0.0]), record(10, [0.0, 0.0])]

    steps, scores = ablation.curve(records, "exact")

    np.testing.assert_array_equal(steps, [10, 20])
    np.testing.assert_array_equal(scores, [[0.0, 0.0], [1.0, 0.0]])


def test_iqm_is_the_mean_of_the_middle_half():
    assert ablation.iqm(np.arange(8.0)) == pytest.approx(3.5)
    assert ablation.iqm(np.array([[1, 1, 1, 1], [100, -100, 1, 1]])) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=(3, 30))

    lo, hi = ablation.bootstrap_ci(scores, reps=500, seed=1)

    assert lo <= ablation.iqm(scores) <= hi
    assert (lo, hi) == ablation.bootstrap_ci(scores, reps=500, seed=1)
    assert ablation.bootstrap_ci(np.full((3, 30), 0.4), reps=200) == pytest.approx((0.4, 0.4))


def test_bootstrap_diff_detects_a_gap_and_its_absence():
    rng = np.random.default_rng(0)
    base = rng.uniform(size=(3, 30))

    same, lo, hi = ablation.bootstrap_diff(base, base.copy(), reps=500)
    assert same == pytest.approx(0.0)
    assert lo <= 0.0 <= hi
    assert ablation.within_noise(lo, hi)

    gap, lo, hi = ablation.bootstrap_diff(base + 0.5, base, reps=500)
    assert gap == pytest.approx(0.5)
    assert lo > 0
    assert not ablation.within_noise(lo, hi)


def test_run_names_carry_config_lr_and_seed():
    assert ablation.parse_run_name("shared-cv0.25__lr1e-3__s2") == ("shared-cv0.25", "1e-3", 2)
    with pytest.raises(ValueError):
        ablation.parse_run_name("shared-cv0.25-s2")


def write_run(root, name: str, exact: list[float]) -> None:
    path = root / name / "held_out" / "ff_ppo" / "20261112000000_seed1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record(10, [0.0] * len(exact))) + "\n" + json.dumps(record(20, exact)) + "\n")


def test_collect_groups_runs_by_config_and_lr(tmp_path):
    write_run(tmp_path, "sep__lr3e-4__s1", [1.0, 0.0])
    write_run(tmp_path, "sep__lr3e-4__s2", [1.0, 1.0])
    write_run(tmp_path, "sep__lr1e-3__s1", [0.0, 0.0])
    (tmp_path / "preflight" / "held_out").mkdir(parents=True)  # not a run name: skipped

    runs = ablation.collect(tmp_path, "exact")

    assert set(runs) == {("sep", "3e-4"), ("sep", "1e-3")}
    np.testing.assert_array_equal(runs[("sep", "3e-4")].scores, [[1.0, 0.0], [1.0, 1.0]])
    assert runs[("sep", "3e-4")].seeds == [1, 2]
    np.testing.assert_array_equal(runs[("sep", "1e-3")].steps, [10, 20])
    assert runs[("sep", "1e-3")].curves.shape == (1, 2, 2)  # seeds, evals, tasks


def test_best_lr_picks_the_highest_iqm():
    rng = np.random.default_rng(0)
    low, high = rng.uniform(0, 0.2, (3, 10)), rng.uniform(0.6, 0.8, (3, 10))
    runs = {
        ("sep", "3e-4"): ablation.Runs([1, 2, 3], low, np.array([1]), low[:, None]),
        ("sep", "1e-3"): ablation.Runs([1, 2, 3], high, np.array([1]), high[:, None]),
    }

    assert ablation.best_lr(runs, "sep") == "1e-3"
