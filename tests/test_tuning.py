"""The two-stage random search: sampling, run matrices, selection and the frozen config."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from jaxarc_baselines import tuning

ROOT = Path(__file__).resolve().parents[1]
SPACES = ROOT / "experiments" / "tuning" / "spaces"
CAMPAIGNS = ROOT / "experiments" / "tuning" / "campaigns"


def ppo_space():
    return tuning.load_space(SPACES / "ppo_shared.yaml")


def mini_campaign():
    return tuning.load_campaign(CAMPAIGNS / "E-ppo-mini.yaml")


# --- search space and sampling ---------------------------------------------------


def test_published_spaces_have_at_most_six_knobs():
    for path in sorted(SPACES.glob("*.yaml")):
        space = tuning.load_space(path)
        assert 1 <= len(space.knobs) <= 6, path.name


def test_sampling_is_deterministic_in_the_seed():
    a = tuning.sample_configs(ppo_space(), 32, seed=7)
    b = tuning.sample_configs(ppo_space(), 32, seed=7)
    c = tuning.sample_configs(ppo_space(), 32, seed=8)

    assert a == b
    assert a != c
    assert [cfg["config_id"] for cfg in a] == [f"c{i:02d}" for i in range(32)]


def test_samples_stay_in_range_and_use_each_distribution():
    space = ppo_space()
    configs = tuning.sample_configs(space, 400, seed=0)
    for knob in space.knobs:
        values = [cfg["values"][knob.name] for cfg in configs]
        if knob.dist == "choice":
            assert {json.dumps(v, sort_keys=True) for v in values} == {
                json.dumps(v, sort_keys=True) for v in knob.values
            }
        else:
            assert min(values) >= knob.low and max(values) <= knob.high
    # Log-uniform: about half the lr samples fall below the geometric midpoint.
    lr = [cfg["values"]["lr"] for cfg in configs]
    mid = math.sqrt(1e-4 * 3e-3)
    assert 0.4 < sum(v < mid for v in lr) / len(lr) < 0.6


def test_sampled_floats_are_rounded_to_three_significant_figures():
    for cfg in tuning.sample_configs(ppo_space(), 20, seed=3):
        lr = cfg["values"]["lr"]
        assert lr == float(f"{lr:.3g}")


def test_knob_overrides_cover_scalar_and_joint_knobs():
    space = ppo_space()
    values = {"lr": 1e-3, "ent_coef": 0.01, "vf_coef": 0.5, "clip_eps": 0.2,
              "gae_lambda": 0.95, "update": {"system.epochs": 4, "system.num_minibatches": 16}}

    overrides = tuning.knob_overrides(space, values)

    assert "system.lr=0.001" in overrides
    assert "system.epochs=4" in overrides and "system.num_minibatches=16" in overrides
    assert len(overrides) == 7


def test_unknown_distribution_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("name: bad\nschedule: lr_horizon\nknobs:\n  lr: {path: system.lr, dist: normal, low: 0, high: 1}\n")

    with pytest.raises(ValueError, match="normal"):
        tuning.load_space(path)


# --- run matrices ---------------------------------------------------------------


def test_stage1_runs_the_final_schedule_and_stops_at_half():
    campaign, space = mini_campaign(), ppo_space()
    configs = tuning.sample_configs(space, campaign["stage1"]["configs"], campaign["sample_seed"])

    rows = tuning.stage_matrix(campaign, space, configs, "s1")

    assert len(rows) == 32 * 3
    row = rows[0]
    assert row["name"] == "c00__s1" and row["config_id"] == "c00" and row["seed"] == 1
    ov = row["overrides"]
    assert "arch.total_timesteps=100000000" in ov
    assert "system.lr_schedule_timesteps=200000000" in ov
    assert "arch.num_evaluation=10" in ov
    assert "+env.eval_scenario=Mini-split_dev" in ov
    assert "+run_meta.stage=tune-s1" in ov and "+run_meta.config_id=c00" in ov
    assert "+run_meta.campaign_id=E-ppo-mini" in ov
    for fixed in ("system=ppo/ff_ppo_shared", "network=arc_spatial_shared", "reward=pbrs"):
        assert fixed in ov


def test_stage2_and_final_run_the_full_budget_on_fresh_seeds():
    campaign, space = mini_campaign(), ppo_space()
    configs = tuning.sample_configs(space, 4, 0)

    s2 = tuning.stage_matrix(campaign, space, configs, "s2")
    final = tuning.stage_matrix(campaign, space, configs[:1], "final")

    assert {r["seed"] for r in s2} == {101, 102, 103, 104, 105}
    assert len(final) == 20 and {r["seed"] for r in final} == set(range(1001, 1021))
    for row in s2 + final:
        assert "arch.total_timesteps=200000000" in row["overrides"]
        assert "arch.num_evaluation=20" in row["overrides"]
        assert not any(o.startswith("system.lr_schedule_timesteps") for o in row["overrides"])
    assert "+env.eval_scenario=Mini-split_test" in final[0]["overrides"]
    assert "+run_meta.stage=final" in final[0]["overrides"]


def test_stage_seeds_never_overlap():
    campaign = mini_campaign()
    s1, s2, fin = (set(tuning.stage_seeds(campaign, s)) for s in ("s1", "s2", "final"))

    assert not (s1 & s2) and not (s1 & fin) and not (s2 & fin)


def test_epsilon_schedule_is_stretched_for_pqn_stage1():
    space = tuning.load_space(SPACES / "pqn.yaml")
    campaign = dict(mini_campaign(), space="pqn")
    cfg = tuning.sample_configs(space, 1, 0)
    frac = cfg[0]["values"]["exploration_fraction"]

    s1 = tuning.stage_matrix(campaign, space, cfg, "s1")[0]["overrides"]
    s2 = tuning.stage_matrix(campaign, space, cfg, "s2")[0]["overrides"]

    # Over the stage-1 run, epsilon decays as it does over the first half of a full run.
    assert f"system.exploration_fraction={frac * 2:.6g}" in s1
    assert f"system.exploration_fraction={frac:.6g}" in s2


def test_matrix_round_trips_through_jsonl(tmp_path):
    campaign, space = mini_campaign(), ppo_space()
    rows = tuning.stage_matrix(campaign, space, tuning.sample_configs(space, 2, 0), "s1")
    path = tmp_path / "s1.jsonl"

    tuning.write_matrix(rows, path)

    assert tuning.read_matrix(path) == rows


# --- results, selection, freezing ----------------------------------------------


S2 = range(101, 106)


def fake_result(config_id, seed, exact, success=0.0, cell=0.0, collapsed=False):
    return tuning.RunResult(config_id=config_id, seed=seed, exact=exact, cell=cell,
                            train_success=success, collapsed=collapsed, wall_s=60.0)


def test_scores_are_seed_means_ranked_by_exact_then_tie_break():
    results = [
        fake_result("c00", 1, 0.2), fake_result("c00", 2, 0.4),
        fake_result("c01", 1, 0.3), fake_result("c01", 2, 0.3),  # same mean, more training success
        fake_result("c02", 1, 0.0, success=9.0), fake_result("c02", 2, 0.0, success=9.0),
    ]
    results[2] = results[2]._replace(train_success=5.0)
    results[3] = results[3]._replace(train_success=5.0)

    scores = tuning.score_configs(results, tie_break="train_success")

    assert [s["config_id"] for s in scores] == ["c01", "c00", "c02"]
    assert scores[0]["score"] == pytest.approx(0.3)
    assert scores[0]["n"] == 2


def test_select_top_refuses_missing_runs_unless_allowed():
    results = [fake_result("c00", 1, 0.5), fake_result("c01", 1, 0.1), fake_result("c01", 2, 0.1)]

    with pytest.raises(ValueError, match="c00"):
        tuning.select_top(results, k=1, expected_seeds=[1, 2], tie_break="train_success")
    top = tuning.select_top(results, k=1, expected_seeds=[1, 2], tie_break="train_success",
                            allow_missing=True)
    assert top == ["c00"]


def test_collapse_alarm_stops_selection():
    results = [fake_result(f"c{i:02d}", 1, 0.1, collapsed=i < 2) for i in range(10)]

    with pytest.raises(tuning.CollapseAlarm):
        tuning.select_top(results, k=4, expected_seeds=[1], tie_break="train_success")
    assert len(tuning.select_top(results, k=4, expected_seeds=[1], tie_break="train_success",
                                 force=True)) == 4
    # One in ten is at the limit, not over it.
    ok = [r._replace(collapsed=r.config_id == "c00") for r in results]
    assert len(tuning.select_top(ok, k=4, expected_seeds=[1], tie_break="train_success")) == 4


def test_frozen_config_round_trips(tmp_path):
    campaign, space = mini_campaign(), ppo_space()
    configs = tuning.sample_configs(space, 4, 0)
    s2 = [fake_result(c["config_id"], s, 0.1 * i, cell=0.5) for i, c in enumerate(configs) for s in S2]

    frozen = tuning.freeze(campaign, space, configs, s2)
    path = tmp_path / "frozen.yaml"
    tuning.write_frozen(frozen, path)
    loaded = tuning.load_frozen(path)

    assert loaded == frozen
    assert loaded["config_id"] == "c03"
    winner = next(c for c in configs if c["config_id"] == "c03")
    assert loaded["values"] == winner["values"]
    assert loaded["overrides"] == tuning.knob_overrides(space, winner["values"])
    final = tuning.stage_matrix(campaign, space, [tuning.frozen_as_config(loaded)], "final")
    assert all(set(loaded["overrides"]) <= set(r["overrides"]) for r in final)
    assert final[0]["name"] == "frozen__s1001"


def test_audit_configs_are_drawn_from_below_the_cut_and_checked():
    campaign = dict(mini_campaign())
    campaign["stage2"] = dict(campaign["stage2"], audit={"configs": 2, "seeds": 3})
    ranked = [f"c{i:02d}" for i in range(32)]

    audit = tuning.audit_ids(campaign, ranked)
    assert len(audit) == 2 and set(audit) <= set(ranked[4:])
    assert audit == tuning.audit_ids(campaign, ranked)

    space = ppo_space()
    configs = tuning.sample_configs(space, 32, 0)
    s2 = [fake_result(c, s, 0.30) for c in ranked[:4] for s in S2]
    s2 += [fake_result(audit[0], s, 0.40) for s in S2[:3]]
    frozen = tuning.freeze(campaign, space, configs, s2, audit=audit)

    assert frozen["config_id"] in ranked[:4]  # audit configs are never picked
    assert frozen["audit"]["extend_stage2"] is True


def write_run(stage_dir, name, exact, submit, trunc, success, n_evals=3):
    held = stage_dir / name / "held_out" / "ff_ppo_shared"
    held.mkdir(parents=True)
    lines = []
    for i in range(n_evals):
        lines.append(json.dumps({"event": "evaluator", "t": (i + 1) * 1000, "submit_rate": submit[i],
                                 "per_task": {"exact": [exact[i], 0.0], "cell_accuracy": [0.5, 0.25]}}))
    (held / "x_seed1.jsonl").write_text("\n".join(lines) + "\n")
    log = "".join(f"\x1b[32mACTOR - Steps per second: 1 | Success rate: {s} | Truncation rate: {t}\x1b[0m\n"
                  for s, t in zip(success, trunc))
    (stage_dir / f"{name}.log").write_text(log)
    (stage_dir / f"{name}.wall").write_text("123\n")


def row(name, cid="c00", seed=1, evals=3):
    return {"name": name, "config_id": cid, "seed": seed, "overrides": [f"arch.num_evaluation={evals}"]}


def test_results_are_read_from_the_run_folders(tmp_path):
    write_run(tmp_path, "c00__s1", exact=[0, 0.5, 1.0], submit=[0.2, 0.3, 0.4],
              trunc=[80, 70, 60], success=[1.0, 2.0, 3.5])

    r = tuning.read_run(tmp_path, row("c00__s1"))

    assert r.exact == pytest.approx(0.5) and r.cell == pytest.approx(0.375)
    assert r.train_success == 3.5 and r.collapsed is False and r.wall_s == 123


def test_collapsed_runs_are_flagged(tmp_path):
    write_run(tmp_path, "c00__s1", exact=[0, 0, 0], submit=[0, 0, 0],
              trunc=[99.95, 100, 100], success=[0, 0, 0])

    assert tuning.read_run(tmp_path, row("c00__s1")).collapsed is True
    assert tuning.is_collapsed([0.1, 0, 0], [100, 100, 100]) is False
    assert tuning.is_collapsed([0, 0, 0], [99.0, 100, 100]) is False


def test_unfinished_runs_are_skipped(tmp_path):
    write_run(tmp_path, "c00__s1", exact=[0, 0], submit=[0.1, 0.1], trunc=[80, 80], success=[0, 0], n_evals=2)

    assert tuning.read_run(tmp_path, row("c00__s1", evals=3)) is None
    assert tuning.read_run(tmp_path, row("c00__s2")) is None
    assert tuning.read_results(tmp_path, [row("c00__s1", evals=2), row("c00__s2")])[0].seed == 1


@pytest.mark.parametrize("stage", ["s1", "s2", "final"])
def test_matrix_overrides_compose_with_hydra(stage):
    from hydra import compose, initialize_config_dir

    campaign, space = mini_campaign(), ppo_space()
    row = tuning.stage_matrix(campaign, space, tuning.sample_configs(space, 1, 0), stage)[0]
    with initialize_config_dir(str(ROOT / "experiments" / "configs"), version_base="1.2"):
        cfg = compose(campaign["config_name"], overrides=row["overrides"])

    assert cfg.system.system_name == "ff_ppo_shared"
    assert cfg.network.actor_network.action_head.submit_bias == 1.0
    assert cfg.run_meta.stage == tuning.RUN_META_STAGE[stage]
    assert cfg.run_meta.config_id == row["config_id"]
    assert cfg.arch.seed == row["seed"]
    assert cfg.system.lr_schedule_timesteps == (200_000_000 if stage == "s1" else None)
