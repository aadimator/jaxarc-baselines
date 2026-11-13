"""Tests for run metadata: project, tags, group, job type and code provenance."""

from __future__ import annotations

import os

import pytest
from omegaconf import OmegaConf

from jaxarc_baselines import run_meta


def base_cfg(**meta):
    cfg = OmegaConf.create(
        {
            "system": {"system_name": "ff_ppo"},
            "env": {"scenario": {"name": "Mini-all"}},
            "logger": {
                "loggers": {
                    "wandb": {"project": "jaxarc-baselines", "tag": ["delete"], "group_tag": ["delete"]}
                }
            },
        }
    )
    if meta:
        cfg.run_meta = meta
    OmegaConf.set_struct(cfg, False)
    return cfg


CLEAN = {
    name: {"commit": "a" * 40, "dirty": False} for name in ("baselines", "jaxarc", "stoix")
}


@pytest.fixture(autouse=True)
def clean_trees(monkeypatch):
    monkeypatch.setattr(run_meta, "provenance", lambda: {k: dict(v) for k, v in CLEAN.items()})
    monkeypatch.delenv("WANDB_JOB_TYPE", raising=False)


def test_config_without_run_meta_is_unchanged():
    cfg = base_cfg()
    before = OmegaConf.to_container(cfg)

    run_meta.apply_run_meta(cfg)

    assert OmegaConf.to_container(cfg) == before
    assert "WANDB_JOB_TYPE" not in os.environ


def test_run_meta_sets_project_tags_group_and_job_type():
    cfg = base_cfg(stage="tune-l1", track="E", question="Q2", campaign_id="E-ppo-mini")

    run_meta.apply_run_meta(cfg)

    wandb = cfg.logger.loggers.wandb
    assert wandb.project == "jaxarc-tmlr"
    assert list(wandb.tag) == ["tune-l1", "E", "Q2", "ff_ppo", "Mini-all"]
    assert list(wandb.group_tag) == ["E-ppo-mini"]
    assert os.environ["WANDB_JOB_TYPE"] == "tune-l1"


def test_provenance_is_recorded_in_the_config():
    cfg = base_cfg(stage="smoke")

    run_meta.apply_run_meta(cfg)

    assert cfg.run_meta.provenance.jaxarc.commit == "a" * 40
    assert cfg.run_meta.provenance.stoix.dirty == "no"
    assert cfg.run_meta.code_state == "clean"


def test_smoke_run_needs_no_track_or_campaign():
    cfg = base_cfg(stage="smoke")

    run_meta.apply_run_meta(cfg)

    assert list(cfg.logger.loggers.wandb.tag) == ["smoke", "ff_ppo", "Mini-all"]
    assert list(cfg.logger.loggers.wandb.group_tag) == []


def test_extra_tags_are_appended():
    cfg = base_cfg(stage="smoke", tags=["gpu-check"])

    run_meta.apply_run_meta(cfg)

    assert list(cfg.logger.loggers.wandb.tag)[-1] == "gpu-check"


@pytest.mark.parametrize(
    "meta",
    [
        {},
        {"stage": "tuning"},
        {"stage": "smoke", "track": "X"},
        {"stage": "smoke", "question": "Q9"},
        {"stage": "final", "track": "E", "question": "Q2"},  # no campaign_id
        {"stage": "ablation", "track": "E", "campaign_id": "c"},  # no question
    ],
)
def test_invalid_run_meta_is_rejected(meta):
    cfg = base_cfg()
    cfg.run_meta = meta

    with pytest.raises(ValueError):
        run_meta.apply_run_meta(cfg)


def test_ablation_is_a_recorded_stage():
    cfg = base_cfg(stage="ablation", track="E", question="Q2", campaign_id="E-net-ablation-mini")

    run_meta.apply_run_meta(cfg)

    assert list(cfg.logger.loggers.wandb.tag)[:3] == ["ablation", "E", "Q2"]
    assert list(cfg.logger.loggers.wandb.group_tag) == ["E-net-ablation-mini"]
    assert os.environ["WANDB_JOB_TYPE"] == "ablation"


@pytest.mark.parametrize("stage", ["tune-l1", "final", "scale", "ablation"])
def test_dirty_tree_blocks_recorded_runs(monkeypatch, stage):
    dirty = {k: dict(v) for k, v in CLEAN.items()}
    dirty["jaxarc"]["dirty"] = True
    monkeypatch.setattr(run_meta, "provenance", lambda: dirty)
    cfg = base_cfg(stage=stage, track="E", question="Q2", campaign_id="c")

    with pytest.raises(RuntimeError, match="jaxarc"):
        run_meta.apply_run_meta(cfg)


def test_dirty_tree_is_allowed_for_smoke_runs_and_when_overridden(monkeypatch):
    dirty = {k: dict(v) for k, v in CLEAN.items()}
    dirty["baselines"]["dirty"] = True
    monkeypatch.setattr(run_meta, "provenance", lambda: dirty)

    smoke = base_cfg(stage="smoke")
    run_meta.apply_run_meta(smoke)
    assert smoke.run_meta.code_state == "dirty"
    assert smoke.run_meta.provenance.baselines.dirty == "yes"
    run_meta.apply_run_meta(
        base_cfg(stage="final", track="E", question="Q2", campaign_id="c", allow_dirty=True)
    )


def test_real_provenance_reads_git(monkeypatch):
    monkeypatch.undo()  # use the real provenance()
    info = run_meta.provenance()

    assert set(info) == {"baselines", "jaxarc", "stoix"}
    for repo in info.values():
        assert len(repo["commit"]) == 40
        assert isinstance(repo["dirty"], bool)
