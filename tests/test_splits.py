"""Tests for the seeded MiniARC task split (train / dev / test)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jaxarc_baselines import splits

ROOT = Path(__file__).resolve().parents[1]
MINIARC = ROOT / "data" / "MiniARC" / "data" / "MiniARC"
SUBSETS = ROOT / "configs" / "env" / "jaxarc" / "subsets" / "Mini"


def pair(value: int) -> dict:
    return {"input": [[value]], "output": [[value + 1]]}


def toy_tasks(n: int = 40) -> dict[str, list[dict]]:
    """n tasks with distinct pairs and distinct titles."""
    return {f"task{i}_id{i}": [pair(10 * i), pair(10 * i + 2)] for i in range(n)}


def test_tasks_sharing_a_pair_are_grouped():
    tasks = toy_tasks(5)
    tasks["task1_id1"].append(pair(30))  # task3's first pair

    groups = splits.task_groups(tasks)

    assert sorted(["task1_id1", "task3_id3"]) in [sorted(g) for g in groups]
    assert len(groups) == 4


def test_tasks_with_the_same_title_are_grouped():
    tasks = toy_tasks(4)
    tasks["task0_other"] = [pair(500)]

    groups = splits.task_groups(tasks)

    assert sorted(["task0_id0", "task0_other"]) in [sorted(g) for g in groups]


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("One_Color_Gravity_l6bk9t38ni3lwbapno", "One_Color_gravity_2_l6bkzzz"),
        ("Tetris_l6ab7fu64lvutswrtbk", "Tetris_edit_l6ae0g871owxvc1baq3"),
        ("flip_based_on_symmetry_part_l6a", "edit__flip_based_on_symmetry_part_l6b"),
        ("Simple_Occlusion_l6a", "Simple_Occlusion_Corrected_l6b"),
        ("Color_Emcoding_l6aftgp22mlspnm1zxb", "Color_encoding_updated_l6b"),
    ],
)
def test_variant_titles_are_one_family(a, b):
    assert splits.task_title(a) == splits.task_title(b)
    assert splits.task_title("Tetris_l6a") != splits.task_title("Tiling_l6a")


def test_split_is_disjoint_complete_and_sized():
    tasks = toy_tasks(40)

    split = splits.split_tasks(tasks, {"test": 8, "dev": 8}, seed=3)

    assert set(split) == {"train", "dev", "test"}
    assert (len(split["train"]), len(split["dev"]), len(split["test"])) == (24, 8, 8)
    names = [t for ids in split.values() for t in ids]
    assert sorted(names) == sorted(tasks)


def test_grouped_tasks_land_in_the_same_split():
    tasks = toy_tasks(40)
    for i in range(0, 40, 4):  # ten groups of two via a shared pair
        tasks[f"task{i}_id{i}"].append(pair(10 * (i + 1)))

    for seed in range(10):
        split = splits.split_tasks(tasks, {"test": 8, "dev": 8}, seed=seed)
        where = {t: name for name, ids in split.items() for t in ids}
        for i in range(0, 40, 4):
            assert where[f"task{i}_id{i}"] == where[f"task{i + 1}_id{i + 1}"]


def test_split_depends_only_on_the_seed():
    tasks = toy_tasks(40)
    reordered = dict(reversed(list(tasks.items())))

    a = splits.split_tasks(tasks, {"test": 8, "dev": 8}, seed=7)
    b = splits.split_tasks(reordered, {"test": 8, "dev": 8}, seed=7)
    c = splits.split_tasks(tasks, {"test": 8, "dev": 8}, seed=8)

    assert a == b
    assert a != c
    assert all(ids == sorted(ids) for ids in a.values())


def test_subset_file_round_trips(tmp_path):
    split = splits.split_tasks(toy_tasks(12), {"test": 3, "dev": 3}, seed=0)

    splits.write_subsets(split, tmp_path, header="toy")

    for name, ids in split.items():
        data = yaml.safe_load((tmp_path / f"split_{name}.yaml").read_text())
        assert data["task_ids"] == ids


@pytest.mark.skipif(not MINIARC.exists(), reason="MiniARC data not downloaded")
def test_committed_miniarc_split_matches_the_generator():
    tasks = splits.load_tasks(MINIARC)
    expected = splits.split_tasks(tasks, splits.MINIARC_SIZES, seed=splits.MINIARC_SEED)

    for name, ids in expected.items():
        data = yaml.safe_load((SUBSETS / f"split_{name}.yaml").read_text())
        assert data["task_ids"] == ids
    assert len(tasks) == 149
    assert {k: len(v) for k, v in expected.items()} == {"train": 89, "dev": 30, "test": 30}
