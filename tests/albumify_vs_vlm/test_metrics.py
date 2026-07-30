"""Spec-102 T4.2 — objective metrics (duplicate survival + coverage)."""

from sim_bench.albumify_vs_vlm.metrics import (
    coverage_from_roster,
    duplicate_survival,
)


def test_all_distinct_scenes_zero_redundancy():
    order = ["a", "b", "c"]
    labels = {"a": 0, "b": 1, "c": 2}
    ds = duplicate_survival(order, labels)
    assert ds.n_scenes_covered == 3
    assert ds.n_redundant == 0
    assert ds.redundancy_rate == 0.0


def test_two_from_same_scene_counts_one_redundant():
    order = ["a", "b", "c"]
    labels = {"a": 0, "b": 0, "c": 1}  # a,b same scene
    ds = duplicate_survival(order, labels)
    assert ds.n_scenes_covered == 2
    assert ds.n_redundant == 1
    assert ds.redundancy_rate == round(1 / 3, 4)


def test_noise_and_unknown_are_singletons():
    order = ["a", "b", "c"]
    labels = {"a": -1, "b": -1}  # c unknown; all treated as distinct singletons
    ds = duplicate_survival(order, labels)
    assert ds.n_redundant == 0
    assert ds.n_scenes_covered == 3


def test_coverage_none_until_roster_labelled():
    roster = {"persons": [{"id": "P1", "images": []}], "scenes": [{"id": "S1", "images": []}]}
    assert coverage_from_roster(["a"], roster) is None


def test_coverage_recall_counts_hits():
    roster = {
        "persons": [{"name": "Ana", "images": ["a", "x"]},
                    {"name": "Bo", "images": ["y"]}],   # Bo not in picks -> missed
        "scenes": [{"label": "Castle", "images": ["a"]}],
    }
    cov = coverage_from_roster(["a", "b"], roster)
    assert cov["persons"]["n_covered"] == 1
    assert cov["persons"]["recall"] == 0.5
    assert cov["persons"]["missed"] == ["Bo"]
    assert cov["scenes"]["recall"] == 1.0
