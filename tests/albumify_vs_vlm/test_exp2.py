"""Spec-102 EXP-2 unit tests (no VLM / no network) — cluster sampling, Albumify best, metrics."""

import json

import pytest

from sim_bench.albumify_vs_vlm.exp2 import (
    ClusterCase,
    Exp2Result,
    load_exp2,
    save_exp2,
    select_cluster_cases,
    top1_accuracy,
)


def test_select_excludes_singletons_and_oversized():
    scene_clusters = {
        1: ["a", "b", "c"],          # kept (3 frames)
        2: ["d"],                    # dropped (singleton — trivial pick)
        3: ["e"] * 40,              # dropped (oversized catch-all)
    }
    scores = {s: 1.0 for c in scene_clusters.values() for s in c}
    cases = select_cluster_cases(scene_clusters, scores, max_size=30, min_size=2)
    assert [c.cluster_id for c in cases] == [1]
    assert cases[0].stems == ["a", "b", "c"]


def test_excludes_noise_cluster_minus_one():
    # cluster -1 is the scene-clustering noise/unclustered bucket -> not a same-moment group
    scene_clusters = {-1: ["a", "b", "c", "d"], 0: ["e", "f"]}
    scores = {s: 1.0 for c in scene_clusters.values() for s in c}
    cases = select_cluster_cases(scene_clusters, scores)
    assert [c.cluster_id for c in cases] == [0]


def test_albumify_best_is_argmax_composite():
    scene_clusters = {5: ["p", "q", "r"]}
    scores = {"p": 0.3, "q": 0.9, "r": 0.5}
    cases = select_cluster_cases(scene_clusters, scores)
    assert cases[0].albumify_best == "q"


def test_unscored_frames_ignored_for_argmax_and_size():
    # 'z' has no composite score -> not counted toward size, not eligible as best
    scene_clusters = {7: ["x", "y", "z"]}
    scores = {"x": 0.4, "y": 0.6}
    cases = select_cluster_cases(scene_clusters, scores)
    assert cases[0].stems == ["x", "y"]
    assert cases[0].albumify_best == "y"


def test_top1_accuracy_scores_both_systems():
    cases = [
        ClusterCase(1, ["a", "b"], albumify_best="a", vlm_best="b", human_best="a"),  # alb hit
        ClusterCase(2, ["c", "d"], albumify_best="c", vlm_best="d", human_best="d"),  # vlm hit
        ClusterCase(3, ["e", "f"], albumify_best="e", vlm_best="e", human_best="e"),  # both hit
    ]
    m = top1_accuracy(cases)
    assert m["n_judged"] == 3
    assert m["albumify_hits"] == 2 and m["albumify_top1"] == round(2 / 3, 3)
    assert m["vlm_hits"] == 2 and m["vlm_top1"] == round(2 / 3, 3)
    # agreement: clusters 1,2 disagree, cluster 3 agrees -> 1/3
    assert m["system_agreement"] == round(1 / 3, 3)


def test_top1_accuracy_handles_unjudged():
    cases = [ClusterCase(1, ["a", "b"], albumify_best="a", vlm_best="b")]  # no human_best
    m = top1_accuracy(cases)
    assert m["n_judged"] == 0
    assert m["albumify_top1"] is None and m["vlm_top1"] is None
    assert m["system_agreement"] == 0.0  # they disagree, and both have a vlm pick


def test_save_load_roundtrip(tmp_path):
    res = Exp2Result(trip="t", input_set_hash="h",
                     cases=[ClusterCase(1, ["a", "b"], albumify_best="a", vlm_best="b")])
    p = tmp_path / "exp2.json"
    save_exp2(res, p)
    back = load_exp2(p)
    assert back.trip == "t"
    assert back.cases[0].albumify_best == "a" and back.cases[0].vlm_best == "b"
