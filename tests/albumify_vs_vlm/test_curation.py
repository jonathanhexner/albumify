"""Spec-102 T2 — pure curation helpers (coverage-first top-K + chronological order)."""

from sim_bench.albumify_vs_vlm.curation import (
    chronological_order,
    select_top_k_coverage_first,
)


def test_coverage_first_takes_one_per_scene_before_seconds():
    # 2 scenes, 2 shots each. k=2 must take the best of EACH scene, not both of scene A.
    scores = {"a1": 0.9, "a2": 0.8, "b1": 0.7, "b2": 0.6}
    labels = {"a1": 0, "a2": 0, "b1": 1, "b2": 1}
    chosen = select_top_k_coverage_first(list(scores), scores, labels, k=2)
    assert set(chosen) == {"a1", "b1"}  # one per scene, each scene's best


def test_coverage_first_fills_seconds_after_all_scenes_covered():
    scores = {"a1": 0.9, "a2": 0.8, "b1": 0.7}
    labels = {"a1": 0, "a2": 0, "b1": 1}
    chosen = select_top_k_coverage_first(list(scores), scores, labels, k=3)
    # round 1: a1 (scene0 best), b1 (scene1 best); round 2: a2
    assert chosen == ["a1", "b1", "a2"]


def test_noise_images_are_each_their_own_scene():
    # all noise (-1): must not collapse into one scene and starve to a single pick
    scores = {"n1": 0.9, "n2": 0.8, "n3": 0.7}
    labels = {"n1": -1, "n2": -1, "n3": -1}
    chosen = select_top_k_coverage_first(list(scores), scores, labels, k=3)
    assert chosen == ["n1", "n2", "n3"]


def test_k_larger_than_pool_returns_all():
    scores = {"a1": 0.9, "b1": 0.7}
    labels = {"a1": 0, "b1": 1}
    assert set(select_top_k_coverage_first(list(scores), scores, labels, k=10)) == {"a1", "b1"}


def test_k_zero_or_empty():
    assert select_top_k_coverage_first(["a1"], {"a1": 1.0}, {"a1": 0}, k=0) == []
    assert select_top_k_coverage_first([], {}, {}, k=5) == []


def test_deterministic_on_score_ties():
    scores = {"a1": 0.5, "a2": 0.5, "b1": 0.5}
    labels = {"a1": 0, "a2": 0, "b1": 1}
    r1 = select_top_k_coverage_first(list(scores), scores, labels, k=3)
    r2 = select_top_k_coverage_first(list(reversed(list(scores))), scores, labels, k=3)
    assert r1 == r2  # ties broken by stem -> stable regardless of input order


def test_chronological_order_by_filename_timestamp():
    stems = ["20250822_190000", "20250822_112331", "20250824_090000"]
    assert chronological_order(stems) == [
        "20250822_112331", "20250822_190000", "20250824_090000",
    ]


def test_chronological_untimestamped_sorts_last():
    stems = ["zzz_no_ts", "20250822_112331"]
    assert chronological_order(stems) == ["20250822_112331", "zzz_no_ts"]
