"""spec-103: the album arm inserts build_scene_distance immediately before cluster_scenes."""

from __future__ import annotations

from sim_bench.albumify_vs_vlm.albumify_arm import _insert_scene_distance


def ut_inserts_before_cluster_scenes():
    steps = ["extract_scene_embedding", "cluster_scenes", "select_best"]
    out = _insert_scene_distance(steps)
    assert out == ["extract_scene_embedding", "build_scene_distance", "cluster_scenes", "select_best"]
    # immediately before, and only once
    assert out.index("build_scene_distance") == out.index("cluster_scenes") - 1


def ut_noop_when_already_present():
    steps = ["extract_scene_embedding", "build_scene_distance", "cluster_scenes"]
    assert _insert_scene_distance(steps) == steps


def ut_noop_when_no_cluster_scenes():
    steps = ["discover_images", "score_iqa"]
    assert _insert_scene_distance(steps) == steps


def ut_does_not_mutate_input():
    steps = ["extract_scene_embedding", "cluster_scenes"]
    _insert_scene_distance(steps)
    assert steps == ["extract_scene_embedding", "cluster_scenes"]  # original untouched


def test_scene_distance_insertion_suite():
    ut_inserts_before_cluster_scenes()
    ut_noop_when_already_present()
    ut_noop_when_no_cluster_scenes()
    ut_does_not_mutate_input()
