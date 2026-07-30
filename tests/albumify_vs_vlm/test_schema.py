"""Spec-102 — shared AlbumResult schema round-trips to JSON identically for both arms."""

import json

from sim_bench.albumify_vs_vlm.schema import AlbumResult, Pick, save_album_result


def _result(arm: str) -> AlbumResult:
    return AlbumResult(
        trip="budapest", arm=arm, input_set_hash="H", k=2, pipeline="p",
        order=["a", "b"],
        picks=[Pick(id="a", score=0.9, scene_cluster=0, reason="best", role="opener"),
               Pick(id="b", score=0.8, scene_cluster=1, reason="ok", role="closer")],
        scene_clusters={0: ["a", "a2"], 1: ["b"]},
    )


def test_scene_clusters_keys_stringified_for_json():
    d = _result("albumify").to_json()
    assert set(d["scene_clusters"]) == {"0", "1"}  # int keys -> str for valid JSON


def test_both_arms_produce_same_top_level_shape():
    a = set(_result("albumify").to_json())
    v = set(_result("vlm").to_json())
    assert a == v  # viewer cannot distinguish arms by structure


def test_save_and_reload(tmp_path):
    res = _result("vlm")
    path = tmp_path / "picks.json"
    save_album_result(res, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["order"] == ["a", "b"]
    assert loaded["picks"][0]["role"] == "opener"
    assert loaded["arm"] == "vlm"
