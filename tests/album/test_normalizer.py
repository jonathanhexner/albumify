"""spec-106 P0 — album.json normalizer: pure classify() + normalize(context)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sim_bench.album.normalizer import classify, normalize
from sim_bench.album.schema import PICK, NEAR_DUP, NOT_SELECTED, SET_ASIDE


class ut_Classify:
    """The drop-reason derivation — the heart. Pure, synthetic data."""

    def test_pick_near_dup_not_selected_set_aside(self):
        # cluster 0: a,b (a picked → b is near-dup). cluster 1: c (not picked, no pick in cluster).
        # s = screenshot (set aside).
        all_ids = {"a", "b", "c", "s"}
        selected = ["a"]
        labels = {"a": 0, "b": 0, "c": 1}
        set_aside = {"s": {"reason": "screenshot (filename)"}}
        scores = {"a": 0.6, "b": 0.5, "c": 0.4}
        out = classify(all_ids, selected, labels, set_aside, scores)
        assert out["a"][0] == PICK
        assert out["b"][0] == NEAR_DUP and out["b"][2] == ["a"]       # beaten by the pick
        assert out["c"][0] == NOT_SELECTED
        assert out["s"][0] == SET_ASIDE and "screenshot" in out["s"][1]

    def test_selected_wins_over_set_aside(self):
        out = classify({"a"}, ["a"], {}, {"a": {"reason": "x"}}, {"a": 0.9})
        assert out["a"][0] == PICK                                     # selected takes priority

    def test_multiple_winners_in_a_cluster(self):
        out = classify({"a", "b", "c"}, ["a", "b"], {"a": 0, "b": 0, "c": 0}, {}, {})
        assert out["c"][0] == NEAR_DUP
        assert set(out["c"][2]) == {"a", "b"}                          # beaten by both picks


class ut_Normalize:
    def _ctx(self):
        return SimpleNamespace(
            straightened_from={},
            composite_scores={"x/a.jpg": 0.6, "x/b.jpg": 0.5, "x/c.jpg": 0.4},
            iqa_scores={"x/a.jpg": 0.8}, ava_scores={"x/a.jpg": 0.7},
            occlusion_scores={}, tilt_angles={},
            image_captions={"x/a.jpg": "a rowboat on a lake"},
            scene_cluster_labels={"x/a.jpg": 0, "x/b.jpg": 0, "x/c.jpg": 1},
            selected_images=["x/a.jpg"],
            set_aside=[{"image": "x/Screenshot_1.jpg", "reason": "screenshot", "tier": "metadata"}],
            people_clusters={0: ["f1", "f2"], 1: ["f3"]},
            album_theme={"album_type": "family trip — towns & nature"},
        )

    def test_album_shape(self):
        alb = normalize(self._ctx(), job_id="job1", created_at="2026-10-05T00:00:00Z")
        assert alb.job_id == "job1"
        assert [p.id for p in alb.picks] == ["a"]
        pick = alb.picks[0]
        assert pick.score.composite == 0.6 and pick.score.sharpness == 0.8 and pick.score.aesthetic == 0.7
        assert pick.beat == ["b"]                                      # b is a's near-dup
        assert pick.caption == "a rowboat on a lake"
        # dropped carries b (near-dup), c (not selected), screenshot (set aside)
        by_id = {d.id: d for d in alb.dropped}
        assert by_id["b"].reason == NEAR_DUP and by_id["b"].beaten_by == ["a"]
        assert by_id["c"].reason == NOT_SELECTED
        assert by_id["Screenshot_1"].reason == SET_ASIDE
        assert all(d.promotable for d in alb.dropped)                 # show-everything
        assert alb.counts.kept == 1 and alb.counts.set_aside == 1 and alb.counts.people == 2
        assert alb.counts.scenes == 2
        assert alb.theme["album_type"].startswith("family trip")

    def test_to_dict_is_json_friendly(self):
        import json
        alb = normalize(self._ctx(), "j", "t")
        json.dumps(alb.to_dict())                                      # must not raise

    def test_straighten_derivative_maps_to_original_stem(self):
        ctx = self._ctx()
        ctx.straightened_from = {"x/a_straight.jpg": "x/a.jpg"}
        ctx.selected_images = ["x/a_straight.jpg"]                      # pick is the derivative
        alb = normalize(ctx, "j", "t")
        assert [p.id for p in alb.picks] == ["a"]                       # mapped back to original stem
