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


class ut_AlbumDeterminism:
    """spec-110 AC4 - the album must not change when only the pipeline's own ordering changes.

    HDBSCAN numbers its clusters by the row order it is handed, and the embedding dict it is handed
    is not order-stable (SIGHTING-126). The partition stays the same; the ids and the selected order
    do not. The album therefore does its own numbering and ordering.
    """

    @staticmethod
    def _context(selected, labels):
        return SimpleNamespace(
            selected_images=[f"/t/{s}.jpg" for s in selected],
            scene_cluster_labels={f"/t/{s}.jpg": c for s, c in labels.items()},
            composite_scores={f"/t/{s}.jpg": 0.5 for s in labels},
            set_aside=[], straightened_from={}, people_clusters={},
            image_captions={}, album_theme={},
        )

    def test_permuted_cluster_ids_and_selected_order_give_the_same_album(self):
        # same partition twice: {a,b} and {c,d}, picks a and c -- only the ids/order differ
        run1 = normalize(self._context(["a", "c"], {"a": 0, "b": 0, "c": 1, "d": 1}),
                         job_id="j", created_at="t").to_dict()
        run2 = normalize(self._context(["c", "a"], {"a": 1, "b": 1, "c": 0, "d": 0}),
                         job_id="j", created_at="t").to_dict()
        assert run1 == run2

    def test_scene_zero_is_the_earliest_scene_not_an_arbitrary_one(self):
        album = normalize(self._context(["c", "a"], {"a": 7, "b": 7, "c": 3, "d": 3}),
                          job_id="j", created_at="t").to_dict()
        by_id = {p["id"]: p["scene_cluster"] for p in album["picks"]}
        assert by_id == {"a": 0, "c": 1}          # a's scene starts first -> scene 0

    def test_noise_stays_minus_one(self):
        album = normalize(self._context(["a", "x"], {"a": 4, "b": 4, "x": -1}),
                          job_id="j", created_at="t").to_dict()
        assert {p["id"]: p["scene_cluster"] for p in album["picks"]} == {"a": 0, "x": -1}

    def test_picks_are_chronological_by_capture_time_when_exif_has_it(self):
        from datetime import datetime

        ctx = self._context(["a", "b"], {"a": 0, "b": 1})
        # EXIF says b was shot first, although 'a' sorts first by name
        ctx.geo_metadata = {
            "/t/a.jpg": SimpleNamespace(timestamp=datetime(2026, 8, 22, 18, 15)),
            "/t/b.jpg": SimpleNamespace(timestamp=datetime(2026, 8, 22, 17, 42)),
        }
        album = normalize(ctx, job_id="j", created_at="t").to_dict()
        assert [p["id"] for p in album["picks"]] == ["b", "a"]
        assert [p["order"] for p in album["picks"]] == [0, 1]
        assert {p["id"]: p["scene_cluster"] for p in album["picks"]} == {"b": 0, "a": 1}

    def test_beat_and_beaten_by_lists_are_ordered(self):
        ctx = self._context(["b"], {"b": 0, "a": 0, "c": 0})
        album = normalize(ctx, job_id="j", created_at="t").to_dict()
        assert album["picks"][0]["beat"] == ["a", "c"]
        assert all(d["beaten_by"] == ["b"] for d in album["dropped"])

    def test_a_duplicate_in_selected_images_is_picked_once(self):
        ctx = self._context(["a", "a"], {"a": 0, "b": 0})
        album = normalize(ctx, job_id="j", created_at="t").to_dict()
        assert [p["id"] for p in album["picks"]] == ["a"]
        assert album["counts"]["kept"] == 1
