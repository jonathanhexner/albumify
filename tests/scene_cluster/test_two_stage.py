"""Unit tests for the two-stage scene clusterer (spec-103 arm A3).

Time comes from the YYYYMMDD_HHMMSS filename stem (no EXIF needed), so these use plain string ids and
empty geo metadata to exercise the time+visual path. Embeddings are 3-D toy vectors: e1/e2 are near-
duplicates (cosine ~1) and e_diff is orthogonal (cosine 0, visual distance 1 > tau).
"""

from __future__ import annotations

import numpy as np

from sim_bench.scene_cluster.two_stage import TwoStageInputs, TwoStageSceneClusterer


E1 = np.array([1.0, 0.0, 0.0])
E1B = np.array([0.99, 0.01, 0.0])   # near-duplicate of E1
E_DIFF = np.array([0.0, 1.0, 0.0])  # orthogonal to E1 -> visual distance 1.0


def _run(emb, ids, **cfg):
    clusterer = TwoStageSceneClusterer(**cfg)
    return clusterer.calc(TwoStageInputs(embeddings=emb, geo_metadata={}, image_ids=ids)).labels


def ut_TwoStage_near_duplicates_same_window_cluster_together():
    ids = ["20250101_120000", "20250101_120030", "20250101_120045"]
    emb = {ids[0]: E1, ids[1]: E1B, ids[2]: E_DIFF}
    labels = _run(emb, ids)
    # the two near-duplicates form a scene; the visually-different shot in the same window does not join
    assert labels[ids[0]] == labels[ids[1]] and labels[ids[0]] != -1
    assert labels[ids[2]] == -1


def ut_TwoStage_time_gap_splits_into_separate_segments():
    # identical-looking photos hours apart must NOT share a scene (different temporal segments)
    ids = ["20250101_120000", "20250101_120030", "20250101_150000", "20250101_150030"]
    emb = {i: E1 for i in ids}
    labels = _run(emb, ids, gap_threshold_min=60.0)
    assert labels[ids[0]] == labels[ids[1]] != -1
    assert labels[ids[2]] == labels[ids[3]] != -1
    assert labels[ids[0]] != labels[ids[2]]  # separate scenes despite identical pixels


def ut_TwoStage_lone_photo_in_segment_is_unclustered():
    ids = ["20250101_120000", "20250101_180000"]  # 6h apart -> two 1-photo segments
    emb = {i: E1 for i in ids}
    labels = _run(emb, ids, gap_threshold_min=60.0)
    assert labels[ids[0]] == -1 and labels[ids[1]] == -1


def ut_TwoStage_visually_different_never_merge_within_window():
    ids = ["20250101_120000", "20250101_120100"]
    emb = {ids[0]: E1, ids[1]: E_DIFF}
    labels = _run(emb, ids)
    assert labels[ids[0]] == -1 and labels[ids[1]] == -1  # not a scene: too visually far


def ut_TwoStage_untimed_photos_share_trailing_segment():
    ids = ["no_timestamp_a", "no_timestamp_b", "20250101_120000"]
    emb = {ids[0]: E1, ids[1]: E1B, ids[2]: E_DIFF}
    labels = _run(emb, ids)
    # the two untimed near-duplicates still cluster (via the trailing untimed segment)
    assert labels[ids[0]] == labels[ids[1]] != -1


def ut_TwoStage_is_deterministic():
    ids = ["20250101_120000", "20250101_120030", "20250101_120045", "20250101_130000"]
    emb = {ids[0]: E1, ids[1]: E1B, ids[2]: E_DIFF, ids[3]: E1}
    a = _run(emb, ids)
    b = _run(emb, ids)
    assert a == b


def test_two_stage_suite():
    ut_TwoStage_near_duplicates_same_window_cluster_together()
    ut_TwoStage_time_gap_splits_into_separate_segments()
    ut_TwoStage_lone_photo_in_segment_is_unclustered()
    ut_TwoStage_visually_different_never_merge_within_window()
    ut_TwoStage_untimed_photos_share_trailing_segment()
    ut_TwoStage_is_deterministic()
