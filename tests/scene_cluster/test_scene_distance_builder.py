"""Unit tests for SceneDistanceBuilder (spec-103 Path A) + the build_scene_distance step.

Time comes from the YYYYMMDD_HHMMSS filename stem (no EXIF needed), so ids are plain strings and geo
metadata is empty. Embeddings are toy 3-D vectors: E1/E1B are near-duplicates, E_DIFF is orthogonal.
"""

from __future__ import annotations

import math

import numpy as np

from sim_bench.scene_cluster.geo_time_fusion import (SceneDistanceBuilder, SceneDistanceInputs,
                                                     SceneDistanceResult)

E1 = np.array([1.0, 0.0, 0.0])
E1B = np.array([0.9, 0.2, 0.0])     # somewhat similar to E1 (not identical)
E_DIFF = np.array([0.0, 1.0, 0.0])  # orthogonal to E1


def _build(emb, ids, **cfg):
    return SceneDistanceBuilder(**cfg).calc(SceneDistanceInputs(embeddings=emb, geo_metadata={}, image_ids=ids))


def ut_SceneDistance_short_range_pull_shrinks_distance():
    # two moderately-similar photos 10 s apart -> boosted distance strictly < raw visual distance
    ids = ["20250101_120000", "20250101_120010"]
    emb = {ids[0]: E1, ids[1]: E1B}
    raw = 1.0 - float(np.dot(E1 / np.linalg.norm(E1), E1B / np.linalg.norm(E1B)))
    r = _build(emb, ids, boost=0.6, tau_sec=60.0)
    d = r.distance_matrix[0, 1]
    assert d < raw and d > 0.0
    # exact factor: discount = 1 - 0.6*exp(-10/60)
    expected = raw * (1.0 - 0.6 * math.exp(-10.0 / 60.0))
    assert abs(d - expected) < 1e-9


def ut_SceneDistance_far_apart_is_pure_visual():
    # 1 hour apart -> discount ~= 1, distance ~= raw visual (time does nothing)
    ids = ["20250101_120000", "20250101_130000"]
    emb = {ids[0]: E1, ids[1]: E1B}
    raw = 1.0 - float(np.dot(E1 / np.linalg.norm(E1), E1B / np.linalg.norm(E1B)))
    d = _build(emb, ids, boost=0.6, tau_sec=60.0).distance_matrix[0, 1]
    assert abs(d - raw) < 1e-3  # exp(-3600/60) ~ 0


def ut_SceneDistance_never_pushes_apart():
    # the boost only ever SHRINKS distance -- fused <= raw visual for every pair, always
    ids = ["20250101_120000", "20250101_120005", "20250101_140000"]
    emb = {ids[0]: E1, ids[1]: E_DIFF, ids[2]: E1B}
    r = _build(emb, ids, boost=0.6, tau_sec=60.0)
    E = np.array([emb[i] / np.linalg.norm(emb[i]) for i in ids])
    raw = 1.0 - np.clip(E @ E.T, -1, 1)
    assert np.all(r.distance_matrix <= raw + 1e-9)  # tol covers the builder's norm-epsilon vs plain norm


def ut_SceneDistance_missing_time_is_pure_visual():
    ids = ["no_time_a", "no_time_b"]  # no parseable timestamp -> no boost
    emb = {ids[0]: E1, ids[1]: E1B}
    raw = 1.0 - float(np.dot(E1 / np.linalg.norm(E1), E1B / np.linalg.norm(E1B)))
    r = _build(emb, ids)
    assert abs(r.distance_matrix[0, 1] - raw) < 1e-9
    assert r.per_image_signal_used[ids[0]] == ["visual"]  # no 'time' recorded


def ut_SceneDistance_matrix_is_symmetric_zero_diag_aligned():
    ids = ["20250101_120000", "20250101_120005", "20250101_120030"]
    emb = {ids[0]: E1, ids[1]: E1B, ids[2]: E_DIFF}
    r = _build(emb, ids)
    D = r.distance_matrix
    assert r.image_ids == ids
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.allclose(np.diag(D), 0.0)
    assert r.per_image_signal_used[ids[0]] == ["visual", "time"]


def ut_SceneDistance_is_deterministic():
    ids = ["20250101_120000", "20250101_120005", "20250101_120030"]
    emb = {ids[0]: E1, ids[1]: E1B, ids[2]: E_DIFF}
    a = _build(emb, ids).distance_matrix
    b = _build(emb, ids).distance_matrix
    assert np.array_equal(a, b)


def ut_SceneDistance_result_type():
    r = _build({"20250101_120000": E1, "20250101_120005": E1B},
               ["20250101_120000", "20250101_120005"])
    assert isinstance(r, SceneDistanceResult)


def test_scene_distance_builder_suite():
    ut_SceneDistance_short_range_pull_shrinks_distance()
    ut_SceneDistance_far_apart_is_pure_visual()
    ut_SceneDistance_never_pushes_apart()
    ut_SceneDistance_missing_time_is_pure_visual()
    ut_SceneDistance_matrix_is_symmetric_zero_diag_aligned()
    ut_SceneDistance_is_deterministic()
    ut_SceneDistance_result_type()
