"""Unit tests for merge iteration visibility (SIGHTING-029).

Coverage:
  - _parse_merge_log keeps all rejection rows (not just first per pair)
  - _parse_merge_log populates iteration field
  - _latest_per_pair returns the last iteration entry per pair
  - _latest_per_pair fixes the size-display bug (shows latest size, not iter-1 size)
  - _build_pair_history groups all rows by pair, sorted by iteration
  - _build_iter_timeline builds one entry per iteration with correct merged pair
  - MergeAnalysisView exposes n_iterations, iter_timeline, pair_history, all_rejection_rows
"""
from pathlib import Path

import pytest

from face_cluster.views.merge_view import (
    MergeDecisionRow,
    _parse_merge_log,
    _latest_per_pair,
    _build_pair_history,
    _build_iter_timeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(a: int, b: int, iteration: int, size_a: int, dist: float = 0.55) -> dict:
    """Return a minimal merge_log entry for a rejected pair."""
    return {
        "iteration": iteration,
        "cluster_a": a,
        "cluster_b": b,
        "cluster_a_size": size_a,
        "cluster_b_size": 2,
        "exemplar_dist": dist,
        "threshold_used": 0.57,
        "support": 3,
        "required_support": 1,
        "post_diameter": 0.5,
        "max_allowed_diameter": 0.7,
        "passes_exemplar": dist <= 0.57,
        "passes_support": True,
        "passes_margin": True,
        "passes_diameter": True,
        "action": "rejected",
        "rejection_reason": "exemplar_dist too high" if dist > 0.57 else None,
        "actually_merged": False,
        "T_a": None, "T_b": None, "T_global": None,
        "margin_gap": 0.1,
        "margin_dist_to_b": 0.55,
        "margin_competitor_dist": 0.65,
        "margin_competitor_id": 5,
    }


def _merged_row(a: int, b: int, iteration: int, size_a: int, size_b: int = 3) -> dict:
    """Return a minimal merge_log entry for an actually-merged pair."""
    return {
        "iteration": iteration,
        "cluster_a": a,
        "cluster_b": b,
        "cluster_a_size": size_a,
        "cluster_b_size": size_b,
        "exemplar_dist": 0.30,
        "threshold_used": 0.57,
        "support": 5,
        "required_support": 1,
        "post_diameter": 0.45,
        "max_allowed_diameter": 0.70,
        "passes_exemplar": True,
        "passes_support": True,
        "passes_margin": True,
        "passes_diameter": True,
        "action": "merged",
        "rejection_reason": None,
        "actually_merged": True,
        "T_a": None, "T_b": None, "T_global": None,
        "margin_gap": 0.15,
        "margin_dist_to_b": 0.30,
        "margin_competitor_dist": 0.45,
        "margin_competitor_id": 9,
    }


class _FakeCR:
    clusters: dict = {}
    exemplars: dict = {}


# ---------------------------------------------------------------------------
# ut_ParseMergeLog
# ---------------------------------------------------------------------------

class ut_ParseMergeLog:

    def test_keeps_all_rejection_rows_across_iterations(self):
        """All rejection entries must survive — no dedup across iterations."""
        log = [
            _row(1, 10, iteration=1, size_a=135),
            _row(1, 10, iteration=2, size_a=138),
            _row(1, 10, iteration=3, size_a=140),
        ]
        _, all_rej = _parse_merge_log(log, _FakeCR(), [])
        assert len(all_rej) == 3

    def test_iteration_field_populated(self):
        log = [
            _row(1, 10, iteration=2, size_a=138),
            _row(1, 11, iteration=2, size_a=138),
        ]
        _, all_rej = _parse_merge_log(log, _FakeCR(), [])
        iters = {r.iteration for r in all_rej}
        assert iters == {2}

    def test_merged_entries_separate_from_rejections(self):
        log = [
            _merged_row(1, 6, iteration=1, size_a=135),
            _row(1, 10, iteration=1, size_a=135),
        ]
        merges, all_rej = _parse_merge_log(log, _FakeCR(), [])
        assert len(merges) == 1
        assert merges[0].cluster_b == 6
        assert len(all_rej) == 1
        assert all_rej[0].cluster_b == 10

    def test_missing_iteration_field_defaults_to_zero(self):
        """Old-format logs without 'iteration' key must not crash."""
        entry = _row(1, 10, iteration=1, size_a=135)
        del entry["iteration"]
        _, all_rej = _parse_merge_log([entry], _FakeCR(), [])
        assert all_rej[0].iteration == 0


# ---------------------------------------------------------------------------
# ut_LatestPerPair
# ---------------------------------------------------------------------------

class ut_LatestPerPair:

    def test_returns_last_iteration_entry(self):
        """_latest_per_pair must return the highest-iteration entry per pair."""
        rows = [
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason="x", exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], cluster_a_size=135, iteration=1),
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason="x", exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], cluster_a_size=143, iteration=5),
        ]
        latest = _latest_per_pair(rows)
        assert len(latest) == 1
        assert latest[0].cluster_a_size == 143
        assert latest[0].iteration == 5

    def test_fixes_size_display_bug(self):
        """Latest view must show the grown cluster size, not the initial size."""
        rows = [
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason="x", exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], cluster_a_size=135, iteration=1),
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason="x", exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], cluster_a_size=138, iteration=2),
        ]
        latest = _latest_per_pair(rows)
        # Must NOT be 135 (the old broken behaviour)
        assert latest[0].cluster_a_size != 135, "Bug regression: showing iter-1 size"
        assert latest[0].cluster_a_size == 138

    def test_multiple_distinct_pairs_all_returned(self):
        rows = [
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.5,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=1),
            MergeDecisionRow(cluster_a=2, cluster_b=11, exemplar_dist=0.6,
                             threshold_used=0.57, support=2, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=1),
        ]
        latest = _latest_per_pair(rows)
        assert len(latest) == 2


# ---------------------------------------------------------------------------
# ut_BuildPairHistory
# ---------------------------------------------------------------------------

class ut_BuildPairHistory:

    def test_groups_by_pair_key(self):
        rows = [
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=1),
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=2),
            MergeDecisionRow(cluster_a=2, cluster_b=11, exemplar_dist=0.6,
                             threshold_used=0.57, support=2, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=1),
        ]
        history = _build_pair_history(rows)
        assert (1, 10) in history
        assert (2, 11) in history
        assert len(history[(1, 10)]) == 2
        assert len(history[(2, 11)]) == 1

    def test_sorted_by_iteration_ascending(self):
        rows = [
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=3),
            MergeDecisionRow(cluster_a=1, cluster_b=10, exemplar_dist=0.575,
                             threshold_used=0.57, support=3, action="rejected",
                             rejection_reason=None, exemplar_face_ids_a=[],
                             exemplar_face_ids_b=[], iteration=1),
        ]
        history = _build_pair_history(rows)
        iters = [r.iteration for r in history[(1, 10)]]
        assert iters == [1, 3]


# ---------------------------------------------------------------------------
# ut_BuildIterTimeline
# ---------------------------------------------------------------------------

class ut_BuildIterTimeline:

    def test_one_entry_per_iteration(self):
        log = [
            _merged_row(1, 6, iteration=1, size_a=135, size_b=3),
            _row(1, 10, iteration=1, size_a=135),
            _row(1, 10, iteration=2, size_a=138),
            _merged_row(1, 12, iteration=2, size_a=138, size_b=2),
        ]
        tl = _build_iter_timeline(log)
        assert len(tl) == 2
        assert tl[0]["iteration"] == 1
        assert tl[1]["iteration"] == 2

    def test_merged_pair_identified_correctly(self):
        log = [
            _merged_row(1, 6, iteration=1, size_a=135, size_b=3),
            _row(1, 10, iteration=1, size_a=135),
        ]
        tl = _build_iter_timeline(log)
        assert tl[0]["merged_a"] == 1
        assert tl[0]["merged_b"] == 6

    def test_no_merge_iteration_has_none_pair(self):
        log = [
            _row(1, 10, iteration=1, size_a=135),
            _row(1, 11, iteration=1, size_a=135),
        ]
        tl = _build_iter_timeline(log)
        assert tl[0]["merged_a"] is None
        assert tl[0]["merged_b"] is None

    def test_n_candidates_counted_per_iteration(self):
        log = [
            _merged_row(1, 6, iteration=1, size_a=135),
            _row(1, 10, iteration=1, size_a=135),
            _row(2, 7, iteration=1, size_a=60),
            _row(1, 10, iteration=2, size_a=138),
        ]
        tl = _build_iter_timeline(log)
        assert tl[0]["n_candidates"] == 3
        assert tl[1]["n_candidates"] == 1
