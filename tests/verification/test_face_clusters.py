"""spec-111 P1 — verify_face_clusters: candidate detection + verify + pairwise gate."""
from __future__ import annotations

from sim_bench.verification.face_clusters import (
    cosine_distance, merge_candidates, split_candidates, verify_face_clusters,
)
from sim_bench.verification.eval import evaluate_clustering
from sim_bench.verification.types import MERGE, SPLIT


class ut_MergeCandidates:
    def test_only_just_above_threshold_pairs(self):
        # 0 & 1 are 0.4 apart (just above 0.35 → borderline); 2 is far from both (1.6–2.0).
        centroids = {0: [1.0, 0.0], 1: [0.6, 0.8], 2: [-1.0, 0.0]}
        cands = merge_candidates(centroids, merge_threshold=0.35, band=0.1)
        assert (0, 1) in cands                      # borderline-not-merged → candidate
        assert (0, 2) not in cands and (1, 2) not in cands   # confidently apart

    def test_confidently_distinct_not_escalated(self):
        centroids = {0: [1.0, 0.0], 1: [0.0, 1.0]}  # orthogonal, distance ~1.0
        assert merge_candidates(centroids, 0.35, 0.1) == []


class ut_SplitCandidates:
    def test_bimodal_cluster_flagged(self):
        # one cluster with two opposite sub-groups → high spread
        members = {0: [[1, 0], [1, 0], [-1, 0], [-1, 0]], 1: [[1, 0], [0.99, 0.1]]}
        out = split_candidates(members, spread_threshold=0.45)
        assert 0 in out and 1 not in out           # tight cluster 1 not flagged

    def test_singleton_ignored(self):
        assert split_candidates({0: [[1, 0]]}, 0.1) == []


class ut_VerifyFaceClusters:
    def _centroids(self):
        return {0: [1.0, 0.0], 1: [0.6, 0.8], 2: [-1.0, 0.0]}  # 0-1 borderline (0.4), 2 far

    def test_merge_when_adjudicator_says_same(self):
        vr = verify_face_clusters(
            self._centroids(), members={},
            adjudicate_merge=lambda i, j: True,     # VLM: same person
            adjudicate_split=lambda c: False,
        )
        assert vr.n_corrections == 1 and vr.corrections[0].kind == MERGE
        assert vr.corrections[0].detail["clusters"] == [0, 1]

    def test_no_merge_when_adjudicator_says_different(self):
        vr = verify_face_clusters(self._centroids(), {}, lambda i, j: False, lambda c: False)
        assert vr.n_corrections == 0
        assert vr.n_escalated == 1                  # still escalated the borderline pair, just didn't merge

    def test_split_when_adjudicator_agrees(self):
        members = {0: [[1, 0], [-1, 0]]}            # bimodal
        vr = verify_face_clusters({0: [1, 0]}, members, lambda i, j: False, lambda c: True)
        assert vr.n_corrections == 1 and vr.corrections[0].kind == SPLIT


class ut_ClusteringGate:
    """The AC1 gate: does the verifier's correction improve pairwise agreement with person-truth?"""

    def test_merge_that_matches_truth_improves(self):
        # truth: all 3 faces are the same person. baseline split them into 2 clusters. verifier merged.
        truth    = {"f1": "P", "f2": "P", "f3": "P"}
        baseline = {"f1": 0, "f2": 0, "f3": 1}      # f3 wrongly in its own cluster
        verified = {"f1": 0, "f2": 0, "f3": 0}      # merged
        r = evaluate_clustering(baseline, verified, truth, "verify_face_clusters")
        assert r.verified_agreement == 1.0 and r.baseline_agreement < 1.0
        assert r.improved is True

    def test_bad_merge_hurts_and_fails_gate(self):
        # truth: f1,f2 same; f3 different. verifier wrongly merged f3 in.
        truth    = {"f1": "P", "f2": "P", "f3": "Q"}
        baseline = {"f1": 0, "f2": 0, "f3": 1}      # correct
        verified = {"f1": 0, "f2": 0, "f3": 0}      # wrong merge
        r = evaluate_clustering(baseline, verified, truth, "verify_face_clusters")
        assert r.hurt >= 1 and r.improved is False  # stays off
