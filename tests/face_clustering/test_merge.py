"""Unit tests for apply_manual_merges() and simplified ConservativeMerger."""
import numpy as np
import pytest

from face_cluster.merge import apply_manual_merges, ConservativeMerger
from face_cluster.config import PipelineConfig
from face_cluster.types import ClusterResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster_result(clusters: dict) -> ClusterResult:
    """Build a minimal ClusterResult from {cluster_id: [node_indices]}."""
    all_nodes = [n for nodes in clusters.values() for n in nodes]
    n = max(all_nodes) + 1 if all_nodes else 0
    labels = np.full(n, -1, dtype=int)
    for cid, nodes in clusters.items():
        for node in nodes:
            labels[node] = cid
    return ClusterResult(
        labels=labels,
        clusters={cid: list(nodes) for cid, nodes in clusters.items()},
        cluster_stats={cid: {"diameter": 0.1, "size": len(nodes)}
                       for cid, nodes in clusters.items()},
        exemplars={cid: list(nodes)[:1] for cid, nodes in clusters.items()},
        n_clusters=len(clusters),
        n_noise=0,
    )


def _identity_distance_matrix(n: int, intra: float = 0.1) -> np.ndarray:
    """n×n distance matrix where intra-cluster distance is `intra`, all others 0.5."""
    dm = np.full((n, n), 0.5, dtype=float)
    np.fill_diagonal(dm, 0.0)
    return dm


# ---------------------------------------------------------------------------
# ut_ApplyManualMerges
# ---------------------------------------------------------------------------

class ut_ApplyManualMerges:

    def test_no_approvals_returns_unchanged(self):
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3], 2: [4, 5]})
        dm = _identity_distance_matrix(6)
        result = apply_manual_merges(cr, [], dm)
        assert result.n_clusters == 3
        assert set(result.clusters.keys()) == {0, 1, 2}

    def test_simple_pair_merges_correctly(self):
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3], 2: [4, 5]})
        dm = _identity_distance_matrix(6)
        result = apply_manual_merges(cr, [(0, 1)], dm)
        assert result.n_clusters == 2
        # Nodes from clusters 0 and 1 must be in the same cluster
        node_to_cluster = {}
        for cid, nodes in result.clusters.items():
            for n in nodes:
                node_to_cluster[n] = cid
        assert node_to_cluster[0] == node_to_cluster[1]  # was cluster 0
        assert node_to_cluster[2] == node_to_cluster[3]  # was cluster 1
        assert node_to_cluster[0] == node_to_cluster[2]  # both merged
        assert node_to_cluster[4] != node_to_cluster[0]  # cluster 2 untouched

    def test_transitive_chain_merges_all(self):
        """Approving (0,1) and (1,2) should merge all three into one cluster."""
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3], 2: [4, 5]})
        dm = _identity_distance_matrix(6)
        result = apply_manual_merges(cr, [(0, 1), (1, 2)], dm)
        assert result.n_clusters == 1
        all_nodes = set(result.clusters[min(result.clusters.keys())])
        assert all_nodes == {0, 1, 2, 3, 4, 5}

    def test_unknown_pair_silently_ignored(self):
        """Cluster IDs that don't exist in the base result are ignored."""
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3]})
        dm = _identity_distance_matrix(4)
        result = apply_manual_merges(cr, [(0, 99)], dm)
        assert result.n_clusters == 2  # unchanged

    def test_all_clusters_merged(self):
        """Merging all pairs into one mega-cluster."""
        cr = _make_cluster_result({0: [0], 1: [1], 2: [2], 3: [3]})
        dm = _identity_distance_matrix(4)
        result = apply_manual_merges(cr, [(0, 1), (2, 3), (0, 2)], dm)
        assert result.n_clusters == 1

    def test_labels_consistency(self):
        """labels array must be consistent with clusters dict after merge."""
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3]})
        dm = _identity_distance_matrix(4)
        result = apply_manual_merges(cr, [(0, 1)], dm)
        for cid, nodes in result.clusters.items():
            for node in nodes:
                assert result.labels[node] == cid

    def test_stats_recomputed(self):
        """Merged cluster should have stats dict with 'diameter' key."""
        cr = _make_cluster_result({0: [0, 1], 1: [2, 3]})
        dm = np.array([
            [0.0, 0.1, 0.4, 0.5],
            [0.1, 0.0, 0.3, 0.4],
            [0.4, 0.3, 0.0, 0.1],
            [0.5, 0.4, 0.1, 0.0],
        ])
        result = apply_manual_merges(cr, [(0, 1)], dm)
        cid = min(result.clusters.keys())
        assert "diameter" in result.cluster_stats[cid]
        # Diameter of merged [0,1,2,3] with max dist 0.5
        assert abs(result.cluster_stats[cid]["diameter"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# ut_SimplifiedMerger
# ---------------------------------------------------------------------------

class ut_SimplifiedMerger:

    # Deleted 2026-05-29 (spec-054, SIGHTING-072):
    # test_adaptive_threshold_fields_removed asserted that PipelineConfig
    # no longer has merge_threshold_alpha / _beta / use_adaptive_merge_threshold
    # / merge_exemplar_percentile / merge_global_percentile. Grep confirmed
    # all 5 fields are live in production code (face_cluster/analysis.py,
    # app/shared/merge_controls.py, etc.). The test encoded an aborted
    # cleanup intent; the codebase chose to keep adaptive thresholds. If
    # we ever decide to remove them, that's a separate spec.

    def test_fixed_threshold_field_present(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "merge_exemplar_threshold")
        assert 0 < cfg.merge_exemplar_threshold <= 1.0

    def test_config_validates(self):
        """PipelineConfig with merge enabled should instantiate without error."""
        cfg = PipelineConfig(merge_enabled=True, merge_exemplar_threshold=0.40)
        assert cfg.merge_enabled
        assert cfg.merge_exemplar_threshold == 0.40


# ---------------------------------------------------------------------------
# ut_MergeTwoClusters — exemplar reselection after merge
# ---------------------------------------------------------------------------

class ut_MergeTwoClusters:
    """Verify that _merge_two_clusters reselects exemplars from ALL merged nodes.

    Topology:
        Cluster A = [0, 1, 2]   exemplars_A = [0]   (node 2 excluded initially)
        Cluster B = [3]          exemplars_B = [3]

    Distance matrix:
        0 ↔ 1 : 0.05  (tight core pair in A)
        0 ↔ 2 : 0.40  (node 2 is peripheral in A)
        1 ↔ 2 : 0.40
        2 ↔ 3 : 0.15  (node 2 is the bridge to B)
        0 ↔ 3 : 0.70  (core of A is far from B)
        1 ↔ 3 : 0.70

    OLD code: combined_exemplars = [0] + [3] = [0, 3]  — node 2 never discovered.
    NEW code: full d10 reselection over [0,1,2,3] → picks [0, 2] (node 2 is bridge).
    """

    # 4-node distance matrix matching the topology above
    DM = np.array([
        [0.00, 0.05, 0.40, 0.70],
        [0.05, 0.00, 0.40, 0.70],
        [0.40, 0.40, 0.00, 0.15],
        [0.70, 0.70, 0.15, 0.00],
    ], dtype=float)

    def _make_cr(self) -> ClusterResult:
        """Cluster A=[0,1,2] exemplar=[0], Cluster B=[3] exemplar=[3]."""
        labels = np.array([0, 0, 0, 1])
        return ClusterResult(
            labels=labels,
            clusters={0: [0, 1, 2], 1: [3]},
            cluster_stats={
                0: {"diameter": 0.40, "size": 3},
                1: {"diameter": 0.00, "size": 1},
            },
            exemplars={0: [0], 1: [3]},
            n_clusters=2,
            n_noise=0,
        )

    def _merger(self) -> ConservativeMerger:
        cfg = PipelineConfig(
            d10_k=1,
            N_exemplars_max=2,
            exemplar_suppression_radius=0.2,
        )
        return ConservativeMerger(cfg)

    def test_bridge_node_becomes_exemplar_after_merge(self):
        """Node 2 (bridge, not in original exemplars) must appear in exemplars post-merge."""
        cr = self._make_cr()
        merger = self._merger()
        result = merger._merge_two_clusters(cr, 0, 1, self.DM)

        exemplars = result.exemplars[0]
        assert 2 in exemplars, (
            f"Bridge node 2 must be in post-merge exemplars, got {exemplars}"
        )

    def test_old_exemplar_pool_nodes_not_forced_in(self):
        """Node 3 (B's original exemplar) should NOT be an exemplar when node 2 is better."""
        cr = self._make_cr()
        merger = self._merger()
        result = merger._merge_two_clusters(cr, 0, 1, self.DM)

        # With N_exemplars_max=2 and suppression_radius=0.2:
        # d10 ranking: 0 (0.05) < 2 (0.15) < 3 (0.15) tie, 1 suppressed by 0.
        # Greedy: select 0, then 2 (first node >= suppression from 0).
        # Node 3 is not selected.
        exemplars = result.exemplars[0]
        assert 3 not in exemplars, (
            f"Node 3 should be displaced by bridge node 2, got {exemplars}"
        )

    def test_merged_cluster_nodes_correct(self):
        """Merged cluster 0 must contain all four nodes; cluster 1 must be gone."""
        cr = self._make_cr()
        merger = self._merger()
        result = merger._merge_two_clusters(cr, 0, 1, self.DM)

        assert set(result.clusters[0]) == {0, 1, 2, 3}
        assert 1 not in result.clusters

    def test_exemplar_count_respects_n_max(self):
        """Exemplar count must not exceed N_exemplars_max."""
        cr = self._make_cr()
        merger = self._merger()
        result = merger._merge_two_clusters(cr, 0, 1, self.DM)

        assert len(result.exemplars[0]) <= merger.config.N_exemplars_max


# ---------------------------------------------------------------------------
# ut_CrossDistGate — spec-019: p25 cross-distance OR gate
# ---------------------------------------------------------------------------

class ut_CrossDistGate:
    """Verify cross-distance OR gate and unique-pair support.

    Topology (6 nodes):
        Cluster A = [0, 1, 2]  exemplars_A = [0]
        Cluster B = [3, 4, 5]  exemplars_B = [3]

    Distances set so exemplar-to-exemplar (0↔3) is high (0.50),
    but nodes 2↔4 are close (0.20), pulling p25_cross_dist low.
    """

    DM = np.array([
        #    0     1     2     3     4     5
        [0.00, 0.05, 0.10, 0.50, 0.55, 0.60],   # 0
        [0.05, 0.00, 0.08, 0.48, 0.50, 0.58],   # 1
        [0.10, 0.08, 0.00, 0.35, 0.20, 0.40],   # 2
        [0.50, 0.48, 0.35, 0.00, 0.06, 0.08],   # 3
        [0.55, 0.50, 0.20, 0.06, 0.00, 0.07],   # 4
        [0.60, 0.58, 0.40, 0.08, 0.07, 0.00],   # 5
    ], dtype=float)

    def _make_cr(self) -> ClusterResult:
        labels = np.array([0, 0, 0, 1, 1, 1])
        return ClusterResult(
            labels=labels,
            clusters={0: [0, 1, 2], 1: [3, 4, 5]},
            cluster_stats={
                0: {"diameter": 0.10, "size": 3},
                1: {"diameter": 0.08, "size": 3},
            },
            exemplars={0: [0], 1: [3]},
            n_clusters=2,
            n_noise=0,
        )

    def test_p25_cross_distance_basic(self):
        """p25_cross_distance uses ALL node pairs, not just exemplars."""
        merger = ConservativeMerger(PipelineConfig())
        val = merger._p25_cross_distance([0, 1, 2], [3, 4, 5], self.DM)
        # 9 cross-pair distances: 0.50, 0.55, 0.60, 0.48, 0.50, 0.58, 0.35, 0.20, 0.40
        # sorted: 0.20, 0.35, 0.40, 0.48, 0.50, 0.50, 0.55, 0.58, 0.60
        # p25 index ≈ 2.0 → value 0.35  (numpy: percentile(arr, 25) ≈ 0.35)
        assert 0.15 <= val <= 0.45

    def test_p25_exemplar_distance_higher_than_cross(self):
        """With distant exemplars but close non-exemplar nodes, cross < exemplar."""
        merger = ConservativeMerger(PipelineConfig())
        ex_dist = merger._p25_exemplar_distance([0], [3], self.DM)
        cross_dist = merger._p25_cross_distance([0, 1, 2], [3, 4, 5], self.DM)
        assert cross_dist < ex_dist, (
            f"cross={cross_dist:.3f} should be < exemplar={ex_dist:.3f}"
        )

    def test_or_gate_passes_via_cross_path(self):
        """Pair fails exemplar gate but passes via cross-dist → Gate A passes."""
        cfg = PipelineConfig(
            merge_enabled=True,
            merge_exemplar_threshold=0.40,  # exemplar dist 0.50 > 0.40 → FAIL
            merge_cross_threshold=0.45,     # cross dist ~0.35 < 0.45 → PASS
            merge_use_cross_gate=True,
            merge_support_min=1,
            merge_support_frac=0.0,
            merge_margin=0.0,
            merge_diameter_expansion_factor=100.0,  # effectively disabled
        )
        merger = ConservativeMerger(cfg)
        evidence = merger._evaluate_merge_evidence(
            0, 1, self._make_cr(), self.DM, {}, cfg.merge_exemplar_threshold
        )
        assert not evidence["passes_exemplar"] or evidence["passes_cross"], (
            "At least one exemplar-side path should pass"
        )
        assert evidence["passes_exemplar"], "OR gate should make passes_exemplar True"
        assert evidence["passes_cross"], "Cross-dist path should pass"

    def test_or_gate_disabled_falls_back_to_exemplar_only(self):
        """When merge_use_cross_gate=False, Gate A is exemplar-only."""
        cfg = PipelineConfig(
            merge_enabled=True,
            merge_exemplar_threshold=0.40,  # exemplar dist 0.50 > 0.40 → FAIL
            merge_cross_threshold=0.45,     # would pass, but gate disabled
            merge_use_cross_gate=False,
        )
        merger = ConservativeMerger(cfg)
        evidence = merger._evaluate_merge_evidence(
            0, 1, self._make_cr(), self.DM, {}, cfg.merge_exemplar_threshold
        )
        assert not evidence["passes_exemplar"], (
            "With cross gate disabled, exemplar should FAIL"
        )

    def test_or_gate_blocked_when_both_clusters_large(self):
        """Cross-dist OR path must not apply when min cluster size > merge_cross_max_size."""
        # Build clusters larger than the max_size threshold
        n_a, n_b = 8, 8
        n = n_a + n_b
        dm = np.full((n, n), 0.50, dtype=float)
        np.fill_diagonal(dm, 0.0)
        # Make intra-cluster distances tight
        for i in range(n_a):
            for j in range(i+1, n_a):
                dm[i, j] = dm[j, i] = 0.05
        for i in range(n_a, n):
            for j in range(i+1, n):
                dm[i, j] = dm[j, i] = 0.05
        # Cross pairs: exemplars (0, n_a) are far, but some non-exemplar pairs are close
        dm[0, n_a] = dm[n_a, 0] = 0.55  # exemplar-to-exemplar: far
        # Make many non-exemplar cross pairs close so p25_cross_dist would pass
        for i in range(1, n_a):
            for j in range(n_a+1, n):
                dm[i, j] = dm[j, i] = 0.25
        labels = np.array([0]*n_a + [1]*n_b)
        cr = ClusterResult(
            labels=labels,
            clusters={0: list(range(n_a)), 1: list(range(n_a, n))},
            cluster_stats={0: {"diameter": 0.05, "size": n_a}, 1: {"diameter": 0.05, "size": n_b}},
            exemplars={0: [0], 1: [n_a]},
            n_clusters=2, n_noise=0,
        )
        cfg = PipelineConfig(
            merge_enabled=True,
            merge_exemplar_threshold=0.40,  # exemplar dist 0.55 > 0.40 → FAIL
            merge_cross_threshold=0.45,     # cross-dist ~0.30 < 0.45 → would PASS
            merge_use_cross_gate=True,
            merge_cross_max_size=5,         # min(8,8)=8 > 5 → OR path blocked
        )
        merger = ConservativeMerger(cfg)
        evidence = merger._evaluate_merge_evidence(0, 1, cr, dm, {}, cfg.merge_exemplar_threshold)
        assert evidence["passes_cross"], "Cross-dist itself should pass"
        assert not evidence["passes_exemplar"], (
            "OR path should be blocked for large-vs-large clusters"
        )

    def test_unique_support_prevents_node_reuse(self):
        """Unique support must not count the same node twice."""
        merger = ConservativeMerger(PipelineConfig())
        # With threshold 0.55:
        # Close pairs: (2,4)=0.20, (2,3)=0.35, (2,5)=0.40, (1,3)=0.48,
        # (0,3)=0.50, (1,4)=0.50, (0,4)=0.55
        # Raw support: all 7 pairs are <= 0.55
        raw = merger._count_support([0, 1, 2], [3, 4, 5], self.DM, 0.55)
        unique = merger._count_unique_support([0, 1, 2], [3, 4, 5], self.DM, 0.55)
        # Greedy: pick (2,4)=0.20 → used {2},{4}
        #         pick (1,3)=0.48 → used {1,2},{3,4}
        #         pick (0,5)=0.60 > threshold, skip
        #         remaining: (0,5) only pair with unused nodes, but 0.60>0.55
        # So unique=2, raw >= 2
        assert unique <= raw, "Unique can never exceed raw"
        assert unique <= 3, "At most 3 pairs (min cluster size)"
        assert unique >= 1, "At least one close pair exists"

    def test_unique_support_greedy_order(self):
        """Greedy picks closest pairs first, maximising quality."""
        merger = ConservativeMerger(PipelineConfig())
        unique = merger._count_unique_support([0, 1, 2], [3, 4, 5], self.DM, 0.55)
        # Greedy assigns: (2,4)=0.20 first, then (1,3)=0.48
        # Node 0 and 5 have no pair <=0.55 with remaining nodes
        assert unique == 2

    def test_evidence_dict_contains_new_fields(self):
        """p25_cross_dist and unique_support must appear in evidence dict."""
        merger = ConservativeMerger(PipelineConfig(merge_enabled=True))
        evidence = merger._evaluate_merge_evidence(
            0, 1, self._make_cr(), self.DM, {}, 0.40
        )
        assert "p25_cross_dist" in evidence
        assert "unique_support" in evidence
        assert "passes_cross" in evidence
        assert isinstance(evidence["p25_cross_dist"], float)
        assert isinstance(evidence["unique_support"], int)
