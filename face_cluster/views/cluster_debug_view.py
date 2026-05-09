"""ClusterDebugView: graph-level diagnostics for a single cluster."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from face_cluster.views._base import EdgeInfo, FaceGraphInfo, _embeddings_matrix, _pairwise_distances


@dataclass
class ClusterDebugView:
    """Graph-level diagnostics for a single cluster.

    Answers: WHY did these faces end up together?  Is this cluster
    a tight clique or a fragile chain?
    """
    cluster_id: int
    n_faces: int
    n_edges: int
    max_possible_edges: int  # n*(n-1)/2
    edge_density: float      # n_edges / max_possible_edges

    diameter: float          # max pairwise distance
    median_dist: float       # median pairwise distance
    chain_score: float       # diameter / (2 * median) — >1.5 suggests chain

    bridge_face_ids: List[int]       # articulation points
    edges: List[EdgeInfo]            # all edges in the cluster
    face_graph: List[FaceGraphInfo]  # per-face connectivity

    distance_matrix: Optional[np.ndarray]  # (n x n) for heatmap
    face_ids_order: List[int]              # face_id ordering matching distance_matrix

    @classmethod
    def compute(cls, result: "PipelineResult", cluster_id: int) -> "ClusterDebugView":
        """Compute graph debug diagnostics for a cluster."""
        import networkx as nx

        from face_cluster.config import PipelineConfig

        faces = result.faces
        cr = result.cluster_result

        if cluster_id not in cr.clusters:
            raise ValueError(f"cluster_id {cluster_id} not found")

        member_indices = cr.clusters[cluster_id]
        n = len(member_indices)

        # Build face_id <-> local index mappings
        fid_list = [faces[i].face_id for i in member_indices]
        global_to_local = {gi: li for li, gi in enumerate(member_indices)}

        # Distance matrix
        mat = _embeddings_matrix(faces, member_indices)
        if mat is not None and n > 1:
            pw = _pairwise_distances(mat)
            upper = pw[np.triu_indices_from(pw, k=1)]
            diameter = float(pw.max())
            median_dist = float(np.median(upper)) if len(upper) > 0 else 0.0
        else:
            pw = np.zeros((n, n)) if n > 0 else np.array([])
            diameter = 0.0
            median_dist = 0.0

        # Rebuild the mutual kNN subgraph for this cluster using the same
        # threshold that was used during clustering.  We read the config from
        # the pipeline_run.json stored in the PipelineResult.
        cfg = PipelineConfig()  # defaults
        run_cfg = result.summary.get("config") or {}
        K = run_cfg.get("K", cfg.K)
        dist_thresh = run_cfg.get("distance_threshold", cfg.distance_threshold)

        # Build local kNN graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        edges_info: List[EdgeInfo] = []

        if mat is not None and n > 1:
            k_actual = min(K, n - 1)
            # Find kNN per local node
            neighbor_sets: List[set] = []
            for i in range(n):
                dists_i = pw[i].copy()
                dists_i[i] = np.inf
                nearest = np.argsort(dists_i)[:k_actual]
                neighbor_sets.append(set(nearest.tolist()))

            for i in range(n):
                for j in range(i + 1, n):
                    if j in neighbor_sets[i] and i in neighbor_sets[j]:
                        d = float(pw[i, j])
                        if d <= dist_thresh:
                            G.add_edge(i, j, distance=d)
                            edges_info.append(EdgeInfo(
                                face_id_a=fid_list[i],
                                face_id_b=fid_list[j],
                                distance=round(d, 4),
                            ))

        n_edges = len(edges_info)
        max_edges = n * (n - 1) // 2 if n > 1 else 1
        edge_density = n_edges / max_edges if max_edges > 0 else 0.0
        chain_score = diameter / (2 * median_dist) if median_dist > 0 else 0.0

        # Articulation points (bridge faces)
        bridge_locals = list(nx.articulation_points(G)) if n > 2 else []
        bridge_fids = [fid_list[li] for li in bridge_locals]

        # Per-face graph info
        face_graph: List[FaceGraphInfo] = []
        for li in range(n):
            nbrs = list(G.neighbors(li))
            nbr_fids = [fid_list[j] for j in nbrs]
            nbr_dists = [round(float(pw[li, j]), 4) for j in nbrs] if mat is not None else []
            face_graph.append(FaceGraphInfo(
                face_id=fid_list[li],
                n_edges=len(nbrs),
                neighbors=nbr_fids,
                neighbor_dists=nbr_dists,
                is_bridge=li in bridge_locals,
            ))
        face_graph.sort(key=lambda fg: -fg.n_edges)

        return cls(
            cluster_id=cluster_id,
            n_faces=n,
            n_edges=n_edges,
            max_possible_edges=max_edges,
            edge_density=round(edge_density, 3),
            diameter=round(diameter, 4),
            median_dist=round(median_dist, 4),
            chain_score=round(chain_score, 2),
            bridge_face_ids=bridge_fids,
            edges=edges_info,
            face_graph=face_graph,
            distance_matrix=pw if mat is not None else None,
            face_ids_order=fid_list,
        )
