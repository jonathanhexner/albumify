"""Analysis view dataclasses for the face clustering app.

Three levels of analysis, each computed from a PipelineResult:
  - RunOverview  : full-run health check (quality gate funnel, cluster table, UMAP)
  - ClusterView  : per-cluster drill-down (exemplars, outliers, split signal, nearest clusters)
  - FaceView     : per-face drill-down (attributes, closest faces same/other clusters)

All computations are pure functions — no Streamlit, no file I/O.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from face_cluster.pipeline import PipelineResult
from face_cluster.types import FaceRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small row types used inside views
# ---------------------------------------------------------------------------

@dataclass
class ClusterRow:
    cluster_id: int
    size: int
    diameter: float          # max intra-cluster distance
    avg_intra_dist: float    # mean intra-cluster distance
    n_exemplars: int
    nearest_cluster_id: int
    nearest_cluster_dist: float
    merge_candidate: bool    # exemplar dist < distance_threshold


@dataclass
class NearestClusterRow:
    cluster_id: int
    size: int
    min_exemplar_dist: float
    p10_cross_dist: float
    merge_threshold: float
    merge_candidate: bool


@dataclass
class FaceRow:
    face_id: int
    image_path: str
    blur_score: float
    area: float
    pose: Optional[Tuple[float, float, float]]  # (yaw, pitch, roll)
    dist_to_exemplar: float
    dist_to_centroid: float
    role: str   # "exemplar" | "core" | "attached" | "holdout"
    cluster_id: int
    is_outlier: bool


@dataclass
class EdgeInfo:
    """One edge in the kNN graph within a cluster."""
    face_id_a: int
    face_id_b: int
    distance: float


@dataclass
class FaceGraphInfo:
    """Per-face graph connectivity within a cluster."""
    face_id: int
    n_edges: int             # direct edges to other cluster members
    neighbors: List[int]     # face_ids of direct neighbors
    neighbor_dists: List[float]
    is_bridge: bool          # articulation point — removal splits cluster


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
        from face_cluster.config import PipelineConfig
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


@dataclass
class CloseFace:
    face_id: int
    image_path: str
    cluster_id: int
    distance: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embeddings_matrix(faces: List[FaceRecord], indices: List[int]) -> Optional[np.ndarray]:
    """Stack L2-normalised embeddings for the given face indices. Returns None if any missing."""
    vecs = []
    for i in indices:
        emb = faces[i].embedding_normalized if faces[i].embedding_normalized is not None else faces[i].embedding
        if emb is None:
            return None
        vecs.append(emb / (np.linalg.norm(emb) + 1e-9))
    return np.stack(vecs).astype(np.float32)


def _pairwise_distances(mat: np.ndarray) -> np.ndarray:
    """Cosine distances from L2-normalised embeddings (1 - dot product)."""
    sims = mat @ mat.T
    return np.clip(1.0 - sims, 0.0, 2.0)


def _centroid(mat: np.ndarray) -> np.ndarray:
    return mat.mean(axis=0)


def _dist_to_centroid(mat: np.ndarray) -> np.ndarray:
    c = _centroid(mat)
    c = c / (np.linalg.norm(c) + 1e-9)
    return np.clip(1.0 - mat @ c, 0.0, 2.0)


def _exemplar_dist(mat: np.ndarray, exemplar_local_indices: List[int]) -> np.ndarray:
    """Distance from each face to the nearest exemplar."""
    if not exemplar_local_indices:
        return np.zeros(len(mat))
    ex_mat = mat[exemplar_local_indices]
    sims = mat @ ex_mat.T
    return np.clip(1.0 - sims.max(axis=1), 0.0, 2.0)


def _face_row(
    face: FaceRecord,
    cluster_id: int,
    dist_to_exemplar: float,
    dist_to_centroid: float,
    exemplar_ids: List[int],
    outlier_threshold: float,
    is_core: bool,
) -> FaceRow:
    if face.face_id in exemplar_ids:
        role = "exemplar"
    elif is_core:
        role = "core"
    else:
        role = "attached"
    return FaceRow(
        face_id=face.face_id,
        image_path=face.image_path or "",
        blur_score=face.blur_score,
        area=face.area,
        pose=face.pose,
        dist_to_exemplar=dist_to_exemplar,
        dist_to_centroid=dist_to_centroid,
        role=role,
        cluster_id=cluster_id,
        is_outlier=dist_to_exemplar > outlier_threshold,
    )


# ---------------------------------------------------------------------------
# RunOverview
# ---------------------------------------------------------------------------

@dataclass
class RunOverview:
    """Full-run health check for the Run Overview tab."""

    # Quality gate funnel
    n_images: int
    n_faces_detected: int
    n_core: int
    n_holdout: int

    # Clustering summary
    n_clusters: int
    n_noise: int
    cluster_rows: List[ClusterRow]

    # UMAP (None if fewer than 5 core faces)
    umap_coords: Optional[np.ndarray]   # shape (n_core, 2)
    umap_labels: Optional[np.ndarray]   # cluster_id per core face (-1 = noise)
    umap_face_ids: Optional[List[int]]  # face_id per UMAP point

    # Stage timing from pipeline_run.json (may be empty)
    stage_timings: Dict[str, float]

    @classmethod
    def compute(cls, result: PipelineResult) -> "RunOverview":
        faces = result.faces
        cr = result.cluster_result
        summary = result.summary
        n_core = summary.get("n_core", 0)
        n_noise = summary.get("n_noise", 0)

        # Cluster rows — need embeddings to compute distances
        cluster_rows: List[ClusterRow] = []
        all_cluster_ids = sorted(cr.clusters.keys())

        # Build per-cluster exemplar embedding vectors for nearest-cluster computation
        exemplar_embeds: Dict[int, np.ndarray] = {}
        for cid, member_indices in cr.clusters.items():
            ex_indices = cr.exemplars.get(cid, member_indices[:1])
            mat = _embeddings_matrix(faces, ex_indices)
            if mat is not None:
                exemplar_embeds[cid] = mat.mean(axis=0)

        for cid in all_cluster_ids:
            member_indices = cr.clusters[cid]
            stats = cr.cluster_stats.get(cid, {})
            mat = _embeddings_matrix(faces, member_indices)

            if mat is not None and len(mat) > 1:
                pw = _pairwise_distances(mat)
                diameter = float(pw.max())
                avg_intra = float(pw[np.triu_indices_from(pw, k=1)].mean())
            elif mat is not None and len(mat) == 1:
                diameter = 0.0
                avg_intra = 0.0
            else:
                diameter = float(stats.get("diameter", 0.0))
                avg_intra = float(stats.get("avg_dist", 0.0))

            # Nearest cluster by exemplar distance
            nearest_cid = -1
            nearest_dist = float("inf")
            my_ex = exemplar_embeds.get(cid)
            if my_ex is not None:
                for other_cid, other_ex in exemplar_embeds.items():
                    if other_cid == cid:
                        continue
                    d = float(np.clip(1.0 - float(my_ex @ other_ex), 0.0, 2.0))
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_cid = other_cid

            ex_ids = [faces[i].face_id for i in cr.exemplars.get(cid, [])]
            cluster_rows.append(ClusterRow(
                cluster_id=cid,
                size=len(member_indices),
                diameter=round(diameter, 4),
                avg_intra_dist=round(avg_intra, 4),
                n_exemplars=len(ex_ids),
                nearest_cluster_id=nearest_cid,
                nearest_cluster_dist=round(min(nearest_dist, 9.999), 4),
                merge_candidate=(nearest_dist < 0.45),
            ))

        # UMAP
        umap_coords = None
        umap_labels = None
        umap_face_ids = None
        core_indices = [i for i, f in enumerate(faces) if f.is_core]
        if len(core_indices) >= 5:
            mat = _embeddings_matrix(faces, core_indices)
            if mat is not None:
                try:
                    import umap as umap_lib
                    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_jobs=1)
                    umap_coords = reducer.fit_transform(mat).astype(np.float32)
                    umap_labels = np.array([cr.labels[i] if i < len(cr.labels) else -1 for i in range(len(core_indices))], dtype=np.int32)
                    umap_face_ids = [faces[i].face_id for i in core_indices]
                except Exception as exc:
                    logger.warning(f"UMAP failed: {exc}")

        # Stage timings from summary
        stage_timings = summary.get("stages_timing", {}) or {}

        n_images = len({f.image_path for f in faces if f.image_path})

        return cls(
            n_images=n_images,
            n_faces_detected=len(faces),
            n_core=n_core,
            n_holdout=len(faces) - n_core,
            n_clusters=cr.n_clusters,
            n_noise=n_noise,
            cluster_rows=cluster_rows,
            umap_coords=umap_coords,
            umap_labels=umap_labels,
            umap_face_ids=umap_face_ids,
            stage_timings=stage_timings,
        )


# ---------------------------------------------------------------------------
# ClusterView
# ---------------------------------------------------------------------------

@dataclass
class ClusterView:
    """Per-cluster drill-down for the Cluster Analysis tab."""

    cluster_id: int
    size: int
    diameter: float
    avg_intra_dist: float

    exemplar_face_ids: List[int]
    faces: List[FaceRow]          # all members, sorted by dist_to_exemplar
    outlier_face_ids: List[int]   # dist_to_exemplar > p90
    split_signal: bool            # bimodal gap detected in intra-cluster distances

    nearest_clusters: List[NearestClusterRow]

    @classmethod
    def compute(cls, result: PipelineResult, cluster_id: int) -> "ClusterView":
        faces = result.faces
        cr = result.cluster_result

        if cluster_id not in cr.clusters:
            raise ValueError(f"cluster_id {cluster_id} not found in result")

        member_indices = cr.clusters[cluster_id]
        ex_face_ids = [faces[i].face_id for i in cr.exemplars.get(cluster_id, member_indices[:1])]
        ex_local = [j for j, i in enumerate(member_indices)
                    if faces[i].face_id in ex_face_ids]

        mat = _embeddings_matrix(faces, member_indices)
        if mat is not None and len(mat) > 1:
            pw = _pairwise_distances(mat)
            diameter = float(pw.max())
            avg_intra = float(pw[np.triu_indices_from(pw, k=1)].mean())
            d_exemplar = _exemplar_dist(mat, ex_local)
            d_centroid = _dist_to_centroid(mat)
            outlier_threshold = float(np.percentile(d_exemplar, 90)) if len(d_exemplar) > 2 else float(d_exemplar.max())
            # Split signal: large gap (>0.05) in sorted intra-distances suggests bimodal
            upper_dists = pw[np.triu_indices_from(pw, k=1)]
            sorted_d = np.sort(upper_dists)
            gaps = np.diff(sorted_d)
            split_signal = bool(gaps.max() > 0.12 and gaps.max() > 3 * gaps.mean()) if len(gaps) > 0 else False
        else:
            diameter = avg_intra = 0.0
            d_exemplar = np.zeros(len(member_indices))
            d_centroid = np.zeros(len(member_indices))
            outlier_threshold = 1.0
            split_signal = False

        core_face_ids = set(faces[i].face_id for i in (result.summary.get("core_face_ids") or []))
        face_rows = []
        for j, i in enumerate(member_indices):
            face = faces[i]
            is_core = face.is_core
            fr = _face_row(
                face=face,
                cluster_id=cluster_id,
                dist_to_exemplar=float(d_exemplar[j]),
                dist_to_centroid=float(d_centroid[j]),
                exemplar_ids=ex_face_ids,
                outlier_threshold=outlier_threshold,
                is_core=is_core,
            )
            face_rows.append(fr)
        face_rows.sort(key=lambda r: r.dist_to_exemplar)

        outlier_face_ids = [r.face_id for r in face_rows if r.is_outlier]

        # Nearest clusters
        my_ex_indices = cr.exemplars.get(cluster_id, member_indices[:1])
        my_ex_mat = _embeddings_matrix(faces, my_ex_indices)

        nearest: List[NearestClusterRow] = []
        for other_cid, other_members in cr.clusters.items():
            if other_cid == cluster_id:
                continue
            other_ex_indices = cr.exemplars.get(other_cid, other_members[:1])
            other_ex_mat = _embeddings_matrix(faces, other_ex_indices)
            other_mat = _embeddings_matrix(faces, other_members)

            if my_ex_mat is None or other_ex_mat is None:
                continue

            # Min exemplar-to-exemplar distance
            cross_sims = my_ex_mat @ other_ex_mat.T
            min_ex_dist = float(np.clip(1.0 - cross_sims.max(), 0.0, 2.0))

            # p10 cross distance (all members vs all members)
            p10_dist = min_ex_dist
            if other_mat is not None and mat is not None:
                cross_all = mat @ other_mat.T
                all_dists = np.clip(1.0 - cross_all, 0.0, 2.0).flatten()
                p10_dist = float(np.percentile(all_dists, 10))

            merge_threshold = 0.45
            nearest.append(NearestClusterRow(
                cluster_id=other_cid,
                size=len(other_members),
                min_exemplar_dist=round(min_ex_dist, 4),
                p10_cross_dist=round(p10_dist, 4),
                merge_threshold=merge_threshold,
                merge_candidate=min_ex_dist < merge_threshold,
            ))

        nearest.sort(key=lambda r: r.min_exemplar_dist)

        return cls(
            cluster_id=cluster_id,
            size=len(member_indices),
            diameter=round(diameter, 4),
            avg_intra_dist=round(avg_intra, 4),
            exemplar_face_ids=ex_face_ids,
            faces=face_rows,
            outlier_face_ids=outlier_face_ids,
            split_signal=split_signal,
            nearest_clusters=nearest[:10],
        )


# ---------------------------------------------------------------------------
# FaceView
# ---------------------------------------------------------------------------

@dataclass
class FaceView:
    """Per-face drill-down for the Face Analysis tab."""

    face_id: int
    image_path: str
    cluster_id: int           # -1 if holdout/noise

    blur_score: float
    area: float
    pose: Optional[Tuple[float, float, float]]
    rank_in_image: int        # ordinal position among faces in the same source image
    gate_result: str          # "core" | "holdout"
    gate_rejection_reason: Optional[str]

    closest_same_cluster: List[CloseFace]    # top 5
    closest_other_clusters: List[CloseFace]  # top 5
    coimage_faces: List[FaceRow]             # other faces in the same source image

    @classmethod
    def compute(cls, result: PipelineResult, face_id: int) -> "FaceView":
        faces = result.faces
        cr = result.cluster_result

        # Find the face
        face_map = {f.face_id: (i, f) for i, f in enumerate(faces)}
        if face_id not in face_map:
            raise ValueError(f"face_id {face_id} not found")

        idx, face = face_map[face_id]

        # Determine cluster
        cluster_id = -1
        for cid, members in cr.clusters.items():
            if idx in members:
                cluster_id = cid
                break

        # Gate result and rejection reason
        gate_result = "core" if face.is_core else "holdout"
        gate_rejection_reason: Optional[str] = None
        if not face.is_core:
            reasons = []
            if face.blur_score < 50.0:
                reasons.append(f"blur {face.blur_score:.1f} < min 50.0")
            if face.area < 1000:
                reasons.append(f"area {face.area:.0f} < min 1000")
            gate_rejection_reason = "; ".join(reasons) if reasons else "unknown"

        # Rank within source image
        if face.image_path:
            same_image_faces = [f for f in faces if f.image_path == face.image_path]
            rank_in_image = same_image_faces.index(face) + 1
        else:
            rank_in_image = 1

        # All embeddings for distance computation
        all_embs = _embeddings_matrix(faces, list(range(len(faces))))

        closest_same: List[CloseFace] = []
        closest_other: List[CloseFace] = []

        if all_embs is not None:
            my_emb = all_embs[idx]
            sims = all_embs @ my_emb
            dists = np.clip(1.0 - sims, 0.0, 2.0)

            # Sort all faces by distance (exclude self)
            order = np.argsort(dists)
            for j in order:
                if j == idx:
                    continue
                f2 = faces[j]
                f2_cluster = -1
                for cid, members in cr.clusters.items():
                    if j in members:
                        f2_cluster = cid
                        break
                cf = CloseFace(
                    face_id=f2.face_id,
                    image_path=f2.image_path or "",
                    cluster_id=f2_cluster,
                    distance=round(float(dists[j]), 4),
                )
                if f2_cluster == cluster_id and cluster_id != -1:
                    if len(closest_same) < 5:
                        closest_same.append(cf)
                else:
                    if len(closest_other) < 5:
                        closest_other.append(cf)
                if len(closest_same) >= 5 and len(closest_other) >= 5:
                    break

        # Co-image faces
        coimage: List[FaceRow] = []
        if face.image_path:
            for j, f2 in enumerate(faces):
                if f2.image_path == face.image_path and f2.face_id != face_id:
                    f2_cid = -1
                    for cid, members in cr.clusters.items():
                        if j in members:
                            f2_cid = cid
                            break
                    f2_ex = [faces[k].face_id for k in cr.exemplars.get(f2_cid, [])] if f2_cid >= 0 else []
                    coimage.append(FaceRow(
                        face_id=f2.face_id,
                        image_path=f2.image_path or "",
                        blur_score=f2.blur_score,
                        area=f2.area,
                        pose=f2.pose,
                        dist_to_exemplar=0.0,
                        dist_to_centroid=0.0,
                        role="core" if f2.is_core else "holdout",
                        cluster_id=f2_cid,
                        is_outlier=False,
                    ))

        return cls(
            face_id=face_id,
            image_path=face.image_path or "",
            cluster_id=cluster_id,
            blur_score=face.blur_score,
            area=face.area,
            pose=face.pose,
            rank_in_image=rank_in_image,
            gate_result=gate_result,
            gate_rejection_reason=gate_rejection_reason,
            closest_same_cluster=closest_same,
            closest_other_clusters=closest_other,
            coimage_faces=coimage,
        )
