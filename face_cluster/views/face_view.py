"""FaceView: per-face drill-down."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from face_cluster.pipeline import PipelineResult
from face_cluster.views._base import (
    CloseFace, FaceRow, _embeddings_matrix,
)


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

        # Gate result and rejection reason — prefer persisted verdict; heuristic as fallback
        gate_result = "core" if face.is_core else "holdout"
        gate_rejection_reason: Optional[str] = None
        if not face.is_core:
            if face.rejection_reason:
                gate_rejection_reason = face.rejection_reason
            else:
                # Fallback heuristic for legacy runs without quality_verdict
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
                        area_ratio=f2.area_ratio,
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
