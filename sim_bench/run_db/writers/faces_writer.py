"""Per-table writer for the `faces` and `face_scores` tables (spec-057).

Extracted from the monolithic ``RunExporter._write_faces_and_scores`` so
each table has a focused, individually-testable writer module.

Public surface:
    write_faces(conn, faces, crop_manifest, image_scores) -> None
"""
from __future__ import annotations

from typing import Dict, List, Optional
import sqlite3

import pandas as pd

from face_cluster.db import FACES_SCHEMA, FACE_SCORES_SCHEMA
from face_cluster.types import FaceRecord
from sim_bench.run_db.writers._common import maybe_float


_FACES_COLUMNS = [
    "face_id", "image_path", "image_id", "face_index",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h", "crop_path",
    "det_score", "blur_score", "area", "yaw", "pitch", "roll",
    "is_core", "rejection_reason",
    "iqa_score", "ava_score", "sharpness_score", "scene_cluster_id",
    "area_ratio", "bbox_x_ratio", "bbox_y_ratio", "bbox_w_ratio", "bbox_h_ratio",
]

_SCORES_COLUMNS = [
    "face_id", "pose_score", "eyes_score", "expression_score",
    "frontal_score", "is_clusterable",
]


def write_faces(
    conn: sqlite3.Connection,
    faces: List[FaceRecord],
    crop_manifest: Dict[int, str],
    image_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Insert one row per face into the `faces` table and one matching row
    into `face_scores`. Pandera-validates both DataFrames before commit
    (spec-033 P-H) — a NULL in a non-nullable column raises ValidationError
    rather than landing as a corrupt row (SIGHTING-059 guard).
    """
    image_scores = image_scores or {}
    face_rows = []
    score_rows = []
    for face in faces:
        bbox = face.bbox or (0.0, 0.0, 0.0, 0.0)
        yaw, pitch, roll = (face.pose or (None, None, None))
        # spec-033 P-C C-3: per-image scores looked up by image_path
        # (the canonical key). Falls back to image_id for FC-App-style runs.
        img_key = face.image_path or face.image_id or ""
        img_score = image_scores.get(img_key) or image_scores.get(face.image_id, {})
        face_rows.append((
            face.face_id,
            face.image_path or face.image_id,
            face.image_id,
            face.face_index,
            maybe_float(bbox[0] if len(bbox) > 0 else None),
            maybe_float(bbox[1] if len(bbox) > 1 else None),
            maybe_float(bbox[2] if len(bbox) > 2 else None),
            maybe_float(bbox[3] if len(bbox) > 3 else None),
            crop_manifest.get(face.face_id, ""),
            maybe_float(face.det_score),
            float(face.blur_score),
            float(face.area),
            maybe_float(yaw),
            maybe_float(pitch),
            maybe_float(roll),
            1 if face.is_core else 0,
            face.rejection_reason,
            maybe_float(img_score.get("iqa")),
            maybe_float(img_score.get("ava")),
            maybe_float(img_score.get("sharpness")),
            img_score.get("scene_cluster_id"),
            # spec-040 Phase 4 (schema v5): canonical unit-normalized geometry.
            maybe_float(getattr(face, "area_ratio", None)),
            maybe_float(getattr(face, "bbox_x_ratio", None)),
            maybe_float(getattr(face, "bbox_y_ratio", None)),
            maybe_float(getattr(face, "bbox_w_ratio", None)),
            maybe_float(getattr(face, "bbox_h_ratio", None)),
        ))

        pose_score = None
        if face.pose and face.pose[0] is not None:
            pose_score = max(0.0, 1.0 - abs(face.pose[0]) / 90.0)
        score_rows.append((
            face.face_id, pose_score, None, None, None,
            1 if face.is_core else 0,
        ))

    FACES_SCHEMA.validate(pd.DataFrame(face_rows, columns=_FACES_COLUMNS))
    FACE_SCORES_SCHEMA.validate(pd.DataFrame(score_rows, columns=_SCORES_COLUMNS))

    conn.executemany(
        "INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        face_rows,
    )
    conn.executemany(
        "INSERT INTO face_scores VALUES (?,?,?,?,?,?)",
        score_rows,
    )
