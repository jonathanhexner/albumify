"""Per-table writer for the `images` table (spec-040 Phase 4 / spec-057).

Extracted from RunExporter._write_images.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from face_cluster.db import IMAGES_SCHEMA
from face_cluster.types import FaceRecord
from sim_bench.run_db.writers._common import maybe_float


def _norm(p: str) -> str:
    """Normalize path separators so face_records (forward slashes) and
    caller-supplied paths (potentially backslash on Windows) align.
    """
    return str(p).replace("\\", "/")


def write_images(
    conn: sqlite3.Connection,
    faces: List[FaceRecord],
    image_paths: Optional[List[str]] = None,
    image_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> None:
    """Persist one row per distinct image to the `images` table.

    Rows derived from two sources unioned:
      * every distinct face.image_path in faces (image had >=1 face)
      * every entry in image_paths (images discovered upstream, even if
        they produced zero faces)

    Pandera-validates the resulting DataFrame before INSERT (spec-033 P-H).
    Empty input still validates an empty DataFrame so the schema contract
    is exercised on every run.
    """
    image_scores = image_scores or {}
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    by_path: Dict[str, Dict[str, object]] = {}
    for face in faces:
        p = _norm(face.image_path) if face.image_path else None
        if not p:
            continue
        entry = by_path.setdefault(p, {
            "n_faces": 0,
            "width_px": None,
            "height_px": None,
        })
        entry["n_faces"] = int(entry["n_faces"]) + 1
        if entry["width_px"] is None and face.image_width_px:
            entry["width_px"] = int(face.image_width_px)
        if entry["height_px"] is None and face.image_height_px:
            entry["height_px"] = int(face.image_height_px)

    all_paths = sorted(set(by_path) | {_norm(p) for p in (image_paths or [])})

    rows: List[Tuple] = []
    for path in all_paths:
        entry = by_path.get(path, {"n_faces": 0, "width_px": None, "height_px": None})
        scores = image_scores.get(path, {}) or {}
        rows.append((
            path,
            Path(path).name,
            entry["width_px"],
            entry["height_px"],
            int(entry["n_faces"]),
            maybe_float(scores.get("iqa")),
            maybe_float(scores.get("ava")),
            maybe_float(scores.get("sharpness")),
            maybe_float(scores.get("composite")),
            scores.get("scene_cluster_id"),
            1,
            now,
        ))

    df = pd.DataFrame(rows, columns=[
        "image_path", "image_id", "width_px", "height_px", "n_faces",
        "iqa_score", "ava_score", "sharpness_score", "composite_score",
        "scene_cluster_id", "filter_passed", "created_at",
    ])
    IMAGES_SCHEMA.validate(df)

    if rows:
        conn.executemany(
            "INSERT INTO images "
            "(image_path, image_id, width_px, height_px, n_faces, "
            " iqa_score, ava_score, sharpness_score, composite_score, "
            " scene_cluster_id, filter_passed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
