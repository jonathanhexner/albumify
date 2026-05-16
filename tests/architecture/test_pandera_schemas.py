"""spec-033 P-H: enforce Pandera schemas on DB I/O boundary.

Rules:
  1. Each schema rejects rows that violate nullable=False on the columns
     spec-033 P-C plumbs (blur_score, area, is_core).
  2. Each schema accepts a fully-populated synthetic row.
  3. RunExporter._write_faces_and_scores invokes the schemas (validation at
     write time, not just a passive contract).
"""
from __future__ import annotations

import inspect
import re

import pandas as pd
import pandera.errors as paerr
import pytest

from face_cluster.db import (
    FACES_SCHEMA,
    FACE_SCORES_SCHEMA,
    FILTER_DECISIONS_SCHEMA,
)


def _good_face_row(face_id: int = 0) -> dict:
    return {
        "face_id":          face_id,
        "image_path":       f"img_{face_id}.jpg",
        "image_id":         f"img_{face_id}",
        "face_index":       0,
        "bbox_x":           0.0, "bbox_y": 0.0, "bbox_w": 100.0, "bbox_h": 100.0,
        "crop_path":        f"crops/{face_id}.jpg",
        "det_score":        0.95,
        "blur_score":       100.0,
        "area":             10000.0,
        "yaw":              5.0, "pitch": 2.0, "roll": 1.0,
        "is_core":          1,
        "rejection_reason": None,
        "iqa_score":        0.8, "ava_score": 0.7, "sharpness_score": 0.6,
        "scene_cluster_id": 0,
    }


def test_faces_schema_accepts_complete_row():
    df = pd.DataFrame([_good_face_row(0), _good_face_row(1)])
    FACES_SCHEMA.validate(df)


def test_faces_schema_rejects_null_blur_score():
    row = _good_face_row()
    row["blur_score"] = None
    with pytest.raises(paerr.SchemaError):
        FACES_SCHEMA.validate(pd.DataFrame([row]))


def test_faces_schema_rejects_null_area():
    row = _good_face_row()
    row["area"] = None
    with pytest.raises(paerr.SchemaError):
        FACES_SCHEMA.validate(pd.DataFrame([row]))


def test_faces_schema_rejects_invalid_is_core():
    row = _good_face_row()
    row["is_core"] = 2  # only 0/1 allowed
    with pytest.raises(paerr.SchemaError):
        FACES_SCHEMA.validate(pd.DataFrame([row]))


def test_faces_schema_rejects_negative_area():
    row = _good_face_row()
    row["area"] = -1.0
    with pytest.raises(paerr.SchemaError):
        FACES_SCHEMA.validate(pd.DataFrame([row]))


def test_filter_decisions_schema_rejects_unknown_item_type():
    df = pd.DataFrame([{
        "item_id": "x", "item_type": "ghost", "parent_id": None,
        "filter_name": "image_quality", "rejected": 0,
        "reason": "ok", "measured_json": "{}",
    }])
    with pytest.raises(paerr.SchemaError):
        FILTER_DECISIONS_SCHEMA.validate(df)


def test_face_scores_schema_accepts_nullable_columns():
    df = pd.DataFrame([{
        "face_id": 0, "pose_score": None, "eyes_score": None,
        "expression_score": None, "frontal_score": None, "is_clusterable": None,
    }])
    FACE_SCORES_SCHEMA.validate(df)


def test_exporter_invokes_faces_schema():
    """spec-033 P-H: the WRITER must call FACES_SCHEMA, not just import it."""
    from face_cluster import run_exporter
    src = inspect.getsource(run_exporter._write_faces_and_scores
                            if hasattr(run_exporter, "_write_faces_and_scores")
                            else run_exporter.RunExporter._write_faces_and_scores)
    assert re.search(r"FACES_SCHEMA\.validate\b", src), (
        "RunExporter._write_faces_and_scores must call FACES_SCHEMA.validate "
        "(spec-033 P-H — validation at write time, not just an unused import)."
    )


def test_exporter_invokes_face_scores_schema():
    """Mirror: FACE_SCORES_SCHEMA must be called from the writer."""
    from face_cluster import run_exporter
    src = inspect.getsource(run_exporter.RunExporter._write_faces_and_scores)
    assert re.search(r"FACE_SCORES_SCHEMA\.validate\b", src), (
        "RunExporter._write_faces_and_scores must call FACE_SCORES_SCHEMA.validate."
    )
