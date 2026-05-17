"""spec-033 P-H: Pandera DataFrame schemas for the run DB I/O boundaries.

These schemas are the contract on the *DataFrame* side of the boundary
(Pydantic FaceRecord is the contract on the *object* side — see
`face_cluster/types.py`).

Lives next to the DDL in `face_cluster/db/` so a change to a table's
columns and its validator are reviewed together. Both writer
(`RunExporter`) and reader (`RunStore`) import from here.

Migration note: existing v4 runs were exported before this contract
existed, so columns we now declare ``nullable=False`` may contain NULLs.
Read-side validation calls live downstream of P-D / P-H rollout and may
warn-then-validate rather than hard-fail; that decision lives at the
call site, not in this module.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

# Pandas nullable Int64 — required for columns that mix int values with None
# (regular int64 can't hold NaN; float64 loses int semantics).
_NULLABLE_INT = pd.Int64Dtype()


# ---------------------------------------------------------------------------
# faces — one row per face in the run.
# spec-033 P-C C-1 + C-3: bridge plumbs blur/det/pose/landmarks/aligned_face
# through, so these columns become non-nullable on fresh runs.  Legacy v4
# DBs predate this and will fail validation — readers may opt out.
# ---------------------------------------------------------------------------
FACES_SCHEMA = DataFrameSchema(
    {
        "face_id":      Column(int, nullable=False, unique=True),
        "image_path":   Column(str, nullable=True),
        "image_id":     Column(str, nullable=True),
        "face_index":   Column(_NULLABLE_INT, nullable=True),
        "bbox_x":       Column(float, nullable=True),
        "bbox_y":       Column(float, nullable=True),
        "bbox_w":       Column(float, nullable=True),
        "bbox_h":       Column(float, nullable=True),
        "crop_path":    Column(str, nullable=True),
        # The 5 fields that SIGHTING-059 lost.  spec-033 P-C C-1 plumbs them.
        "det_score":    Column(float, nullable=True),  # nullable while bridge still permits absence
        "blur_score":   Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        "area":         Column(float, nullable=False, checks=pa.Check.ge(0.0)),
        "yaw":          Column(float, nullable=True),  # nullable when pose wasn't computed
        "pitch":        Column(float, nullable=True),
        "roll":         Column(float, nullable=True),
        "is_core":      Column(int, nullable=False, checks=pa.Check.isin([0, 1])),
        "rejection_reason": Column(str, nullable=True),
        # spec-033 P-C C-3: image-level joined fields
        "iqa_score":    Column(float, nullable=True),
        "ava_score":    Column(float, nullable=True),
        "sharpness_score": Column(float, nullable=True),
        "scene_cluster_id": Column(_NULLABLE_INT, nullable=True),
        # spec-040 Phase 4 (schema v5): canonical unit-normalized geometry.
        # SIGHTING-064 fix; ∈ [0,1]. Nullable on writes from legacy code paths
        # that don't yet compute them; required on the unified path.
        "area_ratio":     Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "bbox_x_ratio":   Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "bbox_y_ratio":   Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "bbox_w_ratio":   Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "bbox_h_ratio":   Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
    },
    strict=False,  # tolerate extra columns; existing readers add more
    coerce=True,
)


# ---------------------------------------------------------------------------
# face_scores — one row per face_id with optional per-scorer values.
# All scorer columns nullable because not every scorer runs on every face
# (this is by design).
# ---------------------------------------------------------------------------
FACE_SCORES_SCHEMA = DataFrameSchema(
    {
        "face_id":          Column(int, nullable=False, unique=True),
        "pose_score":       Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "eyes_score":       Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "expression_score": Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "frontal_score":    Column(float, nullable=True, checks=pa.Check.in_range(0.0, 1.0, include_min=True, include_max=True)),
        "is_clusterable":   Column(_NULLABLE_INT, nullable=True, checks=pa.Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# images — spec-040 Phase 4 (schema v5) · SIGHTING-065 fix
# ---------------------------------------------------------------------------
IMAGES_SCHEMA = DataFrameSchema(
    {
        "image_path":       Column(str, nullable=False, unique=True),
        "image_id":         Column(str, nullable=True),
        "width_px":         Column(_NULLABLE_INT, nullable=True),
        "height_px":        Column(_NULLABLE_INT, nullable=True),
        "n_faces":          Column(int, nullable=False, checks=pa.Check.ge(0)),
        "iqa_score":        Column(float, nullable=True),
        "ava_score":        Column(float, nullable=True),
        "sharpness_score":  Column(float, nullable=True),
        "composite_score":  Column(float, nullable=True),
        "scene_cluster_id": Column(_NULLABLE_INT, nullable=True),
        "filter_passed":    Column(int, nullable=False, checks=pa.Check.isin([0, 1])),
        "created_at":       Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# scene_clusters — spec-040 Phase 4 · SIGHTING-066 fix
# ---------------------------------------------------------------------------
SCENE_CLUSTERS_SCHEMA = DataFrameSchema(
    {
        "scene_cluster_id":    Column(int, nullable=False),
        "iteration":           Column(int, nullable=False, checks=pa.Check.ge(0)),
        "size":                Column(int, nullable=False, checks=pa.Check.ge(0)),
        "method":              Column(str, nullable=False),
        "exemplar_image_path": Column(str, nullable=True),
        "avg_intra_distance":  Column(float, nullable=True),
        "created_at":          Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# scene_cluster_assignments — spec-040 Phase 4
# ---------------------------------------------------------------------------
SCENE_CLUSTER_ASSIGNMENTS_SCHEMA = DataFrameSchema(
    {
        "image_path":           Column(str, nullable=False),
        "scene_cluster_id":     Column(int, nullable=False),
        "iteration":            Column(int, nullable=False, checks=pa.Check.ge(0)),
        "distance_to_centroid": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


# ---------------------------------------------------------------------------
# filter_decisions — spec-032 typed filter log.
# ---------------------------------------------------------------------------
FILTER_DECISIONS_SCHEMA = DataFrameSchema(
    {
        "item_id":       Column(str, nullable=False),
        "item_type":     Column(str, nullable=False, checks=pa.Check.isin(["image", "face", "cluster"])),
        "parent_id":     Column(str, nullable=True),
        "filter_name":   Column(str, nullable=False),
        "rejected":      Column(int, nullable=False, checks=pa.Check.isin([0, 1])),
        "reason":        Column(str, nullable=False),
        "measured_json": Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)
