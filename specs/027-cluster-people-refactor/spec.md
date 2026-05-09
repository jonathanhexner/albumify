# spec-027: Refactor cluster_people.py

**Status**: Implemented
**Date**: 2026-05-05

## Problem

`sim_bench/pipeline/steps/cluster_people.py` is 687 lines with 24 buried imports. It handles four separate responsibilities:
1. Pipeline step class (ClusterPeopleStep) with 6 clustering methods
2. FaceForClustering ↔ FaceRecord bridge
3. Face crop generation from bounding boxes
4. Export to disk (CSVs + DB + crops + merge artifacts)

## Solution

Split into 4 files:

| File | Responsibility | ~Lines |
|------|---------------|--------|
| `cluster_people.py` | Step class, method dispatch, label building | ~150 |
| `face_cluster_bridge.py` | `_faces_to_face_records()`, `_run_face_cluster_knn()` | ~150 |
| `face_cluster_export.py` | `_export_for_analysis()`, `_generate_crops_from_bboxes()`, DB write call | ~200 |
| (existing) `face_cluster/result_db.py` | DB writer | unchanged |

All imports at module top. No function-level imports except genuine lazy imports for optional heavy dependencies (hdbscan, sklearn).

## Acceptance Criteria

1. All 4 files have imports at the top
2. No file exceeds 200 lines
3. All existing tests pass unchanged
4. Pipeline behavior identical (same output for same input)
5. E2E test passes
