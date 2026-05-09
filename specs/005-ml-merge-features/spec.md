# Feature Specification: ML Merge Features (V3)

**Feature Branch**: `005-ml-merge-features`
**Created**: 2026-04-16
**Status**: Implemented
**Source plan**: `docs/ML_MERGE_FEATURES_PLAN.md`, Cursor plan `ml_merge_features_modules_a2826991`

## Overview

Compute a rich feature vector for every candidate cluster pair immediately after base clustering, and persist those features alongside human merge/reject labels so they can later train a merge classifier. This replaces the V2 17-feature scalar set with a V3 ~40-feature package of focused modules.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Collect labeled training data while reviewing merges (Priority: P1)

A user reviews proposed cluster merges in the Merge Analysis tab. They approve or reject pairs. On save, the system automatically stores the rich feature vector for each decision — no extra action needed. A data scientist can later load `merge_features.parquet` from any completed run and train a classifier.

**Acceptance Scenarios**:

1. **Given** a completed clustering run with merge candidates, **When** the user opens the Merge Analysis tab, **Then** feature vectors for all candidate pairs are computed (without user action) and cached for the session.
2. **Given** the user has made approve/reject decisions, **When** they click "Save Decisions", **Then** `merge_features.parquet` appears in the run output directory, containing one row per labeled pair with all feature columns plus `label`, `run_id`, `timestamp`, `feature_version`.
3. **Given** a saved `merge_features.parquet`, **When** loaded with `load_merge_features(run_dir)`, **Then** all expected feature columns are present with correct dtypes (float for distances, int for counts).

---

### User Story 2 - Feature schema is stable across runs (Priority: P1)

A data scientist combines parquet files from multiple runs. The schema must be consistent so DataFrames can be concatenated without manual cleanup.

**Acceptance Scenarios**:

1. **Given** two parquet files from different runs, **When** loaded and concatenated, **Then** column names and dtypes match exactly.
2. **Given** a parquet with some `None` feature values (e.g., quality features for faces without pose data), **When** loaded, **Then** those columns are float with NaN, not object dtype.

---

### Edge Cases

- Cluster pair with only 1 face per side: percentiles use single value; no crash.
- Faces with no `image_path`: source image features gracefully return 0 / NaN, not exception.
- Faces with no `pose` data: quality features return defaults (frontal_frac=1.0, pose_diff=0.0).
- No candidate pairs under threshold: `compute_all_pairs` returns empty dict; parquet not written.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST compute feature vectors for all candidate pairs (min exemplar dist ≤ `candidate_threshold`) immediately when the Merge Analysis tab is opened.
- **FR-002**: Feature computation MUST be asynchronous — must not block the Streamlit render thread.
- **FR-003**: Features MUST be organized into focused modules: `distance.py` (Group A), `geometry.py` (Groups B+C), `source_images.py` (Group D), `quality.py` (Group G).
- **FR-004**: `save_merge_features()` MUST write `merge_features.parquet` with columns: all `ClusterPairFeatures` fields + `cluster_a`, `cluster_b`, `label`, `run_id`, `timestamp`, `feature_version`.
- **FR-005**: `load_merge_features()` MUST return `None` (not raise) when file is absent.
- **FR-006**: `MergeFeatureContext` MUST assert `distance_matrix.shape[0] == len(faces)` at construction (index-space guard).
- **FR-007**: `ClusterPairFeatures` fields MUST all be `Optional` for backward compatibility with V1/V2 consumers.
- **FR-008**: Deferred modules (`context.py`, `graph.py`) MUST exist as stubs so import paths are stable.

### Key Entities

- **`MergeFeatureContext`**: Input container (cluster_result, faces, distance_matrix, graph_result).
- **`ClusterPairFeatures`**: Output dataclass (~40 Optional fields, V3).
- **`FeatureComputer`**: Orchestrator — `compute_all_pairs()`, `compute_pair_features()`, `to_dataframe()`.
- **`merge_features.parquet`**: Persistence artifact in run output directory.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 19 unit and contract tests pass (`tests/face_clustering/test_features_v3.py`, `test_merge_features_contract.py`).
- **SC-002**: `merge_features.parquet` is written on save with correct schema (verified by `test_save_load_roundtrip`).
- **SC-003**: Feature computation completes in < 1s for album-scale runs (≤ 200 faces, ≤ 50 clusters).
- **SC-004**: No `None` values for Group A or B features — all distance/geometry fields are always populated for valid pairs.

## Feature Catalog

### Implemented (V3)

| Group | Module | Features |
|-------|--------|----------|
| A | `distance.py` | `min_exemplar_dist`, `exemplar_dist_mean`, `exemplar_dist_std`, `min_cross_dist`, `p10/p25/p50/p75/p90_cross_dist`, `cross_dist_iqr`, `support_fraction`, `n_cross_pairs_below_threshold` |
| B+C | `geometry.py` | `size_a/b`, `size_min/ratio/sum`, `diameter_a/b/max/ratio`, `post_merge_diameter`, `diameter_expansion`, `mean_intra_dist_a/b`, `exemplar_count_a/b`, `t_a/b`, `t_local/global`, `dist_to_threshold_ratio` |
| D | `source_images.py` | `n_images_a/b`, `shared_source_images`, `shared_source_ratio`, `same_image_min_dist` |
| G | `quality.py` | `mean_blur_a/b`, `blur_min_a/b`, `frontal_frac_a/b/min`, `pose_diff`, `yaw_std_a/b`, `mean_area_a/b`, `area_ratio` |

### Deferred (stubs only)

| Group | Module | Features |
|-------|--------|----------|
| E+I | `context.py` | margin_a/b, second_nearest_dist, rank_among_candidates, global context (n_total_clusters, dist_zscore, etc.) |
| F | `graph.py` | knn_edge_count, edge_density, bidirectional_edges, hub_node_count |

### Known Gaps vs. `docs/ML_MERGE_FEATURES_PLAN.md`

These features appear in the spec document but were **not included** in V3 implementation:

| Feature | Group | Reason |
|---------|-------|--------|
| `exemplar_density_a` | B | Omitted in Cursor plan's geometry.py scope |
| `images_per_face_a/b` | D (spec: C) | Omitted in source_images.py scope |
| `n_images_post_merge` | D (spec: C) | Omitted in source_images.py scope |

These are low-priority additions — derivable from already-present fields (`exemplar_count_a / size_a`, `n_images_a / size_a`, `n_images_a + n_images_b - shared_source_images`).

## Assumptions

- `distance_matrix` is face-list indexed (row/col = face index in `faces` list), not core-only.
- `FaceRecord.image_path` is always non-None (pipeline validates this at Stage 1).
- Quality features degrade gracefully when `blur_score`, `area`, or `pose` are missing/zero — no crash.
- kNN graph (`graph_result`) is optional; Group F features are all null when absent.
