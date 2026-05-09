# Face Clustering DB — Data Model Design

**Date**: 2026-05-04
**Status**: Draft — for discussion

## Overview

One SQLite DB per run: `{output_dir}/face_clustering.db`

Replaces: `faces.csv`, `clusters.csv`, `clusters_stage_base.csv`, `faces_merged.csv`, `clusters_merged.csv`, `merge_log.json`, `merge_metadata.json`, `crop_manifest.json`, `embeddings.npy`, `embedding_face_ids.npy`

## Tables

### faces
One row per detected face. Immutable after detection stage.

| Column | Type | Description |
|--------|------|-------------|
| face_id | INTEGER PK | Unique within run |
| image_path | TEXT NOT NULL | Source image file |
| face_index | INTEGER | Index within image (0, 1, 2...) |
| bbox_x | REAL | Normalized bounding box |
| bbox_y | REAL | |
| bbox_w | REAL | |
| bbox_h | REAL | |
| crop_path | TEXT | Path to 112x112 crop JPEG |
| det_score | REAL | Detection confidence |
| blur_score | REAL | Laplacian blur score |
| area | REAL | Face area in pixels |
| yaw | REAL | Pose yaw |
| pitch | REAL | Pose pitch |
| roll | REAL | Pose roll |
| is_core | BOOLEAN | Passed quality gating? |
| rejection_reason | TEXT | Why rejected from core (null if core) |

### embeddings
One row per face. Binary embedding stored as blob.

| Column | Type | Description |
|--------|------|-------------|
| face_id | INTEGER PK FK→faces | |
| embedding | BLOB | 512-dim float32 (2048 bytes) |
| model_name | TEXT | "buffalo_l", "custom", etc. |
| l2_norm | REAL | For quick sanity check (should be ~1.0) |

### cluster_assignments
One row per face per iteration. This is the core traceability table.

| Column | Type | Description |
|--------|------|-------------|
| face_id | INTEGER FK→faces | |
| cluster_id | INTEGER | Cluster ID at this iteration |
| iteration | INTEGER | 0 = base clustering, 1+ = after each merge |
| is_exemplar | BOOLEAN | Is this face an exemplar at this iteration? |
| d10_score | REAL | Exemplar ranking score (null if not exemplar) |

**Primary key**: (face_id, iteration)

Queries:
- Base assignment: `WHERE iteration = 0`
- Final assignment: `WHERE iteration = (SELECT MAX(iteration) FROM cluster_assignments)`
- Face history: `WHERE face_id = 42 ORDER BY iteration`
- Changed faces: `SELECT ... WHERE iteration = N AND cluster_id != (SELECT cluster_id FROM ... WHERE iteration = N-1 AND face_id = ...)`

### clusters
One row per cluster per iteration. Tracks cluster-level metrics across merge iterations.

| Column | Type | Description |
|--------|------|-------------|
| cluster_id | INTEGER | |
| iteration | INTEGER | 0 = base, 1+ = after merge |
| size | INTEGER | Number of faces |
| diameter | REAL | Max pairwise distance |
| avg_intra_dist | REAL | Average intra-cluster distance |
| origin | TEXT | "base", "auto_merge", "manual_merge" |
| parent_ids | TEXT | Comma-separated parent cluster IDs (if merged) |

**Primary key**: (cluster_id, iteration)

Queries:
- Base clusters: `WHERE iteration = 0`
- Final clusters: `WHERE iteration = (SELECT MAX(iteration) ...)`
- Cluster 6 history: `WHERE cluster_id = 6 ORDER BY iteration`
- Which clusters got merged: `WHERE origin = 'auto_merge'`

### merge_decisions
One row per candidate pair per iteration. Complete merge evidence.

| Column | Type | Description |
|--------|------|-------------|
| iteration | INTEGER | Which merge iteration |
| cluster_a | INTEGER | |
| cluster_b | INTEGER | |
| action | TEXT | "merged" or "rejected" |
| exemplar_dist | REAL | p25 exemplar distance |
| cross_dist | REAL | p25 cross distance (null if not computed) |
| support | INTEGER | Support count |
| margin_gap | REAL | Margin gap |
| post_diameter | REAL | Diameter if merged |
| passes_exemplar | BOOLEAN | Gate A: exemplar path |
| passes_cross | BOOLEAN | Gate A: cross path |
| passes_support | BOOLEAN | Gate B |
| passes_margin | BOOLEAN | Gate C |
| passes_diameter | BOOLEAN | Gate D |
| threshold_used | REAL | Actual threshold for this pair |
| rejection_reason | TEXT | Why rejected (null if merged) |

**Primary key**: (iteration, cluster_a, cluster_b)

### face_scores
Per-face scoring from the pipeline. One row per face.

| Column | Type | Description |
|--------|------|-------------|
| face_id | INTEGER PK FK→faces | |
| pose_score | REAL | Frontal score (0-1) |
| eyes_score | REAL | Eyes open score (0-1) |
| expression_score | REAL | Smile score (0-1) |
| frontal_score | REAL | Combined frontal (from landmarks) |
| is_clusterable | BOOLEAN | Passed frontal threshold? |

### run_metadata
Single-row table with run configuration and summary.

| Column | Type | Description |
|--------|------|-------------|
| run_id | TEXT PK | Timestamp-based ID |
| source_album | TEXT | Source directory |
| config | TEXT | Full PipelineConfig as JSON (one exception — config IS a blob) |
| n_images | INTEGER | |
| n_faces | INTEGER | |
| n_core | INTEGER | |
| n_clusters_base | INTEGER | |
| n_clusters_final | INTEGER | |
| n_merges | INTEGER | |
| n_iterations | INTEGER | |
| started_at | TEXT | ISO timestamp |
| finished_at | TEXT | ISO timestamp |

## What This Replaces

| Old file | New table(s) | Notes |
|----------|-------------|-------|
| faces.csv | faces + face_scores | No more CSV parsing |
| clusters.csv | clusters WHERE iteration=0 | No more overwriting |
| clusters_stage_base.csv | Not needed — clusters has all iterations |
| clusters_merged.csv | clusters WHERE iteration=max | |
| faces_merged.csv | cluster_assignments WHERE iteration=max | |
| merge_log.json | merge_decisions | Proper columns, not JSON blob |
| merge_metadata.json | run_metadata | |
| embeddings.npy | embeddings table | Queryable by face_id |
| embedding_face_ids.npy | Not needed — embeddings has face_id PK |
| crop_manifest.json | faces.crop_path column | |

## What Still Lives on Disk

- `crops/` directory with JPEG files — binary images don't belong in SQLite
- `face_clustering.db` — the single DB file

## Example Queries

**Full traceability chain for a face:**
```sql
SELECT f.image_path, f.face_index, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
       f.crop_path, e.model_name, e.l2_norm,
       ca.cluster_id, ca.iteration, ca.is_exemplar,
       fs.pose_score, fs.eyes_score, fs.expression_score
FROM faces f
JOIN embeddings e ON e.face_id = f.face_id
JOIN cluster_assignments ca ON ca.face_id = f.face_id
LEFT JOIN face_scores fs ON fs.face_id = f.face_id
WHERE f.face_id = 42
ORDER BY ca.iteration
```

**Verify embedding matches face (for spec-025 Part C):**
```sql
SELECT f.image_path, f.face_index, f.crop_path, e.embedding, e.l2_norm
FROM faces f
JOIN embeddings e ON e.face_id = f.face_id
WHERE f.face_id = 42
-- App loads crop, re-extracts embedding, compares to e.embedding
```

**Cosine distance between two faces:**
```sql
SELECT e1.embedding, e2.embedding
FROM embeddings e1, embeddings e2
WHERE e1.face_id = 42 AND e2.face_id = 89
-- App computes: 1 - dot(e1, e2) / (norm(e1) * norm(e2))
```

**Faces that changed cluster during merge:**
```sql
SELECT ca1.face_id, ca1.cluster_id AS before, ca2.cluster_id AS after
FROM cluster_assignments ca1
JOIN cluster_assignments ca2 ON ca1.face_id = ca2.face_id
WHERE ca1.iteration = 0 AND ca2.iteration = (SELECT MAX(iteration) FROM cluster_assignments)
AND ca1.cluster_id != ca2.cluster_id
```

**Suspicious faces (nearest neighbor in different cluster):**
```sql
-- Done in Python: load all embeddings, compute pairwise distances,
-- for each face find nearest neighbor, flag if different cluster_id
```

## Migration Path

1. Add `face_clustering_db.py` writer module alongside existing `export.py`
2. Pipeline writes BOTH old CSVs AND new DB during transition
3. Standalone app loader reads DB when available, falls back to CSVs
4. After validation: remove CSV writing, delete old files

## Open Questions

1. Should the main app's `UniversalCache` be unified with this DB? Or keep them separate (UniversalCache for pipeline caching, this DB for analysis output)?
2. Should `embeddings` store the raw float32 blob or a base64 string? Blob is smaller and faster but not human-readable.
3. Should per-iteration cluster scores (diameter, avg_intra_dist) be stored, or recomputed on demand from embeddings? Storing is faster but uses more space.
