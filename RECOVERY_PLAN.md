# Face Clustering Architecture Spec

## Goal

Produce reliable face clusters from a photo album, with full traceability and each
stage independently testable. This module is standalone — integration into the main
album pipeline comes later.

---

## Component Responsibilities

| Component | Owns | Does NOT own |
|---|---|---|
| `face_cluster/` | Pure algorithms (detection, embedding, clustering, export) | File I/O, paths, CLI |
| `scripts/run_face_clustering.py` | Orchestration: read album → call stages → write results | Algorithm logic |
| `app/face_clustering_labeling.py` | User corrections: merge/split clusters, save labels | Running clustering |
| `app/face_clustering_debug/` | Visualization: why did clustering decide X | Modifying any data |
| `tests/face_clustering/` | Verifying correctness of each stage | Real images or albums |

**Rule**: if you are unsure where code belongs, check this table. If it doesn't fit, ask.

---

## Pipeline Stages

Each stage has a typed input, typed output, output file, and an independent test.

```
Album path
   │
   ▼
[Stage 1] Detect & Embed          face_cluster/embedding.py → InsightFaceEmbedder
   │  Input:  image directory
   │  Output: List[FaceRecord]    (face_id, image_path, bbox, embedding, aligned_face)
   │  File:   face_records.json
   │
   ▼
[Stage 2] Quality Gate            face_cluster/quality.py → QualityGater
   │  Input:  List[FaceRecord]
   │  Output: List[FaceRecord]    (is_core flag set)
   │  File:   (updates face_records.json in-place)
   │
   ▼
[Stage 3] Save Crops              face_cluster/crops.py  (NEW - thin wrapper)
   │  Input:  List[FaceRecord]
   │  Output: crops/face_XXXX_aligned.jpg per face
   │  File:   crop_manifest.json  (face_id → crop_path)
   │
   ▼
[Stage 4] Build kNN + Cluster     face_cluster/knn_graph.py + clustering.py
   │  Input:  List[FaceRecord] (core set)
   │  Output: ClusterResult    (labels, clusters, exemplars)
   │  File:   cluster_result.json
   │
   ▼
[Stage 5] Export                  face_cluster/export.py  (NEW - thin wrapper)
      Input:  FaceRecord list + ClusterResult + crop_manifest
      Output: faces.csv, clusters.csv, export_summary.json
      File:   see Data Contract below
```

Each stage saves its output file **before** the next stage starts.
If stage 4 crashes, stages 1–3 output is intact and resumable.

---

## Data Contract

Both apps (`labeling`, `debug`) read exactly this directory structure:

```
results/{album_name}/{run_id}/
├── face_records.json        face_id, image_path (never null), bbox, is_core
├── crop_manifest.json       face_id → crop_path (face_XXXX_aligned.jpg)
├── cluster_result.json      face_id → cluster_id, n_clusters, n_noise
├── export_summary.json      source_album, run_id, config, created_at, n_faces
├── faces.csv                face_id, image_path, crop_path, cluster_id, quality scores
└── clusters.csv             cluster_id, size, exemplar_face_id, diameter
```

`export_summary.json` is the manifest. Apps always read it first to validate the run.

---

## Typed Interfaces (existing in `face_cluster/types.py`)

```
FaceRecord      face_id, image_path, bbox, embedding, aligned_face, is_core
GraphResult     neighbors, edges, distance_matrix
ClusterResult   labels, clusters, exemplars, n_clusters, n_noise
PipelineConfig  K, distance_threshold, quality thresholds, merge/attach flags
```

`image_path` on `FaceRecord` is **required** (not Optional) for production runs.
Any record with `image_path=None` must be rejected at stage 1 output.

---

## Stage Tests (independent — no real images needed)

```
tests/face_clustering/
├── test_stage1_detection.py     synthetic image → FaceRecord has image_path, norm > 0.1
├── test_stage2_quality.py       known pose angles → correct is_core assignment
├── test_stage3_crops.py         known bbox → crop file exists, filename matches face_id
├── test_stage4_clustering.py    3×3 synthetic embeddings → exactly 3 clusters, 0 noise
├── test_stage5_export.py        known ClusterResult → no nulls in faces.csv
└── test_e2e.py                  all stages in sequence → full lineage trace for every face
```

Each test uses fixtures from `tests/face_clustering/fixtures/` (synthetic data only).
No test touches a real album path.

---

## What to Build

| Item | Status | Notes |
|---|---|---|
| `face_cluster/crops.py` | NEW | Save aligned crops, write crop_manifest.json |
| `face_cluster/export.py` | NEW | Produce faces.csv, clusters.csv, export_summary.json |
| `scripts/run_face_clustering.py` | NEW | Replaces benchmark_face_clustering.py |
| `tests/face_clustering/` | NEW | All 6 test files above |
| `face_cluster/types.py` | MODIFY | Make `image_path` non-optional |

---

## What to Archive

Move to `archive/` — do not delete yet:

```
scripts/benchmark_face_clustering.py    → replaced by run_face_clustering.py
scripts/export_clustering_data.py       → replaced by face_cluster/export.py
scripts/regenerate_embeddings_from_crops.py
scripts/cluster_knn_components.py
scripts/debug_*.py (all)
scripts/compare_*.py (all)
scripts/trace_*.py (all)
scripts/analyze_*.py (all)
scripts/validate_export.py
app/face_clustering_comparison.py
app/debug_hybrid_closest.py
notebooks/ (all except debug_embeddings_comparison.ipynb)
```

---

## Enforcement Rules

1. **No new scripts** for face clustering. Investigations go in notebooks. Production logic goes in `face_cluster/`.
2. **No algorithm code** in `scripts/` or `app/`. Scripts call `face_cluster/`, never implement logic.
3. **No null `image_path`**. Stage 1 validates this before writing `face_records.json`.
4. **No stage reads from memory**. Every stage reads its input from the previous stage's file.
5. **No test uses a real album path**. All tests use synthetic fixtures.
6. **Any notebook insight** that is useful → promote to `face_cluster/` with a test before merging.

---

## Implementation Order

1. `face_cluster/types.py` — make `image_path` required
2. `face_cluster/crops.py` — save crops + manifest
3. `face_cluster/export.py` — produce CSV + summary
4. `tests/face_clustering/` — all stage tests (verify each before moving on)
5. `scripts/run_face_clustering.py` — orchestrator (thin, only I/O)
6. Archive old scripts/notebooks
7. Verify labeling app loads new output format correctly
