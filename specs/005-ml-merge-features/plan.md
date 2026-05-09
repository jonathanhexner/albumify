# ML Merge Features — Implementation Plan

**Date**: 2026-04-14 (updated)  
**Depends on**: `docs/ML_CLUSTER_MERGING.md` (feature catalog), `specs/002-interactive-merge-approval/` (UI)

---

## Goal

Compute a rich feature vector for every candidate cluster pair **immediately after base clustering**, and persist those features alongside human merge/reject labels so they can later be used to train a merge classifier.

---

## Current State

| What | Where | Status |
|------|-------|--------|
| `FeatureComputer` (V2, 17 features) | `face_cluster/features.py` | Exists but **not wired** into pipeline or UI |
| Merge Approval UI | `app/face_clustering.py` | Saves ~8 scalar fields per decision, **no feature vectors** |
| `merge_decisions.json` | `face_cluster/export.py` | Reader/writer exists |

### Key Constraints

1. **Index space**: After export/load, everything is face-list indices (0..N_faces-1). **FeatureComputer V3 must use face-list space.**
2. **GraphResult lost after export**: kNN edge features (Group F) are optional — compute when available, null otherwise.
3. **Distance matrix**: Not stored on disk. Recomputed from embeddings in the UI (~fast for album-scale).

---

## Data Flow

```
Pipeline run → base clusters
        |
        v
  FeatureComputer.compute_all_pairs(context)     ← happens once, right after clustering
        |
        v
  Dict[(cid_a, cid_b), ClusterPairFeatures]      ← cached in session, displayed in UI
        |
        v
  User labels approve/reject per pair
        |
        v
  save_merge_features(features + labels)          ← parquet to disk
```

Features are computed **before** the user sees any candidates. The user's label is just one column joined to the precomputed feature row at save time.

---

## What Needs to Change

### Two files, two concerns:

| File | What changes |
|------|-------------|
| `face_cluster/features.py` | Extend to V3: add ~25 new features (Groups A–D), accept `MergeFeatureContext` container, add `compute_all_pairs()` |
| `face_cluster/export.py` | Add `save_merge_features()` / `load_merge_features()` — write/read parquet with features + labels |

### UI wiring (small):

| File | What changes |
|------|-------------|
| `app/face_clustering.py` | Call `compute_all_pairs()` when building merge analysis view. On save, join labels to precomputed features and call `save_merge_features()`. |

That's it. No new modules until we actually train a classifier.

---

## Architecture

### Input Container

```python
@dataclass
class MergeFeatureContext:
    cluster_result: ClusterResult
    faces: List[FaceRecord]
    distance_matrix: np.ndarray           # (N_faces x N_faces), face-list indexed
    graph_result: Optional[GraphResult]   # for kNN edge features (optional)
```

Entry-point assertion: `distance_matrix.shape[0] == len(faces)`.

### Output

`compute_all_pairs(context, candidate_threshold) → Dict[Tuple[int,int], ClusterPairFeatures]`

Each `ClusterPairFeatures` has ~40 fields. Converts to a DataFrame row via `.to_dict()`.

---

## Implementation Tasks

### Step 1: `MergeFeatureContext` + V3 `ClusterPairFeatures`

Add the container dataclass and extend `ClusterPairFeatures` with new P1 fields (all `Optional[float]` for backward compat).

### Step 2: Extend `compute_cluster_stats`

Per-cluster stats needed by pair features: `n_images`, `blur_min`, `yaw_std`, `mean_area`, `diameter`, `mean_intra_dist`, `exemplar_count`. Accepts `MergeFeatureContext`.

### Step 3: Extend `compute_pair_features`

Add: cross-dist percentiles (p25/p75/p90), IQR, min_cross_dist, exemplar_dist_mean/std, post_merge_diameter, diameter_expansion, shared_source_images, same_image_min_dist.

### Step 4: `compute_all_pairs` + `to_dataframe`

Iterate candidate pairs, call `compute_pair_features` for each, return dict. Helper converts to DataFrame.

### Step 5: Persistence in `export.py`

`save_merge_features(df, output_dir)` → writes `merge_features.parquet`.  
`load_merge_features(run_dir)` → reads it back.

### Step 6: Wire into UI

- Compute features when merge analysis view is built.
- On save: join labels → call `save_merge_features()`.

### Step 7: Tests

- Unit test: synthetic embeddings → verify feature values.
- Contract test: save → load → assert dtypes and field presence.

---

## Later (not now)

These are deferred until we have labeled data:

- P2 features (margin, graph topology, quality/pose) — Groups E–G
- P3 features (interactions, global context) — Groups H–I
- Training script / classifier class
- UI "ML Score" column
- Saving kNN edges to disk for Group F features on loaded runs

---

## Risks

| Risk | Mitigation |
|------|------------|
| Index space confusion | V3 asserts `distance_matrix.shape[0] == len(faces)` at entry |
| Feature schema drift between runs | `feature_version` column in parquet |
| Slow feature computation | O(|Ci|*|Cj|) per pair; <1s total for album-scale |
