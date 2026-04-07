# Sprint Plans: Clustering Algorithm Documentation & Face Debug Improvements

## Overview

These sprints add structured documentation and decision parameters to all clustering algorithms, plus improve the face clustering debug app with filename visibility, landmarks overlay, and proper 5-point face alignment.

---

## Sprint 1: Base Class Enhancement [DONE]

**Goal**: Add `doc_explanation` and `decision_parameters` attributes to `ClusteringMethod` base class.

**Files**: `sim_bench/clustering/base.py`

**Changes**:
1. Add class attributes to `ClusteringMethod`:
   ```python
   # 5-6 line explanation of how the algorithm works
   doc_explanation: str = ""

   # Dict of decision parameters: {param_name: {description, default, used_for}}
   decision_parameters: Dict[str, Dict[str, Any]] = {}
   ```

2. Add method `get_decision_info()` that returns:
   - Algorithm name
   - doc_explanation
   - decision_parameters with current values
   - Threshold values used in last clustering run

3. Update `cluster()` signature to store computed thresholds in `self.last_run_thresholds`

**Acceptance Criteria**:
- Base class has `doc_explanation` and `decision_parameters` attributes
- `get_decision_info()` returns structured info for UI display
- Existing tests pass

---

## Sprint 2: HDBSCAN Documentation [DONE]

**Goal**: Add documentation attributes to `HDBSCANClusterer`.

**Files**: `sim_bench/clustering/hdbscan.py`

**Changes**:
```python
doc_explanation = """
HDBSCAN uses density-based clustering to find groups without specifying k.
Decision: A point joins a cluster if it's in a dense region (mutual reachability).

Key Parameters:
- min_cluster_size: Minimum points to form a cluster
- cluster_selection_epsilon: Merge clusters closer than this distance
- metric: Distance metric (cosine recommended for face embeddings)

Threshold: Points with mutual_reachability > epsilon become noise.
"""

decision_parameters = {
    "min_cluster_size": {
        "description": "Minimum faces to form a valid cluster",
        "default": 5,
        "decision_role": "Clusters smaller than this become noise"
    },
    "cluster_selection_epsilon": {
        "description": "Distance threshold for cluster merging",
        "default": 0.045,
        "decision_role": "Clusters with min distance < epsilon are merged"
    },
    "min_samples": {
        "description": "Core point density requirement",
        "default": None,
        "decision_role": "Points need this many neighbors to be core points"
    }
}
```

**Acceptance Criteria**:
- `HDBSCANClusterer.doc_explanation` returns 5-6 line explanation
- `decision_parameters` dict documents each parameter's role in decisions
- Values match what's actually used in `cluster()` method

---

## Sprint 3: hybrid_hdbscan_knn Documentation [DONE]

**Goal**: Add documentation attributes to `HybridHDBSCANKNN`.

**Files**: `sim_bench/clustering/hybrid_hdbscan_knn.py`

**Changes**:
```python
doc_explanation = """
Hybrid method: HDBSCAN creates initial clusters, then iteratively merges/attaches.
Threshold T computed per cluster: percentile(exemplar_pairwise_distances).

Merge Decision: Clusters A & B merge if >=3 exemplar pairs have distance <= min(T_A, T_B),
with >=2 distinct exemplars from each side.

Attach Decision: Noise point attaches if >=2 exemplars are within cluster's T.
"""

decision_parameters = {
    "threshold_floor": {
        "description": "Minimum allowed threshold T",
        "default": 0.125,
        "decision_role": "Prevents over-splitting (T cannot go below this)"
    },
    "threshold_ceiling": {
        "description": "Maximum allowed threshold T",
        "default": 0.405,
        "decision_role": "Prevents over-merging (T cannot exceed this)"
    },
    "merge_min_pairs": {
        "description": "Required exemplar pairs within T to merge",
        "default": 3,
        "decision_role": "Merge if cross_pairs >= merge_min_pairs"
    },
    "merge_min_distinct": {
        "description": "Required distinct exemplars per side",
        "default": 2,
        "decision_role": "Both clusters must contribute >= this many exemplars"
    },
    "attach_min_exemplars": {
        "description": "Exemplars within T to attach noise",
        "default": 2,
        "decision_role": "Noise attaches if >= this many exemplars within T"
    },
    "threshold_percentile": {
        "description": "Percentile of exemplar distances for T",
        "default": 90,
        "decision_role": "T = percentile(exemplar_dists, this value)"
    }
}
```

**Acceptance Criteria**:
- Explanation focuses on merge/attach decisions
- Parameters clearly show what drives each decision
- UI can display "Threshold T=0.15, merge_min_pairs=3, actual_pairs=4 -> MERGE"

---

## Sprint 4: hybrid_closest_face Documentation [DONE]

**Goal**: Add documentation attributes to `HybridHDBSCANClosestFace`.

**Files**: `sim_bench/clustering/hybrid_closest_face.py`

**Changes**:
```python
doc_explanation = """
Like hybrid_hdbscan_knn but uses ALL faces (not just exemplars) for merge decision.
Threshold T computed from d3 (k-th neighbor distance) of all faces in cluster.

Merge Decision: For each face in A, compute d3_cross (distance to k-th nearest in B).
Face "fits" if d3_cross <= T_A * merge_threshold_multiplier.
Merge if fits_A + fits_B >= merge_min_faces.

Better for pose variation where exemplars may be biased toward frontal faces.
"""

decision_parameters = {
    "threshold_floor": {
        "description": "Minimum allowed threshold T",
        "default": 0.045,
        "decision_role": "T cannot go below this"
    },
    "threshold_ceiling": {
        "description": "Maximum allowed threshold T",
        "default": 0.405,
        "decision_role": "T cannot exceed this"
    },
    "merge_min_faces": {
        "description": "Total faces that must fit to merge",
        "default": 2,
        "decision_role": "Merge if fits_A + fits_B >= this"
    },
    "merge_threshold_multiplier": {
        "description": "Relaxation factor for d3_cross check",
        "default": 1.5,
        "decision_role": "Face fits if d3_cross <= T * this multiplier"
    },
    "early_exit_multiplier": {
        "description": "Skip pair if exemplars too far apart",
        "default": 2.0,
        "decision_role": "Skip merge check if min_exemplar_dist > max(T_A,T_B) * this"
    }
}
```

**Note**: Remove "min_distance" from UI display - it's not the decision parameter. Show `d3_cross` values and `merge_min_faces` instead.

---

## Sprint 5: Hybrid Variants Documentation [DONE]

**Goal**: Add documentation to `Tcore2all`, `merge_twotier`, `attach_strong1`.

**Files**:
- `sim_bench/clustering/hybrid_hdbscan_knn_Tcore2all.py`
- `sim_bench/clustering/hybrid_hdbscan_knn_merge_twotier.py`
- `sim_bench/clustering/hybrid_hdbscan_knn_attach_strong1.py`

**Pattern**: Same structure as Sprint 3/4 - add `doc_explanation` and `decision_parameters`.

---

## Sprint 6: Other Clustering Methods [DONE]

**Goal**: Add documentation to `mutual_knn`, `dbscan`, `kmeans`, `hierarchical`.

**Files**:
- `sim_bench/clustering/mutual_knn.py`
- `sim_bench/clustering/dbscan.py`
- `sim_bench/clustering/kmeans.py`
- `sim_bench/clustering/hierarchical.py`

**Pattern**: Same structure - `doc_explanation` and `decision_parameters`.

---

## Sprint 7: Face Grid - Add Image Filename [DONE]

**Goal**: Show image filename in face gallery captions.

**Files**:
- `app/face_clustering_debug/components/face_grid.py`
- `app/face_clustering_debug/models/schemas.py` (if needed)

**Changes**:
1. Update `FaceInfo` schema to ensure `image_path` is available
2. Update caption in `render_face_grid()`:
   ```python
   # Before: caption = f"{star}#{face.index}"
   # After:
   filename = Path(face.image_path).stem if face.image_path else ""
   caption = f"{star}#{face.index} | {filename}"
   ```

**Acceptance Criteria**:
- Each face shows `#42 | IMG_1234` format
- Truncate long filenames if needed

---

## Sprint 8: Face Detail - Enhanced View [DONE]

**Goal**: Click on face shows detailed info with landmarks, filename, alignment data.

**Files**: `app/face_clustering_debug/components/face_detail.py`

**Changes**:
1. Add filename display prominently
2. Draw 5-point landmarks with labels (LE, RE, N, LM, RM)
3. Show roll angle, frontal score
4. Show bbox coordinates
5. Add "Copy path" button for debugging

**Layout**:
```
┌─────────────────────────────────────┐
│  [Face Image with Landmarks]        │
│  LE=red, RE=red, N=green, LM/RM=blue│
├─────────────────────────────────────┤
│  File: IMG_1234.heic                │
│  Path: D:\Photos\IMG_1234.heic      │
│  Face #42 in image                  │
├─────────────────────────────────────┤
│  Roll: 12.3°  Frontal: 0.85         │
│  Bbox: (100, 200, 150, 150)         │
│  Confidence: 0.92                   │
└─────────────────────────────────────┘
```

---

## Sprint 9: 5-Point Face Alignment [DONE]

**Goal**: Replace 2-point (eye-only) rotation with proper 5-point affine alignment.

**Files**:
- `sim_bench/pipeline/utils/face_alignment.py`
- `sim_bench/pipeline/steps/score_face_frontal.py` (if needed)

**Current Problem**:
- `compute_roll_angle()` only uses 2 points (left_eye, right_eye)
- `align_and_crop_face()` just rotates by this angle
- Does not normalize for face position/scale

**Solution - Standard 5-point alignment**:
```python
# Reference template (normalized coordinates for 112x112 or 256x256)
ARCFACE_REF_POINTS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041],   # right mouth
], dtype=np.float32)

def align_face_5point(image, landmarks, target_size=256):
    """Compute affine transform from 5 landmarks to reference template."""
    src_pts = np.array(landmarks, dtype=np.float32)

    # Scale reference to target size
    scale = target_size / 112.0
    dst_pts = ARCFACE_REF_POINTS * scale

    # Estimate affine transform (similarity: rotation + scale + translation)
    tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

    # Apply transform
    aligned = cv2.warpAffine(image, tform, (target_size, target_size))
    return aligned
```

**Acceptance Criteria**:
- All 5 landmarks used for alignment
- Faces appear upright regardless of head tilt
- Existing embeddings cache may need clearing (alignment changed)

---

## Sprint 10: Debug UI - Show Decision Values [DONE]

**Goal**: Update debug UI to show actual threshold values and decision outcomes.

**Files**:
- `app/face_clustering_debug/components/algorithm_explanation.py`
- `app/face_clustering_debug/components/decision_card.py` (if exists)
- `app/face_clustering_debug/pages/merge_decisions.py`
- `app/face_clustering_debug/pages/attach_decisions.py`

**Changes**:
1. For each clustering method, display:
   - `doc_explanation` (from Sprint 2-6)
   - Current parameter values
   - Per-decision: threshold used, actual values, outcome

2. Merge decision display:
   ```
   Cluster 0 ↔ Cluster 1
   ─────────────────────
   T_A = 0.15  T_B = 0.20
   Cross-exemplar pairs within T: 4 (need ≥3) ✓
   Distinct exemplars: A=3, B=2 (need ≥2) ✓
   → MERGED (reason: b_fits_a)
   ```

3. Attach decision display:
   ```
   Face #42 (noise)
   ─────────────────
   Cluster 0: T=0.15, matches=3 (need ≥2) ✓
   Cluster 1: T=0.20, matches=1 (need ≥2) ✗
   → ATTACHED to Cluster 0
   ```

---

## Execution Order

1. **Sprint 1** (base class) - foundation for all others
2. **Sprints 2-6** (algorithm docs) - can be parallelized
3. **Sprint 7** (filename in grid) - quick win
4. **Sprint 8** (face detail) - enhances debugging
5. **Sprint 9** (5-point alignment) - fixes core issue
6. **Sprint 10** (decision UI) - ties everything together

---

## Dependencies

- Sprint 10 depends on Sprints 1-6 (needs `doc_explanation` and `decision_parameters`)
- Sprint 9 is independent but affects cached embeddings
- Sprints 7-8 are independent UI improvements

