# Pipeline Design

## Executive Summary

**Goal:** Select best 1-2 photos per "moment" while handling face and non-face images differently.

**Key Principles:**
1. Detect faces **early** (right after discovery)
2. Cluster by **scene** first, then sub-cluster by **face identity** and **face count**
3. Different selection logic for **face clusters** (composite score) vs **non-face clusters** (AVA + Siamese)
4. **Smart selection** - not just top N, but intelligent decisions about keeping 1 vs 2 images

**Config:** `configs/pipeline.yaml` - all parameters are configurable

---

## Pipeline Steps

```
1. discover_images        Find all images in folder
2. detect_faces           Detect faces + pose + significance (EARLY!)
3. score_iqa              Technical quality
4. score_ava              Aesthetic quality
5. score_face_pose        Face frontal-ness (face images only)
6. score_face_eyes        Eyes open score (face images only)
7. score_face_smile       Smile score (face images only)
8. filter_quality         Remove technically bad images
9. extract_scene_embedding   Scene features (DINOv2)
10. cluster_scenes           Group similar scenes
11. extract_face_embedding   Face identity features (ArcFace)
12. cluster_by_identity      Sub-cluster by person + face count
13. select_best              Smart selection with branching logic
```

---

## Clustering Hierarchy

```
All Images
    │
    ├── Scene Cluster: Beach
    │       ├── Mom alone (1 face)           → select best 1-2
    │       ├── Dad alone (1 face)           → select best 1-2
    │       ├── Mom+Dad (2 faces)            → select best 1-2  ← SEPARATE!
    │       ├── Family group (3+ faces)      → select best 1-2
    │       └── No faces (landscape)         → select best 1-2
    │
    ├── Scene Cluster: Museum
    │       ├── Child alone                  → select best 1-2
    │       └── Artwork shots (no faces)     → select best 1-2
    │
    └── ...
```

**Key:** Multi-person photos are separate clusters (Mom+Dad ≠ Mom alone)

---

## Selection Logic

### Face Clusters

```python
composite_score = (
    w_eyes * eyes_open_score +
    w_pose * pose_score +         # Back of head = 0
    w_smile * smile_score +
    w_ava * ava_score
)

# Smart selection (not just top N):
# - Always take #1
# - Take #2 only if:
#   - Score above threshold
#   - Not near-duplicate of #1 (Siamese check)
#   - Score gap not too large
```

### Non-Face Clusters

```python
# 1. Rank by AVA score
# 2. If top 3 scores within 5%: use Siamese CNN as tiebreaker
# 3. Apply same smart selection rules
```

---

## Face Significance

A face is "significant" if:

| Check | Threshold | Configurable |
|-------|-----------|--------------|
| Area ratio | ≥ 3% of image | `face_detection.min_area_ratio` |
| Confidence | ≥ 0.7 | `face_detection.min_confidence` |
| Yaw angle | ≤ 75° (not back of head) | `face_detection.max_yaw` |

**Pose scoring:**
- Frontal (0-15°) → 1.0
- 3/4 view (45°) → 0.7
- Profile (90°) → 0.3
- Back of head (>100°) → 0.0 or reject

---

## Configurable Knobs

All in `configs/pipeline.yaml`:

| Section | Key Parameters |
|---------|----------------|
| `face_detection` | min_area_ratio, min_confidence, max_yaw |
| `face_scoring.weights` | eyes_open, pose, smile, ava (must sum to 1.0) |
| `quality_filter` | min_iqa_score, min_sharpness |
| `clustering.scene` | method, feature_method, min_cluster_size |
| `clustering.face_identity` | distance_threshold |
| `selection` | max_images_per_cluster, min_score_threshold, max_score_gap |
| `siamese` | enabled, duplicate_similarity, tiebreaker_score_range |

---

## Architecture Support

### What Exists ✅

| Component | Location | Status |
|-----------|----------|--------|
| Face detection step | `pipeline/steps/detect_faces.py` | Exists |
| Face pose scoring | `pipeline/steps/score_face_pose.py` | Exists |
| Face eyes scoring | `pipeline/steps/score_face_eyes.py` | Exists |
| Face smile scoring | `pipeline/steps/score_face_smile.py` | Exists |
| Scene clustering | `pipeline/steps/cluster_scenes.py` | Exists |
| ArcFace embeddings | `pipeline/steps/extract_face_embeddings.py` | Exists |
| Siamese model | `models/album_app/siamese_comparison_model.pt` | Exists |
| Pipeline context | `pipeline/context.py` | Exists |
| Caching system | `pipeline/cache_handler.py` | Exists |

### What Needs Work 🔧

| Component | Current State | Needed |
|-----------|---------------|--------|
| `detect_faces` | ✅ Now step 2 in pipeline | Done |
| `cluster_by_identity` | ✅ Created | Done |
| `select_best` | ✅ Rewritten with branching logic | Done |
| Pipeline config | Uses `global_config.yaml` | Use new `configs/pipeline.yaml` |
| DEFAULT_PIPELINE | ✅ Updated to 13 steps | Done |
| Siamese integration | ✅ Fully integrated | Done - tiebreaker + duplicate detection |

---

## Implementation Plan

### Phase 1: Config & Pipeline Order ✅
1. [x] Create `configs/pipeline.yaml` ✅ Done
2. [x] Update `DEFAULT_PIPELINE` in `pipeline_service.py` ✅ Done
3. [x] Move `detect_faces` to run after `discover_images` ✅ Done

### Phase 2: Face Sub-Clustering ✅
4. [x] Create `cluster_by_identity` step ✅ Done
   - Group by face count (0, 1, 2, 3+)
   - Within same count, cluster by ArcFace identity
   - Store in `context.face_clusters`

### Phase 3: Smart Selection ✅
5. [x] Rewrite `select_best` step ✅ Done
   - Branch on face vs non-face cluster
   - Composite scoring for face clusters
   - AVA + IQA for non-face clusters
   - Smart 1-2 rules (not just top N)
   - Near-duplicate detection using scene embeddings

### Phase 4: Integration ✅
6. [x] Wire up Siamese model in selection ✅ Done
   - Tiebreaker when top scores within threshold
   - Near-duplicate detection (low confidence = duplicates)
   - Falls back to embedding similarity if Siamese unavailable
7. [ ] Load config from `pipeline.yaml` (using step configs for now)
8. [x] Test end-to-end ✅ Done

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| Mom+Dad = separate cluster | Want best of each person AND best couple shot |
| Pose check in significance | Back-of-head is useless, filter early |
| Siamese for near-duplicates | Burst photos should keep only best |
| Rules first, NN later | Start simple, add learned decisions when we have data |
