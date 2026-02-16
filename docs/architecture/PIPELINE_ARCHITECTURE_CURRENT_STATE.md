# Pipeline Architecture - Current State

## TL;DR - What's Actually Being Used?

**Active (API Default)**:
- ✅ InsightFace pipeline steps (NEW)
- ✅ `select_best.py` with composite scoring (NEW)
- ✅ Hardcoded in `sim_bench/api/services/pipeline_service.py`

**Legacy (Not Used by API)**:
- ⚠️ `sim_bench/face_pipeline/` module (OLD MediaPipe-based)
- ⚠️ `FacePipelineService` (OLD all-in-one service)
- ⚠️ `configs/pipeline.yaml` default_pipeline (OLD MediaPipe steps)

---

## The Confusion

There are **TWO parallel face processing systems** in the codebase:

### System 1: Legacy `face_pipeline/` Module (OLD)

**Location**: `sim_bench/face_pipeline/`

**Components**:
```
face_pipeline/
├── pipeline.py          # FacePipelineService - all-in-one face processing
├── quality_scorer.py    # FaceQualityScorer - MediaPipe-based scoring
├── crop_service.py      # Face cropping
├── pose_estimator.py    # SixDRepNet pose estimation
└── types.py             # Data types
```

**What it does**:
- Complete face processing pipeline in a single service
- Uses MediaPipe for landmark detection
- Scores faces using: pose, eyes, smile, sharpness
- Returns `AlbumFaceResult` with faces, clusters, stats

**Used by** (OLD pipeline steps):
- `detect_faces` (step)
- `score_face_quality` (step)
- `score_face_pose` (step)

**Status**: ⚠️ **NOT used by the API anymore**

---

### System 2: InsightFace Pipeline Steps (NEW)

**Location**: `sim_bench/pipeline/steps/`

**Components**:
```
pipeline/steps/
├── detect_persons.py                # NEW: YOLOv8-Pose person detection
├── insightface_detect_faces.py      # NEW: InsightFace SCRFD face detection
├── insightface_score_expression.py  # NEW: Expression scoring
├── insightface_score_eyes.py        # NEW: Eye state scoring
├── insightface_score_pose.py        # NEW: Face pose scoring
└── select_best.py                   # NEW: Composite scoring (redesigned)
```

**What it does**:
- Modular pipeline steps (each does one thing)
- Uses InsightFace for face detection
- Uses YOLOv8 for person detection
- Writes to common context attributes
- `select_best.py` reads from context and computes composite scores

**Defined in** (API service):
```python
# sim_bench/api/services/pipeline_service.py
DEFAULT_PIPELINE = [
    "discover_images",
    "score_iqa",
    "score_ava",
    "detect_persons",              # YOLOv8-Pose
    "insightface_detect_faces",    # InsightFace SCRFD
    "insightface_score_expression",
    "insightface_score_eyes",
    "insightface_score_pose",
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_people",
    "cluster_by_identity",
    "select_best"                  # Composite scoring
]
```

**Status**: ✅ **ACTIVE - Used by the API**

---

## How They Relate to `select_best.py`

### OLD System → select_best

```
┌─────────────────────────────────────┐
│   detect_faces (MediaPipe)          │
│   → writes to context.faces         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   score_face_pose (SixDRepNet)      │
│   → writes to context.face_pose_*   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   score_face_eyes (MediaPipe)       │
│   → writes to context.face_eyes_*   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   score_face_smile (MediaPipe)      │
│   → writes to context.face_smile_*  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   select_best (reads from context)  │
│   OLD: Branching face vs non-face   │
└─────────────────────────────────────┘
```

### NEW System → select_best

```
┌─────────────────────────────────────┐
│   detect_persons (YOLOv8)           │
│   → writes to context.persons       │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   insightface_detect_faces          │
│   → writes to context.faces         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   insightface_score_expression      │
│   → writes to context.face_smile_*  │ ← Common interface!
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   insightface_score_eyes            │
│   → writes to context.face_eyes_*   │ ← Common interface!
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   insightface_score_pose            │
│   → writes to context.face_pose_*   │ ← Common interface!
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│   select_best (reads from context)  │
│   NEW: Composite scoring             │
│   quality + penalty                  │
└─────────────────────────────────────┘
```

**Key Insight**: Both systems write to the SAME context attributes, so `select_best.py` doesn't need to know which system provided the data!

---

## The Role of `quality_scorer.py`

### `face_pipeline/quality_scorer.py` (OLD)

**What it does**:
- Takes a `CroppedFace` object
- Computes: pose_score, eyes_open_score, smile_score, sharpness_score
- Returns `FaceQualityScore` object
- Attaches scores directly to face object

**Used by**:
- `face_pipeline/pipeline.py` → `FacePipelineService`
- OLD pipeline step: `score_face_quality.py`

**Status**: ⚠️ **Not used by active pipeline**

---

### `pipeline/scoring/quality_strategy.py` (NEW)

**What it does**:
- Computes IMAGE quality (not face quality)
- Strategies: WeightedAverage, SiameseRefinement, SiameseTournament
- Returns quality score in [0, 1]
- Used by `select_best.py` for composite scoring

**Used by**:
- `select_best.py` → Computes `image_quality_score`

**Status**: ✅ **Active, used by select_best**

---

### `pipeline/scoring/person_penalty.py` (NEW)

**What it does**:
- Computes person/portrait penalties
- Reads face scores from context (eyes, smile, pose)
- Returns penalty in [-0.7, 0]
- Used by `select_best.py` for composite scoring

**Used by**:
- `select_best.py` → Computes `person_penalty`

**Status**: ✅ **Active, used by select_best**

---

## Configuration Files

### `configs/pipeline.yaml`

```yaml
# OLD: MediaPipe pipeline (NOT used by API)
default_pipeline:
  - discover_images
  - detect_faces           # ← OLD MediaPipe
  - score_face_pose        # ← OLD
  - score_face_eyes        # ← OLD
  - score_face_smile       # ← OLD
  - filter_quality
  - extract_scene_embedding
  - cluster_scenes
  - extract_face_embeddings
  - cluster_by_identity
  - select_best

# NEW: InsightFace pipeline
insightface_pipeline:
  - discover_images
  - detect_persons         # ← NEW YOLOv8
  - insightface_detect_faces # ← NEW InsightFace
  - insightface_score_expression
  - insightface_score_eyes
  - insightface_score_pose
  - filter_quality
  - extract_scene_embedding
  - cluster_scenes
  - extract_face_embeddings
  - cluster_by_identity
  - select_best            # ← NEW composite scoring
```

### `sim_bench/api/services/pipeline_service.py`

```python
# ACTIVE: Hardcoded default used by API
DEFAULT_PIPELINE = [
    "discover_images",
    "score_iqa",
    "score_ava",
    "detect_persons",              # NEW
    "insightface_detect_faces",    # NEW
    "insightface_score_expression",# NEW
    "insightface_score_eyes",      # NEW
    "insightface_score_pose",      # NEW
    "filter_quality",
    "extract_scene_embedding",
    "cluster_scenes",
    "extract_face_embeddings",
    "cluster_people",
    "cluster_by_identity",
    "select_best"
]
```

**Discrepancy**: The API uses the NEW pipeline by default, but `configs/pipeline.yaml` still shows the OLD pipeline as "default_pipeline".

---

## What's Actually Running?

When you run the API (`python -m uvicorn sim_bench.api.main:app --reload`):

1. **API receives request** → `POST /api/v1/pipeline/run`
2. **PipelineService.start_pipeline()** → Uses `DEFAULT_PIPELINE` (NEW InsightFace steps)
3. **Pipeline executes**:
   - YOLOv8 detects persons
   - InsightFace detects faces
   - InsightFace scores expression, eyes, pose
   - Writes to common context attributes
4. **select_best.py runs**:
   - Reads scores from context (doesn't care about source)
   - Computes `image_quality_score` using quality strategy
   - Computes `person_penalty` using penalty computer
   - Returns `composite_score = quality + penalty`
5. **Results saved** to database

**None of the `face_pipeline/` module is involved!**

---

## Summary Table

| Component | Type | Status | Used By |
|-----------|------|--------|---------|
| `face_pipeline/pipeline.py` | Service | Legacy | Not used |
| `face_pipeline/quality_scorer.py` | Scorer | Legacy | Not used |
| `face_pipeline/crop_service.py` | Service | Legacy | Some steps use it |
| `face_pipeline/pose_estimator.py` | Model | Active | InsightFace steps use it |
| `pipeline/steps/detect_persons.py` | Step | Active | API pipeline |
| `pipeline/steps/insightface_*.py` | Steps | Active | API pipeline |
| `pipeline/steps/select_best.py` | Step | Active | API pipeline |
| `pipeline/scoring/quality_strategy.py` | Strategy | Active | select_best |
| `pipeline/scoring/person_penalty.py` | Computer | Active | select_best |

---

## Why Is This Confusing?

1. **Two systems with similar names**:
   - `face_pipeline/quality_scorer.py` (OLD)
   - `pipeline/scoring/quality_strategy.py` (NEW)

2. **Config file mismatch**:
   - `configs/pipeline.yaml` → Shows OLD pipeline as default
   - `pipeline_service.py` → Uses NEW pipeline as default

3. **Legacy code still present**:
   - `face_pipeline/` module still exists
   - OLD pipeline steps still registered
   - Could still be used if someone explicitly requests them

---

## Recommendations

### 1. Update Config File ✅

Update `configs/pipeline.yaml` to reflect reality:

```yaml
# Active pipeline (used by API)
default_pipeline:
  - discover_images
  - score_iqa
  - score_ava
  - detect_persons
  - insightface_detect_faces
  - insightface_score_expression
  - insightface_score_eyes
  - insightface_score_pose
  - filter_quality
  - extract_scene_embedding
  - cluster_scenes
  - extract_face_embeddings
  - cluster_people
  - cluster_by_identity
  - select_best

# Legacy MediaPipe pipeline (kept for compatibility)
mediapipe_pipeline:
  - discover_images
  - detect_faces
  - score_face_pose
  - score_face_eyes
  - score_face_smile
  # ... etc
```

### 2. Mark Legacy Code 📝

Add deprecation warnings to legacy components:

```python
# face_pipeline/pipeline.py
class FacePipelineService:
    """
    DEPRECATED: Legacy all-in-one face processing service.
    
    Use the modular InsightFace pipeline steps instead:
    - detect_persons
    - insightface_detect_faces
    - insightface_score_expression
    - insightface_score_eyes
    - insightface_score_pose
    """
```

### 3. Document Active Architecture 📚

Create clear docs showing:
- What's active vs legacy
- How components relate
- Migration path for old code

---

## Quick Answer to Your Question

**Q: What is the relation between `face_pipeline/pipeline.py`, `select_best.py`, and `quality_scorer.py`?**

**A:**
- `face_pipeline/pipeline.py` + `face_pipeline/quality_scorer.py` = **OLD system (legacy, not used by API)**
- `pipeline/steps/select_best.py` + `pipeline/scoring/quality_strategy.py` = **NEW system (active, used by API)**
- They are **two different implementations** of face processing
- The NEW system replaced the OLD system
- Both write to common context attributes, so they're theoretically interchangeable

**Q: Which is actually being used?**

**A:**
- **API uses**: NEW InsightFace steps + NEW select_best with composite scoring
- **Legacy**: OLD face_pipeline module exists but isn't used by default
- **Your recent work**: Updated `select_best.py` (the NEW one), which is the active component

**Bottom line**: You've been working on the RIGHT code! The `face_pipeline/` stuff is legacy and not relevant to the active system. 🎯
