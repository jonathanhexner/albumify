# Face Filtering and Frontal Scoring Plan

## Problem Statement

Face recognition and clustering has issues with:
1. Small/low-confidence faces causing noise
2. Rotated/occluded faces producing unreliable embeddings
3. Non-frontal faces degrading clustering quality

## Solution Overview

Two-stage filtering approach:

**Stage 1 - Filter Small/Unreliable Faces**
- Discard faces that are too small or low confidence
- Image treated as "no people" if all faces filtered
- No face penalty applied in this case

**Stage 2 - Score Frontal/Symmetry**
- Compute frontal score for remaining faces
- Non-frontal faces: penalized in selection, excluded from clustering
- Frontal faces: normal processing, included in clustering

## Implementation Plan

### Task 1: Create `filter_faces` Step

**File**: `sim_bench/pipeline/steps/filter_faces.py`

**Purpose**: Remove faces that are too small or unreliable

**Config Schema**:
```yaml
filter_faces:
  min_confidence: 0.5           # detection confidence threshold
  min_bbox_ratio: 0.02          # min bbox_width / image_width
  min_relative_size: 0.3        # min bbox_width / max_bbox_width in image
  min_eye_ratio: 0.01           # min inter_eye_distance / image_width
```

**Logic**:
```python
def process(self, context, config):
    for image_path, face_data in context.insightface_faces.items():
        image_width = get_image_width(image_path)
        faces = face_data.get('faces', [])

        # Find max bbox width in this image
        max_bbox_width = max(f['bbox']['w_px'] for f in faces) if faces else 0

        filtered_faces = []
        for face in faces:
            bbox_width = face['bbox']['w_px']
            confidence = face['confidence']
            inter_eye = compute_inter_eye_distance(face['landmarks'])

            # Compute filter scores
            filter_scores = {
                'confidence': confidence,
                'bbox_ratio': bbox_width / image_width,
                'relative_size': bbox_width / max_bbox_width if max_bbox_width > 0 else 0,
                'eye_ratio': inter_eye / image_width,
            }

            # Check thresholds
            passes = (
                filter_scores['confidence'] >= config['min_confidence'] and
                filter_scores['bbox_ratio'] >= config['min_bbox_ratio'] and
                filter_scores['relative_size'] >= config['min_relative_size'] and
                filter_scores['eye_ratio'] >= config['min_eye_ratio']
            )

            if passes:
                face['filter_scores'] = filter_scores
                face['filter_passed'] = True
                filtered_faces.append(face)
            else:
                # Store why it was filtered (for debugging)
                face['filter_scores'] = filter_scores
                face['filter_passed'] = False

        face_data['faces'] = filtered_faces
        face_data['filtered_count'] = len(faces) - len(filtered_faces)
```

**Metadata**:
```python
StepMetadata(
    name="filter_faces",
    display_name="Filter Small Faces",
    description="Remove faces that are too small or low confidence",
    category="filtering",
    requires={"insightface_faces"},
    produces={"insightface_faces"},  # modified in place
    depends_on=["insightface_detect_faces"],
)
```

---

### Task 2: Create `score_face_frontal` Step

**File**: `sim_bench/pipeline/steps/score_face_frontal.py`

**Purpose**: Compute frontal score, mark faces as clusterable

**Config Schema**:
```yaml
score_face_frontal:
  min_eye_bbox_ratio: 0.20      # inter_eye / bbox_width - below = profile
  max_asymmetry: 1.8            # nose-eye asymmetry - above = profile
  min_frontal_score: 0.4        # below = not clusterable
```

**Logic**:
```python
def compute_frontal_score(face, config):
    landmarks = face['landmarks']  # 5-point: [left_eye, right_eye, nose, left_mouth, right_mouth]
    bbox = face['bbox']

    # 1. Inter-eye ratio (detects yaw)
    left_eye, right_eye = landmarks[0], landmarks[1]
    inter_eye = distance(left_eye, right_eye)
    bbox_width = bbox['w_px']
    eye_bbox_ratio = inter_eye / bbox_width

    # Normalize: 0.25+ is ideal frontal, 0.15 is profile
    eye_score = clip((eye_bbox_ratio - 0.15) / (0.25 - 0.15), 0, 1)

    # 2. Asymmetry ratio (detects yaw from different angle)
    nose = landmarks[2]
    dl = distance(nose, left_eye)
    dr = distance(nose, right_eye)
    asymmetry = max(dl, dr) / (min(dl, dr) + 1e-6)

    # Normalize: 1.0 is ideal, 2.0+ is profile
    asym_score = clip(1.0 - (asymmetry - 1.0) / 1.0, 0, 1)

    # Combine scores (average)
    frontal_score = (eye_score + asym_score) / 2

    return {
        'frontal_score': frontal_score,
        'eye_bbox_ratio': eye_bbox_ratio,
        'eye_score': eye_score,
        'asymmetry': asymmetry,
        'asym_score': asym_score,
    }

def process(self, context, config):
    min_frontal = config.get('min_frontal_score', 0.4)

    for image_path, face_data in context.insightface_faces.items():
        for face in face_data.get('faces', []):
            scores = compute_frontal_score(face, config)
            face['frontal_scores'] = scores
            face['frontal_score'] = scores['frontal_score']
            face['is_clusterable'] = scores['frontal_score'] >= min_frontal
```

**Metadata**:
```python
StepMetadata(
    name="score_face_frontal",
    display_name="Score Face Frontal",
    description="Compute frontal score and mark faces as clusterable",
    category="analysis",
    requires={"insightface_faces"},
    produces={"insightface_faces"},  # adds frontal_score, is_clusterable
    depends_on=["filter_faces"],
)
```

---

### Task 3: Add Roll Alignment to `extract_face_embeddings`

**File**: `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Purpose**: Rotate face crops to align eyes horizontally before embedding extraction (improves clustering for tilted faces)

**Logic**:
```python
def compute_roll_angle(landmarks):
    """Compute roll angle from eye landmarks."""
    left_eye, right_eye = landmarks[0], landmarks[1]
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    roll_angle = math.atan2(dy, dx) * 180 / math.pi  # degrees
    return roll_angle

def align_face_crop(face_crop, roll_angle):
    """Rotate face crop to make eyes horizontal."""
    if abs(roll_angle) < 5:  # Skip small rotations
        return face_crop

    h, w = face_crop.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
    aligned = cv2.warpAffine(face_crop, rotation_matrix, (w, h))
    return aligned
```

**Integration**: Apply `align_face_crop()` after cropping, before passing to embedding extractor.

**Store**: Add `roll_angle` to face data for debugging/display.

---

### Task 4: Modify `extract_face_embeddings` to Skip Non-Clusterable

**File**: `sim_bench/pipeline/steps/extract_face_embeddings.py`

**Changes**:
1. Only extract embeddings for faces where `is_clusterable == True`
2. Skip faces that don't have this flag (backwards compatibility)

**Modified `_get_all_faces()` method**:
```python
def _get_all_faces(self, context, config):
    # ... existing code ...

    for face_info in face_data.get('faces', []):
        # Skip non-clusterable faces
        if not face_info.get('is_clusterable', True):  # default True for backwards compat
            skipped_non_frontal += 1
            continue

        # ... rest of existing code ...

    logger.info(f"Skipped {skipped_non_frontal} non-frontal faces")
```

---

### Task 4: Modify Penalty Calculation

**File**: `sim_bench/pipeline/scoring/person_penalty.py` (or relevant scoring file)

**Changes**:
1. Add `frontal_penalty_weight` config
2. Use best face's `frontal_score` for penalty
3. No penalty if no faces passed filtering

**Config**:
```yaml
select_best:
  penalties:
    frontal_penalty_weight: 0.3  # weight for non-frontal penalty
```

**Logic**:
```python
def compute_face_penalty(self, image_path, context, config):
    face_data = context.insightface_faces.get(image_path, {})
    faces = face_data.get('faces', [])

    # No faces = no penalty
    if not faces:
        return 0.0

    # Find best frontal score among all faces
    best_frontal_score = max(f.get('frontal_score', 1.0) for f in faces)

    # Penalty: higher when frontal_score is low
    frontal_penalty_weight = config.get('frontal_penalty_weight', 0.3)
    penalty = (1.0 - best_frontal_score) * frontal_penalty_weight

    return penalty
```

---

### Task 5: Update Pipeline Config

**File**: `configs/pipeline.yaml`

Add new steps to pipeline:
```yaml
steps:
  # ... existing steps ...

  filter_faces:
    min_confidence: 0.5
    min_bbox_ratio: 0.02
    min_relative_size: 0.3
    min_eye_ratio: 0.01

  score_face_frontal:
    min_eye_bbox_ratio: 0.20
    max_asymmetry: 1.8
    min_frontal_score: 0.4

pipelines:
  default_pipeline:
    - discover_images
    - detect_persons
    - insightface_detect_faces
    - filter_faces              # NEW - Stage 1: remove small/low-conf
    - score_face_frontal        # NEW - Stage 2: compute frontal score
    - extract_face_embeddings   # MODIFIED - skip non-clusterable, align roll
    - cluster_people
    - select_best               # MODIFIED - frontal penalty
    # ... rest of steps ...
```

---

### Task 6: Register New Steps

**File**: `sim_bench/pipeline/steps/all_steps.py`

Add imports:
```python
from sim_bench.pipeline.steps.filter_faces import FilterFacesStep
from sim_bench.pipeline.steps.score_face_frontal import ScoreFaceFrontalStep
```

---

### Task 7: Add UI Controls (Optional)

**File**: `app/streamlit/components/pipeline_runner.py`

Add sliders for:
- Min confidence
- Min face size ratio
- Min frontal score

---

## TODO List

- [ ] Task 1: Create `filter_faces` step
- [ ] Task 2: Create `score_face_frontal` step
- [ ] Task 3: Add roll alignment to `extract_face_embeddings`
- [ ] Task 4: Modify `extract_face_embeddings` to skip non-clusterable
- [ ] Task 5: Add frontal penalty to scoring
- [ ] Task 6: Update `pipeline.yaml` config
- [ ] Task 7: Register new steps in `all_steps.py`
- [ ] Task 8: Update UI to display all new scores
- [ ] Task 9: Delete database to clear stale cache

### Task 8: Update UI to Display All New Scores

**Files**:
- `sim_bench/api/services/pipeline_service.py` - Add new scores to `image_metrics`
- `app/streamlit/components/metrics.py` - Display in metrics table

**New columns in metrics table**:
| Column | Source |
|--------|--------|
| Confidence | `face['confidence']` |
| BBox Ratio | `face['filter_scores']['bbox_ratio']` |
| Relative Size | `face['filter_scores']['relative_size']` |
| Eye Ratio | `face['filter_scores']['eye_ratio']` |
| Frontal Score | `face['frontal_score']` |
| Eye/BBox Ratio | `face['frontal_scores']['eye_bbox_ratio']` |
| Asymmetry | `face['frontal_scores']['asymmetry']` |
| Roll Angle | `face['roll_angle']` |
| Clusterable | `face['is_clusterable']` |

---

### Task 9: Delete Database

**Action**: Delete `~/.sim_bench/sim_bench.db` to clear stale cache

**Script**: `scripts/reset_database.py`
```python
import os
from pathlib import Path

db_path = Path.home() / ".sim_bench" / "sim_bench.db"
if db_path.exists():
    os.remove(db_path)
    print(f"Deleted {db_path}")
else:
    print(f"Database not found: {db_path}")
```

**Note**: User will need to re-run pipeline after deletion.

---

## Testing

1. Run on test data: `D:\sim-bench\test_data\budapest_2025`
2. Verify filtering removes small/low-confidence faces
3. Verify frontal score computed correctly
4. Verify non-frontal faces excluded from clustering
5. Verify penalty applied correctly in selection

## Success Criteria

- Small background faces no longer clustered
- Profile/rotated faces excluded from clustering
- Non-frontal faces get selection penalty
- Images with only small faces treated as "no people" (no penalty)
- Clustering quality improved
