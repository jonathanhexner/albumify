# Proposal: Face Alignment Pipeline Refactor

**Status**: Pending Approval
**Author**: Claude (for Senior SW Engineer review)
**Date**: 2026-02-19
**Related Sighting**: SIGHTING-001

---

## Problem Statement

Face alignment is failing for rotated faces (especially upside-down). The root cause is architectural:

1. **`compute_roll_angle()` doesn't detect face orientation** - only measures eye-line tilt
2. **`extract_face_embeddings.py` has mixed responsibilities** - detection conversion, cropping, alignment, embedding extraction
3. **No validation** that alignment actually worked
4. **No unit tests** for alignment correctness

## Objective

Refactor face processing into single-responsibility steps that can be tested independently.

## Constraints

- Must not break existing pipeline functionality
- Must be backwards compatible with cached data (or invalidate cache cleanly)
- Must work with InsightFace landmark format (5-point)
- Must handle 0°, 90°, 180°, 270° rotations

## Proposed Architecture

### Current Flow (Broken)
```
insightface_detect_faces
    → filter_faces
    → score_face_frontal (computes roll_angle - WRONG)
    → extract_face_embeddings (alignment + crop + embed - TOO MUCH)
```

### Proposed Flow (Clean)
```
insightface_detect_faces
    → filter_faces
    → detect_face_orientation   [NEW]
    → align_faces               [NEW]
    → validate_alignment        [NEW]
    → crop_faces                [NEW]
    → extract_face_embeddings   [SIMPLIFIED]
```

## New Pipeline Steps

### 1. `detect_face_orientation` (NEW)

**Purpose**: Determine if face needs 0°/90°/180°/270° pre-rotation.

**Logic**:
```python
def detect_face_orientation(landmarks):
    """
    Determine face orientation from landmark positions.

    Normal face: eyes at top, nose below eyes, mouth below nose
    Upside-down: eyes at bottom, nose above eyes, mouth above nose
    Rotated 90° CW: eyes on right, mouth on left
    Rotated 90° CCW: eyes on left, mouth on right

    Returns: 0, 90, 180, or 270 (degrees to rotate to upright)
    """
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks

    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
    nose_y = nose[1]

    # Check vertical relationships
    eyes_above_nose = eye_center_y < nose_y
    nose_above_mouth = nose_y < mouth_center_y

    if eyes_above_nose and nose_above_mouth:
        return 0  # Upright
    elif not eyes_above_nose and not nose_above_mouth:
        return 180  # Upside down
    else:
        # Check horizontal for 90° rotations
        # ... (similar logic for left/right relationships)
```

**Output**: Adds `orientation_angle` to each face in context.

### 2. `align_faces` (NEW)

**Purpose**: Apply rotation + 5-point affine alignment.

**Logic**:
```python
def align_face(image, landmarks, orientation_angle, target_size=256):
    """
    1. Pre-rotate image by orientation_angle to make face upright
    2. Transform landmarks to rotated coordinates
    3. Apply 5-point affine alignment to ArcFace template
    4. Return aligned face crop
    """
    if orientation_angle != 0:
        image, landmarks = rotate_image_and_landmarks(image, landmarks, orientation_angle)

    aligned = align_face_5point(image, landmarks, target_size)
    return aligned
```

**Output**: Stores aligned face crops in `context.aligned_faces`.

### 3. `validate_alignment` (NEW)

**Purpose**: Check that alignment worked correctly.

**Logic**:
```python
def validate_alignment(aligned_crop, expected_landmarks=ARCFACE_REF_POINTS):
    """
    Run face detection on aligned crop.
    Check that detected landmarks are close to expected positions.
    Flag faces with high alignment error.
    """
    detected = detect_landmarks(aligned_crop)
    error = compute_landmark_error(detected, expected_landmarks)

    return {
        "alignment_valid": error < THRESHOLD,
        "alignment_error": error,
        "detected_landmarks": detected
    }
```

**Output**: Marks faces with `alignment_valid=False` for review.

### 4. `crop_faces` (NEW)

**Purpose**: Just crop faces from images (no alignment).

**Logic**: Simple bbox cropping with margin. Useful for debug comparison.

### 5. `extract_face_embeddings` (SIMPLIFIED)

**Purpose**: Extract embeddings from pre-aligned face crops.

**Changes**:
- Remove all alignment logic
- Remove MediaPipe/InsightFace branching
- Just take aligned crops from context and extract embeddings

## Unit Tests Required

### test_face_orientation_detection.py
```python
def test_upright_face():
    landmarks = [[100, 50], [200, 50], [150, 100], [110, 150], [190, 150]]
    assert detect_face_orientation(landmarks) == 0

def test_upside_down_face():
    landmarks = [[100, 150], [200, 150], [150, 100], [110, 50], [190, 50]]
    assert detect_face_orientation(landmarks) == 180

def test_90_degree_cw_face():
    # Eyes on right, mouth on left
    ...

def test_90_degree_ccw_face():
    # Eyes on left, mouth on right
    ...
```

### test_face_alignment.py
```python
def test_alignment_produces_reference_landmarks():
    """Aligned face should have landmarks at ArcFace reference positions."""
    aligned = align_face(test_image, test_landmarks, orientation=0)
    detected = detect_landmarks(aligned)
    error = compute_error(detected, ARCFACE_REF_POINTS)
    assert error < 5.0  # pixels

def test_alignment_with_180_rotation():
    """Upside-down face should be correctly aligned."""
    aligned = align_face(upside_down_image, upside_down_landmarks, orientation=180)
    # Should produce upright face
    detected = detect_landmarks(aligned)
    assert is_upright(detected)
```

### test_alignment_validation.py
```python
def test_good_alignment_passes_validation():
    ...

def test_bad_alignment_fails_validation():
    ...
```

## Integration Points

- Context keys: `orientation_angles`, `aligned_faces`, `alignment_validations`
- Config: `configs/pipeline.yaml` - add new steps to pipeline
- Caching: New cache keys for aligned faces (invalidate old face_embedding cache)

## Risks and Trade-offs

1. **Performance**: Additional detection pass for validation (can be optional)
2. **Cache invalidation**: Old cached embeddings may be invalid (faces were misaligned)
3. **Complexity**: More pipeline steps (but each is simpler and testable)

## Migration Plan

1. Implement new steps alongside existing code
2. Add feature flag to enable new pipeline
3. Run benchmark comparison (old vs new)
4. Deprecate old code path
5. Remove old code

## Acceptance Criteria

- [ ] Face #118 (upside-down) is correctly aligned
- [ ] All unit tests pass
- [ ] Benchmark shows improvement or no regression
- [ ] Debug panel shows correct landmarks for all orientations
- [ ] `extract_face_embeddings.py` is under 100 lines
