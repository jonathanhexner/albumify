# Quality Gating Algorithm

**Purpose**: Filter low-quality faces before clustering to improve accuracy and reduce noise.

---

## Overview

Quality gating is a **pre-clustering filter** that separates detected faces into two groups:
- **Core set**: High-quality faces suitable for clustering
- **Holdout set**: Low-quality faces that can be attached later (optional)

By clustering only high-quality faces, we avoid contaminating clusters with poor detections, blurry faces, or extreme poses.

---

## Algorithm

### Input
- List of detected faces with:
  - Bounding box (x, y, w, h)
  - Landmarks (5-point or more)
  - Aligned face crop (112×112 RGB)
  - Optional: Pose angles (yaw, pitch, roll)

### Output
- **core_indices**: Indices of high-quality faces → use for clustering
- **holdout_indices**: Indices of low-quality faces → exclude from clustering

### Process

```python
def select_core_set(faces: List[FaceRecord], config: PipelineConfig) -> Tuple[List[int], List[int]]:
    """
    Apply quality filters to select core faces for clustering.

    Filtering criteria (applied in order):
    1. Compute blur scores (Laplacian variance)
    2. Optionally compute pose scores (requires SixDRepNet)
    3. Select top K faces per image by area
    4. Apply quality thresholds:
       - abs(yaw) <= yaw_max
       - abs(pitch) <= pitch_max
       - abs(roll) <= roll_max
       - blur_score >= blur_min
       - area >= min_face_area (optional)

    Returns:
        (core_indices, holdout_indices)
    """
```

**Step 1: Compute Blur Scores**
```python
# Variance of Laplacian on grayscale crop
gray = cv2.cvtColor(face.aligned_face, cv2.COLOR_RGB2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
blur_score = laplacian.var()

# Typical values:
# - Sharp face: 500-2000
# - Moderate blur: 100-500
# - Severe blur: < 100
```

**Step 2: Optionally Compute Pose** (requires SixDRepNet)
```python
# Estimate head pose from aligned crop
pose_estimator = PoseEstimator(device='cpu')
yaw, pitch, roll = pose_estimator.estimate_pose(face.aligned_face)

# Typical ranges:
# - Frontal face: yaw ≈ 0°, pitch ≈ 0°, roll ≈ 0°
# - Profile view: yaw ≈ ±90°
# - Looking up/down: pitch ≈ ±30°
```

**Step 3: Select Top K Per Image**
```python
# Group faces by image_id
image_groups = group_by(faces, key=lambda f: f.image_id)

# For each image, keep only top K faces by area
for image_id, face_list in image_groups.items():
    sorted_by_area = sorted(face_list, key=lambda f: f.area, reverse=True)
    candidates.extend(sorted_by_area[:config.max_faces_per_image_core])
```

**Step 4: Apply Thresholds**
```python
core_indices = []
holdout_indices = []

for i, face in enumerate(faces):
    if i not in candidates:
        holdout_indices.append(i)
        continue

    # Check all criteria
    passes = True

    if face.pose is not None:
        yaw, pitch, roll = face.pose
        if abs(yaw) > config.yaw_max:
            passes = False
        if abs(pitch) > config.pitch_max:
            passes = False
        if abs(roll) > config.roll_max:
            passes = False

    if face.blur_score < config.blur_min:
        passes = False

    if config.min_face_area and face.area < config.min_face_area:
        passes = False

    if passes:
        core_indices.append(i)
        face.is_core = True
    else:
        holdout_indices.append(i)
```

---

## Configuration Parameters

```yaml
filter_quality_gate:
  yaw_max: 45.0              # Max absolute yaw (degrees)
  pitch_max: 30.0            # Max absolute pitch (degrees)
  roll_max: 30.0             # Max absolute roll (degrees)
  blur_min: 100.0            # Min blur score (Laplacian variance)
  min_face_area: null        # Min face area in pixels (optional)
  max_faces_per_image_core: 10  # Max faces per image in core set
  use_pose_estimation: false # Use SixDRepNet for pose (slow, requires GPU)
```

### Tuning Guidelines

**For high-precision clustering** (fewer faces, higher quality):
- `yaw_max: 30.0` (frontal faces only)
- `blur_min: 200.0` (sharp faces only)
- `max_faces_per_image_core: 3` (top 3 faces per image)

**For high-recall clustering** (more faces, tolerate lower quality):
- `yaw_max: 60.0` (allow more profile views)
- `blur_min: 50.0` (tolerate some blur)
- `max_faces_per_image_core: 20` (keep more faces per image)

**For balanced clustering** (recommended):
- `yaw_max: 45.0`
- `blur_min: 100.0`
- `max_faces_per_image_core: 10`

---

## Why Each Filter Matters

### Pose Filtering (yaw, pitch, roll)
**Problem**: Extreme poses produce different embeddings
- Frontal vs profile views of same person → high distance
- Upside-down faces → incorrect landmarks → bad alignment

**Solution**: Filter faces with extreme poses
- Keeps only near-frontal faces (yaw < 45°)
- Allows slight head tilt (pitch < 30°, roll < 30°)

### Blur Filtering
**Problem**: Blurry faces produce noisy embeddings
- Camera shake, motion blur → unstable features
- Low-quality embeddings → false negatives (same person not matched)

**Solution**: Filter faces below blur threshold
- Laplacian variance < 100 indicates severe blur
- Removes ~20-30% of faces in typical albums

### Area Filtering
**Problem**: Tiny faces have insufficient detail
- Faces < 40×40 pixels → poor feature extraction
- Upscaling tiny faces → artifacts, noise

**Solution**: Optionally filter small faces
- Set `min_face_area: 1600` (40×40 pixels)
- Or rely on detection confidence instead

### Top-K Per Image
**Problem**: Group photos with many faces
- 20 faces in one image → most are background/distant
- Clustering 20 faces from same image → computational waste

**Solution**: Keep only top K faces by area
- Prioritizes foreground faces (larger area)
- Reduces redundancy in group photos

---

## Statistics & Validation

After quality gating, log these statistics:

```
Quality Gating Results:
  Total faces: 147
  Core faces: 98 (66.7%)
  Holdout faces: 49 (33.3%)

  Core breakdown:
    - High quality: 73 (49.7%)
    - Moderate quality: 25 (17.0%)

  Holdout reasons:
    - Extreme pose: 21 (14.3%)
    - Low blur: 18 (12.2%)
    - Small size: 7 (4.8%)
    - Not in top-K: 3 (2.0%)
```

**Validation**: Assert at least one core face
```python
assert len(core_indices) > 0, \
    "No faces passed quality gate - check thresholds (all faces filtered out)"
```

---

## Edge Cases

### No Faces Detected
```python
if len(faces) == 0:
    return [], []  # Empty core and holdout
```

### All Faces Filtered
```python
if len(core_indices) == 0:
    # Either:
    # 1. Raise error (recommended - indicates bad thresholds)
    raise ValueError("No faces passed quality gate")

    # 2. Fall back to best face (lenient)
    best_face = max(faces, key=lambda f: f.blur_score)
    return [faces.index(best_face)], list(range(len(faces) - 1))
```

### Single Face Per Image
```python
# If every image has exactly 1 face, top-K filter has no effect
# All faces are candidates, only quality thresholds apply
```

---

## Integration with Pipeline

```python
# Pipeline step: filter_quality_gate.py
class FilterQualityGateStep(BaseStep):
    def process(self, context: PipelineContext, config: dict):
        # Create FaceRecord objects from context
        face_records = self._create_face_records(context)

        # Create QualityGater
        gater = QualityGater(config, use_pose_estimation=config['use_pose_estimation'])

        # Compute blur scores
        face_records = gater.compute_blur_scores(face_records)

        # Optionally compute pose
        if config['use_pose_estimation']:
            face_records = gater.compute_pose_scores(face_records)

        # Apply quality gate
        core_indices, holdout_indices = gater.select_core_set(face_records)

        # Store results
        context.core_indices = core_indices
        context.holdout_indices = holdout_indices
        context.face_records = face_records
```

---

## Performance

**Typical timing** (CPU):
- Blur computation: 1-2 ms per face
- Pose estimation: 50-100 ms per face (if enabled)
- Quality filtering: < 1 ms per face

**For 100 faces**:
- Without pose: ~0.2 seconds
- With pose: ~5-10 seconds

**Recommendation**: Disable pose estimation unless needed (use blur + area filters instead)

---

## References

- **Blur metric**: Laplacian variance ([Pech-Pacheco et al., 2000](https://ieeexplore.ieee.org/document/877643))
- **Pose estimation**: SixDRepNet ([Hempel et al., 2022](https://arxiv.org/abs/2202.13370))
- **Face quality**: ISO/IEC 29794-5 standard

---

**See Also**:
- [Pipeline Overview](../pipeline/overview.md) - Where quality gating fits in pipeline
- [Configuration Reference](../pipeline/configuration.md) - All config parameters
- [Troubleshooting](../pipeline/troubleshooting.md) - Common quality gating issues
