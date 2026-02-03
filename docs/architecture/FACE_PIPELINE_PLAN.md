# Face Processing Pipeline Plan

## Overview

Build a comprehensive face processing pipeline for album organization that:
1. Detects and crops faces from images
2. Assesses face quality (pose, smile, eyes open)
3. Extracts face embeddings using trained ArcFace model
4. Clusters faces by identity
5. (Future) Enables "People" view like Google Photos

---

## Phase 1: Face Detection & Cropping

### Current State
- MediaPipe face detection already integrated in `portrait_analysis/analyzer.py`
- Returns relative bounding boxes (`x`, `y`, `w`, `h` in 0-1 range)
- Has `face_ratio` (face area / image area) to determine face dominance

### Required Work

#### 1.1 Face Dominance Threshold
**Question**: What `face_ratio` threshold should we use to consider a face "sufficiently dominant"?
- Current portrait threshold: `0.0005` (very low)
- Suggested for face processing: `0.02` (face is at least 2% of image area)
- Or use bounding box minimum size: `min(w, h) > 0.1` (face > 10% of image dimension)

#### 1.2 Face Cropping Service
Create `FaceCropService` that:
- Takes image path, returns list of cropped face images
- Adds padding around bounding box (e.g., 20% margin)
- Handles edge cases (face near image border)
- Returns both PIL Image and crop metadata (original bbox, padding used)

```python
@dataclass
class CroppedFace:
    image: Image.Image  # Cropped face image
    original_path: str
    bbox: dict  # Original bounding box (relative)
    bbox_pixels: dict  # Absolute pixel coordinates
    padding: float  # Padding used
    face_ratio: float
    detection_confidence: float
```

---

## Phase 2: Head Pose Estimation (SixDRepNet)

### What is SixDRepNet?
- 6DoF head pose estimation (yaw, pitch, roll)
- Pretrained model available on GitHub/HuggingFace
- Input: Cropped face image (typically 224x224)
- Output: Euler angles in degrees

### Required Work

#### 2.1 SixDRepNet Integration
Options:
1. **sixdrepnet PyPI package** - `pip install sixdrepnet`
2. **Direct model loading** from HuggingFace/weights

#### 2.2 Pose Quality Scoring
```python
@dataclass
class PoseScore:
    yaw: float    # Left-right rotation (degrees)
    pitch: float  # Up-down rotation (degrees)
    roll: float   # Tilt rotation (degrees)
    is_frontal: bool  # Within acceptable range
    penalty: float    # 0-1, higher = worse pose

# Thresholds (from your requirements):
YAW_THRESHOLD = 20.0   # |yaw| > 20° is penalized
PITCH_THRESHOLD = 15.0 # |pitch| > 15° is penalized
ROLL_THRESHOLD = 10.0  # |roll| > 10° is penalized
```

**Question**: How should we calculate the penalty?
- Option A: Binary (0 if all within threshold, 1 otherwise)
- Option B: Continuous penalty based on deviation:
  ```python
  penalty = max(0, |yaw|/20 - 1) + max(0, |pitch|/15 - 1) + max(0, |roll|/10 - 1)
  ```
- Option C: Soft threshold with gradual increase

---

## Phase 3: Face Quality Metrics (Using Existing MediaPipe)

### 3.1 Smile Score
Already implemented in `portrait_analysis/smile_detection.py`:
- `smile_score`: 0-1 normalized
- `is_smiling`: boolean

### 3.2 Eyes Open Score
Already implemented in `portrait_analysis/eye_state.py`:
- `left_ear`, `right_ear`: Eye Aspect Ratio
- `both_eyes_open`: boolean

### Required Work
- Ensure these work on cropped face images (currently analyze full image)
- May need to adjust for cropped face input

---

## Phase 4: Face Embeddings

### Current State
- `FaceEmbeddingService` created (from today's work)
- Uses trained ArcFace model
- Expects 112x112 input

### Required Work

#### 4.1 Resize Cropped Faces
- Cropped faces need to be resized to 112x112
- Use bilinear/bicubic interpolation
- Apply same normalization as training (mean=0.5, std=0.5)

#### 4.2 Batch Processing
- Process multiple faces efficiently
- Cache embeddings to avoid recomputation

---

## Phase 5: Face Clustering

### Current State
- `FaceEmbeddingService.cluster_faces()` already implemented
- Uses Agglomerative Clustering with cosine distance

### Required Work

#### 5.1 Clustering Threshold Tuning
**Question**: What clustering parameters work best?
- `distance_threshold`: Currently 0.5 (cosine distance)
- May need tuning based on actual results

#### 5.2 Cluster Metadata
```python
@dataclass
class FaceCluster:
    cluster_id: int
    face_images: List[CroppedFace]
    centroid_embedding: np.ndarray
    representative_face: CroppedFace  # Best quality face in cluster
    num_images: int  # Total images containing this person
```

---

## Phase 6: Integration into Album Pipeline

### New Service: `FacePipelineService`

```python
class FacePipelineService:
    """Complete face processing pipeline."""

    def __init__(self, config):
        self._face_detector = MediaPipePortraitAnalyzer(config)
        self._pose_estimator = SixDRepNetEstimator(config)
        self._face_embedder = FaceEmbeddingService(config)

    def process_album(self, image_paths: List[Path]) -> AlbumFaceResult:
        """
        Process all images in album.

        Returns:
            - All detected faces with quality scores
            - Face clusters (identities)
            - Best face per person
        """

    def get_face_quality_score(self, face: CroppedFace) -> FaceQualityScore:
        """
        Compute overall face quality score.

        Combines: pose, smile, eyes_open, detection_confidence
        """
```

### Quality Score Formula
```python
@dataclass
class FaceQualityScore:
    pose_score: float      # 0-1, higher = more frontal
    smile_score: float     # 0-1, higher = more smile
    eyes_open_score: float # 0-1, higher = more open
    sharpness_score: float # 0-1, from IQA
    overall: float         # Weighted combination
```

**Question**: What weights for overall score?
- Pose: 0.3 (most important for face recognition)
- Eyes open: 0.3 (closed eyes = bad photo)
- Smile: 0.2 (nice to have)
- Sharpness: 0.2 (technical quality)

---

## Future Work (Phase 7+)

### 7.1 Sub-clusters by Scene/Time
Within each identity cluster, create sub-groups based on:
- Photo timestamp (photos from same event)
- Scene similarity (using existing DINOv2 features)
- Clothing/appearance similarity

### 7.2 Best Image Selection Per Person
For each identity cluster:
1. Rank faces by quality score
2. Consider diversity (different poses, expressions)
3. Select representative set for "People" view

### 7.3 "People" Album View
- Show all identified people (clusters)
- Thumbnail: best face per person
- Click to see all photos of that person
- Manual merge/split clusters

---

## Implementation Order

1. **Face Cropping** - Foundation for all subsequent steps
2. **SixDRepNet Integration** - New capability needed
3. **Unified Face Quality Score** - Combine all metrics
4. **Face Embeddings on Crops** - Connect to existing service
5. **Clustering Pipeline** - End-to-end flow
6. **Album Integration** - Connect to existing workflow

---

## Design Decisions (Resolved)

1. **Face dominance threshold**: **2% of image area**
   - Allows processing of group photos while filtering tiny background faces

2. **Pose penalty calculation**: **Soft sigmoid**
   - Smooth transition around threshold boundaries
   - Formula: `penalty = sigmoid(k * (angle - threshold))` where k controls steepness

3. **Quality score weights**: 0.3 pose, 0.3 eyes, 0.2 smile, 0.2 sharpness
   - Can be tuned later based on results

4. **SixDRepNet installation**: **PyPI package** (`pip install sixdrepnet`)

---

## File Structure (Proposed)

```
sim_bench/
├── face_pipeline/
│   ├── __init__.py
│   ├── crop_service.py        # Face cropping
│   ├── pose_estimator.py      # SixDRepNet wrapper
│   ├── quality_scorer.py      # Combined quality score
│   ├── pipeline.py            # Main FacePipelineService
│   └── types.py               # CroppedFace, FaceQualityScore, etc.
├── album/services/
│   └── face_embedding_service.py  # (existing)
└── portrait_analysis/
    └── analyzer.py  # (existing, may need minor updates)
```
