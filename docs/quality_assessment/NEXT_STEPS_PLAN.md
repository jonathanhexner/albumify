# PhotoTriage Training - Next Steps Plan

**Date**: 2025-11-28
**Current Status**: Multi-feature pairwise ranker implemented (CLIP + CNN + IQA)

## Overview

This document outlines the next development steps for PhotoTriage quality assessment, focusing on two main directions:
1. **Bradley-Terry (BT) ranking model** for more principled probabilistic ranking
2. **Facial feature analysis** for portrait-specific quality assessment

---

## Phase 1: Bradley-Terry Ranking Model

### Background

The current model uses **margin ranking loss** which learns:
- `score(better) > score(worse) + margin`

**Bradley-Terry (BT)** is a more principled probabilistic approach used in successful papers (Best-Photo-From-Burst, Series Photo Selection):
- Models probability that image i beats image j: `P(i > j) = exp(score_i) / (exp(score_i) + exp(score_j))`
- Equivalent to logistic regression on score differences
- Provides calibrated probabilities (useful for confidence estimates)
- Naturally handles ranking across multiple pairs

### Implementation Plan

#### 1.1 Create BT Loss Module

**File**: `sim_bench/quality_assessment/trained_models/losses.py`

```python
class BradleyTerryLoss(nn.Module):
    """
    Bradley-Terry loss for pairwise ranking.

    P(i > j) = sigmoid(score_i - score_j)
    Loss = -log(P(winner > loser))
    """

    def forward(self, score_i, score_j, winner):
        """
        Args:
            score_i: Scores for first images (batch_size,)
            score_j: Scores for second images (batch_size,)
            winner: 0 if i wins, 1 if j wins

        Returns:
            loss: Scalar loss
        """
        # P(i > j) = sigmoid(score_i - score_j)
        prob_i_wins = torch.sigmoid(score_i - score_j)

        # winner = 0 means i wins, winner = 1 means j wins
        # loss = -log(P(correct outcome))
        prob_correct = torch.where(
            winner == 0,
            prob_i_wins,
            1 - prob_i_wins
        )

        loss = -torch.log(prob_correct + 1e-8).mean()
        return loss
```

**Advantages over margin loss**:
- Probabilistic interpretation
- Better calibration for confidence estimates
- Used in competition-winning papers

#### 1.2 Update Training Script

**File**: `train_multifeature_ranker.py`

Changes:
- Import BT loss
- Add `--loss_type` argument: `margin` or `bradley_terry`
- Update training loop to use selected loss

**Comparison metrics**:
- Add calibration plots (predicted probability vs actual win rate)
- Compare margin loss vs BT loss on same architecture

#### 1.3 Expected Results

Based on literature:
- BT loss typically gives **1-3% improvement** over margin loss
- Better probability calibration (important for downstream applications)
- More stable training (sigmoid smoother than hinge)

**Timeline**: 1-2 days implementation + testing

---

## Phase 2: Facial Feature Analysis

### Background

For portrait photos, **facial quality** is critical:
- Eyes open/closed
- Smiling vs neutral
- Face sharpness
- Facial occlusions
- Head pose/angle
- Lighting on face

Current model uses **global CNN features** which don't specifically target faces. We need **face-specific features**.

### Implementation Plan

#### 2.1 Face Detection & Alignment

**Use existing libraries** (don't re-implement):

**Option A: RetinaFace** (Recommended)
```python
# Already in PyTorch, fast, accurate
from facenet_pytorch import MTCNN

detector = MTCNN(keep_all=True, device='cuda')
boxes, probs, landmarks = detector.detect(img, landmarks=True)

# landmarks: (N, 5, 2) - left_eye, right_eye, nose, mouth_left, mouth_right
```

**Option B: MediaPipe** (Google's solution)
```python
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    min_detection_confidence=0.5
)

# Returns 468 facial landmarks
```

#### 2.2 Facial Quality Features

**File**: `sim_bench/specialized_models/faces.py` (already exists!)

Current implementation has:
- Face detection count
- Face bounding boxes

**Extend with**:

```python
class FacialQualityAnalyzer:
    """Analyze facial quality attributes."""

    def __init__(self, device='cuda'):
        self.detector = MTCNN(keep_all=True, device=device)
        # Load eye closure detector (pretrained model)
        self.eye_classifier = self._load_eye_classifier()
        # Load smile detector
        self.smile_detector = self._load_smile_detector()

    def analyze_face(self, face_crop, landmarks):
        """Extract facial quality features."""
        return {
            'eyes_open': self._check_eyes_open(face_crop, landmarks),
            'smiling': self._check_smile(face_crop, landmarks),
            'face_sharpness': self._compute_face_sharpness(face_crop),
            'lighting_quality': self._assess_face_lighting(face_crop),
            'head_pose': self._estimate_head_pose(landmarks),
            'occlusion_score': self._check_occlusions(face_crop)
        }

    def _check_eyes_open(self, face_crop, landmarks):
        """Detect if eyes are open (EAR - Eye Aspect Ratio)."""
        # Use landmarks to compute eye aspect ratio
        # EAR < 0.2 typically means closed
        left_eye = landmarks[0]  # Left eye landmarks
        right_eye = landmarks[1]  # Right eye landmarks

        ear_left = self._compute_ear(left_eye)
        ear_right = self._compute_ear(right_eye)

        return (ear_left + ear_right) / 2 > 0.2

    def _check_smile(self, face_crop, landmarks):
        """Detect smile using mouth landmarks."""
        # Compute mouth aspect ratio
        # Higher ratio = wider mouth = likely smiling
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)

        # Or use pretrained smile classifier
        return self.smile_detector(face_crop)

    def _compute_face_sharpness(self, face_crop):
        """Compute sharpness specifically on face region."""
        # Laplacian variance on face (more reliable than full image)
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()
```

**Pretrained Models to Use**:
1. **Eye closure**: Use EAR (Eye Aspect Ratio) - no model needed, pure geometry
2. **Smile detection**:
   - Option 1: Use mouth landmark geometry (simple, fast)
   - Option 2: Fine-tuned CNN on smile/neutral (CelebA dataset)
3. **Head pose**: Solve PnP problem with landmarks (OpenCV)

#### 2.3 Integration with Multi-Feature Ranker

**File**: `sim_bench/quality_assessment/trained_models/phototriage_multifeature.py`

**Current feature extraction**:
```python
features = [CLIP (512), CNN (1024), IQA (4)] = 1540-dim
```

**Add facial features**:
```python
features = [
    CLIP (512),
    CNN (1024),
    IQA (4),
    FACE (8)  # NEW: eyes_open, smiling, face_sharp, lighting, pose_x, pose_y, pose_z, occlusion
] = 1548-dim
```

**Implementation**:
```python
class MultiFeatureExtractor(nn.Module):
    def __init__(self, config):
        # ... existing code ...

        # Add face analyzer
        if config.use_face_features:
            from sim_bench.specialized_models.faces import FacialQualityAnalyzer
            self.face_analyzer = FacialQualityAnalyzer(device=config.device)
            self.face_dim = 8
        else:
            self.face_analyzer = None
            self.face_dim = 0

    def extract_face_features(self, image_path):
        """Extract facial quality features."""
        if self.face_analyzer is None:
            return torch.zeros(0)

        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect faces
        boxes, probs, landmarks = self.face_analyzer.detector.detect(
            img_rgb, landmarks=True
        )

        if boxes is None or len(boxes) == 0:
            # No faces detected - return neutral values
            return torch.zeros(self.face_dim)

        # Use largest face
        largest_idx = np.argmax([(b[2]-b[0])*(b[3]-b[1]) for b in boxes])
        face_box = boxes[largest_idx]
        face_landmarks = landmarks[largest_idx]

        # Crop face
        x1, y1, x2, y2 = face_box.astype(int)
        face_crop = img_rgb[y1:y2, x1:x2]

        # Analyze
        face_attrs = self.face_analyzer.analyze_face(face_crop, face_landmarks)

        # Convert to tensor
        features = torch.tensor([
            float(face_attrs['eyes_open']),
            float(face_attrs['smiling']),
            face_attrs['face_sharpness'] / 1000.0,  # Normalize
            face_attrs['lighting_quality'],
            face_attrs['head_pose'][0],  # Yaw
            face_attrs['head_pose'][1],  # Pitch
            face_attrs['head_pose'][2],  # Roll
            face_attrs['occlusion_score']
        ], dtype=torch.float32)

        return features

    def extract_all(self, image_path):
        clip_feat = self.extract_clip(img_pil)
        cnn_feat = self.extract_cnn(img_pil)
        iqa_feat = self.extract_iqa(image_path)
        face_feat = self.extract_face_features(image_path)  # NEW

        features = torch.cat([clip_feat, cnn_feat, iqa_feat, face_feat])
        return features
```

#### 2.4 Facial Feature Configuration

**File**: `sim_bench/quality_assessment/trained_models/phototriage_multifeature.py`

Add to `MultiFeatureConfig`:
```python
@dataclass
class MultiFeatureConfig:
    # ... existing fields ...

    # Face features
    use_face_features: bool = True
    face_detection_confidence: float = 0.5
    max_faces_per_image: int = 1  # Use largest face
```

#### 2.5 Analysis & Visualization

**Update Streamlit app** to show facial features:
- Display detected faces with bounding boxes
- Show per-face quality scores
- Compare model performance on portraits vs non-portraits

**Expected insights**:
- Does model learn that eyes open → better quality?
- Does smiling improve predicted scores?
- Face sharpness vs overall sharpness correlation

#### 2.6 Expected Performance Impact

Based on literature and intuition:

**For portrait photos**:
- **+5-10% accuracy improvement** on portrait-heavy pairs
- Eyes closed should strongly predict lower quality
- Face sharpness often more important than background sharpness

**For non-portrait photos**:
- No impact (face features will be zero/neutral)
- Should not hurt performance

**Overall**:
- Expect **+2-5% overall accuracy** (assuming ~30-50% of PhotoTriage photos contain faces)

**Timeline**: 3-5 days implementation + testing

---

## Phase 3: Combined BT + Facial Features

Once both Phase 1 and Phase 2 are complete:

**Best configuration**:
```python
config = MultiFeatureConfig(
    # Features
    use_face_features=True,
    use_sharpness=True,
    use_exposure=True,
    use_colorfulness=True,
    use_contrast=True,

    # Architecture
    mlp_hidden_dims=[512, 256],
    dropout=0.3,
    use_layernorm=True,

    # Training
    loss_type='bradley_terry',  # NEW
    learning_rate=1e-4,
    margin=1.0
)
```

**Expected cumulative results**:
- Current baseline (multi-feature): 60-70% (target)
- + BT loss: +1-3% → **61-73%**
- + Facial features: +2-5% → **63-78%**
- **Final target**: **65-78% pairwise accuracy**

Compare to benchmarks:
- Random: 50%
- Sharpness only: 56.4%
- **Our target: 65-78%** ✓ Significantly better

---

## Implementation Priority

### High Priority (Do Next)

1. **Fix current training** (get baseline working)
   - Current run has data issues
   - Need to see actual baseline performance first

2. **Bradley-Terry loss** (Quick win, 1-2 days)
   - Easy to implement
   - Well-studied, proven approach
   - Immediate comparison with margin loss

### Medium Priority

3. **Facial features** (3-5 days)
   - Requires integration with face detection
   - Need to extend existing `faces.py` module
   - Good for portrait-heavy datasets

### Lower Priority (Future)

4. **Partial CLIP fine-tuning**
   - Only if frozen features plateau
   - Unfreeze last 1-2 transformer blocks
   - Requires careful learning rate scheduling

5. **Auxiliary task learning**
   - Multi-task learning (predict attributes + ranking)
   - May improve feature learning

6. **Data augmentation**
   - Random crops, flips
   - Color jittering (careful - affects quality!)
   - Only if overfitting is observed

---

## Success Criteria

### Phase 1 (BT Loss)
- ✓ Implement BT loss module
- ✓ Compare with margin loss (same architecture)
- ✓ Show probability calibration improvement
- **Target**: +1-3% accuracy

### Phase 2 (Facial Features)
- ✓ Extend faces.py with quality analysis
- ✓ Integrate into multi-feature ranker
- ✓ Show performance improvement on portrait photos
- **Target**: +2-5% overall accuracy

### Overall Success
- **Baseline (current)**: 60-70% pairwise accuracy
- **After Phase 1**: 61-73%
- **After Phase 2**: 63-78%
- **Beat rule-based sharpness baseline (56.4%)**: ✓ YES

---

## Technical Notes

### Bradley-Terry Math

Given scores $s_i, s_j$ for images i and j:

$$P(i > j) = \frac{\exp(s_i)}{\exp(s_i) + \exp(s_j)} = \sigma(s_i - s_j)$$

Where $\sigma$ is the sigmoid function.

**Loss**:
$$\mathcal{L} = -\log P(\text{winner} > \text{loser})$$

**Equivalence to logistic regression**:
- Predict: $y_{ij} = 1$ if $i$ beats $j$
- Model: $P(y_{ij} = 1) = \sigma(s_i - s_j)$
- This is exactly logistic regression on score differences

### Face Detection Libraries

**Comparison**:

| Library | Speed | Accuracy | Landmarks | Notes |
|---------|-------|----------|-----------|-------|
| MTCNN | Fast | High | 5 points | Best for faces in wild |
| RetinaFace | Medium | Very High | 5 points | SOTA, slightly slower |
| MediaPipe | Very Fast | Medium | 468 points | Overkill for our needs |
| Dlib | Slow | High | 68 points | CPU-only, legacy |

**Recommendation**: Use **MTCNN** (facenet-pytorch)
- Good balance of speed/accuracy
- 5 landmarks sufficient for our features
- Already integrated in `sim_bench/specialized_models/faces.py`

### Facial Features Reference

**Eye Aspect Ratio (EAR)**:
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Where p1-p6 are eye landmarks. EAR < 0.2 typically means closed.

**Smile Detection**:
```
Mouth Aspect Ratio = (mouth_height / mouth_width)
```
Higher ratio = wider mouth = likely smiling.

Or use pretrained classifier (simpler).

---

## References

1. **Bradley-Terry Model**:
   - Bradley, R. A., & Terry, M. E. (1952). "Rank analysis of incomplete block designs"
   - Used in: Best-Photo-From-Burst (ECCV 2019)

2. **Facial Quality Assessment**:
   - "Face Quality Assessment in Video" (CVPR 2016)
   - MediaPipe Face Mesh (Google, 2020)

3. **PhotoTriage**:
   - Mihai et al. (2023) - Original dataset paper
   - Uses pairwise comparisons with quality attributes

---

## Questions & Decisions

### Decision Points

1. **BT vs Margin Loss**: Try both, pick better
2. **Face features**: Start with geometric (EAR, smile ratio), add learned models if needed
3. **Feature normalization**: Layer norm after concatenation (current approach is good)

### Open Questions

1. **How many faces to analyze per image?**
   - Proposal: Just largest face (simpler, portrait photos typically have 1 main subject)
   - Alternative: Average features across all faces

2. **What to do when no faces detected?**
   - Proposal: Return zeros (neutral features)
   - Alternative: Skip face features entirely for that image

3. **Should we use different models for different photo types?**
   - Proposal: Single unified model (use face features=0 for non-portraits)
   - Alternative: Separate models for portraits vs landscapes (more complex)

---

## Summary

**Immediate next steps**:
1. Fix current training data issues
2. Get baseline performance with existing multi-feature model
3. Implement Bradley-Terry loss (1-2 days)
4. Implement facial features (3-5 days)

**Expected outcome**: **65-78% pairwise accuracy**, beating all rule-based methods.
