# select_best Redesign Implementation

## Overview

The `select_best` step has been completely redesigned to use a **simple additive scoring model**:

```
composite_score = image_quality_score + person_penalty
```

This replaces the previous complex branching logic (face vs non-face clusters) with a unified, extensible approach.

---

## Key Components

### 1. Image Quality Scoring (`quality_strategy.py`)

**Purpose**: Compute technical/aesthetic quality independent of content.

**Strategies Available**:

- **`WeightedAverageQuality`** (baseline): Simple weighted average of IQA and AVA
  ```python
  quality = 0.3 * iqa + 0.7 * ava
  ```

- **`SiameseRefinementQuality`** (recommended, Option 3): Refines top candidates with Siamese comparisons
  ```python
  # For top 3 candidates only
  base_quality = 0.3 * iqa + 0.7 * ava
  # Compare against best image
  boost = ±0.1 * siamese_confidence
  quality = base_quality + boost
  ```

- **`SiameseTournamentQuality`**: Full pairwise tournament among top candidates
  ```python
  # All pairs compared, wins accumulated
  siamese_score = wins / total_comparisons
  quality = 0.4 * base_quality + 0.6 * siamese_score
  ```

**Key Features**:
- Strategy pattern for easy switching
- Configurable via YAML
- Caches base quality computations
- Logs warnings for unexpected score ranges (e.g., AVA > 1.0)

---

### 2. Person Penalty Scoring (`person_penalty.py`)

**Purpose**: Apply portrait-specific penalties (0 to -0.7).

**Penalty Table**:

| Condition           | Default Penalty | Configurable |
|---------------------|-----------------|--------------|
| No person           | 0.0             | N/A          |
| Person, no face     | -0.3            | ✅            |
| Body not facing     | -0.1            | ✅            |
| Eyes closed         | -0.15           | ✅            |
| Not smiling         | -0.05           | ✅            |
| Face turned         | -0.1            | ✅            |

**Accumulation**:
- Penalties are **additive** (cumulative)
- Example: Eyes closed + not smiling = -0.15 + -0.05 = -0.2
- Capped at `max_penalty` (-0.7 by default)

**Key Features**:
- No branching logic (uses early returns and threshold checks)
- All thresholds configurable
- Defaults to neutral (0.5) if scores unavailable

---

### 3. Composite Scoring (`select_best.py`)

**Flow**:

```python
# 1. Initialize strategies
quality_strategy = ImageQualityStrategyFactory.create("siamese_refinement", config)
penalty_computer = PersonPenaltyFactory.create(config)

# 2. For each image
for image in cluster:
    quality = quality_strategy.compute_quality(image, context, siamese_model, all_images)
    penalty = penalty_computer.compute_penalty(image, context)
    composite_score = quality + penalty

# 3. Filter by threshold
filtered = [img for img, score in scored if score >= min_threshold]

# 4. Select best + dissimilar
selected = []
selected.append(best_image)  # Always select #1
for image in remaining:
    if is_dissimilar(image, selected):
        selected.append(image)
```

**Removed**:
- ❌ `_score_face_cluster()` - face-specific weighted scoring
- ❌ `_score_non_face_cluster()` - non-face AVA scoring
- ❌ `_cluster_has_faces()` - face presence checking
- ❌ `has_faces` parameter to `_select_from_cluster()`
- ❌ Branching logic for face vs non-face
- ❌ Old `InsightFaceScorer` and `MediaPipeScorer` classes

**Added**:
- ✅ `_compute_composite_scores()` - unified scoring
- ✅ `_select_dissimilar()` - best + dissimilar selection
- ✅ `_check_dissimilar()` - dissimilarity checker
- ✅ `_build_quality_config()` - strategy config builder

---

## Configuration

### New Config Structure (`configs/pipeline.yaml`)

```yaml
select_best:
  max_images_per_cluster: 2
  min_score_threshold: 0.4
  dissimilarity_threshold: 0.85

  # Image quality strategy
  quality_strategy: siamese_refinement  # Options: weighted_average, siamese_refinement, siamese_tournament
  
  quality_weights:
    iqa: 0.3
    ava: 0.7

  siamese_refinement:
    top_n: 3
    boost_range: 0.1

  # Person penalties
  person_penalties:
    face_occlusion: -0.3
    body_not_facing: -0.1
    eyes_closed: -0.15
    not_smiling: -0.05
    face_turned: -0.1
    max_penalty: -0.7
    body_facing_threshold: 0.5
    eyes_threshold: 0.5
    smile_threshold: 0.5
    pose_threshold: 0.5

  siamese:
    enabled: true
    checkpoint_path: models/album_app/siamese_comparison_model.pt

  use_face_subclusters: true
  include_noise: true
```

### Old Config (Removed)

```yaml
# ❌ Removed
scoring_backend: insightface
scoring_strategy: insightface_penalty
penalties:
  body_orientation: 0.1
  face_occlusion: 0.2
  eyes_closed: 0.15
  no_smile: 0.1
  face_turned: 0.15

face_weights:
  eyes_open: 0.30
  pose: 0.30
  smile: 0.20
  ava: 0.20

max_score_gap: 0.25
tiebreaker_threshold: 0.05
duplicate_similarity_threshold: 0.95
```

---

## Benefits

### ✅ Simplicity
- No branching logic (face vs non-face)
- Same formula for all images
- Easy to understand and explain

### ✅ Extensibility
- Add new quality metrics: Implement new strategy class
- Add new penalties: Update penalty config
- A/B test: Just change YAML config

### ✅ Maintainability
- Strategy pattern: swap implementations easily
- Factory pattern: centralized creation
- No if/else: threshold-based checks and early returns

### ✅ Configurability
- All weights/penalties in YAML
- Easy to tune per use-case
- Can learn optimal values from feedback

### ✅ Adherence to Project Guidelines
- No if/else (strategy pattern, early returns)
- No try/except (let errors propagate)
- Absolute imports
- Logger instead of prints
- Methods under 50 lines

---

## Examples

### Example 1: Landscape (No Person)
```
iqa_score = 0.8
ava_score = 0.7

image_quality_score = 0.3*0.8 + 0.7*0.7 = 0.73
person_penalty = 0.0  (no person)

composite_score = 0.73 + 0.0 = 0.73 ✅
```

### Example 2: Good Portrait
```
iqa_score = 0.8
ava_score = 0.7
person_detected = True, face_detected = True
eyes_open = True, smiling = True, facing = True

image_quality_score = 0.73
person_penalty = 0.0  (no issues)

composite_score = 0.73 + 0.0 = 0.73 ✅
```

### Example 3: Portrait, Eyes Closed
```
iqa_score = 0.8
ava_score = 0.7
eyes_open = False

image_quality_score = 0.73
person_penalty = -0.15  (eyes closed)

composite_score = 0.73 - 0.15 = 0.58 ⚠️
```

### Example 4: Portrait, Face Occluded
```
iqa_score = 0.8
ava_score = 0.7
person_detected = True, face_detected = False

image_quality_score = 0.73
person_penalty = -0.3  (face occluded)

composite_score = 0.73 - 0.3 = 0.43 ⚠️
```

### Example 5: Poor Quality + Multiple Issues
```
iqa_score = 0.3
ava_score = 0.4
eyes_open = False, smiling = False, body_turned = True

image_quality_score = 0.3*0.3 + 0.7*0.4 = 0.37
person_penalty = -0.15 - 0.05 - 0.1 = -0.3

composite_score = 0.37 - 0.3 = 0.07 ❌ (very low, likely discarded)
```

---

## Migration

### Phase 1: ✅ Implement New Scoring
- New strategy classes created
- New penalty computer created
- `select_best.py` refactored
- Config updated

### Phase 2: 🔄 Test & Validate
- Run on sample datasets
- Compare results with old approach
- Tune penalty values if needed
- Validate scoring distributions

### Phase 3: ⏳ Production
- Deploy new scoring to production
- Monitor composite score distributions
- Collect user feedback
- Iterate on penalty values

---

## Files Changed

### New Files
- `sim_bench/pipeline/scoring/quality_strategy.py` - Quality scoring strategies
- `sim_bench/pipeline/scoring/person_penalty.py` - Person penalty computer
- `sim_bench/pipeline/scoring/__init__.py` - Module exports
- `docs/architecture/SELECT_BEST_REDESIGN_IMPLEMENTATION.md` - This file

### Modified Files
- `sim_bench/pipeline/steps/select_best.py` - Complete refactor
- `configs/pipeline.yaml` - Updated select_best config

---

## Testing Checklist

- [ ] Test with landscape images (no person)
- [ ] Test with good portraits (all scores high)
- [ ] Test with poor portraits (multiple penalties)
- [ ] Test with face occlusion
- [ ] Test with eyes closed
- [ ] Test with not smiling
- [ ] Test with Siamese refinement enabled
- [ ] Test with Siamese refinement disabled
- [ ] Test with different quality strategies
- [ ] Verify score distributions are reasonable
- [ ] Verify best images are selected correctly
- [ ] Verify dissimilarity filtering works

---

## Future Enhancements

### Neural Network Scoring
The current design makes it easy to replace with NN-based scoring:

```python
class NeuralQualityStrategy(ImageQualityStrategy):
    """Use neural network to predict image quality."""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def compute_quality(self, image_path, context, siamese_model, all_images):
        features = extract_features(image_path, context)
        return self.model.predict(features)

class NeuralPenaltyComputer(PersonPenaltyComputer):
    """Use neural network to compute person penalties."""
    
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def compute_penalty(self, image_path, context):
        features = extract_person_features(image_path, context)
        return self.model.predict(features)
```

Just update config:
```yaml
quality_strategy: neural_quality
person_penalty_strategy: neural_penalty
```

No code changes to `select_best.py` needed! 🎉

---

## Summary

✅ **Simpler**: One formula, no branching  
✅ **Extensible**: Easy to add strategies  
✅ **Configurable**: All params in YAML  
✅ **Maintainable**: Clean separation of concerns  
✅ **Future-proof**: Ready for ML/NN scoring
