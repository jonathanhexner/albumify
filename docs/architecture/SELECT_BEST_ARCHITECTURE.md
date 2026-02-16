# select_best Architecture

## Overview

The `select_best` pipeline step selects the best representative images from each cluster using a **composite scoring model** that combines technical image quality with portrait-specific penalties.

**Core Formula**:
```
composite_score = image_quality_score + person_penalty
```

Where:
- `image_quality_score` ∈ [0, 1] - Technical/aesthetic quality (higher is better)
- `person_penalty` ∈ [-0.7, 0] - Portrait-specific penalties (0 = perfect, negative = issues)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SelectBestStep.process()                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  Initialize Strategies from Config          │
        ├─────────────────────────────────────────────┤
        │  • ImageQualityStrategy                     │
        │  • PersonPenaltyComputer                    │
        │  • Siamese Model (if enabled)               │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │  For Each Cluster (Scene/Identity)          │
        └─────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────────┐
        │      _select_from_cluster()                 │
        └─────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Compute Composite    │   │  Compute Composite    │
    │  Scores for Image 1   │   │  Scores for Image N   │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
            ┌─────────────────────────────────────┐
            │  _compute_composite_scores()        │
            └─────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  Quality Strategy     │   │  Penalty Computer     │
    │  .compute_quality()   │   │  .compute_penalty()   │
    └───────────────────────┘   └───────────────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
            ┌─────────────────────────────────────┐
            │  composite = quality + penalty      │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  Filter by min_score_threshold      │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  Sort by composite_score DESC       │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  _select_dissimilar()               │
            │  • Always select #1                 │
            │  • Add dissimilar images            │
            └─────────────────────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │  Return Selected Images             │
            └─────────────────────────────────────┘
```

---

## Component 1: Image Quality Scoring

### Purpose
Compute technical and aesthetic quality independent of image content.

### Strategies

#### A. WeightedAverageQuality (Baseline)
Simple weighted combination of IQA and AVA scores.

```python
quality = w_iqa * iqa_score + w_ava * ava_score
```

**Default Weights**: IQA = 0.3, AVA = 0.7

**Pros**:
- Fast (no Siamese needed)
- Predictable
- No dependencies

**Cons**:
- May not capture subtle quality differences
- No learned refinement

**Config**:
```yaml
quality_strategy: weighted_average
quality_weights:
  iqa: 0.3
  ava: 0.7
```

---

#### B. SiameseRefinementQuality (Recommended)
Refines top candidates using Siamese CNN comparisons against the best image.

**Algorithm**:
```python
# Step 1: Compute base quality for all images
base_quality = 0.3 * iqa + 0.7 * ava

# Step 2: Get top N candidates by base quality
top_candidates = sort_by_base_quality(images)[:top_n]

# Step 3: Refine only top candidates
if image not in top_candidates:
    return base_quality
else:
    # Compare against best image
    best = top_candidates[0]
    result = siamese.compare_images(image, best)
    
    if result['prediction'] == 1:  # Current is better
        boost = +boost_range * result['confidence']
    else:
        boost = -boost_range * result['confidence']
    
    return base_quality + boost
```

**Default Config**: top_n=3, boost_range=0.1 (±10%)

**Pros**:
- Selective refinement (only top images)
- Fast (N-1 comparisons max)
- Learned quality assessment
- Minimal overhead

**Cons**:
- Requires Siamese model
- Only refines top candidates

**Config**:
```yaml
quality_strategy: siamese_refinement
quality_weights:
  iqa: 0.3
  ava: 0.7
siamese_refinement:
  top_n: 3
  boost_range: 0.1
siamese:
  enabled: true
  checkpoint_path: models/album_app/siamese_comparison_model.pt
```

**Example**:
```
Image A: base_quality = 0.75
Image B: base_quality = 0.73 (top 3)
Image C: base_quality = 0.60 (not in top 3)

Siamese: B vs A → B is better (confidence=0.8)

Final:
  A: 0.75 (best, no comparison)
  B: 0.73 + (0.1 * 0.8) = 0.81  ← Boosted!
  C: 0.60 (not refined)
```

---

#### C. SiameseTournamentQuality (Most Accurate)
Runs full pairwise tournament among top candidates.

**Algorithm**:
```python
# Step 1: Compute base quality
base_quality = 0.3 * iqa + 0.7 * ava

# Step 2: Get top N candidates
top_candidates = sort_by_base_quality(images)[:top_n]

# Step 3: Run tournament for top candidates only
if image not in top_candidates:
    return base_quality
else:
    # Compare against all other candidates
    wins = 0.0
    comparisons = 0
    
    for opponent in top_candidates:
        if opponent == image:
            continue
        
        result = siamese.compare_images(image, opponent)
        if result['prediction'] == 1:  # Image wins
            wins += result['confidence']
        comparisons += 1
    
    siamese_score = wins / comparisons
    
    # Combine with base quality
    return w_base * base_quality + w_siamese * siamese_score
```

**Default Config**: top_n=4, base_weight=0.4, siamese_weight=0.6

**Pros**:
- Most accurate quality assessment
- All pairs compared
- Learned preferences

**Cons**:
- Slower (n² comparisons)
- Requires Siamese model
- More complex

**Config**:
```yaml
quality_strategy: siamese_tournament
quality_weights:
  iqa: 0.3
  ava: 0.7
siamese_tournament:
  top_n: 4
  base_weight: 0.4
  siamese_weight: 0.6
siamese:
  enabled: true
  checkpoint_path: models/album_app/siamese_comparison_model.pt
```

**Comparisons for top_n=4**: 6 total (4 choose 2)

---

### Strategy Pattern

All strategies implement the `ImageQualityStrategy` interface:

```python
class ImageQualityStrategy(ABC):
    @abstractmethod
    def compute_quality(
        self,
        image_path: str,
        context: PipelineContext,
        siamese_model: Optional[object],
        all_images: List[str],
    ) -> float:
        """Compute quality score in [0, 1]."""
        pass
```

**Factory**:
```python
strategy = ImageQualityStrategyFactory.create(
    strategy_name="siamese_refinement",
    config={"iqa_weight": 0.3, "ava_weight": 0.7, "top_n": 3, "boost_range": 0.1}
)

quality = strategy.compute_quality(image_path, context, siamese_model, all_images)
```

---

## Component 2: Person Penalty Computation

### Purpose
Apply portrait-specific penalties for images with people.

### Penalty Table

| Condition                | Default Penalty | Threshold        | Notes                    |
|--------------------------|-----------------|------------------|--------------------------|
| No person detected       | 0.0             | N/A              | No portrait penalties    |
| Person, but no face      | -0.3            | N/A              | Strong penalty           |
| Body not facing camera   | -0.1            | facing < 0.5     | Moderate penalty         |
| Eyes closed              | -0.15           | eyes < 0.5       | Moderate penalty         |
| Not smiling              | -0.05           | smile < 0.5      | Light penalty            |
| Face turned away         | -0.1            | pose < 0.5       | Moderate penalty         |

**Maximum Penalty**: -0.7 (cap to prevent extreme negatives)

### Penalty Accumulation

Penalties are **additive** (cumulative):

```python
if not person_detected:
    penalty = 0.0
elif not face_detected:
    penalty = -0.3  # Face occlusion penalty only
else:
    penalty = 0.0
    
    # Accumulate individual penalties
    if body_facing_score < body_facing_threshold:
        penalty += body_not_facing_penalty
    
    if eyes_score < eyes_threshold:
        penalty += eyes_closed_penalty
    
    if smile_score < smile_threshold:
        penalty += not_smiling_penalty
    
    if pose_score < pose_threshold:
        penalty += face_turned_penalty
    
    # Cap at max_penalty
    penalty = max(penalty, max_penalty)

return penalty
```

### Examples

**Example 1: No Person**
```
person_detected = False
→ penalty = 0.0
```

**Example 2: Person, Good Portrait**
```
person_detected = True
face_detected = True
body_facing = 0.9, eyes = 0.9, smile = 0.8, pose = 0.8
→ All above thresholds
→ penalty = 0.0
```

**Example 3: Eyes Closed**
```
person_detected = True
face_detected = True
eyes = 0.3 (< 0.5)
→ penalty = -0.15
```

**Example 4: Multiple Issues**
```
person_detected = True
face_detected = True
body_facing = 0.4 (< 0.5) → -0.1
eyes = 0.3 (< 0.5)        → -0.15
smile = 0.2 (< 0.5)       → -0.05
→ penalty = -0.1 + -0.15 + -0.05 = -0.3
```

**Example 5: Face Occluded**
```
person_detected = True
face_detected = False
→ penalty = -0.3 (immediate, no other checks)
```

### Configuration

```yaml
person_penalties:
  # Penalty values
  face_occlusion: -0.3
  body_not_facing: -0.1
  eyes_closed: -0.15
  not_smiling: -0.05
  face_turned: -0.1
  max_penalty: -0.7
  
  # Thresholds for applying penalties
  body_facing_threshold: 0.5
  eyes_threshold: 0.5
  smile_threshold: 0.5
  pose_threshold: 0.5
```

### Score Sources

| Penalty Type    | Context Attribute        | Source Step                  |
|-----------------|--------------------------|------------------------------|
| Person presence | `context.persons`        | `detect_persons`             |
| Body facing     | `context.persons`        | `detect_persons`             |
| Face detection  | `context.face_*_scores`  | `insightface_detect_faces`   |
| Eyes open       | `context.face_eyes_scores` | `insightface_score_eyes`   |
| Smile           | `context.face_smile_scores` | `insightface_score_expression` |
| Face pose       | `context.face_pose_scores` | `insightface_score_pose`   |

**Note**: Works with both MediaPipe and InsightFace pipelines (common interface).

---

## Component 3: Composite Scoring

### Formula

```
composite_score = image_quality_score + person_penalty
```

### Range

**Theoretical**: [-0.7, 1.0]
- Best: quality=1.0, penalty=0.0 → 1.0
- Worst: quality=0.0, penalty=-0.7 → -0.7

**Practical**: [0.0, 1.0]
- Most images fall in this range
- Negative scores are rare (terrible quality + major issues)

### Threshold Filtering

```python
min_threshold = 0.4  # Configurable

filtered = [
    (img, score) for img, score in scored
    if score >= min_threshold
]

# If all below threshold, keep best one anyway
if not filtered and scored:
    filtered = [max(scored, key=lambda x: x[1])]
```

**Rationale**: Users expect at least one photo per cluster.

### Example Scores

| Scenario                    | Quality | Penalty | Composite | Keep? |
|-----------------------------|---------|---------|-----------|-------|
| High quality landscape      | 0.85    | 0.0     | 0.85      | ✅     |
| Good portrait               | 0.75    | 0.0     | 0.75      | ✅     |
| Portrait, eyes closed       | 0.75    | -0.15   | 0.60      | ✅     |
| Portrait, not smiling       | 0.75    | -0.05   | 0.70      | ✅     |
| Portrait, face occluded     | 0.75    | -0.30   | 0.45      | ✅     |
| Low quality portrait        | 0.35    | -0.20   | 0.15      | ❌     |
| Multiple issues             | 0.40    | -0.30   | 0.10      | ❌     |

---

## Component 4: Dissimilarity Selection

### Purpose
Select best + dissimilar images from each cluster to maximize variety.

### Algorithm

```python
def _select_dissimilar(scored_images, max_per_cluster, threshold):
    selected = []
    
    # Step 1: Always select best image
    best = scored_images[0]
    selected.append(best)
    
    # Step 2: Add dissimilar images
    for image, score in scored_images[1:]:
        if len(selected) >= max_per_cluster:
            break
        
        # Check dissimilarity against all selected
        is_dissimilar = all(
            embedding_similarity(image, s) < threshold
            for s in selected
        )
        
        if is_dissimilar:
            selected.append(image)
    
    return selected
```

### Dissimilarity Check

Uses **scene embedding cosine similarity**:

```python
def embedding_similarity(img1, img2):
    emb1 = context.scene_embeddings[img1]
    emb2 = context.scene_embeddings[img2]
    
    return cosine_similarity(emb1, emb2)
```

**Threshold**: 0.85 (default)
- similarity < 0.85 → dissimilar (keep)
- similarity ≥ 0.85 → too similar (skip)

### Configuration

```yaml
max_images_per_cluster: 2
dissimilarity_threshold: 0.85
```

### Example

```
Cluster: 5 images, max_per_cluster=2

Scored (sorted by composite):
1. img_A: 0.85
2. img_B: 0.78 (similarity to A = 0.90) ← Too similar
3. img_C: 0.75 (similarity to A = 0.70) ← Dissimilar!
4. img_D: 0.70
5. img_E: 0.65

Selected: [img_A, img_C]
```

---

## Cluster Splitting by Identity

### Purpose
When a scene contains multiple people, split by identity and select independently.

### Flow

```
Scene Cluster: 10 images
  ↓
Split by Face Embedding Similarity
  ↓
├─ Identity A: 7 images
│  └─ Select best 2 from Identity A
│
└─ Identity B: 3 images
   └─ Select best 2 from Identity B

Result: Fair representation (both people get photos)
```

### Configuration

```yaml
use_face_subclusters: true
```

When `true`:
- Process `context.face_clusters` (identity-split)
- Each identity cluster processed independently

When `false`:
- Process `context.scene_clusters` (no identity split)
- All images in scene compete together

---

## Full Processing Flow

### 1. Initialization

```python
def process(context, config):
    # Create quality strategy
    quality_strategy = ImageQualityStrategyFactory.create(
        config['quality_strategy'],
        config
    )
    
    # Create penalty computer
    penalty_computer = PersonPenaltyFactory.create(
        config['person_penalties']
    )
    
    # Load Siamese model if needed
    siamese_model = load_siamese_if_needed(config)
```

### 2. Cluster Iteration

```python
    selected_images = []
    
    # Iterate over clusters
    for cluster_id, images in clusters:
        # Select from this cluster
        cluster_selected = _select_from_cluster(
            context, images, max_per_cluster, siamese_model
        )
        
        selected_images.extend(cluster_selected)
```

### 3. Cluster Selection

```python
def _select_from_cluster(context, images, max_per_cluster, siamese_model):
    # Step 1: Compute composite scores
    scored = _compute_composite_scores(context, images, siamese_model)
    
    # Step 2: Filter by threshold
    filtered = [(img, score) for img, score in scored if score >= min_threshold]
    
    # Keep best if all below threshold
    if not filtered:
        filtered = [max(scored, key=lambda x: x[1])]
    
    # Step 3: Sort by composite score
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    # Step 4: Select best + dissimilar
    selected = _select_dissimilar(filtered, max_per_cluster)
    
    return selected
```

### 4. Composite Scoring

```python
def _compute_composite_scores(context, images, siamese_model):
    scored = []
    
    for image in images:
        # Compute quality
        quality = quality_strategy.compute_quality(
            image, context, siamese_model, images
        )
        
        # Compute penalty
        penalty = penalty_computer.compute_penalty(
            image, context
        )
        
        # Combine
        composite = quality + penalty
        
        scored.append((image, composite))
    
    return scored
```

---

## Configuration Reference

### Complete Example

```yaml
select_best:
  # Selection limits
  max_images_per_cluster: 2
  min_score_threshold: 0.4
  dissimilarity_threshold: 0.85
  
  # Image quality strategy
  quality_strategy: siamese_refinement
  
  quality_weights:
    iqa: 0.3
    ava: 0.7
  
  # Strategy-specific config
  siamese_refinement:
    top_n: 3
    boost_range: 0.1
  
  siamese_tournament:
    top_n: 4
    base_weight: 0.4
    siamese_weight: 0.6
  
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
  
  # Siamese model
  siamese:
    enabled: true
    checkpoint_path: models/album_app/siamese_comparison_model.pt
  
  # Clustering options
  use_face_subclusters: true
  include_noise: true
```

### Minimal Config (Defaults)

```yaml
select_best: {}
```

Uses:
- `siamese_refinement` quality strategy
- IQA=0.3, AVA=0.7
- Default penalties
- max_per_cluster=2, threshold=0.4

---

## Design Patterns

### Strategy Pattern
Quality scoring strategies are interchangeable:

```python
# Easy to swap
strategy = ImageQualityStrategyFactory.create(strategy_name, config)

# Easy to add new strategies
class NeuralQualityStrategy(ImageQualityStrategy):
    def compute_quality(self, ...):
        return neural_model.predict(...)
```

### Factory Pattern
Centralized creation of strategies and computers:

```python
quality_strategy = ImageQualityStrategyFactory.create(name, config)
penalty_computer = PersonPenaltyFactory.create(config)
```

### No Branching
Threshold-based checks replace if/else:

```python
# Instead of:
if eyes_score < 0.5:
    penalty += eyes_penalty

# We use:
penalty_check = eyes_score < eyes_threshold
penalty_amount = eyes_penalty if penalty_check else 0.0
penalty += penalty_amount
```

---

## Performance Considerations

### Quality Strategy Performance

| Strategy            | Images/sec | Comparisons | Notes              |
|---------------------|------------|-------------|--------------------|
| WeightedAverage     | ~1000      | 0           | No Siamese         |
| SiameseRefinement   | ~300       | N-1         | Only top N         |
| SiameseTournament   | ~100       | N*(N-1)/2   | All pairs in top N |

**Cluster of 10 images, top_n=3**:
- WeightedAverage: 0 comparisons
- SiameseRefinement: 2 comparisons (top 2 vs best)
- SiameseTournament: 3 comparisons (top 3, all pairs)

### Caching

Quality strategies cache base quality computations:

```python
self._base_quality_cache[image_path] = base_quality
```

Avoids recomputing IQA+AVA for multiple strategy invocations.

---

## Future Extensions

### Neural Network Scoring

Easy to add ML-based scoring:

```python
class NeuralQualityStrategy(ImageQualityStrategy):
    def __init__(self, model_path):
        self.model = load_model(model_path)
    
    def compute_quality(self, image_path, context, siamese_model, all_images):
        features = {
            'iqa': context.iqa_scores.get(image_path),
            'ava': context.ava_scores.get(image_path),
            'sharpness': context.sharpness_scores.get(image_path),
            # ... more features
        }
        return self.model.predict(features)
```

Config:
```yaml
quality_strategy: neural
neural_model_path: models/quality_predictor.onnx
```

### Learned Penalties

Replace rule-based penalties with learned model:

```python
class NeuralPenaltyComputer:
    def compute_penalty(self, image_path, context):
        features = extract_person_features(image_path, context)
        return self.model.predict(features)
```

### Context-Aware Penalties

Adjust penalties based on image context:

```python
# Candid photos: don't penalize not smiling
if context.scene_type == 'candid':
    not_smiling_penalty = 0.0
else:
    not_smiling_penalty = -0.05
```

---

## Comparison with Old Architecture

| Aspect | Old (Branching) | New (Composite) |
|--------|----------------|-----------------|
| **Scoring** | Different for face vs non-face | Unified for all images |
| **Face weights** | eyes=0.3, pose=0.3, smile=0.2, ava=0.2 | Replaced by penalties |
| **Extensibility** | Hard to add metrics | Easy (strategy pattern) |
| **Configuration** | Mixed concerns | Clean separation |
| **Code size** | ~600 lines | ~300 lines |
| **Branching** | `if has_faces:` everywhere | None (strategy dispatch) |
| **Testability** | Hard to isolate logic | Easy (mock strategies) |

---

## Summary

**Key Improvements**:
- ✅ **Simpler**: One formula, no face/non-face branching
- ✅ **Cleaner**: Strategy pattern, no if/else
- ✅ **Extensible**: Easy to add strategies/penalties
- ✅ **Configurable**: All parameters in YAML
- ✅ **Maintainable**: Clear separation of concerns
- ✅ **Future-proof**: Ready for ML/NN integration

**Core Insight**: Separate technical quality (universal) from portrait preferences (penalties).

**Deployment**: Drop-in replacement, backward compatible via config.
