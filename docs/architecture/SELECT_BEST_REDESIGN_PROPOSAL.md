# select_best Redesign Proposal

## Current Problems

### 1. Overly Complex Branching Logic
```python
if has_faces:
    score = w_eyes*eyes + w_pose*pose + w_smile*smile + w_ava*ava
else:
    score = 0.7*ava + 0.3*iqa
```

**Problems**:
- Different scoring logic for face vs non-face
- Weights are arbitrary and hard to tune
- Face attributes mixed with quality metrics
- Can't easily add new scoring methods

### 2. Confusing Score Semantics
- Are face attributes quality indicators or penalties?
- Why does "no smile" reduce score? (Not always a quality issue)
- Mixed signals: technical quality vs portrait preferences

### 3. Hard to Extend
- Want to add new quality metrics? Need to modify weighted formula
- Want to change portrait preferences? Need to retune all weights
- Can't easily A/B test different strategies

---

## Proposed Design: Simple Additive Model

### Core Concept

```
composite_score = image_quality_score + person_penalty
```

Where:
- `image_quality_score` ∈ [0, 1] - Technical/aesthetic quality (higher is better)
- `person_penalty` ∈ [-∞, 0] - Portrait-specific penalties (0 = perfect, negative = issues)

### Why This is Better

✅ **Clearer separation of concerns**: Quality vs portrait preferences  
✅ **Simpler to tune**: Adjust penalties independently of quality  
✅ **Easier to extend**: Add new quality metrics or penalties without touching each other  
✅ **No branching**: Same formula for all images  
✅ **Intuitive**: Quality goes up, penalties go down  

---

## Component 1: Image Quality Score

### Definition
**Technical and aesthetic quality**, independent of content.

### Problem: Current Methods Insufficient

**User Observation**: AVA + IQA + sharpness alone aren't sufficient for quality assessment.

**Solution**: Pluggable quality scoring strategy that's easy to change.

### Quality Scoring Strategies

#### Strategy A: Weighted Average (Baseline)
```python
image_quality_score = 0.3 * iqa_score + 0.7 * ava_score
```

**Pros**: Simple, fast  
**Cons**: May not capture all quality aspects

#### Strategy B: Siamese Tournament (Top-4)
```python
# 1. Get top 4 candidates by AVA+IQA
candidates = top_n_by_ava_iqa(images, n=4)

# 2. Run pairwise Siamese comparisons
siamese_scores = {}
for img in candidates:
    wins = 0
    for opponent in candidates:
        if img != opponent:
            result = siamese.compare(img, opponent)
            if result['prediction'] == 1:  # img is better
                wins += result['confidence']
    siamese_scores[img] = wins / (len(candidates) - 1)

# 3. Combine with base quality
image_quality_score = 0.4 * base_quality + 0.6 * siamese_scores[img]
```

**Pros**: Siamese directly influences quality, more accurate  
**Cons**: Slower (n² comparisons), only for top candidates

#### Strategy C: Siamese Refinement (Hybrid)
```python
# 1. Base quality from AVA+IQA
base_quality = 0.3 * iqa + 0.7 * ava

# 2. For top candidates, refine with Siamese
if img in top_candidates:
    # Compare against best image
    result = siamese.compare(img, best_image)
    if result['prediction'] == 1:
        boost = 0.1 * result['confidence']
    else:
        boost = -0.1 * result['confidence']
    
    image_quality_score = base_quality + boost
else:
    image_quality_score = base_quality
```

**Pros**: Fast (only compare top candidates), selective refinement  
**Cons**: Only top images get Siamese scoring

#### Strategy D: Ensemble (Multiple Models)
```python
# Future: Add more quality models
image_quality_score = (
    0.2 * iqa +
    0.3 * ava +
    0.2 * siamese_score +
    0.2 * clip_aesthetic +  # Future
    0.1 * inception_score   # Future
)
```

### Recommended Implementation: Strategy Pattern

```python
class ImageQualityStrategy(ABC):
    @abstractmethod
    def compute_quality(self, image_path: str, context: PipelineContext, 
                       siamese_model, all_images: List[str]) -> float:
        """Compute quality score for image."""
        pass

class WeightedAverageQuality(ImageQualityStrategy):
    """Simple weighted average of IQA and AVA."""
    
    def compute_quality(self, image_path, context, siamese_model, all_images):
        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)
        return 0.3 * iqa + 0.7 * ava

class SiameseTournamentQuality(ImageQualityStrategy):
    """Run Siamese tournament among top candidates."""
    
    def compute_quality(self, image_path, context, siamese_model, all_images):
        # Get base quality
        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)
        base_quality = 0.3 * iqa + 0.7 * ava
        
        # Get top 4 candidates
        top_4 = self._get_top_n_by_base_quality(all_images, context, n=4)
        
        # If not in top 4, just use base quality
        if image_path not in top_4:
            return base_quality
        
        # Run tournament for top candidates
        siamese_score = self._run_tournament(image_path, top_4, siamese_model)
        
        # Combine
        return 0.4 * base_quality + 0.6 * siamese_score

class SiameseRefinementQuality(ImageQualityStrategy):
    """Refine top candidates with Siamese comparisons."""
    
    def compute_quality(self, image_path, context, siamese_model, all_images):
        # Get base quality
        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)
        base_quality = 0.3 * iqa + 0.7 * ava
        
        # Get top 3 candidates
        top_3 = self._get_top_n_by_base_quality(all_images, context, n=3)
        
        # If not in top 3, just use base quality
        if image_path not in top_3:
            return base_quality
        
        # Compare against best
        best_image = top_3[0]
        if image_path == best_image:
            return base_quality
        
        # Compare
        result = siamese_model.compare_images(image_path, best_image)
        if result['prediction'] == 1:  # Current image is better
            boost = 0.1 * result['confidence']
        else:
            boost = -0.1 * result['confidence']
        
        return base_quality + boost
```

### Configuration

```yaml
select_best:
  # Quality scoring strategy
  quality_strategy: 'weighted_average'  # Options: weighted_average, siamese_tournament, siamese_refinement
  
  # Weighted average config
  quality_weights:
    iqa: 0.3
    ava: 0.7
  
  # Siamese tournament config (if using tournament strategy)
  siamese_tournament:
    top_n: 4
    base_weight: 0.4
    siamese_weight: 0.6
  
  # Siamese refinement config (if using refinement strategy)
  siamese_refinement:
    top_n: 3
    boost_range: 0.1  # ±10% adjustment
```

### Why This is Easy to Change

✅ **Swap strategies via config**: Just change `quality_strategy` setting  
✅ **Add new strategies**: Implement new class, register in factory  
✅ **A/B test**: Run pipeline with different strategies, compare results  
✅ **No code changes**: All tuning via YAML  

### Questions

**Q1**: Which strategy should be the default?
- **Recommendation**: Start with `weighted_average`, add others as options

**Q2**: Should Siamese tournament be cached?
- Pairwise comparisons can be cached and reused
- **Recommendation**: Yes, cache comparison results

**Q3**: For tournament, should we compare all pairs or use elimination?
- All pairs: More accurate, slower (n²)
- Elimination: Faster, less accurate
- **Recommendation**: All pairs for top-4 (only 6 comparisons)

---

## Component 2: Person Penalty

### Definition
**Penalties for portrait-specific issues**. Only applies when person detected.

### Penalty Table

| Condition | Penalty | Rationale |
|-----------|---------|-----------|
| No person detected | 0.0 | No portrait issues |
| Person, but no face | -0.3 | Strong penalty - face expected |
| Body not facing camera | -0.1 | Moderate - turned away |
| Eyes closed | -0.15 | Moderate - important for portraits |
| Not smiling | -0.05 | Light - preference, not quality |
| Face turned away | -0.1 | Moderate - not engaging |

### Formula

```python
if not person_detected:
    person_penalty = 0.0
elif not face_detected:
    person_penalty = -0.3  # Occluded/hidden face
else:
    person_penalty = (
        body_not_facing * -0.1 +
        eyes_closed * -0.15 +
        not_smiling * -0.05 +
        face_turned * -0.1
    )
```

**Range**: [-0.7, 0.0]  
**Meaning**: 0.0 = perfect portrait, negative = issues

### Penalty Accumulation

Penalties are **additive** (cumulative):
- Person facing, eyes open, smiling, frontal face: `0.0`
- Person turned + eyes closed: `-0.1 + -0.15 = -0.25`
- No face: `-0.3` (regardless of other attributes)

### Questions

**Q4**: Are penalty values reasonable?
- Face occlusion: -0.3 (30% reduction)
- Eyes closed: -0.15 (15% reduction)
- Not smiling: -0.05 (5% reduction)
- **Concern**: Do these feel right? Should they be configurable?

**Q5**: Should penalties be multiplicative or additive?
- **Additive** (proposed): `-0.1 + -0.15 = -0.25`
- **Multiplicative**: `score * (1 - 0.1) * (1 - 0.15) = score * 0.765`
- **Recommendation**: Additive (simpler, more predictable)

**Q6**: Maximum penalty cap?
- Should total penalty be capped at -0.5 or -0.7?
- Or allow unlimited stacking?
- **Recommendation**: Cap at -0.7 (prevents extreme negatives)

**Q7**: What if face scoring is unavailable (MediaPipe/InsightFace not run)?
- **Recommendation**: `person_penalty = 0.0` (no penalties without data)

---

## Component 3: Composite Score

### Formula

```python
composite_score = image_quality_score + person_penalty
```

### Examples

#### Example 1: Landscape (No Person)
```
iqa_score = 0.8
ava_score = 0.7
person_detected = False

image_quality_score = 0.3*0.8 + 0.7*0.7 = 0.73
person_penalty = 0.0

composite_score = 0.73 + 0.0 = 0.73
```

#### Example 2: Good Portrait
```
iqa_score = 0.8
ava_score = 0.7
person_detected = True, face_detected = True
eyes_open = True, smiling = True, facing = True

image_quality_score = 0.73
person_penalty = 0.0

composite_score = 0.73 + 0.0 = 0.73
```

#### Example 3: Portrait, Eyes Closed
```
iqa_score = 0.8
ava_score = 0.7
person_detected = True, face_detected = True
eyes_open = False, smiling = True, facing = True

image_quality_score = 0.73
person_penalty = -0.15

composite_score = 0.73 - 0.15 = 0.58
```

#### Example 4: Portrait, Face Occluded
```
iqa_score = 0.8
ava_score = 0.7
person_detected = True, face_detected = False

image_quality_score = 0.73
person_penalty = -0.3

composite_score = 0.73 - 0.3 = 0.43
```

#### Example 5: Poor Quality Portrait
```
iqa_score = 0.3
ava_score = 0.4
person_detected = True, face_detected = True
eyes_open = False, smiling = False, body_turned = True

image_quality_score = 0.3*0.3 + 0.7*0.4 = 0.37
person_penalty = -0.15 - 0.05 - 0.1 = -0.3

composite_score = 0.37 - 0.3 = 0.07  ← Very low!
```

### Range

**Theoretical**: [-0.7, 1.0]  
- Best possible: High quality (1.0) + no penalties (0.0) = 1.0
- Worst possible: Low quality (0.0) + max penalties (-0.7) = -0.7

**Practical**: [0.0, 1.0]
- Most images will be in this range
- Negative scores are rare (terrible quality + major issues)

---

## Component 4: Cluster Splitting by Identity

### Terminology Clarification

**OLD**: "Subclusters" (confusing)  
**NEW**: "Split clusters" or "identity-based clusters"

### How It Works

**Before splitting**:
- Cluster 1: 10 images (8 of Person A, 2 of Person B)

**After splitting**:
- Cluster 1A: 8 images (Person A only)
- Cluster 1B: 2 images (Person B only)

**Selection**: Choose best from 1A and best from 1B **independently** (they don't compete)

### Why Split?

Without splitting, Person A would dominate selection (8 out of 10 images). Splitting ensures both people are represented.

### Implementation

```python
def split_cluster_by_identity(cluster_images, context):
    """Split cluster into sub-clusters by person identity."""
    
    # Group images by face embedding similarity
    identity_clusters = {}
    
    for image in cluster_images:
        # Get face embedding
        face_embedding = context.face_embeddings.get(image)
        
        if face_embedding is None:
            # No face detected, goes to "no_person" cluster
            identity_clusters.setdefault('no_person', []).append(image)
            continue
        
        # Find matching identity cluster
        matched = False
        for identity_id, members in identity_clusters.items():
            if identity_id == 'no_person':
                continue
            
            # Compare with representative image
            representative = members[0]
            rep_embedding = context.face_embeddings.get(representative)
            
            # Check similarity
            similarity = cosine_similarity(face_embedding, rep_embedding)
            if similarity > 0.6:  # Same person threshold
                identity_clusters[identity_id].append(image)
                matched = True
                break
        
        if not matched:
            # Create new identity cluster
            new_id = f"person_{len(identity_clusters)}"
            identity_clusters[new_id] = [image]
    
    return identity_clusters
```

### Result

Each identity cluster is processed **independently**:
```python
for scene_cluster_id, images in scene_clusters.items():
    # Split by identity
    identity_clusters = split_cluster_by_identity(images, context)
    
    # Process each identity cluster separately
    for identity_id, identity_images in identity_clusters.items():
        selected = select_best_from_cluster(identity_images, max_per_cluster=2)
        all_selected.extend(selected)
```

**Key Point**: Person A's 8 images compete only with each other. Person B's 2 images compete only with each other. Fair representation guaranteed!

---

## Component 5: Selection Logic

### Step 1: Split Cluster by Identity (if applicable)

```python
identity_clusters = split_cluster_by_identity(images, context)
```

### Step 2: For Each Identity Cluster

Process independently:

#### 2.1 Compute Composite Scores

```python
scored = [
    (img, image_quality(img) + person_penalty(img))
    for img in identity_images
]
```

#### 2.2 Filter Low Quality

```python
MIN_THRESHOLD = 0.4
filtered = [(img, score) for img, score in scored if score >= MIN_THRESHOLD]

# If all below threshold, keep best one anyway
if not filtered and scored:
    filtered = [max(scored, key=lambda x: x[1])]
```

#### 2.3 Sort by Composite Score

```python
sorted_images = sorted(filtered, key=lambda x: x[1], reverse=True)
```

#### 2.4 Select Best + Dissimilar

```python
selected = []
MAX_PER_CLUSTER = 2
DISSIMILARITY_THRESHOLD = 0.85

# Always select #1
selected.append(sorted_images[0][0])

# Select additional dissimilar images
for img, score in sorted_images[1:]:
    if len(selected) >= MAX_PER_CLUSTER:
        break
    
    # Check dissimilarity against all selected
    is_dissimilar = all(
        embedding_similarity(img, s) < DISSIMILARITY_THRESHOLD
        for s in selected
    )
    
    if is_dissimilar:
        selected.append(img)
```

### Questions

**Q8**: Should we have different thresholds for face vs non-face clusters?
- **Current proposal**: Same threshold for all (0.4)
- **Alternative**: Lower threshold for portraits (0.3) since penalties reduce scores
- **Recommendation**: Same threshold (simpler), adjust penalty values instead

**Q9**: Should max_per_cluster depend on cluster size?
- Small cluster (2-3 images): max = 1
- Medium cluster (4-10 images): max = 2
- Large cluster (10+ images): max = 3
- **Recommendation**: Keep fixed at 2 (simpler), but make configurable

**Q10**: Identity similarity threshold?
- Current: 0.6 (ArcFace distance)
- Should this be configurable?
- **Recommendation**: Yes, make configurable (different for family events vs strangers)

**Q11**: What if all images below threshold?
- **Recommendation**: Select best one anyway (user expects at least one photo per cluster)
- This ensures every identity gets at least one photo

---

## Implementation Changes

### Simplified Architecture

```python
class SelectBestStep:
    def _select_from_cluster(self, context, images, max_per_cluster, siamese_model):
        # 1. Compute composite scores
        scored = [
            (img, self._compute_composite_score(img, context))
            for img in images
        ]
        
        # 2. Filter low quality
        filtered = [(img, score) for img, score in scored if score >= self._config['min_threshold']]
        
        # 3. Sort by score
        sorted_images = sorted(filtered, key=lambda x: x[1], reverse=True)
        
        # 4. Select best + dissimilar
        selected = self._select_dissimilar(sorted_images, max_per_cluster)
        
        # 5. Apply Siamese tiebreaker if needed
        if siamese_model:
            selected = self._apply_siamese_tiebreaker(selected, siamese_model)
        
        return selected
    
    def _compute_composite_score(self, image_path, context):
        # Compute image quality
        quality = self._compute_image_quality(image_path, context)
        
        # Compute person penalty
        penalty = self._compute_person_penalty(image_path, context)
        
        return quality + penalty
    
    def _compute_image_quality(self, image_path, context):
        iqa = context.iqa_scores.get(image_path, 0.5)
        ava = context.ava_scores.get(image_path, 0.5)
        
        weights = self._config.get('quality_weights', {'iqa': 0.3, 'ava': 0.7})
        return weights['iqa'] * iqa + weights['ava'] * ava
    
    def _compute_person_penalty(self, image_path, context):
        persons = context.persons.get(image_path, {})
        
        if not persons.get('person_detected'):
            return 0.0
        
        penalties = self._config.get('person_penalties', {
            'face_occlusion': -0.3,
            'body_not_facing': -0.1,
            'eyes_closed': -0.15,
            'not_smiling': -0.05,
            'face_turned': -0.1
        })
        
        # Check face visibility
        faces = context.face_eyes_scores.get(f"{image_path}:face_0")
        if not faces:
            return penalties['face_occlusion']
        
        # Accumulate penalties
        total_penalty = 0.0
        
        # Body orientation
        body_facing = persons.get('body_facing_score', 1.0)
        if body_facing < 0.5:
            total_penalty += penalties['body_not_facing']
        
        # Eyes
        eyes_score = context.face_eyes_scores.get(f"{image_path}:face_0", 1.0)
        if eyes_score < 0.5:
            total_penalty += penalties['eyes_closed']
        
        # Smile
        smile_score = context.face_smile_scores.get(f"{image_path}:face_0", 1.0)
        if smile_score < 0.5:
            total_penalty += penalties['not_smiling']
        
        # Face pose
        pose_score = context.face_pose_scores.get(f"{image_path}:face_0", 1.0)
        if pose_score < 0.5:
            total_penalty += penalties['face_turned']
        
        # Cap total penalty
        max_penalty = self._config.get('max_penalty', -0.7)
        return max(total_penalty, max_penalty)
```

### Configuration

```yaml
select_best:
  max_per_cluster: 2
  min_threshold: 0.4
  
  # Image quality weights
  quality_weights:
    iqa: 0.3
    ava: 0.7
  
  # Person penalties
  person_penalties:
    face_occlusion: -0.3
    body_not_facing: -0.1
    eyes_closed: -0.15
    not_smiling: -0.05
    face_turned: -0.1
  
  max_penalty: -0.7
  
  # Dissimilarity
  dissimilarity_threshold: 0.85
  
  # Siamese tiebreaker
  siamese:
    enabled: true
    tiebreaker_threshold: 0.05
    checkpoint_path: models/siamese.pt
```

---

## Benefits of New Design

### 1. Clarity
- ✅ Clear what each component does
- ✅ Quality vs preferences separated
- ✅ Penalties are explicit and tunable

### 2. Simplicity
- ✅ No branching logic (face vs non-face)
- ✅ Same formula for all images
- ✅ Easy to understand and explain

### 3. Extensibility
- ✅ Add new quality metrics: Just update quality formula
- ✅ Add new penalties: Just add to penalty table
- ✅ A/B test: Change config, no code changes

### 4. Tunability
- ✅ All weights/penalties in config
- ✅ Can adjust per use-case (weddings vs vacation)
- ✅ Can learn optimal values from user feedback

### 5. Testability
- ✅ Each component independently testable
- ✅ Clear expectations for edge cases
- ✅ Easier to debug issues

---

## Migration Path

### Phase 1: Add New Scoring (Keep Old)
- Implement new `_compute_composite_score()` method
- Keep existing `_score_face_cluster()` and `_score_non_face_cluster()`
- Add config flag: `scoring_method: 'legacy'` or `'additive'`

### Phase 2: Test & Compare
- Run both methods in parallel
- Compare selections
- Tune penalty values based on results

### Phase 3: Switch Default
- Change default to `scoring_method: 'additive'`
- Keep legacy as fallback

### Phase 4: Remove Legacy
- Delete old scoring methods
- Clean up code

---

## Questions for Discussion

### Critical Questions (Need Answers)

1. **Quality strategy**: Which should be default?
   - Weighted average (simple)
   - Siamese tournament (accurate but slower)
   - Siamese refinement (hybrid)
   - **USER INPUT**: You mentioned AVA+IQA aren't sufficient. Should we default to one of the Siamese strategies?

2. **Penalty values**: Are the proposed penalties (-0.3, -0.15, -0.1, -0.05) reasonable?

3. **Minimum threshold**: Is 0.4 a good cutoff for "too low"?

4. **Identity threshold**: Is 0.6 face embedding similarity good for "same person"?

5. **Penalty accumulation**: Should it be additive or multiplicative?
   - **ANSWER**: Additive (user confirmed)

### Clarified Questions

6. **Cluster splitting**: ✅ Split scene clusters by identity, process independently (user confirmed)

7. **Quality scoring flexibility**: ✅ Use strategy pattern, make easy to change (user confirmed)

### Nice-to-Have Questions

8. Should quality weights differ for portraits vs landscapes?

9. Should penalties be context-aware? (e.g., candid shots allow not smiling)

10. Should we add positive bonuses? (e.g., +0.1 for particularly good composition)

11. How to handle missing data? (e.g., no AVA score)

---

## Recommendation

**Proceed with additive model** with these parameters:

- **Quality**: 30% IQA + 70% AVA
- **Penalties**: As specified in table
- **Threshold**: 0.4 minimum composite score
- **Accumulation**: Additive with -0.7 cap
- **All values configurable** in YAML

Implement in phases to allow testing and comparison with legacy method.

**Do you agree? Any concerns or changes?**
