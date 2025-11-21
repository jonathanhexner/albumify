# Where Aggregation Happens: Code Path

## Call Stack

### 1. Entry Point: Pairwise Evaluation
**File**: `sim_bench/quality_assessment/pairwise_evaluator.py`
**Line**: 133-134

```python
# Assess both images
score_a = self.method.assess_image(image_a_path)  # <-- Entry point
score_b = self.method.assess_image(image_b_path)
```

### 2. CLIP Aesthetic Wrapper
**File**: `sim_bench/quality_assessment/clip_aesthetic.py`
**Line**: 210-231

```python
def assess_image(self, image_path: str) -> float:
    # Check cache
    if self.enable_cache and image_path in self._score_cache:
        return self._score_cache[image_path]
    
    # Assess using underlying aesthetic assessor
    score = self.aesthetic_assessor.assess_image(image_path)  # <-- Delegates
    
    # Cache result
    if self.enable_cache:
        self._score_cache[image_path] = score
    
    return score
```

### 3. Aesthetic Assessor: Single Image
**File**: `sim_bench/vision_language/applications/aesthetic.py`
**Line**: 106-116

```python
def assess_image(self, image_path: str) -> float:
    """
    Assess aesthetic quality of single image.
    
    Returns:
        Aesthetic score (higher = better quality)
    """
    return self.assess_batch([image_path])[0]  # <-- Calls batch version
```

### 4. Aesthetic Assessor: Batch Processing
**File**: `sim_bench/vision_language/applications/aesthetic.py`
**Line**: 118-145

```python
def assess_batch(self, image_paths: List[str]) -> np.ndarray:
    """
    Assess aesthetic quality of multiple images.
    
    Returns:
        Array of aesthetic scores [n_images]
    """
    # Encode images
    image_embs = self.model.encode_images(image_paths)  # <-- Step 1: Encode images
    
    # Compute similarities with all prompts
    similarities = self.model.compute_similarity(  # <-- Step 2: Compute similarities
        image_embs,
        self.prompt_embeddings
    )
    # similarities shape: [n_images, n_prompts]
    # For learned variant: [1, 18] (1 image, 18 prompts)
    
    # Aggregate based on method
    if self.aggregation == "contrastive_only":
        scores = self._aggregate_contrastive(similarities)  # <-- AGGREGATION HERE
    elif self.aggregation == "weighted":
        scores = self._aggregate_weighted(similarities)  # <-- AGGREGATION HERE (default)
    else:  # mean
        scores = np.mean(similarities, axis=1)  # <-- AGGREGATION HERE
    
    return scores  # <-- Returns [n_images] array, e.g., [0.25]
```

### 5. Weighted Aggregation (Default Method)
**File**: `sim_bench/vision_language/applications/aesthetic.py`
**Line**: 212-228

```python
def _aggregate_weighted(self, similarities: np.ndarray) -> np.ndarray:
    """
    Weighted aggregation of all prompts.
    
    Args:
        similarities: [n_images, n_prompts] array
                     For learned variant: [1, 18]
    
    Returns:
        [n_images] array of final scores
    """
    # Contrastive component (50% weight)
    contrastive = self._aggregate_contrastive(similarities)
    # For learned variant with 9 pairs:
    #   - similarities[:, 0:2] = pair 1 (pos, neg)
    #   - similarities[:, 2:4] = pair 2 (pos, neg)
    #   - ...
    #   - similarities[:, 16:18] = pair 9 (pos, neg)
    # Returns: mean of 9 contrastive scores
    
    # Positive component (30% weight)
    start_idx = self.n_contrasts * 2  # 9 * 2 = 18 (for learned variant)
    end_idx = start_idx + self.n_positive  # 18 + 0 = 18 (no positive in learned)
    positive = np.mean(similarities[:, start_idx:end_idx], axis=1)
    # For learned variant: empty slice, returns 0
    
    # Negative component (20% weight) - inverted
    start_idx = end_idx  # 18
    end_idx = start_idx + self.n_negative  # 18 + 0 = 18 (no negative in learned)
    negative = -np.mean(similarities[:, start_idx:end_idx], axis=1)
    # For learned variant: empty slice, returns 0
    
    # Weighted combination
    return 0.5 * contrastive + 0.3 * positive + 0.2 * negative
    # For learned variant: 0.5 * contrastive + 0 + 0
```

### 6. Contrastive Aggregation (Sub-component)
**File**: `sim_bench/vision_language/applications/aesthetic.py`
**Line**: 201-210

```python
def _aggregate_contrastive(self, similarities: np.ndarray) -> np.ndarray:
    """
    Aggregate using only contrastive pairs.
    
    Args:
        similarities: [n_images, n_prompts] array
                     For learned variant: [1, 18]
    
    Returns:
        [n_images] array of contrastive scores
    """
    contrastive_scores = []
    
    for i in range(self.n_contrasts):  # 9 iterations for learned variant
        pos_sim = similarities[:, i*2]      # Even indices: positive prompts
        neg_sim = similarities[:, i*2 + 1]  # Odd indices: negative prompts
        contrastive_scores.append(pos_sim - neg_sim)
        # For each pair: [similarity(pos) - similarity(neg)]
    
    return np.mean(contrastive_scores, axis=0)
    # Returns: mean of 9 contrastive scores
    # Shape: [n_images], e.g., [0.50]
```

## Data Flow Example

For a single image with learned variant (9 pairs, 18 prompts):

```
1. Image encoding:
   image.jpg → CLIP encoder → [512-dim vector]

2. Prompt encoding (done once at initialization):
   18 prompts → CLIP encoder → [18, 512-dim] matrix

3. Similarity computation:
   cosine_similarity(image_emb, prompt_embs) → [18] array
   Example: [0.65, 0.20, 0.70, 0.15, ..., 0.60, 0.25]

4. Contrastive aggregation (in _aggregate_contrastive):
   Pair 1: 0.65 - 0.20 = 0.45
   Pair 2: 0.70 - 0.15 = 0.55
   ...
   Pair 9: 0.60 - 0.25 = 0.35
   Mean: (0.45 + 0.55 + ... + 0.35) / 9 = 0.50

5. Weighted aggregation (in _aggregate_weighted):
   contrastive = 0.50
   positive = 0 (no positive attributes in learned variant)
   negative = 0 (no negative attributes in learned variant)
   final = 0.5 * 0.50 + 0.3 * 0 + 0.2 * 0 = 0.25

6. Return:
   score_a = 0.25
```

## Key Files and Line Numbers

| Step | File | Line | Function |
|------|------|------|----------|
| Entry | `pairwise_evaluator.py` | 133 | `score_a = self.method.assess_image(...)` |
| Wrapper | `clip_aesthetic.py` | 225 | `self.aesthetic_assessor.assess_image(...)` |
| Single→Batch | `aesthetic.py` | 116 | `assess_batch([image_path])[0]` |
| Encode & Similarity | `aesthetic.py` | 129-135 | `encode_images()` + `compute_similarity()` |
| **AGGREGATION** | `aesthetic.py` | 138-143 | `_aggregate_weighted(similarities)` |
| Weighted Formula | `aesthetic.py` | 228 | `0.5 * contrastive + 0.3 * positive + 0.2 * negative` |
| Contrastive Sub | `aesthetic.py` | 201-210 | `_aggregate_contrastive()` |

## Summary

**Aggregation happens in**:
- **File**: `sim_bench/vision_language/applications/aesthetic.py`
- **Function**: `_aggregate_weighted()` (line 212-228)
- **Called from**: `assess_batch()` (line 141)
- **Input**: `similarities` array [n_images, n_prompts]
- **Output**: Single score per image [n_images]

The aggregation combines all prompt similarities into one number using weighted averaging.

