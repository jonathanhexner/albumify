# select_best.py Issues and Required Fixes

## Issues Identified

### Issue 1: Silent Score Normalization (CRITICAL)

**Location**: Lines 464-466 and 519-521

**Problem**:
```python
ava_score = context.ava_scores.get(path, 5.0)
if ava_score > 1.0:  # AVA is typically 1-10 scale
    ava_score = ava_score / 10.0  # ❌ SILENTLY MODIFYING SCORES!
```

**Why This is Wrong**:
- Silently changing scores without user knowledge
- Making assumptions about score ranges
- If AVA step already normalizes, this breaks it
- Inconsistent behavior depending on upstream

**Fix**:
```python
# Get AVA score (expected to be 0-1 normalized by score_ava step)
ava_score = context.ava_scores.get(path, 0.5)

# Warn if score looks unnormalized
if ava_score > 1.0:
    logger.warning(
        f"AVA score for {Path(path).name} is {ava_score:.2f} "
        f"(expected 0-1 range). Check if score_ava step is normalizing correctly."
    )
```

**Action**: Remove normalization, add warning, fix default from 5.0 to 0.5

---

### Issue 2: Parameter Passing Anti-Pattern

**Location**: Lines 167-178, 197-208

**Problem**:
```python
cluster_selected = self._select_from_cluster(
    context=context,
    image_paths=images,
    has_faces=has_faces,
    max_per_cluster=max_per_cluster,
    min_score_threshold=min_score_threshold,  # ❌
    max_score_gap=max_score_gap,              # ❌
    tiebreaker_threshold=tiebreaker_threshold,# ❌
    duplicate_threshold=duplicate_threshold,  # ❌
    face_weights=face_weights,                # ❌
    siamese_model=siamese_model
)
```

**Why This is Wrong**:
- Passing 11 parameters is ridiculous
- Parameters are just config values, not computed data
- Violates "don't pass what you can access" principle
- Makes code harder to read and maintain

**Fix**:
```python
class SelectBestStep(BaseStep):
    def __init__(self):
        # ...
        self._config = None  # Store config here
    
    def process(self, context: PipelineContext, config: dict):
        self._config = config  # Store for use in methods
        # ...
        
        cluster_selected = self._select_from_cluster(
            context=context,
            image_paths=images,
            has_faces=has_faces,
            max_per_cluster=max_per_cluster,
            siamese_model=siamese_model  # Only pass computed/loaded things
        )
    
    def _select_from_cluster(self, context, image_paths, has_faces, max_per_cluster, siamese_model):
        # Get config values locally when needed
        min_score_threshold = self._config.get("min_score_threshold", 0.4)
        max_score_gap = self._config.get("max_score_gap", 0.25)
        # ... etc
```

**Action**: Store config as instance variable, only pass non-config parameters

---

### Issue 3: Misleading Siamese Log Message

**Location**: Lines 143-147

**Problem**:
```python
if siamese_model:
    logger.info("Siamese CNN enabled for comparison and duplicate detection")
    #                                                   ^^^^^^^^^^^^^^^^^^^ ❌ WRONG!
```

**Why This is Wrong**:
The code at lines 388-391 explicitly states:

```python
def _check_near_duplicate(...):
    """
    NOTE: Siamese CNN is NOT used for duplicate detection because it compares
    image QUALITY (which is better), not image SIMILARITY (are they the same).
    Low Siamese confidence means it can't tell which is better quality,
    not that they are duplicates.
    """
    # Always use embedding similarity for duplicate detection  ← NOT SIAMESE!
    similarity = self._get_embedding_similarity(context, path1, path2)
```

**Siamese CNN is used for**:
- ✅ **Quality tiebreaking** - When two images have similar scores, determine which is higher quality

**Siamese CNN is NOT used for**:
- ❌ **Duplicate detection** - Uses embedding similarity (DINOv2/OpenCLIP) instead

**Fix**:
```python
if siamese_model:
    logger.info("Siamese CNN enabled for quality tiebreaking")
    logger.debug(f"Loaded from: {siamese_checkpoint}")
else:
    logger.info("Siamese CNN not available - will not perform quality tiebreaking")
```

**Action**: Fix log message to accurately reflect what Siamese does

---

### Issue 4: What is `self._siamese_checkpoint`?

**Location**: Lines 102-103, 115-119

**Question**: "what is self._siamese_checkpoint???!"

**Answer**: It's a **cache variable** to avoid reloading the Siamese model:

```python
def __init__(self):
    self._siamese_model = None
    self._siamese_checkpoint = None  # Tracks which checkpoint is loaded

def _get_siamese_model(self, checkpoint_path: str):
    # Only reload if checkpoint path changes
    if self._siamese_model is None or self._siamese_checkpoint != checkpoint_path:
        self._siamese_model = SiameseQualityModel(checkpoint, device='cpu')
        self._siamese_checkpoint = checkpoint_path  # Remember what we loaded
    return self._siamese_model
```

**Purpose**: Avoid reloading the same model multiple times if `process()` is called repeatedly

**Is this needed?**: Debatable. The step is typically instantiated once per pipeline run, so this optimization may be unnecessary. Could simplify to just reload each time.

---

## Summary of Required Fixes

### 1. Remove Silent Normalization

```diff
- ava_score = context.ava_scores.get(path, 5.0)
- if ava_score > 1.0:
-     ava_score = ava_score / 10.0
+ ava_score = context.ava_scores.get(path, 0.5)
+ if ava_score > 1.0:
+     logger.warning(f"AVA score for {Path(path).name} is {ava_score:.2f} (expected 0-1)")
```

**Impact**: Requires `score_ava` step to output normalized scores (0-1 range)

### 2. Store Config as Instance Variable

```diff
  def __init__(self):
      self._siamese_model = None
      self._siamese_checkpoint = None
+     self._config = None

  def process(self, context, config):
+     self._config = config
-     min_score_threshold = config.get(...)
-     max_score_gap = config.get(...)
      # ... extract only siamese_model, not all config values

  def _select_from_cluster(self, context, image_paths, ...):
      # Remove config parameters from signature
+     min_score_threshold = self._config.get("min_score_threshold", 0.4)
+     max_score_gap = self._config.get("max_score_gap", 0.25)
```

**Impact**: Cleaner method signatures, easier to maintain

### 3. Fix Misleading Log

```diff
  if siamese_model:
-     logger.info("Siamese CNN enabled for comparison and duplicate detection")
+     logger.info("Siamese CNN enabled for quality tiebreaking")
  else:
-     logger.info("Siamese CNN not available, using embedding similarity")
+     logger.info("Siamese CNN not available")
```

**Impact**: Accurate information in logs

---

## Correct Understanding of Siamese Usage

### Siamese CNN Purpose

**What it does**: Compares two images and predicts which one is **higher quality**

**Output**:
```python
{
    'prediction': 1,      # 1 = img1 is better, 0 = img2 is better
    'confidence': 0.89    # How confident (0-1)
}
```

### Two Use Cases in select_best

#### 1. Quality Tiebreaker (Lines 306-376)

**When**: Top images have very similar scores (within `tiebreaker_threshold`, default 0.05)

**Why**: Composite scores might be close, but Siamese can distinguish quality differences

**Example**:
```
Image A: composite_score = 0.752
Image B: composite_score = 0.748  (diff = 0.004 < 0.05 threshold)

→ Use Siamese to determine which is actually better
→ Siamese says B is better quality with 85% confidence
→ Promote B to #1 position
```

#### 2. Duplicate Detection (Lines 377-414)

**Method**: Uses **embedding similarity** (NOT Siamese!)

**Why**: Siamese compares quality, not similarity. Two identical photos might have different quality scores (one is sharper).

**Example**:
```
Image A and Image B are near-identical (same moment, slight camera movement)

Siamese: "A is slightly better quality" ← Not useful for duplicate detection
Embedding: "Cosine similarity = 0.93 > 0.85 threshold" ← Correct method
```

---

## Testing After Fixes

### 1. Test Score Range Handling

```python
# Verify AVA scores are 0-1
for path, score in context.ava_scores.items():
    assert 0 <= score <= 1, f"AVA score out of range: {score}"
```

### 2. Test Config Access

```python
# Verify config is accessible in all methods
step = SelectBestStep()
context = PipelineContext()
config = {"min_score_threshold": 0.6}
step.process(context, config)
# Should not crash when accessing self._config in submethods
```

### 3. Test Siamese Behavior

```python
# With Siamese enabled
config = {
    "siamese": {
        "enabled": True,
        "checkpoint_path": "models/siamese.pt",
        "tiebreaker_range": 0.05
    }
}
# Check logs show correct message
# Verify tiebreaker logic activates when scores are close
```

---

## Recommendation

Apply all three fixes:

1. ✅ Remove silent normalization → Add warning
2. ✅ Store config in instance variable → Simplify method signatures  
3. ✅ Fix Siamese log message → Accurate information

This will make the code:
- More honest (no silent modifications)
- More maintainable (less parameter passing)
- More clear (accurate logging)
