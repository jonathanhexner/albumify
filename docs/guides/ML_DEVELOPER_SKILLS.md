# ML Developer Skills & Best Practices

Lessons learned from face clustering pipeline development. These principles apply to any ML pipeline.

---

## Core Principle: Benchmark Everything, Trust Nothing

> "If you can't measure it, you can't improve it."

Every ML pipeline step should be:
1. **Measurable** - Have clear metrics
2. **Reproducible** - Same input → same output
3. **Debuggable** - Can inspect intermediate results
4. **Comparable** - Can compare against baselines

---

## 1. Pipeline Transparency

### Save Intermediate Results
For every transformation step, save:
- **Input** - What went in
- **Output** - What came out
- **Parameters** - What settings were used
- **Metrics** - How good was the result

**Example from face pipeline:**
```
Face Detection → Raw Crop → Aligned Crop → Embedding → Clustering
     ↓              ↓           ↓            ↓           ↓
  bbox.json    raw_crop/   aligned_crop/  embeddings.npy  labels.json
  landmarks    face_XXX.jpg face_XXX.jpg               + debug_data
```

### Debug Panels
Build visualizations that show all pipeline stages side-by-side:
- Original image with annotations
- Each intermediate transformation
- Final output

This catches bugs like coordinate system mismatches immediately.

---

## 2. Coordinate System Discipline

**Lesson learned:** Landmark-face alignment mismatch occurred because:
- Face crops used 5-point affine transform (landmarks → reference template)
- Displayed landmarks were original pixel coordinates

### Best Practice
Document coordinate systems explicitly:
```python
# Pixel coordinates (absolute, from top-left of image)
landmarks_px: List[Tuple[int, int]]  # e.g., [(234, 156), (298, 154), ...]

# Normalized coordinates (0-1 range, relative to bbox or image)
landmarks_norm: List[Tuple[float, float]]  # e.g., [(0.34, 0.46), ...]

# Reference template coordinates (fixed positions after alignment)
ARCFACE_REF_112 = [(38.29, 51.70), (73.53, 51.50), ...]  # For 112x112
```

When transforming, always track which coordinate system you're in.

---

## 3. Algorithm Documentation Standard

Every ML algorithm should have:

### doc_explanation (5-6 lines)
```python
doc_explanation = """
HDBSCAN finds clusters based on density without requiring k.
It builds a hierarchy of clusters and extracts the most stable ones.
Decision: A point becomes noise if it lacks sufficient nearby neighbors.
Key insight: Works well for clusters of varying density.
Threshold: Points with low local density become noise (-1 label).
"""
```

### decision_parameters dict
```python
decision_parameters = {
    "min_cluster_size": {
        "description": "Minimum points to form a cluster",
        "default": 5,
        "decision_role": "Clusters smaller than this become noise"
    },
    "threshold_floor": {
        "description": "Minimum allowed merge threshold",
        "default": 0.125,
        "decision_role": "Prevents over-splitting tight clusters"
    }
}
```

### get_decision_info() method
Returns current parameter values + last run statistics for debugging.

---

## 4. Benchmark Framework

### Required Components

```
benchmarks/
├── configs/           # Reproducible experiment configs
│   ├── baseline.yaml
│   └── experiment_v2.yaml
├── results/           # Timestamped results
│   ├── 2026-02-19_baseline/
│   │   ├── metrics.json
│   │   ├── predictions.npy
│   │   └── debug_data.json
│   └── 2026-02-19_experiment_v2/
└── compare.py         # Comparison tool
```

### Benchmark Output Format
```json
{
  "timestamp": "2026-02-19T14:30:00",
  "config": { /* full config snapshot */ },
  "git_commit": "abc123",
  "metrics": {
    "precision": 0.85,
    "recall": 0.82,
    "f1": 0.835,
    "n_clusters": 42,
    "noise_ratio": 0.15
  },
  "timing": {
    "total_seconds": 45.2,
    "step_timings": {
      "detection": 12.1,
      "embedding": 28.4,
      "clustering": 4.7
    }
  },
  "debug_data": { /* detailed per-decision data */ }
}
```

### Comparison Checklist
When comparing two runs:
- [ ] Same dataset (or controlled split)
- [ ] Same preprocessing
- [ ] Config diff documented
- [ ] Statistical significance (if applicable)
- [ ] Failure case analysis

---

## 5. Reproducibility Checklist

Before claiming "this works":

- [ ] **Seed fixed** - `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`
- [ ] **Config saved** - Full config in results folder
- [ ] **Git commit recorded** - Know exact code version
- [ ] **Dependencies pinned** - `requirements.txt` with versions
- [ ] **Data versioned** - Hash or version of input data
- [ ] **Run twice** - Verify same results

---

## 6. Debugging Workflow

When something looks wrong:

### Step 1: Isolate
```python
# Save intermediate result
np.save("debug_step3_embeddings.npy", embeddings)
# Check shape, range, NaNs
print(f"Shape: {embeddings.shape}, Range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
print(f"NaNs: {np.isnan(embeddings).sum()}, Zeros: {(embeddings == 0).all(axis=1).sum()}")
```

### Step 2: Visualize
Build debug panels showing:
- Input vs output
- Distribution histograms
- Sample cases (good and bad)

### Step 3: Trace Back
Start from the bad output and trace backwards:
- Which input produced this bad output?
- At which step did it go wrong?
- What was different about this input?

### Step 4: Hypothesis Test
```python
# Don't just fix - verify the fix
# Before: 15% zero-vector embeddings
# Hypothesis: Alignment was cutting off faces
# Test: Compare aligned vs raw crops
assert zero_vector_rate < 0.01, f"Still have {zero_vector_rate:.1%} zero vectors"
```

---

## 7. Testing ML Code

### Unit Tests for Transforms
```python
def test_5point_alignment_identity():
    """Reference landmarks should map to themselves."""
    ref_landmarks = ARCFACE_REF_POINTS_112 * 2  # Scale to 224
    aligned = align_face_5point(test_image, ref_landmarks, target_size=224)
    # Output landmarks should be at reference positions
    assert_landmarks_at_reference(aligned)

def test_alignment_preserves_identity():
    """Same face, different poses should have similar embeddings."""
    emb1 = extract_embedding(face_frontal)
    emb2 = extract_embedding(face_tilted)
    similarity = cosine_similarity(emb1, emb2)
    assert similarity > 0.7, f"Same person similarity too low: {similarity}"
```

### Regression Tests
```python
def test_clustering_matches_baseline():
    """Clustering should match saved baseline results."""
    result = run_clustering(test_embeddings, config)
    baseline = load_baseline("clustering_v2_baseline.json")

    # Allow small differences but catch major regressions
    assert abs(result.n_clusters - baseline.n_clusters) <= 2
    assert abs(result.noise_ratio - baseline.noise_ratio) < 0.05
```

---

## 8. Quick Reference: Debug Checklist

When ML pipeline produces unexpected results:

1. **Check data flow**
   - Are inputs in expected format/range?
   - Any NaN/Inf/zero values?
   - Correct batch dimensions?

2. **Check coordinate systems**
   - Pixel vs normalized?
   - Which reference frame?
   - Transformation applied correctly?

3. **Check parameters**
   - Using intended config?
   - Defaults overridden correctly?
   - Thresholds in sensible range?

4. **Check caching**
   - Stale cached results?
   - Cache key includes all relevant params?
   - Clear cache and re-run?

5. **Visualize**
   - Plot intermediate results
   - Compare good vs bad cases
   - Overlay predictions on inputs

---

## Applied Example: Face Clustering Debug

From recent work, we built:

| Debug Tool | Purpose |
|------------|---------|
| 3-version panel | See original→raw→aligned at each step |
| Landmark overlay | Verify coordinate transforms |
| Decision cards | See merge/attach thresholds vs actual values |
| Algorithm docs | Understand what each method does |
| Benchmark comparison | Compare methods on same data |

This caught:
- 2-point vs 5-point alignment mismatch
- Landmark coordinate system confusion
- Stale cached embeddings
- Parameter documentation gaps

---

## Summary

1. **Save everything** - Inputs, outputs, configs, metrics
2. **Visualize stages** - Debug panels for each transform
3. **Document decisions** - What thresholds, why those values
4. **Test reproducibility** - Same input → same output
5. **Compare baselines** - Know if changes help or hurt
6. **Trace failures** - Follow bad outputs back to root cause
