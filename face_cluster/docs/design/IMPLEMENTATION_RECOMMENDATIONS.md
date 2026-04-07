# Implementation Recommendations - Face Clustering
**Date:** 2026-03-27
**Source:** Expert review panel (CV Researcher + SW Engineer + UI Expert)

---

## ✅ Architecture Approved

**Verdict:** Solid foundation, ready to implement with Priority 1 changes.

---

## 🎯 Priority 1: Must Implement Before Testing

### 1. Validation Checks
Add to each pipeline step:

```python
# After quality gate
assert len(context.core_indices) > 0, \
    "No faces passed quality gate - check thresholds"

# After embedding extraction
assert all(0.9 < np.linalg.norm(emb) < 1.1 for emb in context.embeddings), \
    "Embeddings not normalized - check embedding model"

# After clustering
assert len(context.faces) == len(context.embeddings), \
    f"Face/embedding mismatch: {len(context.faces)} vs {len(context.embeddings)}"
```

### 2. Distance Metric Specification
**Document in architecture:** "Cosine distance on L2-normalized 512-dim embeddings"

**Add to PipelineConfig:**
```yaml
extract_face_embeddings:
  backend: 'insightface'
  normalize: true  # Enforce L2 normalization
  verify_norm: true  # Check norm ≈ 1.0
```

### 3. Logging with Timing
```python
import time

start = time.time()
# ... stage processing ...
duration = time.time() - start

logger.info(f"Stage completed: {self.name}", extra={
    "duration_sec": round(duration, 2),
    "n_faces": len(context.faces),
    "n_clusters": context.initial_clusters.n_clusters
})
```

### 4. UI: Color-Coded Distances
```python
def distance_color(distance: float) -> str:
    if distance < 0.25:
        return "🟢 green"   # High confidence same person
    elif distance < 0.40:
        return "🟡 yellow"  # Uncertain
    else:
        return "🔴 red"     # High confidence different person
```

### 5. UI: Progress Tracking
```python
# In labeling tab
n_labeled = sum(1 for c in clusters if c.label is not None)
n_total = len(clusters)
st.progress(n_labeled / n_total)
st.write(f"Progress: {n_labeled}/{n_total} clusters labeled ({n_labeled/n_total*100:.0f}%)")
```

---

## 🎯 Priority 2: Should Have (Early Phase 2)

### 6. Caching Strategy
```python
# Cache key: (image_path, face_index, model_name)
# Cache: Aligned crop (112x112) + embedding (512-dim)
# Benefit: Recompute embeddings if model changes, skip detection
```

### 7. Performance Benchmarks
```python
# tests/face_clustering/test_performance.py
def test_pipeline_performance():
    for n_faces in [100, 500, 1000]:
        start = time.time()
        run_pipeline(n_faces)
        duration = time.time() - start

        assert duration < n_faces * 0.1, \
            f"Too slow: {duration}s for {n_faces} faces (max: {n_faces*0.1}s)"
```

### 8. Inline Purity Display (After Labeling)
```python
# For cluster with label:
purity = count(most_common_label) / cluster_size
if purity < 0.8:
    st.warning(f"⚠️ Low purity: {purity*100:.0f}% - Review outliers")
```

### 9. Keyboard Shortcuts
```javascript
// In Streamlit (via components.html)
document.addEventListener('keydown', (e) => {
    if (e.key === 'n') nextCluster();
    if (e.key === 'p') previousCluster();
    if (e.key === 'Enter') confirmLabel();
});
```

---

## 📋 Implementation Checklist

### Phase 1A: Pipeline Steps (2-3 days)

- [ ] Create `filter_quality_gate.py` step
  - [ ] Add validation: assert core_indices not empty
  - [ ] Add logging with timing

- [ ] Create `build_knn_graph.py` step
  - [ ] Add validation: embeddings normalized
  - [ ] Add logging: n_faces, n_edges, duration

- [ ] Create `cluster_connected_components.py` step
  - [ ] Add logging: n_clusters, n_noise

- [ ] Create `select_exemplars.py` step
  - [ ] Add logging: exemplars per cluster

- [ ] Create `compute_debug_distances.py` step
  - [ ] Store: 5 closest within, 5 furthest within, 5 closest cross
  - [ ] Compute: exemplar distance matrix
  - [ ] Add validation: all core faces have neighbors

- [ ] Update `export_for_labeling.py` step
  - [ ] Export debug_neighbors to JSON
  - [ ] Add validation checks to export_summary.json

### Phase 1B: UI Updates (1-2 days)

- [ ] Add "🔍 Debug" tab to workbench
  - [ ] Face selector (dropdown)
  - [ ] Show within-cluster neighbors with colored distances
  - [ ] Show cross-cluster neighbors with colored distances
  - [ ] Show cluster exemplars

- [ ] Enhance "🏷️ Label" tab
  - [ ] Add progress bar
  - [ ] Add keyboard shortcuts (n/p/Enter)
  - [ ] Color-code neighbor distances
  - [ ] Show only 3 neighbors by default (expand button)

### Phase 1C: Testing (1 day)

- [ ] Test on test_data/face_clustering
  - [ ] Verify: 3 clusters, 9 faces, 0 noise
  - [ ] Verify: No embedding/crop mismatches
  - [ ] Verify: All validations pass
  - [ ] Verify: Debug neighbors computed correctly

- [ ] Performance benchmark
  - [ ] Measure: Time per 100 faces
  - [ ] Measure: Memory usage
  - [ ] Document: In README or architecture doc

### Phase 1D: Documentation (0.5 day)

- [ ] Update FACE_CLUSTERING_ARCHITECTURE.md
  - [ ] Add distance metric specification
  - [ ] Add validation checks section
  - [ ] Add performance benchmarks

- [ ] Update FACE_CLUSTERING_WORKBENCH_GUIDE.md
  - [ ] Add keyboard shortcuts
  - [ ] Add color coding explanation
  - [ ] Add troubleshooting section

- [ ] Update README.md
  - [ ] Add face clustering workbench section
  - [ ] Add quick start guide

---

## 🚫 Explicitly NOT Implementing (Phase 1)

- ❌ Split/merge (Phase 2)
- ❌ Evaluation metrics (Phase 2 - after labeling)
- ❌ Batch operations (Phase 2)
- ❌ Ablation study UI (Phase 2)
- ❌ Export PDF reports (Phase 3)
- ❌ Uncertainty-based queue (Phase 2)

---

## 📊 Success Criteria (Phase 1)

1. ✅ Pipeline runs on test_data without errors
2. ✅ Produces 3 clusters (3 people, 3 faces each)
3. ✅ All validations pass (embeddings normalized, no mismatches)
4. ✅ Debug UI shows distances with color coding
5. ✅ User can label clusters and save to CSV
6. ✅ Performance: < 10 sec for 100 faces on CPU
7. ✅ Tests pass: Unit tests + integration test

---

## 🎓 Key Insights from Review

**From CV Researcher:**
- Distance metric MUST be documented (cosine on normalized)
- Validation is critical (check embeddings are normalized)
- Focus on initial clustering first (correct approach)

**From SW Engineer:**
- Add timing logs to every stage
- Validation checks prevent silent failures
- Performance benchmarks catch regressions

**From UI Expert:**
- Color-code distances for instant visual feedback
- Show 3 neighbors by default (avoid overwhelm)
- Progress tracking keeps user motivated
- Keyboard shortcuts speed up labeling 10x

---

## ✅ Approved to Proceed

**Next step:** Implement Phase 1A (pipeline steps) with Priority 1 requirements.

**Estimated time:** 4-5 days total for Phase 1.

**Review point:** After Phase 1C testing, review with user before Phase 2.
