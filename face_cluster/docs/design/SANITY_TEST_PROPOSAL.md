# Sanity Test Proposal: High Exemplar Distance Analysis

**Date**: 2026-03-29
**Dataset**: D:\Google_Germany (788 images, top-level only)
**Purpose**: Validate face clustering quality and identify potential bugs vs expected limitations

---

## Objective

Run face clustering pipeline on real-world dataset and analyze clusters with high exemplar inner distances to determine:

1. Are high-distance clusters (> 0.8) due to bugs or natural limitations?
2. What causes these clusters to form?
3. Will split/merge phases improve results, or is there a systematic issue?

---

## Test Plan

### Phase 1: Run Clustering Pipeline

**Input**:
- Directory: `D:\Google_Germany` (788 images: JPG + HEIC)
- Scope: Top-level only (no subdirectories)
- Pipeline: `face_cluster` standalone module (kNN graph + connected components)

**Configuration** (from `face_cluster/config.py`):
```yaml
quality_gate:
  yaw_max: 45.0
  blur_min: 100.0

knn_graph:
  K: 5
  distance_threshold: 0.35

clustering:
  min_cluster_size: 2
```

**Expected Output**:
- `faces.csv`: Face metadata with cluster assignments
- `clusters.csv`: Cluster statistics (size, diameter, exemplar distances)
- `face_crops/`: Aligned 112×112 crops
- `debug_neighbors.json`: Pre-computed distance data

### Phase 2: Identify High-Distance Clusters

**Metric**: Exemplar inner distance = max distance between exemplars in cluster

**Analysis Steps**:
1. Load `clusters.csv` and sort by `max_exemplar_distance` descending
2. Identify clusters with `max_exemplar_distance > 0.8` (if any)
3. For top 3 high-distance clusters:
   - Extract all faces in cluster
   - Compute pairwise distances within cluster
   - Identify the 2 faces with max distance (the "problematic pair")

### Phase 3: Cross-Cluster Distance Analysis

**For each problematic pair** (faces A, B in same cluster):

1. **Within-cluster context**:
   - Distance(A, B) = ? (the high distance)
   - What's the median within-cluster distance?
   - Are A and B outliers, or is whole cluster wide?

2. **Cross-cluster context**:
   - Find top 5 closest faces to A from OTHER clusters
   - Find top 5 closest faces to B from OTHER clusters
   - Compare: Are cross-cluster distances < within-cluster distance?

3. **Decision Matrix**:

| Scenario | Within-Cluster | Cross-Cluster | Diagnosis |
|----------|---------------|---------------|-----------|
| 1 | d(A,B) = 0.85 | d(A, other) = 0.90 | **Natural wide cluster** - faces legitimately similar, will benefit from split phase |
| 2 | d(A,B) = 0.85 | d(A, other) = 0.30 | **Bug: A mis-clustered** - should be in different cluster |
| 3 | d(A,B) = 0.85 | d(A, other_A) = 0.30<br>d(B, other_B) = 0.25 | **Bug: Both mis-clustered** - cluster is merge of 2+ people |
| 4 | d(A,B) = 0.85 | d(A, noise) = 0.60 | **Edge case: Noise attachment** - faces are borderline quality |

### Phase 4: Visual Inspection

**For top 3 problematic clusters**:
1. Display exemplar crops side-by-side
2. Display problematic pair (A, B) crops
3. Display closest cross-cluster neighbors for A and B
4. Manual judgment: Same person or different people?

### Phase 5: Root Cause Analysis

**Possible Causes**:

1. **Algorithm Limitation** (Expected):
   - Conservative threshold (0.35) prevents splits
   - kNN graph creates "chain" connections (A→B→C even if d(A,C) large)
   - Will be fixed by split phase (hierarchical clustering on wide clusters)

2. **Quality Gate Issue** (Potential Bug):
   - Low-quality faces passing gate (blur, extreme pose)
   - Embeddings unreliable for these faces
   - Fix: Stricter quality thresholds

3. **Embedding Issue** (Potential Bug):
   - Embedding extraction bug (wrong face crop, misalignment)
   - Cache corruption (stale embeddings)
   - Fix: Regenerate embeddings, verify alignment

4. **Graph Construction Bug** (Potential Bug):
   - Incorrect distance computation
   - Mutual kNN logic error
   - Fix: Verify graph construction code

---

## Expert Review Questions

### Question 1: Threshold Selection

**Context**: Using `distance_threshold = 0.35` for kNN graph edges.

**Question**: For a dataset of 788 images (~2000-3000 faces expected):
- Is 0.35 reasonable, or should we use a stricter threshold (e.g., 0.30)?
- At what exemplar distance should we be concerned? (0.6? 0.8? 1.0?)

**Expert Input Needed**:
- Typical distribution of exemplar distances in well-formed clusters
- When does high distance indicate bug vs algorithm limitation?

### Question 2: Chain Connection Problem

**Context**: kNN graph can create "chains" where A connects to B, B connects to C, but d(A,C) is large.

**Scenario**:
```
Face A (frontal) → d=0.30 → Face B (slight profile) → d=0.35 → Face C (strong profile)
d(A, C) = 0.80 (too large for same person)
```

**Question**: Is this expected behavior that split phase will fix, or should kNN graph prevent this?

**Expert Input Needed**:
- Should we add diameter constraint to connected components?
- Should we use single-linkage with max-diameter pruning?
- Or trust split phase to handle this?

### Question 3: Exemplar Selection Impact

**Context**: Using d10-based exemplar selection (distance to 10th nearest neighbor).

**Question**: If cluster has diameter 0.80:
- Will exemplar selection still work correctly?
- Could poor exemplar selection hide clustering errors?

**Expert Input Needed**:
- Should we compute "exemplar coverage" metric?
- Should exemplars be selected from different density regions?

### Question 4: Benchmark Expectations

**Context**: Real-world dataset, no ground truth labels.

**Question**: What's a "good" result for Phase 1 (kNN clustering only)?
- Expected % of clusters with diameter > 0.6?
- Expected % of clusters needing split?
- Expected % of noise faces?

**Expert Input Needed**:
- Baseline expectations for kNN-only clustering
- When to be concerned vs when to say "this will be fixed in split/merge"

---

## Tools Required

### Existing Tools
- ✅ `scripts/export_clustering_data.py` - Run clustering and export
- ✅ `app/face_clustering_labeling.py` - Streamlit UI for viewing clusters
- ✅ `face_cluster/analysis.py` - Distance matrix computation

### New Tool Needed
- ❓ **Sanity test script** - Automated analysis of high-distance clusters

**Script Requirements**:
```python
# scripts/sanity_test_clustering.py

def main(album_dir: str, output_dir: str):
    # 1. Run clustering
    run_clustering_pipeline(album_dir, output_dir)

    # 2. Load results
    clusters = load_clusters(output_dir / 'clusters.csv')
    faces = load_faces(output_dir / 'faces.csv')
    embeddings = load_embeddings(output_dir / 'embeddings_*.npy')

    # 3. Identify high-distance clusters
    high_dist_clusters = find_high_distance_clusters(clusters, threshold=0.6)

    # 4. For each high-distance cluster:
    for cluster in high_dist_clusters[:3]:  # Top 3
        # Analyze within-cluster distances
        analyze_within_cluster(cluster, faces, embeddings)

        # Analyze cross-cluster distances
        analyze_cross_cluster(cluster, faces, embeddings, clusters)

        # Visualize problematic pairs
        visualize_problematic_pairs(cluster, output_dir / 'face_crops')

        # Generate HTML report
        generate_report(cluster, output_dir / f'report_cluster_{cluster.id}.html')

    # 5. Summary report
    generate_summary_report(high_dist_clusters, output_dir / 'sanity_test_summary.html')
```

---

## Streamlit App Verification

**Test**: Launch labeling app and verify:
1. App loads without errors
2. Can navigate between clusters
3. Exemplar images display correctly
4. Debug view shows distance information
5. Can identify high-distance clusters visually

**Command**:
```bash
streamlit run app/face_clustering_labeling.py -- --data-dir results/Google_Germany_clustering
```

**Verification Checklist**:
- [ ] App loads without errors
- [ ] Cluster list displays all clusters
- [ ] Exemplar images load and display
- [ ] Can switch between "All Faces" and "Exemplars" tabs
- [ ] Debug tab shows within-cluster and cross-cluster neighbors
- [ ] Distance color coding works (🟢 green < 0.25, 🟡 yellow 0.25-0.40, 🔴 red > 0.40)
- [ ] Can sort clusters by size, diameter, or exemplar distance

---

## Expected Timeline

1. **Expert Review**: 10-15 minutes (review this proposal)
2. **Script Creation**: 30-45 minutes
3. **Pipeline Execution**: 15-20 minutes (788 images)
4. **Analysis**: 10-15 minutes (automated + manual inspection)
5. **Streamlit Verification**: 5-10 minutes
6. **Report Generation**: 10 minutes

**Total**: ~1.5-2 hours

---

## Deliverables

1. **Clustering Results**:
   - `results/Google_Germany_clustering/faces.csv`
   - `results/Google_Germany_clustering/clusters.csv`
   - `results/Google_Germany_clustering/embeddings_*.npy`
   - `results/Google_Germany_clustering/face_crops/`

2. **Sanity Test Reports**:
   - `results/Google_Germany_clustering/sanity_test_summary.html` - Overall summary
   - `results/Google_Germany_clustering/report_cluster_*.html` - Per-cluster analysis

3. **Findings Document**:
   - Root cause analysis
   - Bug vs limitation determination
   - Recommendations for next steps

---

## Risk Assessment

**Low Risk**:
- Read-only analysis (no data modification)
- Uses existing pipeline (battle-tested)
- Automated report generation

**Medium Risk**:
- Large dataset (788 images) - may take time
- HEIC format support (requires pillow-heif)
- High memory usage for distance matrix computation

**Mitigation**:
- Run on subset first (50 images) to verify
- Check HEIC support before full run
- Use sparse distance matrix if needed

---

## Success Criteria

**Test Passes If**:
1. Clustering completes without errors
2. All 788 images processed (or valid skip reasons logged)
3. Clusters with diameter > 0.8 are analyzed
4. Root cause determined (bug vs limitation)
5. Streamlit app displays results correctly
6. Clear recommendation provided

**Test Fails If**:
- Pipeline crashes or hangs
- Embeddings are all zeros (cache corruption)
- All clusters have diameter > 0.8 (systematic bug)
- Streamlit app doesn't load

---

**Next Step**: Wait for expert panel review, then implement sanity test script.
