# Face Clustering Algorithms and Debug Guide

## Overview

This document summarizes the face clustering algorithms used in sim-bench, their parameters, and how to debug clustering decisions.

---

## Two Hybrid Algorithms

We have two hybrid clustering approaches that both start with HDBSCAN, then refine:

| Algorithm | File | Key Idea |
|-----------|------|----------|
| `hybrid_hdbscan_knn` | `sim_bench/clustering/hybrid_hdbscan_knn.py` | Compare **exemplars** between clusters |
| `hybrid_closest_face` | `sim_bench/clustering/hybrid_closest_face.py` | Compare **all faces** using cross-cluster d3 |

---

## Algorithm 1: hybrid_hdbscan_knn (Exemplar-Based)

### How It Works

```
1. HDBSCAN → initial clusters + noise points
2. For each cluster:
   - Compute d3 (distance to 3rd nearest neighbor) for each face
   - T = Q3(d3) + 1.5 × IQR(d3), clamped to [floor, ceiling]
   - Select top 10 exemplars (faces with smallest d3)
3. Merge: If ≥3 exemplar pairs between clusters have distance ≤ min(T_A, T_B)
4. Attach: If noise point is within T of ≥2 exemplars
5. Repeat until no changes
```

### What Are Exemplars?

**Exemplars = faces with smallest d3 (most central faces)**

```
Cluster with 8 faces:
  Face 1: d3 = 0.32  ← exemplar (small d3 = well-connected)
  Face 2: d3 = 0.35  ← exemplar
  Face 3: d3 = 0.38  ← exemplar
  Face 4: d3 = 0.40  ← exemplar
  Face 5: d3 = 0.42  ← exemplar
  Face 6: d3 = 0.45  ← exemplar
  Face 7: d3 = 0.48  ← exemplar
  Face 8: d3 = 0.55  (peripheral, not exemplar if max_exemplars < 8)
```

### The Threshold T

**T = "maximum acceptable distance for a face to belong to this cluster"**

Calculated using Tukey fence (same as box plot whiskers):
```
Q1 = 25th percentile of d3 values
Q3 = 75th percentile of d3 values
IQR = Q3 - Q1
T_raw = Q3 + 1.5 × IQR
T = clamp(T_raw, floor, ceiling)
```

### Known Limitation: Units Mismatch

**Issue identified during debug:**
- T is computed from d3 (distance to 3rd nearest neighbor)
- Merge criteria compare arbitrary pairwise distances between exemplars
- These are different measurements, even though same units (Euclidean distance)

This is a heuristic that works empirically but isn't mathematically rigorous.

---

## Algorithm 2: hybrid_closest_face (Cross-Cluster d3)

### How It Works

This algorithm addresses the "units mismatch" by comparing d3-to-d3:

```
1. HDBSCAN → initial clusters + noise points
2. For each cluster:
   - Compute d3 for all faces
   - T = median(d3) + iqr_multiplier × IQR(d3), clamped to [floor, ceiling]
   - Select exemplars (for early exit optimization only)
3. Merge check between clusters A and B:
   - Stage 1 (Early Exit): Skip if min(exemplar distances) > 2 × max(T_A, T_B)
   - Stage 2 (Cross-Cluster d3):
     - For each face in A, compute d3 using B's faces as neighbors
     - For each face in B, compute d3 using A's faces as neighbors
     - Merge if ≥2 faces have cross-cluster d3 ≤ min(T_A, T_B)
4. Attach: If noise point's cross-cluster d3 ≤ cluster's T
```

### Why This Is More Rigorous

```
hybrid_hdbscan_knn:
  T computed from: d3 (3rd nearest neighbor within cluster)
  Merge compares:  arbitrary pairwise distances between exemplars
  Problem:         different measurements

hybrid_closest_face:
  T computed from: d3 (3rd nearest neighbor within cluster)
  Merge compares:  d3 (3rd nearest neighbor in OTHER cluster)
  Result:          same measurement = fair comparison
```

### Cross-Cluster d3 Explained

```
Cluster A has 5 faces: A1, A2, A3, A4, A5
Cluster B has 4 faces: B1, B2, B3, B4

To check if A1 "fits" into cluster B:
  1. Compute distances from A1 to all B faces: [0.42, 0.55, 0.61, 0.48]
  2. Sort: [0.42, 0.48, 0.55, 0.61]
  3. d3_cross = 3rd smallest = 0.55
  4. If 0.55 ≤ min(T_A, T_B), then A1 "fits" B

Repeat for all faces in A and B, count how many fit.
If ≥ merge_min_faces fit → merge the clusters.
```

---

## Parameter Reference

### hybrid_hdbscan_knn

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 3 | HDBSCAN: minimum faces to form a cluster |
| `min_samples` | 2 | HDBSCAN: core point density |
| `cluster_selection_epsilon` | 0.35 | HDBSCAN: merge clusters within this distance |
| `knn_k` | 3 | k for d_k neighbor distance |
| `threshold_floor` | 0.50 | Minimum T (prevents over-strict clusters) |
| `threshold_ceiling` | 0.90 | Maximum T (prevents over-permissive clusters) |
| `max_exemplars` | 10 | Number of exemplars per cluster |
| `merge_min_pairs` | 3 | Required exemplar pairs within T to merge |
| `merge_min_distinct` | 2 | Required distinct exemplars per cluster |
| `attach_min_exemplars` | 2 | Required exemplars within T to attach |

### hybrid_closest_face

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_cluster_size` | 3 | HDBSCAN: minimum faces to form a cluster |
| `knn_k` | 3 | k for d_k neighbor distance |
| `iqr_multiplier` | 2.5 | Multiplier for IQR (higher = more permissive) |
| `threshold_floor` | 0.30 | Minimum T |
| `threshold_ceiling` | 1.50 | Maximum T |
| `max_exemplars` | 10 | Exemplars (for early exit only) |
| `merge_min_faces` | 2 | Required faces with cross-d3 ≤ T to merge |
| `early_exit_multiplier` | 2.0 | Skip check if exemplars > this × max(T) apart |

---

## Debug Page Usage

The Streamlit debug page (`app/face_clustering_comparison.py`) provides 6 tabs:

### Tab 1: Cluster Overview
- Shows all clusters with their computed T values
- Shows d3 statistics (Q1, Q3, IQR, raw threshold)
- Shows all faces in each cluster (no truncation)
- Exemplars marked with ⭐

### Tab 2: Inter-Cluster Distances
- Heatmap of minimum distances between cluster exemplars
- Table comparing distance vs merge threshold
- Green = could merge, Red = too far

### Tab 3: Merge Decisions
- Every cluster pair that was evaluated
- Shows: threshold used, pairs within T, distinct exemplars involved
- **Reason codes:**
  - `merged` - all criteria met
  - `not_enough_pairs` - too few exemplar pairs close enough
  - `not_enough_distinct_a` - not enough distinct exemplars from cluster A
  - `not_enough_distinct_b` - not enough distinct exemplars from cluster B
- Cross-distance matrix visualization for selected pair

### Tab 4: Attachment Decisions
- Every noise point that was evaluated
- Shows which clusters qualified and why
- Shows why it attached to chosen cluster (or remained noise)

### Tab 5: Parameter Tuning
- Sliders to adjust all parameters
- "Re-run Clustering" button to see immediate effect
- Before/after comparison

### Tab 6: Face Distance Lookup
- Select any two faces by index
- Shows Euclidean and cosine distance
- Shows whether distance is within each cluster's threshold

---

## Debugging Common Issues

### Clusters Not Merging

1. **Check Merge Decisions tab** → find the pair → see which criterion failed
2. **Common causes:**
   - `not_enough_pairs`: exemplars too far apart
   - `not_enough_distinct_a/b`: only 1 exemplar from one side is close
3. **Solutions:**
   - Lower `threshold_floor` to allow more variation
   - Lower `merge_min_pairs` to require fewer close pairs
   - Lower `merge_min_distinct` to allow less diversity

### Wrong Faces Merging

1. **Check Merge Decisions tab** → see which pair merged
2. **Look at cross-distance matrix** → see which exemplars were close
3. **Solutions:**
   - Raise `threshold_floor` to be more strict
   - Raise `merge_min_pairs` to require more evidence
   - Use `hybrid_closest_face` which checks all faces, not just exemplars

### Noise Points Not Attaching

1. **Check Attachment Decisions tab** → see why each cluster didn't qualify
2. **Common cause:** face is too far from all cluster exemplars
3. **Solutions:**
   - Lower `attach_min_exemplars` to 1
   - Check if face might be a different person (correct to leave as noise)

### All Thresholds at Floor or Ceiling

- **All at floor:** clusters are unusually tight (consistent faces)
- **All at ceiling:** clusters are unusually loose (high variation)
- **Solutions:**
   - Adjust floor/ceiling to match your data's characteristics
   - Use `iqr_multiplier` to scale the threshold calculation

---

## Running the Benchmark

```bash
# Generate clustering results with debug data
python scripts/benchmark_face_clustering.py --album-path D:\YourAlbum

# Launch debug UI
streamlit run app/face_clustering_comparison.py
```

Navigate to "Debug: Hybrid kNN" page to explore decisions.

---

## Key Files

| File | Purpose |
|------|---------|
| `sim_bench/clustering/hybrid_hdbscan_knn.py` | Exemplar-based algorithm |
| `sim_bench/clustering/hybrid_closest_face.py` | Cross-cluster d3 algorithm |
| `sim_bench/clustering/base.py` | Base class and factory |
| `configs/clustering_benchmark.yaml` | Parameter configuration |
| `scripts/benchmark_face_clustering.py` | Run benchmark and save results |
| `app/face_clustering_comparison.py` | Debug UI |
| `tests/clustering/test_hybrid_hdbscan_knn.py` | Unit tests |

---

## Algorithm Comparison

| Aspect | hybrid_hdbscan_knn | hybrid_closest_face |
|--------|-------------------|---------------------|
| Merge criteria | Exemplar pairwise distances | Cross-cluster d3 for all faces |
| Units consistency | Heuristic (d3 → pairwise) | Rigorous (d3 → d3) |
| Computational cost | O(E² per pair) where E=exemplars | O(N×M per pair) where N,M=cluster sizes |
| Early exit | No | Yes (skip if exemplars very far) |
| Sensitivity | More sensitive to exemplar selection | More robust (checks all faces) |

---

## Changelog

- **2026-02-16**: Added debug page with 6 tabs
- **2026-02-16**: Fixed API consistency (collect_debug_data parameter)
- **2026-02-16**: Added input validation for embeddings
- **2026-02-16**: Created hybrid_closest_face with cross-cluster d3 matching
- **2026-02-16**: Updated algorithm explanation with concrete examples
