# Face Clustering: Complete Guide

High-precision face clustering pipeline for small batches (10-300 faces) using mutual kNN graphs.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Pipeline Overview](#pipeline-overview)
3. [Configuration Parameters](#configuration-parameters)
4. [Stage-by-Stage Guide](#stage-by-stage-guide)
5. [Merge Criteria Deep Dive](#merge-criteria-deep-dive)
6. [Analysis & Debugging](#analysis--debugging)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```python
from face_cluster import (
    PipelineConfig, QualityGater, KNNGraphBuilder,
    ConnectedComponentsClusterer, D10ExemplarSelector,
    ConservativeMerger
)

# Configure
config = PipelineConfig(
    K=5,                    # kNN neighbors
    distance_threshold=0.35, # Edge threshold
    blur_min=50.0,          # Quality gating
    merge_enabled=True,     # Enable merge
    merge_margin=0.0,       # Disable margin check
)

# Stage A: Quality gating
gater = QualityGater(config)
faces = gater.compute_blur_scores(faces)
core_indices, holdout_indices = gater.select_core_set(faces)

# Stage B-C: Distance matrix + mutual kNN graph
graph_builder = KNNGraphBuilder(config)
distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)
graph_result = graph_builder.build_mutual_knn_graph(
    distance_matrix, config.K, config.distance_threshold
)

# Stage D: Connected components clustering
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)

# Stage E: Exemplar selection
exemplar_selector = D10ExemplarSelector(config)
cluster_result = exemplar_selector.select_exemplars(cluster_result, graph_result)

# Stage F2: Conservative merge
merger = ConservativeMerger(config)
cluster_result = merger.merge_clusters(cluster_result, graph_result)

print(f"Final: {cluster_result.n_clusters} clusters, {cluster_result.n_noise} noise")
```

---

## Pipeline Overview

```
Faces (embeddings + crops)
    ↓
[A] Quality Gating → Core set (high quality) + Holdout set (low quality)
    ↓
[B] Distance Matrix (cosine distance, core set only)
    ↓
[C] Mutual kNN Graph (bidirectional edges + threshold)
    ↓
[D] Connected Components → Initial clusters
    ↓
[E] Exemplar Selection (d10 metric)
    ↓
[F2] Conservative Merge (optional, multi-evidence) → Final clusters
    ↓
[F4] Holdout Attachment (optional) → Attach low-quality faces
    ↓
Output: Clusters + exemplars
```

---

## Configuration Parameters

### Core Clustering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 5 | Number of nearest neighbors for mutual kNN |
| `distance_threshold` | 0.35 | Max cosine distance for edge creation |
| `min_cluster_size` | 2 | Min faces per cluster (smaller = noise) |

### Quality Gating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `blur_min` | 50.0 | Min Laplacian variance (blur score) |
| `yaw_max` | 30.0° | Max absolute yaw angle |
| `pitch_max` | 25.0° | Max absolute pitch angle |
| `roll_max` | 25.0° | Max absolute roll angle |

### Exemplar Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d10_k` | 3 | K for d10 computation (density metric) |
| `exemplars_d10_threshold` | 0.35 | Max d10 to be exemplar candidate |
| `N_exemplars_max` | 10 | Max exemplars per cluster |
| `exemplar_suppression_radius` | 0.2 | Min distance between exemplars |

### Conservative Merge

| Parameter | Default | Recommended | Description |
|-----------|---------|-------------|-------------|
| `merge_enabled` | False | **True** | Enable merge stage |
| `merge_use_adaptive_threshold` | True | True | Use adaptive per-cluster thresholds |
| `merge_exemplar_percentile` | 90 | 90 | Percentile for per-cluster threshold (P90) |
| `merge_global_percentile` | 50 | 50-75 | Percentile for global threshold (50=median, 75=more permissive) |
| `merge_threshold_alpha` | 0.7 | 0.7 | Local vs global weight (0.7 = 70% local, 30% global) |
| `merge_margin` | 0.05 | **0.0** | Margin for "clear best" check (0.0 = disable) |
| `merge_support_frac` | 0.3 | 0.3 | Fraction of smaller cluster for support |
| `merge_support_min` | 2 | 2 | Absolute min support |
| `merge_diameter_expansion_factor` | 1.5 | 1.5 | Max diameter growth factor |

---

## Stage-by-Stage Guide

### Stage A: Quality Gating

**Purpose**: Select high-quality "core" faces for clustering

**Process**:
1. Compute blur score (Laplacian variance on aligned face)
2. Optionally compute pose (yaw/pitch/roll) using SixDRepNet
3. Filter: Core = passes all thresholds, Holdout = fails any

**Output**: `core_indices`, `holdout_indices`

### Stage B: Distance Matrix

**Purpose**: Compute pairwise cosine distances for core faces

**Process**: `distance(i,j) = 1 - cosine_similarity(i,j)`

**Output**: Symmetric N×N matrix (N = core set size)

### Stage C: Mutual kNN Graph

**Purpose**: Build sparse graph connecting similar faces

**Process**: Create edge (i,j) if:
- i in j's top-K neighbors **AND**
- j in i's top-K neighbors (mutual) **AND**
- distance(i,j) ≤ threshold

**Output**: NetworkX graph with bidirectional edges only

### Stage D: Connected Components Clustering

**Purpose**: Assign clusters based on graph connectivity

**Process**:
1. Find connected components (NetworkX)
2. Each component = one cluster
3. Mark clusters < min_cluster_size as noise (-1)

**Output**: `cluster_result` with labels and cluster membership

### Stage E: Exemplar Selection (d10 Metric)

**Purpose**: Select representative faces from each cluster

**d10 Metric**: For face i in cluster C, d10(i) = distance to its K-th nearest neighbor within C (lower = denser region)

**Process**:
1. Compute d10 for all faces in each cluster
2. Candidates: faces with d10 ≤ threshold
3. Greedy suppression (select diverse exemplars):
   - Sort by d10 ascending
   - Add if distance to all selected ≥ suppression_radius
   - Stop at N_exemplars_max

**Output**: `cluster_result.exemplars` (cluster_id → exemplar nodes)

### Stage F2: Conservative Merge (Optional)

**Purpose**: Merge over-fragmented clusters using multi-evidence approach

**See**: [Merge Criteria Deep Dive](#merge-criteria-deep-dive) for full details

**Output**: Merged `cluster_result`, `merger.last_thresholds`, `merger.last_candidates`

---

## Merge Criteria Deep Dive

### Adaptive Threshold Formula

For each cluster pair (A, B):

```
T_A = P90(pairwise distances between exemplars in A)
T_B = P90(pairwise distances between exemplars in B)
T_local = MAX(T_A, T_B)
T_global = percentile(T_1, T_2, ..., T_n) across all clusters
T_merge = α × T_local + (1-α) × T_global

where:
  α = merge_threshold_alpha (default 0.7)
  percentile = merge_global_percentile (default 50 = median)
```

**Why MAX?** Allows merging across different density regions (looser cluster can merge with tighter one).

---

### Four Criteria Summary

**ALL four must pass for merge to happen.**

| Criterion | What It Checks | When It Fails | Action |
|-----------|----------------|---------------|--------|
| **1. Exemplar Distance** | min(exemplar_dist) ≤ T_merge | Exemplars too far apart | Most common - clusters genuinely different |
| **2. Support Count** | Enough cross-cluster pairs ≤ T_merge | Only exemplars close | Prevents merging on weak evidence |
| **3. Margin** | B is clear closest cluster | Another cluster equally close | Set `merge_margin=0.0` to disable |
| **4. Diameter** | Post-merge cluster not too wide | Would create over-wide cluster | Prevents precision loss |

---

### Criterion 1: Exemplar Distance ✓

**Check**:
```python
exemplar_dist = min(distance[ex_a, ex_b]
                    for ex_a in exemplars_A
                    for ex_b in exemplars_B)
passes = exemplar_dist <= T_merge
```

**Purpose**: Don't merge if representative faces are too far apart

**Common failure**: Clusters genuinely different (e.g., different people)

---

### Criterion 2: Support Count ✓

**Check**:
```python
support = count(pairs with distance ≤ T_merge)
required = max(merge_support_frac × min(|A|, |B|), merge_support_min)
passes = support >= required
```

**Parameters**:
- `merge_support_frac=0.3` (30% of smaller cluster)
- `merge_support_min=2` (absolute minimum)

**Purpose**: Require enough evidence beyond just close exemplars

**Common failure**: Only exemplars close, but rest of clusters far apart (correct rejection)

---

### Criterion 3: Margin (Recommend: Disable) ⚠️

**Check**: For each exemplar in A, require B is the **clear closest** cluster:
```python
for exemplar in A:
    dist_to_B = min(distance[exemplar, n] for n in cluster_B)
    dist_to_any_other = min(distance[exemplar, n]
                           for other_cluster in all_others
                           for n in other_cluster)

    if dist_to_B + merge_margin >= dist_to_any_other:
        return False  # Another cluster equally close
```

**Parameter**: `merge_margin=0.05` (default)

**Example**:
```
Exemplar X from cluster A:
  Distance to B: 0.233  ← proposed merge partner
  Distance to C: 0.240  ← another cluster

Check (margin=0.05):
  0.233 + 0.05 = 0.283
  Is 0.283 < 0.240? NO → FAIL

Reason: C is within margin (gap is only 0.007)
```

**Problem**: Often too strict - blocks valid merges when multiple clusters have similar distances

**Recommendation**: Set `merge_margin=0.0` to disable entirely

---

### Criterion 4: Post-Merge Diameter ✓

**Check**:
```python
current_max = max(diameter_A, diameter_B)
post_diameter = max(distance[n1, n2]
                   for n1 in A+B
                   for n2 in A+B)
max_allowed = current_max × merge_diameter_expansion_factor
passes = post_diameter <= max_allowed
```

**Parameter**: `merge_diameter_expansion_factor=1.5`

**Purpose**: Don't create over-wide clusters (maintain precision)

---

### Iterative Merging Process

```python
while True:
    # 1. Propose candidates (exemplar distance < 0.45)
    candidates = get_close_pairs()

    # 2. Evaluate ALL 4 criteria for each
    valid = [pair for pair in candidates if all_4_pass(pair)]

    if not valid:
        break  # Done

    # 3. Merge best pair (smallest exemplar distance)
    merge(best_pair)

    # 4. Recompute adaptive thresholds
    update_thresholds()
```

---

## Analysis & Debugging

### PRIMARY: Get Cluster-to-Cluster Distances

**Use `get_cluster_distances()` to understand why clusters didn't merge:**

```python
from face_cluster.analysis import ClusterSnapshot

# Create snapshot
snapshot = ClusterSnapshot.from_result(
    cluster_result, faces, core_indices, distance_matrix,
    stage="after_merge",
    config=config,
    cluster_thresholds=merger.last_thresholds,
    merge_candidates=merger.last_candidates
)

# Get distance matrix between all cluster pairs
df = snapshot.get_cluster_distances()
print(df.head(20))  # Sorted by Exemplar_Dist
```

**DataFrame Columns**:
- `Exemplar_Dist`: Min distance between exemplars (used for merge proposal, must be ≤ 0.45)
- `Min_Dist`: Min distance between any two faces
- `Mean_Dist`: Average distance across all face pairs
- `Max_Dist`: Max distance between any two faces

**Why pair (4, 23) didn't merge:**
```python
row = df[(df['C1']==4) & (df['C2']==23)].iloc[0]
if row['Exemplar_Dist'] > 0.45:
    print("→ Not even proposed (exemplar distance too large)")
```

---

### Get Pairwise Distances Between Clusters

```python
# All pairwise distances between two clusters
nodes_a = snapshot.clusters[4]
nodes_b = snapshot.clusters[23]
pairwise = snapshot.distance_matrix[np.ix_(nodes_a, nodes_b)]
print(f"Shape: {pairwise.shape}")  # e.g., (27, 3)

# Exemplar distances only
exemplars_a = snapshot.exemplars[4]
exemplars_b = snapshot.exemplars[23]
ex_dists = snapshot.distance_matrix[np.ix_(exemplars_a, exemplars_b)]
print(f"Min exemplar dist: {ex_dists.min():.3f}")  # Used for merge
```

---

### Understanding Distance Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Exemplar_Dist** | Min distance between **exemplars only** | Merge proposal (must be ≤ 0.45) |
| **Min_Dist** | Min distance between **ANY two faces** | Shows if outliers are close |
| **Mean_Dist** | Average across all face pairs | Overall cluster separation |

**Key**: Min_Dist can be small while Exemplar_Dist is large (outliers close, but cores far apart).

**Example**:
```
Pair (4, 23):
  Exemplar_Dist = 0.520  ← Cores far apart
  Min_Dist = 0.278       ← Some outliers close

Result: NOT proposed (exemplar_dist > 0.45)
Interpretation: Outliers close, cores different → correct behavior
```

---

### SECONDARY: Merge Decision Details

**Use `get_merge_decisions_df()` to see which criteria failed for PROPOSED pairs:**

```python
# Get merge decisions (only shows pairs with exemplar_dist < 0.45)
df = snapshot.get_merge_decisions_df()
print(df)

# Filter to failed merges
failed = df[~df['Merged']]
print(failed[['C1', 'C2', 'Exemplar_Dist', 'T_merge', 'Gap', 'Failed']])
```

**DataFrame Columns**:
- `Exemplar_Dist`: Distance between closest exemplars
- `T_A, T_B`: Per-cluster thresholds (P90 of intra-cluster exemplar distances)
- `T_local`: MAX(T_A, T_B)
- `T_global`: Percentile across all clusters (configurable via `merge_global_percentile`)
- `T_merge`: Final threshold = α×T_local + (1-α)×T_global
- `Gap`: Exemplar_Dist - T_merge (negative = passes exemplar check)
- `Failed`: Which criteria failed (Exemplar, Support, Margin, Diameter)

**Note**: This only shows pairs that were PROPOSED (exemplar_dist < 0.45). Use `get_cluster_distances()` to see ALL pairs.

---

### Interpreting Failure Patterns

| Failed Criteria | Meaning | Is This Bad? | Action |
|----------------|---------|--------------|--------|
| **Exemplar only** | Distance slightly above threshold | Maybe | If Gap < 0.05, consider lower alpha |
| **Support only** | Only exemplars close | No | Correct - prevents outlier-driven merges |
| **Margin only** | Another cluster equally close | Usually not | Set `merge_margin=0.0` |
| **Diameter only** | Merged cluster too wide | Probably not | Increase expansion factor if needed |
| **Exemplar + Support + Diameter** | Genuinely different clusters | **No - correct!** | Don't merge these |

**Example: Three Failures (Correct Behavior)**:
```
C1 C2  Exemplar_Dist  T_merge  Gap   Failed
3  6   0.413         0.233    +0.18  Exemplar, Support, Diameter

Analysis:
  - Gap = +0.18 (fails exemplar by a lot)
  - Exemplar_Dist is 1.8× the threshold
  - Failed 3 out of 4 criteria

Conclusion: Clusters 3 and 6 are DIFFERENT (probably different people)

Would lowering alpha help?
  alpha=0.7: T_merge = 0.233 (current)
  alpha=0.5: T_merge = 0.255
  alpha=0.3: T_merge = 0.276
  alpha=0.0: T_merge = 0.307

  NO! Even with alpha=0, exemplar_dist (0.413) > T_merge (0.307)

Why this is good:
  - T_A = 0.202, T_B = 0.191 (both tight internally)
  - But exemplars are 0.413 apart (far between clusters)
  - High precision - preserving genuinely different identities
```

---

## Troubleshooting

### Decision Tree

```
Clusters not merging?
│
├─ Step 1: Check cluster distances
│  │  df = snapshot.get_cluster_distances()
│  │
│  ├─ Exemplar_Dist > 0.45?
│  │  → Not even proposed
│  │  → Check if Min_Dist << Exemplar_Dist:
│  │     Outliers close, cores far apart (correct behavior)
│  │
│  └─ Exemplar_Dist ≤ 0.45?
│     → Was proposed, check merge decisions
│     │
│     ├─ Failed: "Margin" only?
│     │  → Set merge_margin=0.0
│     │
│     ├─ Failed: "Exemplar" only, Gap < 0.05?
│     │  → Consider raising merge_global_percentile (e.g., 75)
│     │
│     └─ Failed: Multiple criteria?
│        → Clusters genuinely different - DON'T merge
│
└─ Too many small clusters overall?
   → Lower distance_threshold in initial clustering (Stage C)
```

---

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Too many small clusters | Over-fragmentation | Enable merge (`merge_enabled=True`) with `merge_margin=0.0` |
| All merges fail on "Margin" | Margin too strict | Set `merge_margin=0.0` |
| Different people in same cluster | Threshold too high | Lower `distance_threshold` (e.g., 0.30) |
| Same person in multiple clusters | Threshold too low OR merge disabled | Raise threshold OR enable merge |
| Clusters too wide (low precision) | Distance threshold too loose | Lower `distance_threshold` OR enable splitting |

---

### Performance Tuning

**For higher precision** (fewer false positives):
- Lower `distance_threshold` (e.g., 0.30)
- Increase `min_cluster_size` (e.g., 3)
- Tighten quality gating (`blur_min=75`, `yaw_max=20`)

**For higher recall** (fewer false negatives):
- Raise `distance_threshold` (e.g., 0.40)
- Enable merge with lower alpha (e.g., 0.5)
- Loosen quality gating

**For balanced** (recommended):
- `distance_threshold=0.35`
- `merge_enabled=True, merge_margin=0.0, alpha=0.7`
- Default quality gating

**Experimenting with global threshold**:
- `merge_global_percentile=25` - Conservative (use lower percentile of cluster thresholds)
- `merge_global_percentile=50` - Balanced (median, default)
- `merge_global_percentile=75` - Permissive (use higher percentile, more merging)
- Higher values allow more merging by raising the global component of T_merge

---

## Key Design Decisions

1. **Mutual kNN** (not regular kNN): Bidirectional edges = higher precision
2. **Connected components** (not HDBSCAN): Deterministic, interpretable
3. **d10 exemplar selection**: Density-based, selects core of each cluster
4. **Adaptive merge thresholds**: Per-cluster P90 + global median = robust across densities
5. **Multi-evidence merge**: Four criteria prevent both under-merging and over-merging
6. **Quality gating first**: Cluster on high-quality faces, attach low-quality later

---

## Production Checklist

- [ ] Set `merge_enabled=True`
- [ ] Set `merge_margin=0.0` (disable margin check)
- [ ] Tune `distance_threshold` on validation set (try 0.30, 0.35, 0.40)
- [ ] Enable pose estimation for quality gating
- [ ] Save `merger.last_thresholds` and `merger.last_candidates` for debugging
- [ ] Monitor cluster size distribution (watch for giant clusters)
- [ ] Manually review exemplar faces from each cluster
- [ ] Check `Failed` column in merge decisions for patterns

---

**Questions?** See individual sections above or refer to code in `face_cluster/` module.
