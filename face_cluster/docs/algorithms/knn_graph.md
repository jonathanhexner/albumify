# KNN Graph Face Clustering Pipeline

Complete documentation of the high-precision face clustering pipeline for small batches (10-20 faces).

## Pipeline Overview

```
Input: Face embeddings (512-dim, L2-normalized)
  ↓
Stage A: Quality Gating (blur + pose) → Core set + Holdout set
  ↓
Stage B: Distance Matrix (cosine distance on core set)
  ↓
Stage C: Mutual kNN Graph (bidirectional edges + threshold)
  ↓
Stage D: Connected Components Clustering
  ↓
Stage E: Exemplar Selection (d10 metric)
  ↓
Stage F2: Conservative Merge (optional, multi-evidence)
  ↓
Stage F3: Cluster Splitting (optional, safety valve)
  ↓
Stage F4: Holdout Attachment (optional, attach low-quality faces)
  ↓
Output: Cluster assignments + exemplars
```

---

## Stage A: Quality Gating

**Purpose**: Select high-quality "core" faces for clustering, defer low-quality to "holdout"

**Input**: List of `FaceRecord` objects with embeddings and aligned face crops

**Steps**:
1. Compute blur score (Laplacian variance on aligned face)
2. (Optional) Compute pose (yaw, pitch, roll) using SixDRepNet
3. Filter faces:
   - Core: `blur ≥ blur_min AND |yaw| ≤ yaw_max AND |pitch| ≤ pitch_max AND |roll| ≤ roll_max`
   - Holdout: All others

**Parameters**:
- `blur_min` (default 50.0) - Minimum blur score
- `yaw_max` (default 30.0°) - Max absolute yaw
- `pitch_max` (default 25.0°) - Max absolute pitch
- `roll_max` (default 25.0°) - Max absolute roll

**Output**:
- `core_indices`: Indices of high-quality faces
- `holdout_indices`: Indices of low-quality faces

**Code**:
```python
gater = QualityGater(config)
faces = gater.compute_blur_scores(faces)
faces = gater.compute_pose_scores(faces)  # Optional
core_indices, holdout_indices = gater.select_core_set(faces)
```

---

## Stage B: Distance Matrix

**Purpose**: Compute pairwise cosine distances for core faces

**Input**:
- `faces`: List of FaceRecord
- `core_indices`: Which faces to use

**Steps**:
1. Extract embeddings for core faces
2. Compute cosine distance: `d(i,j) = 1 - similarity(i,j)`
3. Result is symmetric matrix with zeros on diagonal

**Parameters**: None

**Output**: `distance_matrix` (N×N where N = len(core_indices))

**Code**:
```python
graph_builder = KNNGraphBuilder(config)
distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)
```

---

## Stage C: Mutual kNN Graph

**Purpose**: Build sparse graph connecting similar faces

**Input**: `distance_matrix`

**Steps**:
1. For each face, find K nearest neighbors
2. Create edge only if:
   - i is in j's top-K neighbors AND
   - j is in i's top-K neighbors (mutual) AND
   - distance(i,j) ≤ distance_threshold
3. Build NetworkX graph with edges

**Parameters**:
- `K` (default 5) - Number of nearest neighbors
- `distance_threshold` (default 0.35) - Max distance for edge

**Output**: `GraphResult` with NetworkX graph and edge list

**Code**:
```python
graph_result = graph_builder.build_mutual_knn_graph(
    distance_matrix, config.K, config.distance_threshold
)
```

---

## Stage D: Connected Components Clustering

**Purpose**: Assign cluster labels based on graph connectivity

**Input**: `graph_result` from Stage C

**Steps**:
1. Find connected components in graph (NetworkX)
2. Each component = one cluster
3. Mark clusters smaller than `min_cluster_size` as noise (-1)

**Parameters**:
- `min_cluster_size` (default 2) - Minimum faces per cluster

**Output**: `ClusterResult` with labels, clusters dict, stats

**Code**:
```python
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)
```

---

## Stage E: Exemplar Selection (d10 Metric)

**Purpose**: Select representative "exemplar" faces from each cluster

**d10 Metric**: For face i in cluster C, d10(i) = distance to its K-th nearest neighbor within C (density measure)

**Steps**:
1. For each cluster, compute d10 for all members
2. Select exemplar candidates: faces with `d10 ≤ exemplars_d10_threshold`
3. Greedy suppression:
   - Sort candidates by d10 (ascending)
   - Add candidate if distance to all selected exemplars ≥ suppression_radius
   - Stop at N_exemplars_max

**Parameters**:
- `d10_k` (default 3) - K for d10 computation
- `exemplars_d10_threshold` (default 0.35) - Max d10 to be candidate
- `N_exemplars_max` (default 10) - Max exemplars per cluster
- `exemplar_suppression_radius` (default 0.2) - Min distance between exemplars

**Output**: `cluster_result.exemplars` dict mapping cluster_id → exemplar node indices

**Code**:
```python
exemplar_selector = D10ExemplarSelector(config)
cluster_result = exemplar_selector.select_exemplars(cluster_result, graph_result)
```

---

## Stage F2: Conservative Merge (Optional)

**Purpose**: Merge over-fragmented clusters using multi-evidence approach

**When to use**: Connected components can over-fragment (many small clusters). Merge reduces fragmentation while staying conservative.

**Adaptive Threshold Formula**:
```
For cluster pair (A, B):
  T_A = P90(pairwise distances between exemplars in A)
  T_B = P90(pairwise distances between exemplars in B)
  T_local = MAX(T_A, T_B)
  T_global = median(T_1, T_2, ..., T_n) across all clusters
  T_merge = α × T_local + (1-α) × T_global

  where α = merge_threshold_alpha (default 0.7)
```

**Why MAX?** Allows merging across different density regions (looser cluster can merge with tighter one).

### Four Merge Criteria Summary

| Criterion | What It Checks | Default Params | When It Fails | Impact |
|-----------|----------------|----------------|---------------|--------|
| **1. Exemplar Distance** | Min distance between cluster exemplars ≤ T_merge | `merge_threshold_alpha=0.7` | Exemplars too far apart | Most common - clusters genuinely different |
| **2. Support Count** | Enough cross-cluster pairs ≤ T_merge | `merge_support_frac=0.3`<br>`merge_support_min=2` | Only exemplars close, but not enough other faces | Prevents merging on weak evidence |
| **3. Margin** | Cluster B is clear closest (not ambiguous) | `merge_margin=0.05`<br>**Recommend: 0.0** | Another cluster equally close | Often too strict - disable with 0.0 |
| **4. Diameter** | Post-merge cluster not too wide | `merge_diameter_expansion_factor=1.5` | Would create over-wide cluster | Prevents precision loss |

**Critical**: ALL four must pass. If any fails, merge is rejected.

---

### Four Merge Criteria (ALL must pass):

#### 1. Exemplar Distance ✓
**What**: Min distance between cluster exemplars ≤ T_merge

**Check**:
```python
exemplar_dist = min(distance_matrix[ex_a, ex_b]
                    for ex_a in exemplars_A
                    for ex_b in exemplars_B)
passes = exemplar_dist <= T_merge
```

**Purpose**: Don't merge if representative faces are too far apart

---

#### 2. Support Count ✓
**What**: Count cross-cluster pairs with distance ≤ T_merge, require minimum support

**Check**:
```python
support = sum(1 for n_a in cluster_A for n_b in cluster_B
              if distance_matrix[n_a, n_b] <= T_merge)
required = max(merge_support_frac × min(|A|, |B|), merge_support_min)
passes = support >= required
```

**Parameters**:
- `merge_support_frac` (default 0.3) - Fraction of smaller cluster
- `merge_support_min` (default 2) - Absolute minimum

**Purpose**: Require enough evidence beyond just close exemplars

---

#### 3. Margin (Clear Best) ⚠️
**What**: For each exemplar in cluster A, require that cluster B is the **clear closest** cluster

**Check** (for each exemplar `ex` in A):
```python
dist_to_B = min(distance_matrix[ex, n] for n in cluster_B)
dist_to_others = {
    cluster_id: min(distance_matrix[ex, n] for n in nodes)
    for cluster_id, nodes in all_other_clusters
}

# B must be closer by at least merge_margin
for other_id, dist_to_other in dist_to_others.items():
    if dist_to_B + merge_margin >= dist_to_other:
        return False  # Another cluster is equally close!

return True  # B is clear winner for all exemplars
```

**Parameter**: `merge_margin` (default 0.05)

**Concrete Example**:
```
Exemplar X from cluster A:
  - Distance to cluster B: 0.233
  - Distance to cluster C: 0.240
  - Distance to cluster D: 0.280

Check margin (merge_margin = 0.05):
  0.233 + 0.05 = 0.283
  Is 0.283 < 0.240? NO → FAIL

Reason: Cluster C is "too close" (within 0.05 margin of cluster B)
Result: Even though B is closest, the margin is too small → reject merge
```

**Purpose**: Prevent ambiguous merges when cluster is equally close to multiple others

**Problem**: Often too strict - blocks merges even when distances are small

**Recommendation**: Set `merge_margin = 0.0` to disable (threshold already controls conservatism)

---

#### 4. Post-Merge Diameter ✓
**What**: Check that merged cluster won't be too wide

**Check**:
```python
current_max_diameter = max(diameter_A, diameter_B)
post_diameter = max(distance_matrix[n1, n2]
                   for n1 in cluster_A + cluster_B
                   for n2 in cluster_A + cluster_B)
max_allowed = current_max_diameter × merge_diameter_expansion_factor
passes = post_diameter <= max_allowed
```

**Parameter**: `merge_diameter_expansion_factor` (default 1.5)

**Purpose**: Don't create over-wide clusters

---

### Iterative Merging

```python
while True:
    # Propose candidates (close exemplar pairs)
    candidates = [(A, B) for A, B in all_pairs
                  if min_exemplar_dist(A, B) <= merge_candidate_threshold]

    # Evaluate all 4 criteria for each candidate
    valid_merges = [(A, B) for A, B in candidates
                    if all_4_criteria_pass(A, B)]

    if not valid_merges:
        break  # No more valid merges

    # Merge best pair (smallest exemplar distance)
    best = min(valid_merges, key=lambda x: exemplar_dist(x[0], x[1]))
    merge_clusters(best[0], best[1])

    # Recompute thresholds (adaptive)
    recompute_cluster_thresholds()
```

**Parameters**:
- `merge_enabled` (default False) - Enable merge stage
- `merge_use_adaptive_threshold` (default True) - Use adaptive thresholds
- `merge_threshold_alpha` (default 0.7) - Local vs global weight
- `merge_candidate_threshold` (default 0.45) - Loose threshold for proposals
- `merge_margin` (default 0.05) - **Set to 0.0 to disable margin check**

**Code**:
```python
merger = ConservativeMerger(config)
cluster_result = merger.merge_clusters(cluster_result, graph_result)

# Access decision metadata
thresholds = merger.last_thresholds  # Per-cluster thresholds
candidates = merger.last_candidates  # All proposed merges with evidence
```

---

## Stage F3: Cluster Splitting (Optional)

**Purpose**: Safety valve to split over-wide clusters

**When to use**: If a cluster has diameter > threshold, try splitting

**Steps**:
1. Find clusters with `diameter > split_diameter_threshold`
2. For each, rebuild mutual kNN graph internally with tighter threshold
3. Split into connected components

**Parameters**:
- `split_enabled` (default False)
- `split_diameter_threshold` (default 0.6)
- `split_distance_threshold` (default 0.30)
- `split_K` (default 3)

---

## Stage F4: Holdout Attachment (Optional)

**Purpose**: Attach low-quality holdout faces to clusters

**Steps**:
1. For each holdout face, find K nearest neighbors in core set
2. Vote: which cluster do they belong to?
3. Attach if:
   - ≥ vote_min votes for one cluster AND
   - distance ≤ attach_distance_threshold AND
   - margin to second-best cluster ≥ margin

**Parameters**:
- `attach_enabled` (default False)
- `K_attach` (default 5)
- `vote_min` (default 3)
- `attach_distance_threshold` (default 0.35)
- `margin` (default 0.1)

**Code**:
```python
attacher = HoldoutAttacher(config)
cluster_result = attacher.attach_holdout(cluster_result, faces, core_indices, holdout_indices, distance_matrix)
```

---

## Complete Production Pipeline

```python
from face_cluster import (
    PipelineConfig,
    QualityGater,
    KNNGraphBuilder,
    ConnectedComponentsClusterer,
    D10ExemplarSelector,
    ConservativeMerger,
    HoldoutAttacher,
)

# Config
config = PipelineConfig(
    K=5,
    distance_threshold=0.35,
    min_cluster_size=2,
    blur_min=50.0,
    yaw_max=30.0,
    merge_enabled=True,
    merge_margin=0.0,  # Disable margin check
)

# Stage A: Quality gating
gater = QualityGater(config)
faces = gater.compute_blur_scores(faces)
faces = gater.compute_pose_scores(faces)
core_indices, holdout_indices = gater.select_core_set(faces)

# Stage B: Distance matrix
graph_builder = KNNGraphBuilder(config)
distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)

# Stage C: Mutual kNN graph
graph_result = graph_builder.build_mutual_knn_graph(
    distance_matrix, config.K, config.distance_threshold
)

# Stage D: Connected components
clusterer = ConnectedComponentsClusterer(config)
cluster_result = clusterer.cluster(graph_result, core_indices)

# Stage E: Exemplar selection
exemplar_selector = D10ExemplarSelector(config)
cluster_result = exemplar_selector.select_exemplars(cluster_result, graph_result)

# Stage F2: Conservative merge (optional)
if config.merge_enabled:
    merger = ConservativeMerger(config)
    cluster_result = merger.merge_clusters(cluster_result, graph_result)

# Stage F4: Holdout attachment (optional)
if config.attach_enabled:
    attacher = HoldoutAttacher(config)
    cluster_result = attacher.attach_holdout(
        cluster_result, faces, core_indices, holdout_indices, distance_matrix
    )

# Result
print(f"Clusters: {cluster_result.n_clusters}")
print(f"Noise: {cluster_result.n_noise}")
```

---

## Analysis & Debugging

### Understanding Merge Decisions vs Close Clusters

Two different DataFrames show different perspectives:

| DataFrame | Distance Metric | Purpose | What It Shows |
|-----------|----------------|---------|---------------|
| **Merge Decisions** | `Exemplar_Dist`<br>(min distance between **exemplars only**) | Why did/didn't merges happen? | Only pairs where exemplar_dist ≤ 0.45<br>Shows which criteria failed |
| **Close Clusters** | `Min_Dist`<br>(min distance between **any two faces**) | Which clusters are geometrically close? | All cluster pairs<br>Shows sensitivity to threshold |

**Key Insight**: A cluster pair can have small `Min_Dist` (some faces are close) but large `Exemplar_Dist` (exemplars are far apart). Such pairs won't be considered for merging.

**Example**:
```
Cluster pair (4, 23):
  Min_Dist = 0.278       ← Some face in 4 is close to some face in 23
  Exemplar_Dist = 0.520  ← But exemplars are far apart (not in merge decisions)

  Why not merged? Exemplar distance > 0.45 (not even proposed as candidate)
```

**Example**:
```
Cluster pair (3, 6):
  Min_Dist = 0.379       ← Closest two faces (any)
  Exemplar_Dist = 0.413  ← Closest two exemplars (different faces!)
  T_merge = 0.233

  In Merge Decisions? YES (exemplar_dist < 0.45, so it was proposed)
  Merged? NO (failed Exemplar: 0.413 > 0.233, also Support and Diameter)
```

---

### Interpreting Failure Reasons

When a merge fails, the `Failed` column shows which criteria didn't pass:

| Failure Pattern | Meaning | Common Cause | What To Do |
|----------------|---------|--------------|------------|
| **Exemplar only** | Exemplars too far, but support/diameter OK | Distance > threshold by small margin | Lower alpha (more global) OR these are genuinely different clusters |
| **Support only** | Exemplars close, but not enough other evidence | Only exemplars close, rest of clusters far | Normal - prevents merging on outliers |
| **Margin only** | B not clear closest (ambiguous) | Multiple clusters equally close | Set `merge_margin=0.0` to disable |
| **Diameter only** | Would create too wide cluster | Clusters have different spreads | Increase `merge_diameter_expansion_factor` |
| **Exemplar + Support + Diameter** | Clusters genuinely different | Large distance, different densities | **Don't merge** - likely different people |

**Example Analysis**:
```
Merge Decisions:
C1 C2  Exemplar_Dist  T_merge  Gap   Failed
3  6   0.413         0.233    +0.18  Exemplar, Support, Diameter

Interpretation:
  - Gap = +0.18 (positive = fail exemplar check)
  - Exemplar distance is 1.8× the threshold
  - Failed on 3 out of 4 criteria
  - Conclusion: Clusters 3 and 6 are GENUINELY DIFFERENT (probably different people)

Would lowering alpha help?
  Current (alpha=0.7): T_merge = 0.7×0.202 + 0.3×0.307 = 0.233
  More global (alpha=0.5): T_merge = 0.5×0.202 + 0.5×0.307 = 0.255
  Even more (alpha=0.3): T_merge = 0.3×0.202 + 0.7×0.307 = 0.276

  NO! Even with alpha=0.3, exemplar_dist (0.413) still >> T_merge (0.276)
  These clusters have tight internal structure (T_A=0.202, T_B=0.191) but are
  far apart from each other. This is exactly what we want to preserve.
```

---

### ClusterSnapshot for Analysis

```python
from face_cluster.analysis import ClusterSnapshot

# Create snapshot after any stage
snapshot = ClusterSnapshot.from_result(
    cluster_result, faces, core_indices, distance_matrix,
    stage="after_merge",
    config=config,
    cluster_thresholds=merger.last_thresholds,
    merge_candidates=merger.last_candidates
)

# Standard analyses
snapshot.print_summary()
snapshot.plot_overview()
snapshot.plot_widest_clusters(top_k=3)

# Merge decision analysis
df = snapshot.get_merge_decisions_df()
print(df)  # See why clusters didn't merge

# Close clusters (sensitivity)
df2 = snapshot.get_close_clusters_df()
print(df2)  # See which clusters are nearly merging
```

### Understanding Merge Decisions

DataFrame shows for each cluster pair:
- `Exemplar_Dist`: Distance between closest exemplars
- `T_merge`: Adaptive threshold for this pair
- `Gap`: Exemplar_Dist - T_merge (negative = passes exemplar check)
- `Failed`: Which criteria failed (Exemplar, Support, Margin, Diameter)

If all failing on "Margin" with negative Gap → set `merge_margin=0.0`

---

## Key Design Decisions

1. **Mutual kNN** (not regular kNN): Bidirectional edges = higher precision
2. **Connected components** (not HDBSCAN): Deterministic, interpretable
3. **d10 exemplar selection**: Density-based, selects core of each cluster
4. **Adaptive merge thresholds**: Per-cluster P90 + global median = robust across different densities
5. **Multi-evidence merge**: Four criteria prevent both under-merging and over-merging
6. **Quality gating first**: Defer low-quality faces, cluster on high-quality only

---

## Troubleshooting

**Problem**: Too many small clusters (over-fragmentation)
- **Solution**: Enable merge (`merge_enabled=True`) with `merge_margin=0.0`

**Problem**: All merges failing on "Margin"
- **Solution**: Set `merge_margin=0.0` to disable margin check

**Problem**: Clusters are too wide (low precision)
- **Solution**: Lower `distance_threshold` or enable `split_enabled=True`

**Problem**: Different people in same cluster
- **Solution**: Lower `distance_threshold` or increase `min_cluster_size`

**Problem**: Same person in multiple clusters
- **Solution**: Enable merge with lower `merge_threshold_alpha` (more global)
