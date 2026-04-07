# Exemplar Selection Algorithm

**Purpose**: Select representative faces for each cluster to enable cluster comparison and merging.

---

## Overview

**Exemplars** are high-quality, representative faces within a cluster that:
- Are central to the cluster (low distance to other faces)
- Are diverse enough to cover cluster variation
- Enable efficient cluster-to-cluster comparison

Instead of comparing all faces between clusters, we compare only exemplars, reducing complexity from O(n²) to O(k²) where k << n.

---

## Algorithm: d10-Based Selection

### Intuition

The **d10 metric** measures local density:
- d10(face) = distance to its 10th nearest neighbor within the cluster
- **Low d10** → face is in dense region (many close neighbors)
- **High d10** → face is outlier (far from others)

Exemplars should have **low d10** (high density) to represent the cluster well.

### Process

```python
def select_exemplars(
    cluster: List[int],
    distance_matrix: np.ndarray,
    config: PipelineConfig
) -> List[int]:
    """
    Select up to N exemplars using d10 metric with suppression.

    Steps:
    1. Compute d10 value for each face in cluster
    2. Filter candidates: d10 <= threshold
    3. Sort candidates by d10 (ascending - lower is better)
    4. Greedily select exemplars with suppression radius

    Returns:
        List of exemplar face indices
    """
```

**Step 1: Compute d10 Values**

```python
def compute_d10(cluster_nodes: List[int], distance_matrix: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Compute d10 (distance to kth nearest neighbor) for each node.

    Args:
        cluster_nodes: Face indices in cluster
        distance_matrix: Full pairwise distance matrix
        k: Which neighbor to use (default: 10)

    Returns:
        Array of d10 values (one per face)
    """
    n = len(cluster_nodes)
    if n <= k:
        k = n - 1  # Use furthest neighbor if cluster too small

    # Extract cluster distance submatrix
    indices = np.array(cluster_nodes)
    cluster_dists = distance_matrix[np.ix_(indices, indices)]

    d10_values = []
    for i in range(n):
        # Get distances to all other faces (exclude self)
        dists = cluster_dists[i].copy()
        dists[i] = np.inf  # Exclude self

        # Sort and get kth nearest
        sorted_dists = np.sort(dists)
        d10 = sorted_dists[k]  # 0-indexed, so k = 10th neighbor
        d10_values.append(d10)

    return np.array(d10_values)
```

**Step 2: Filter Candidates**

```python
# Only consider faces with low d10 (high density)
d10_values = compute_d10(cluster_nodes, distance_matrix, k=config.d10_k)
candidate_mask = d10_values <= config.exemplars_d10_threshold
candidate_indices = np.where(candidate_mask)[0]

if len(candidate_indices) == 0:
    # No candidates, use best face
    best_idx = int(np.argmin(d10_values))
    return [cluster_nodes[best_idx]]
```

**Step 3: Sort by d10**

```python
# Sort candidates by d10 (ascending - lower is better)
sorted_candidates = candidate_indices[np.argsort(d10_values[candidate_indices])]
```

**Step 4: Greedy Selection with Suppression**

```python
selected = []

for i in sorted_candidates:
    # Check if this candidate is too close to already selected exemplars
    too_close = False
    for j in selected:
        dist = cluster_dists[i, j]
        if dist < config.exemplar_suppression_radius:
            too_close = True
            break

    if not too_close:
        selected.append(i)

        # Stop if we have enough exemplars
        if len(selected) >= config.N_exemplars_max:
            break

# Convert to original face indices
exemplars = [cluster_nodes[i] for i in selected]
return exemplars
```

---

## Configuration Parameters

```yaml
select_exemplars:
  d10_k: 10                          # K for d10 metric (distance to Kth neighbor)
  exemplars_d10_threshold: 0.25      # Max d10 value for candidates
  exemplar_suppression_radius: 0.15  # Min distance between exemplars
  N_exemplars_max: 5                 # Max exemplars per cluster
```

### Tuning Guidelines

**For high-quality exemplars** (strict, few exemplars):
- `exemplars_d10_threshold: 0.20` (only very dense regions)
- `exemplar_suppression_radius: 0.20` (widely spaced)
- `N_exemplars_max: 3` (top 3 only)

**For diverse coverage** (relaxed, more exemplars):
- `exemplars_d10_threshold: 0.30` (allow moderate density)
- `exemplar_suppression_radius: 0.10` (closer spacing)
- `N_exemplars_max: 10` (up to 10 exemplars)

**Recommended** (balanced):
- `exemplars_d10_threshold: 0.25`
- `exemplar_suppression_radius: 0.15`
- `N_exemplars_max: 5`

---

## Why d10 Works

### Density-Based Quality
- Faces in dense regions → similar to many neighbors → representative
- Outlier faces → high d10 → NOT selected as exemplars

### Noise Robustness
- Single outlier face → high d10 → excluded
- Dense core of cluster → low d10 → selected

### Scale Invariance
- d10 adapts to cluster size and spread
- Large clusters: d10 threshold filters to dense core
- Small clusters: d10 uses fewer neighbors (k = min(10, n-1))

---

## Suppression Radius Explained

**Problem**: Without suppression, exemplars could be very similar
```
Cluster with 3 near-identical faces + 10 others
→ All 3 near-identical faces have low d10
→ All 3 selected as exemplars (redundant!)
```

**Solution**: Enforce minimum distance between exemplars
```python
if distance(candidate, any_selected_exemplar) < suppression_radius:
    skip candidate  # Too similar to existing exemplar
```

This ensures exemplars are **diverse** (cover different parts of cluster).

---

## Example Walkthrough

### Setup
- Cluster of 8 faces
- Distance matrix (cosine distance)
- Config: d10_k=5, threshold=0.25, suppression=0.15, max=3

### Step 1: Compute d10
```
Face   d10 value   (distance to 5th nearest)
0      0.18        ← candidate (d10 <= 0.25)
1      0.22        ← candidate
2      0.31        ✗ rejected (d10 > 0.25)
3      0.15        ← candidate (best!)
4      0.28        ✗ rejected
5      0.20        ← candidate
6      0.42        ✗ rejected (outlier)
7      0.24        ← candidate
```

### Step 2: Sort Candidates by d10
```
Sorted: [3, 0, 5, 1, 7]
        0.15, 0.18, 0.20, 0.22, 0.24
```

### Step 3: Greedy Selection
```
1. Select face 3 (d10=0.15, best)
   selected = [3]

2. Consider face 0 (d10=0.18)
   distance(0, 3) = 0.12 < 0.15 (suppression radius)
   → TOO CLOSE, skip

3. Consider face 5 (d10=0.20)
   distance(5, 3) = 0.22 > 0.15
   → OK, select
   selected = [3, 5]

4. Consider face 1 (d10=0.22)
   distance(1, 3) = 0.18 > 0.15
   distance(1, 5) = 0.25 > 0.15
   → OK, select
   selected = [3, 5, 1]

5. Stop (reached N_exemplars_max = 3)
```

### Result
```
Exemplars: [3, 5, 1]
- Face 3: d10=0.15 (densest)
- Face 5: d10=0.20 (moderate)
- Face 1: d10=0.22 (moderate)

All have low d10 AND are spaced apart (distance > 0.15)
```

---

## Edge Cases

### Small Cluster (n < d10_k)
```python
if cluster_size < d10_k:
    # Use all neighbors except self
    actual_k = cluster_size - 1
    d10_values = compute_d10(cluster, distance_matrix, k=actual_k)
```

### No Candidates Pass Threshold
```python
if len(candidates) == 0:
    # Fall back to single best face
    best_idx = np.argmin(d10_values)
    return [cluster_nodes[best_idx]]
```

### Single Face Cluster
```python
if cluster_size == 1:
    return cluster_nodes  # Only face is the exemplar
```

### All Candidates Suppressed
```python
# If suppression radius too large, no exemplars selected
# → Take the best candidate anyway
if len(selected) == 0:
    best_candidate = sorted_candidates[0]
    return [cluster_nodes[best_candidate]]
```

---

## Performance

**Complexity**:
- d10 computation: O(n² log n) per cluster (n = cluster size)
- Greedy selection: O(k²) where k = number of candidates
- Total: O(n² log n) dominated by distance sorting

**Typical timing**:
- Cluster of 10 faces: < 1 ms
- Cluster of 100 faces: 10-20 ms
- Cluster of 1000 faces: 1-2 seconds

**Optimization**: Use KD-tree or approximate nearest neighbors for very large clusters

---

## Usage in Cluster Merging

Exemplars enable efficient cluster comparison:

```python
def should_merge(cluster_a, cluster_b, exemplars_a, exemplars_b, threshold):
    """
    Decide if two clusters should merge based on exemplar distances.

    Args:
        cluster_a, cluster_b: Cluster face indices
        exemplars_a, exemplars_b: Exemplar face indices
        threshold: Max distance to merge

    Returns:
        True if min exemplar distance <= threshold
    """
    # Compute all pairwise distances between exemplars
    min_dist = np.inf
    for ex_a in exemplars_a:
        for ex_b in exemplars_b:
            dist = distance_matrix[ex_a, ex_b]
            if dist < min_dist:
                min_dist = dist

    return min_dist <= threshold
```

**Efficiency**:
- Without exemplars: Compare all faces → O(|A| × |B|)
- With exemplars: Compare only exemplars → O(|EA| × |EB|)
- Speedup: ~100x for large clusters (100 faces → 5 exemplars)

---

## Statistics & Validation

After exemplar selection, log:

```
Exemplar Selection Results:
  Cluster 0: 3 exemplars from 12 faces
  Cluster 1: 5 exemplars from 28 faces
  Cluster 2: 2 exemplars from 5 faces
  Cluster 3: 1 exemplar from 2 faces

  Average d10 values:
    Exemplars: 0.18
    Non-exemplars: 0.32
  → Exemplars are 44% denser than average
```

**Validation**: Each cluster should have at least 1 exemplar
```python
for cluster_id in clusters:
    assert len(exemplars[cluster_id]) > 0, \
        f"Cluster {cluster_id} has no exemplars"
```

---

## Integration with Pipeline

```python
# Pipeline step: select_exemplars.py
class SelectExemplarsStep(BaseStep):
    def process(self, context: PipelineContext, config: dict):
        cluster_result = context.initial_clusters
        graph_result = context.knn_graph_result

        # Create exemplar selector
        selector = D10ExemplarSelector(config)

        # Select exemplars for each cluster
        cluster_result = selector.select_exemplars(cluster_result, graph_result)

        # Update context
        context.initial_clusters = cluster_result
```

---

## References

- **Density-based selection**: DBSCAN ([Ester et al., 1996](https://dl.acm.org/doi/10.5555/3001460.3001507))
- **Representative selection**: [Elhamifar & Kaluza, 2016](https://arxiv.org/abs/1602.06195)

---

**See Also**:
- [KNN Graph Clustering](knn_graph.md) - How clusters are formed
- [ML-Based Merging](ml_merging.md) - How exemplars are used for merging
- [Pipeline Steps](../pipeline/steps.md) - Pipeline integration details
