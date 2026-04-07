# Merge Analysis Quick Reference

## Question 1: Why aren't these merge candidates?

**Example from your output:**
```
Closest Cluster Pairs:
 C1  C2  Size1  Size2  Min_Dist  T_merge    Gap
  4  23     27      3     0.278    0.444 -0.166
  3   6      6      4     0.379    0.233  0.146
```

**Answer**: Check the **Merge Decisions** DataFrame (not Closest Cluster Pairs):

```python
# Get merge decisions
merge_df = merged_snapshot.get_merge_decisions_df()
print(merge_df)
```

**If a pair is NOT in the Merge Decisions table:**
- Means `Exemplar_Dist > 0.45` (not even proposed)
- **Closest Cluster Pairs** uses `Min_Dist` = minimum distance between ANY two faces
- **Merge Decisions** uses `Exemplar_Dist` = minimum distance between EXEMPLARS only

**Example interpretation:**
- Pair (4, 23): `Min_Dist = 0.278` ← Some outlier faces are close
- But if not in merge decisions → `Exemplar_Dist > 0.45` ← Representative faces (cores) are far apart
- **This is good behavior** - prevents merging based on outliers

---

## Question 2: What criteria failed?

**See the `Failed` column in Merge Decisions DataFrame:**

```python
merge_df = merged_snapshot.get_merge_decisions_df()

# Show only failed merges
failed = merge_df[~merge_df['Merged']]
print(failed[['C1', 'C2', 'Exemplar_Dist', 'T_merge', 'Gap', 'Failed']])
```

**Example output:**
```
C1  C2  Exemplar_Dist  T_merge   Gap    Failed
1   18  0.233         0.339     -0.106  Margin
4   9   0.276         0.318     -0.043  Margin, Support
3   6   0.413         0.233     +0.180  Exemplar, Support, Diameter
```

**Interpretation:**
- **"Margin"** only → Set `merge_margin=0.0` to disable
- **"Support"** only → Only exemplars close, not enough other faces (correct rejection)
- **"Exemplar"** only, small Gap (< 0.05) → Consider lowering alpha or raising global percentile
- **Multiple failures** (Exemplar + Support + Diameter) → Genuinely different clusters (DON'T merge!)

---

## Question 3: How to get cluster_struct after merger?

```python
from face_cluster.analysis import ClusterSnapshot

# After running merger
merger = ConservativeMerger(config)
merged_result = merger.merge_clusters(initial_result, graph_result)

# Create snapshot (this is your "cluster_struct")
merged_snapshot = ClusterSnapshot.from_result(
    merged_result,
    faces,
    core_indices,
    distance_matrix,  # Same distance matrix from before
    stage="after_merge",
    config=config,
    cluster_thresholds=merger.last_thresholds,  # IMPORTANT: Include for analysis
    merge_candidates=merger.last_candidates      # IMPORTANT: Include for analysis
)

# Now use it
merged_snapshot.print_summary()
merge_df = merged_snapshot.get_merge_decisions_df()
merged_snapshot.plot_overview()
```

---

## Question 4: How to get distance matrix?

**Distance matrix is computed once and reused:**

```python
# Compute during graph building (Stage B)
graph_builder = KNNGraphBuilder(config)
distance_matrix = graph_builder.build_distance_matrix(faces, core_indices)

# Same distance matrix used throughout:
graph_result = graph_builder.build_mutual_knn_graph(distance_matrix, K, threshold)
initial_result = clusterer.cluster(graph_result, core_indices)
merged_result = merger.merge_clusters(initial_result, graph_result)

# Access later from snapshot
print(merged_snapshot.distance_matrix.shape)  # (n_core, n_core)

# Or from graph_result
distance_matrix = graph_result.distance_matrix
```

**Example usage:**
```python
# Distance between two specific nodes
node_a = 10
node_b = 15
dist = distance_matrix[node_a, node_b]

# All distances within a cluster
cluster_nodes = merged_snapshot.clusters[cluster_id]
intra_distances = [
    distance_matrix[node_a, node_b]
    for i, node_a in enumerate(cluster_nodes)
    for node_b in cluster_nodes[i+1:]
]
```

---

## New Feature: Experiment with `merge_global_percentile`

**What it does:**
- Controls which percentile of cluster thresholds to use for global threshold
- Default 50 (median)
- Higher values → more permissive merging

**Example:**
```python
config = PipelineConfig(
    merge_enabled=True,
    merge_global_percentile=75,  # Try 25, 50, 75, 90
    merge_margin=0.0,
    merge_threshold_alpha=0.7,
)

# Run merge
merger = ConservativeMerger(config)
merged_result = merger.merge_clusters(initial_result, graph_result)

# Check result
print(f"Clusters: {merged_result.n_clusters}, Noise: {merged_result.n_noise}")
```

**Impact:**
```
If cluster thresholds are [0.20, 0.25, 0.30, 0.35, 0.40]:
  P25 = 0.25 (conservative)
  P50 = 0.30 (median, default)
  P75 = 0.35 (more permissive)
  P90 = 0.39 (very permissive)

For pair with T_local=0.25, alpha=0.7:
  With P25: T_merge = 0.7×0.25 + 0.3×0.25 = 0.250
  With P50: T_merge = 0.7×0.25 + 0.3×0.30 = 0.265
  With P75: T_merge = 0.7×0.25 + 0.3×0.35 = 0.280

Higher T_merge → more pairs pass exemplar distance check → more merging
```

---

## Full Example

See `notebooks/analyze_merge_decisions.ipynb` for complete working example with:
- How to create snapshots before/after merging
- How to analyze failed merge criteria
- How to compare close pairs vs merge candidates
- How to experiment with different global percentiles
- How to visualize merge decisions

**Key code snippet:**
```python
# After running all stages
merged_snapshot = ClusterSnapshot.from_result(
    merged_result, faces, core_indices, distance_matrix,
    stage="after_merge", config=config,
    cluster_thresholds=merger.last_thresholds,
    merge_candidates=merger.last_candidates
)

# Analyze merge decisions
merge_df = merged_snapshot.get_merge_decisions_df()
if merge_df is not None:
    print("\nFailed merges:")
    failed = merge_df[~merge_df['Merged']]
    for _, row in failed.iterrows():
        print(f"({row['C1']}, {row['C2']}): {row['Failed']}")
        if row['Gap'] > 0.1:
            print("  → Genuinely different clusters")
        elif row['Failed'] == 'Margin':
            print("  → Try merge_margin=0.0")

# Compare with close pairs
close_df = merged_snapshot.get_close_clusters_df(top_k=10)
print(close_df)
```
