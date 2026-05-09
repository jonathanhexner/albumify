# ML Merge Classifier — Feature Engineering Plan

**Date**: 2026-04-12
**Status**: Draft — pending review
**Scope**: Features for binary classification: *should cluster pair (Ci, Cj) be merged?*

---

## Existing Features (V2 in `face_cluster/features.py`)

| Feature | Type | Description |
|---------|------|-------------|
| `min_exemplar_dist` | float | Min cosine distance between any two exemplars across clusters |
| `p10_cross_dist` | float | 10th percentile of all cross-cluster face distances |
| `p50_cross_dist` | float | Median cross-cluster distance |
| `support_fraction` | float | Fraction of min(|Ci|,|Cj|) face pairs below support_threshold |
| `diameter_ratio` | float | max(dia_a, dia_b) / min(dia_a, dia_b) — measures imbalance |
| `cluster_size_min` | int | Smaller cluster size |
| `cluster_size_ratio` | float | max(size_a, size_b) / min(size_a, size_b) |
| `T_A`, `T_B` | float | P90 intra-exemplar distances per cluster (per-cluster compactness) |
| `T_local` | float | max(T_A, T_B) — the looser of the two local thresholds |
| `T_global` | float | Global P-based threshold across all clusters |
| `frontal_frac_A`, `frontal_frac_B` | float | Fraction of frontal faces per cluster |
| `pose_diff` | float | Euclidean distance between mean (yaw, pitch) of each cluster |
| `min_exemplar_dist_x_pose` | float | min_exemplar_dist × (1 + pose_diff/90) |
| `p50_cross_dist_x_pose` | float | p50_cross_dist × (1 + pose_diff/90) |

---

## Proposed New Features

### Group A — Cross-Cluster Distance Distribution (high signal)

| Feature | Description | Why |
|---------|-------------|-----|
| `p25_cross_dist` | 25th percentile of cross-cluster distances | Finer distribution sampling between p10 and p50 |
| `p75_cross_dist` | 75th percentile | Measures how the tail looks; two-identity mixtures have high p75 |
| `p90_cross_dist` | 90th percentile | Hard outlier signal |
| `min_cross_dist` | Min distance between ANY face pair (not just exemplars) | May be lower than min_exemplar_dist when exemplar selection misses the closest pair |
| `cross_dist_iqr` | p75 - p25 (interquartile range) | Wide distribution → possibly two different identities with some overlap |
| `n_cross_pairs_under_threshold` | Absolute count of pairs below threshold | Complements support_fraction for small vs large clusters |
| `exemplar_dist_mean` | Mean over all exemplar-pair distances | Smoother than min; less sensitive to one anomalous exemplar |
| `exemplar_dist_std` | Std of exemplar-pair distances | High std → exemplar quality is uneven |

### Group B — Cluster Size & Structure (high signal, easy)

| Feature | Description | Why |
|---------|-------------|-----|
| `size_a`, `size_b` | Individual cluster sizes | Absolute size matters; merging two large clusters is riskier |
| `size_sum` | size_a + size_b | Post-merge total; correlated with post-merge contamination risk |
| `diameter_a`, `diameter_b` | Individual cluster diameters | Need both raw values, not just their ratio |
| `diameter_max` | max(diameter_a, diameter_b) | Upper bound on cluster quality |
| `post_merge_diameter` | Hypothetical diameter of merged cluster | Direct measurement of "would this merge be too wide?" |
| `diameter_expansion_factor` | post_merge_diameter / max(diameter_a, diameter_b) | How much does merging expand the widest cluster? |
| `exemplar_count_a`, `exemplar_count_b` | Number of exemplars per cluster | Clusters with more exemplars have better distance estimates |
| `exemplar_density_a` | exemplar_count_a / size_a | Low density → exemplar selection chose few representatives |

### Group C — Image Diversity (very high signal, not yet used)

These features exploit the fact that **two faces from the same source photo are almost always different people**.

| Feature | Description | Why |
|---------|-------------|-----|
| `n_images_a`, `n_images_b` | Unique source images per cluster | Clusters with many images covering many people vs. one person many times |
| `images_per_face_a`, `images_per_face_b` | n_images / size (diversity ratio) | ~1.0 → each image contributes 1 face (diverse dataset); <0.5 → many face detections per image |
| `shared_source_images_count` | Images appearing in BOTH clusters | **Very strong signal**: if shared > 0, two faces from same photo are split across clusters → suggests they are different people → do NOT merge |
| `shared_source_images_ratio` | shared / min(n_images_a, n_images_b) | Normalized version of above |
| `same_image_min_dist` | Min distance between any face pair sharing a source image (one from Ci, one from Cj) | If small AND same image: two faces from same photo are close → likely detection duplicate, not two people |
| `n_images_post_merge` | n_images_a + n_images_b − shared_source_images_count | Estimated diversity of merged cluster |

> **Implementation note**: Requires `FaceRecord.image_path` (already populated). Group faces by `Path(image_path).name` to compute per-cluster `n_images`.

### Group D — Graph Structure (high signal, requires kNN graph)

| Feature | Description | Why |
|---------|-------------|-----|
| `knn_edge_count` | Number of kNN graph edges between Ci and Cj | Direct graph connectivity: many edges → strong evidence for same identity |
| `edge_density` | knn_edge_count / (size_a × size_b) | Normalized; independent of cluster size |
| `bidirectional_edges` | Edges that are mutual (A→B and B→A) | Mutual edges are much stronger evidence than one-directional |
| `bidirectional_density` | bidirectional_edges / (size_a × size_b) | Normalized version |
| `max_degree_into_b` | Max kNN edges any single Ci node has pointing to Cj | One cluster node highly connected to the other → strong anchor |
| `hub_node_count` | Nodes with ≥ 2 edges into the other cluster | Multiple anchors → more reliable than single-hub connectivity |

> **Implementation note**: kNN graph is already built in `face_cluster/knn_graph.py`. The graph edges reference graph-local (core) indices. Needs mapping to cluster membership.

### Group E — Disambiguation / Context (important for precision)

These features help distinguish "this pair is close AND it's the best match" vs. "this pair is close but there's a better/competing candidate".

| Feature | Description | Why |
|---------|-------------|-----|
| `second_nearest_dist_a` | Min exemplar dist from Ci to its second-nearest cluster (not Cj) | Margin signal: if second_nearest ≈ min_exemplar_dist, identity is ambiguous |
| `second_nearest_dist_b` | Same for Cj | |
| `margin_a` | second_nearest_dist_a − min_exemplar_dist (Ci→Cj) | Large margin → Cj is clearly the best match for Ci |
| `margin_b` | Same from Cj's perspective | |
| `margin_min` | min(margin_a, margin_b) | Overall disambiguation quality |
| `rank_among_candidates` | Rank of this pair by min_exemplar_dist (1 = best) | Candidate ranked #1 is more likely a true merge than #20 |
| `n_candidates_total` | Total number of candidate pairs in this run | Dataset difficulty context |
| `dist_to_threshold_ratio` | min_exemplar_dist / merge_exemplar_threshold | How far below the gate? 0.5 = very safe; 0.99 = borderline |

### Group F — Quality / Pose Distribution

| Feature | Description | Why |
|---------|-------------|-----|
| `mean_blur_a`, `mean_blur_b` | Mean blur score per cluster | Low-blur cluster has higher-quality embeddings → its distances are more reliable |
| `blur_min_a`, `blur_min_b` | Min blur score (worst face) | Very blurry clusters have unreliable embeddings |
| `frontal_frac_min` | min(frontal_frac_a, frontal_frac_b) | Both clusters need frontal faces for a reliable comparison |
| `yaw_std_a`, `yaw_std_b` | Std of yaw within cluster | High std → cluster spans multiple poses; harder to confirm same identity |
| `pose_coverage_a`, `pose_coverage_b` | n_faces_with_pose / size | Low coverage → pose features unreliable |
| `mean_area_a`, `mean_area_b` | Mean face area in pixels | Small faces have lower embedding quality |
| `area_ratio` | max(mean_area_a, mean_area_b) / min(...) | Very different sizes → possibly different camera distances, reduces embedding comparability |

### Group G — Global Context

| Feature | Description | Why |
|---------|-------------|-----|
| `n_total_clusters` | Total clusters in this run | Large albums have more between-cluster competition |
| `global_size_max`, `global_size_min` | Max/min cluster sizes globally | Normalization reference; is this pair unusually small? |
| `size_a_pct_of_max`, `size_b_pct_of_max` | size / global_size_max | Relative size of each cluster |
| `global_dist_p10`, `global_dist_p50` | Global percentiles of all inter-cluster min_exemplar_dists | Where does this pair sit in the overall distance landscape? |
| `dist_zscore` | (min_exemplar_dist − global_dist_mean) / global_dist_std | Standardized distance; pair 3 std below mean is very likely true merge |

---

## Feature Priority Summary

| Priority | Group | Rationale |
|----------|-------|-----------|
| **P1 (MVP)** | Existing V2 + `diameter_a,b` + `size_a,b` + `post_merge_diameter` + `diameter_expansion_factor` | Core geometry; already partially implemented |
| **P1 (MVP)** | C: `n_images_a,b`, `shared_source_images_count`, `shared_source_images_ratio` | Strong identity signal unique to albums; cheap to compute |
| **P2 (Next)** | A: `p25`, `p75`, `p90_cross_dist`, `cross_dist_iqr`, `exemplar_dist_mean/std` | Richer distribution shape; high info density |
| **P2 (Next)** | E: `margin_a,b`, `second_nearest_dist_a,b`, `dist_to_threshold_ratio`, `rank_among_candidates` | Prevents precision errors in ambiguous cases |
| **P3 (Later)** | D: kNN edge features | High signal but requires exposing graph object to feature computer |
| **P3 (Later)** | F: blur/pose distribution, area features | Incremental; useful when embedding quality varies |
| **P4 (Experimental)** | G: global context | Useful for cross-run calibration; adds complexity |

---

## Implementation Notes

### What needs to change in `face_cluster/features.py`

1. `ClusterPairFeatures` dataclass — add new fields (all optional for backwards compat)
2. `FeatureComputer.compute_cluster_stats()` — add `n_images`, `blur_min`, `yaw_std`, `mean_area` to per-cluster stats
3. `FeatureComputer.compute_pair_features()` — add cross-cluster and post-merge computations
4. New `VERSION = 3` feature set constant; keep V1/V2 sets for backwards compat
5. Add `compute_global_context(all_cluster_stats, all_candidates)` helper for Group G features

### Data requirements per feature group

| Group | What's needed |
|-------|---------------|
| A, B, E | Distance matrix + cluster membership (already available) |
| C | `FaceRecord.image_path` per face (already populated) |
| D | `graph_result.edges` from kNN graph (passed separately) |
| F | `FaceRecord.blur_score`, `.pose`, `.area` (already in FaceRecord) |
| G | Computed after all cluster-pair features are ready |

### Training label source

Labels come from `merge_decisions.json` (saved by the Interactive Merge Approval UI):
- `"approve"` → positive (should merge)
- `"reject"` → negative (should not merge)
- No decision → exclude from training

---

## Feature Correlation / Redundancy Notes

- `min_exemplar_dist` and `p10_cross_dist` are strongly correlated (r ≈ 0.85). Keep both; tree models handle redundancy well, but consider dropping one for linear models.
- `T_A`, `T_B`, `T_local` are derived from exemplar distances — redundant with `diameter_a,b` for compact clusters. May be dropped in favor of simpler diameter features.
- `shared_source_images_count` subsumes `same_image_pair_exists` (bool) — include count only.
- Interaction features (`x_pose`) are dataset-specific. Only include if pose data is reliably present (≥70% of faces have pose).

---

## Open Questions

1. **Graph exposure**: kNN graph (`face_cluster/knn_graph.py`) returns a `KNNGraphResult` object. Feature computer would need this passed in. Is this acceptable architecture?
2. **P75/P90 stability**: With small clusters (size < 5), high percentiles are noisy. Cap at available data or add `cross_dist_sample_size` as a feature.
3. **same_image_min_dist**: Requires iterating cross-cluster pairs filtered by shared source image. O(n²) but only for shared-image pairs (usually rare). Acceptable?
4. **Global context recomputation**: Group G features require a two-pass computation (compute per-pair first, then normalize). Acceptable for batch training; may need lazy computation for online inference.
