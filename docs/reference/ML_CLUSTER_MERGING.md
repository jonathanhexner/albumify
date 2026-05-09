# ML-Based Cluster Merging

**Date**: 2026-04-14  
**Status**: Spec / Design  
**Owner**: Jonathan Hexner

---

## 1. Motivation

The current `ConservativeMerger` uses a hand-tuned, gate-based heuristic: four independent gates (exemplar distance, support count, margin, post-merge diameter) must all pass for a merge to be accepted. This works, but:

- **Rigid**: a single tight gate blocks otherwise obvious merges (e.g., margin gate alone rejects 40% of near-miss candidates).
- **Un-learnable**: tuning 6+ threshold knobs across different album sizes and lighting conditions is fragile.
- **No soft trade-offs**: the gates are AND-ed — there's no way for overwhelming evidence on one dimension to compensate for borderline failure on another.

**Proposal**: Replace (or augment) the gate system with a trained binary classifier that takes a rich feature vector for each candidate cluster pair and outputs a merge probability. Human-labeled merge/reject decisions from the Interactive Merge Approval UI provide the training signal.

---

## 2. Related Work

The idea of learning merge/linkage decisions rather than hard-coding thresholds is well-established in the face clustering literature. Key references:

### 2.1 Learned Linkage Functions (face-pair level)

| Paper | Venue | Core Idea |
|-------|-------|-----------|
| Wang et al., *Linkage Based Face Clustering via GCN* | CVPR 2019 | Builds local sub-graphs around each face; a GCN predicts linkage likelihood from neighborhood structure rather than raw cosine distance. |
| Liu et al., *Learn to Cluster Faces via Pairwise Classification* | ICCV 2021 | Reformulates clustering as binary pairwise classification. A rank-weighted density selects which pairs to classify. Context features (weighted-neighbor aggregation) enhance the face-pair representation. |
| Shen et al., *STAR-FC: Structure-Aware Face Clustering* | CVPR 2021 | GCN with structure-preserved subgraph sampling for scalability to 10^7 nodes; uses "node intimacy" as a structural feature for merge decisions. |
| Yang et al., *Ada-NETS: Adaptive Neighbour Discovery* | ICLR 2022 | Transforms faces into a "structure space" via neighbor-embedding aggregation; adaptively selects the optimal number of edges per face to reduce noise. |
| Yang et al., *Doubly Imbalanced Graph Learning* | 2023 | Addresses label imbalance in GCN linkage prediction via reverse-imbalance sampling and augmented graph representations. |

### 2.2 Learned Linkage Functions (cluster-pair level)

| Paper | Venue | Core Idea |
|-------|-------|-----------|
| Yadav et al., *Supervised Hierarchical Clustering with Exponential Linkage* | ICML 2019 | Learns a smooth interpolation between single/average/complete linkage. Jointly learns dissimilarity and linkage functions. Up to 8pt improvement in dendrogram purity. |
| Kulkarni et al., *Unsupervised Face Identification using Heterogeneous Context* | ICMR 2013 | Uses cluster-level context (co-occurrence, scene, clothing, attributes) and learns adaptive merge rules via bootstrapping — closest to our approach. |

### 2.3 How Our Approach Differs

The academic methods above typically operate at the **face-pair** level with deep models (GCNs, transformers). Our approach operates at the **cluster-pair** level with a lightweight classifier (logistic regression, gradient-boosted tree, or small NN). This is a deliberate design choice:

- **Small data**: we collect 50–500 human decisions per album, not millions.
- **Interpretability**: decision-tree or logistic-regression coefficients tell us *which features* drive merges — invaluable for debugging.
- **Low latency**: inference is sub-millisecond; no GCN forward pass.
- **Composability**: the feature vector can incorporate cluster-level statistics (shared source images, quality distributions) that face-pair methods can't naturally express.

The closest analogue in the literature is the Kulkarni et al. approach of bootstrapping adaptive merge rules from heterogeneous cluster-level features.

---

## 3. Architecture Overview

```
  Candidate Cluster Pairs
          |
          v
  FeatureComputer.compute_pair_features(Ci, Cj)
          |
          v
  +-------------------+
  | Feature Vector     |   ~40-60 dimensions
  | (see Section 4)    |
  +-------------------+
          |
          v
  MergeClassifier.predict_proba(features)
          |
          v
  merge_prob >= threshold  -->  MERGE
  merge_prob <  threshold  -->  REJECT
```

**Training loop**:
1. User runs pipeline, gets initial clusters.
2. User reviews candidate pairs in the Merge Approval UI, labeling each as `approve` / `reject`.
3. `FeatureComputer` extracts features for each labeled pair.
4. Features + labels are saved to a training dataset (`merge_training_data.parquet`).
5. Classifier is trained on accumulated data (possibly across multiple albums).
6. On next run, classifier replaces or augments the gate-based merger.

---

## 4. Feature Catalog

Features are organized by group. Priority reflects expected predictive power and implementation cost.

### 4.1 Cross-Cluster Distance Distribution (Group A) — P1

These capture the shape of the distance distribution between all faces in Ci and Cj.

| Feature | Type | Description |
|---------|------|-------------|
| `min_exemplar_dist` | float | Min cosine distance between any two exemplars across clusters |
| `exemplar_dist_mean` | float | Mean of all exemplar-pair distances |
| `exemplar_dist_std` | float | Std of exemplar-pair distances (high = uneven exemplar quality) |
| `min_cross_dist` | float | Min distance between ANY face pair (may be < min_exemplar_dist) |
| `p10_cross_dist` | float | 10th percentile of all cross-cluster distances |
| `p25_cross_dist` | float | 25th percentile |
| `p50_cross_dist` | float | Median cross-cluster distance |
| `p75_cross_dist` | float | 75th percentile (high = tail divergence) |
| `p90_cross_dist` | float | 90th percentile (outlier signal) |
| `cross_dist_iqr` | float | p75 - p25 (wide IQR = possibly mixed identities) |
| `support_fraction` | float | Fraction of cross-pairs below support_threshold |
| `n_cross_pairs_below_threshold` | int | Absolute count (complements fraction for small clusters) |

### 4.2 Cluster Geometry (Group B) — P1

| Feature | Type | Description |
|---------|------|-------------|
| `size_a` | int | Cluster A size |
| `size_b` | int | Cluster B size |
| `size_min` | int | min(size_a, size_b) |
| `size_ratio` | float | max/min size ratio |
| `size_sum` | int | Total faces in merged cluster |
| `diameter_a` | float | Cluster A diameter (max intra-distance) |
| `diameter_b` | float | Cluster B diameter |
| `diameter_max` | float | max(diameter_a, diameter_b) |
| `diameter_ratio` | float | max/min diameter ratio |
| `post_merge_diameter` | float | Diameter of hypothetical merged cluster |
| `diameter_expansion` | float | post_merge_diameter / diameter_max |
| `mean_intra_dist_a` | float | Mean intra-cluster distance A |
| `mean_intra_dist_b` | float | Mean intra-cluster distance B |
| `exemplar_count_a` | int | Number of exemplars in A |
| `exemplar_count_b` | int | Number of exemplars in B |

### 4.3 Compactness & Threshold Features (Group C) — P1

| Feature | Type | Description |
|---------|------|-------------|
| `T_a` | float | P90 intra-exemplar distance for A |
| `T_b` | float | P90 intra-exemplar distance for B |
| `T_local` | float | max(T_a, T_b) |
| `T_global` | float | Global Pxx threshold across all clusters |
| `dist_to_threshold_ratio` | float | min_exemplar_dist / merge_exemplar_threshold |

### 4.4 Source Image Diversity (Group D) — P1

These exploit the constraint that **two faces detected in the same photo are almost always different people**.

| Feature | Type | Description |
|---------|------|-------------|
| `n_images_a` | int | Unique source images in A |
| `n_images_b` | int | Unique source images in B |
| `shared_source_images` | int | Images appearing in BOTH clusters |
| `shared_source_ratio` | float | shared / min(n_images_a, n_images_b) |
| `same_image_min_dist` | float | Min cross-cluster distance for face pairs from the same source image |

**Rationale**: `shared_source_images > 0` is a very strong anti-merge signal. If two clusters each contain a face from the same photo, they're almost certainly different people.

**Edge case — mirrors/reflections**: A person can appear twice in one photo via a mirror, TV screen, or poster. This is rare enough that the feature remains a strong signal. Because the classifier learns soft weights (not hard rules), overwhelming distance evidence can still override a shared-source-image penalty. The model treats it as evidence, not a veto.

### 4.5 Disambiguation / Margin Features (Group E) — P2

These measure whether Ci↔Cj is clearly the best match or if there are competing candidates.

| Feature | Type | Description |
|---------|------|-------------|
| `second_nearest_dist_a` | float | Min exemplar dist from Ci to its second-nearest cluster (not Cj) |
| `second_nearest_dist_b` | float | Same from Cj's perspective |
| `margin_a` | float | second_nearest_dist_a − min_exemplar_dist |
| `margin_b` | float | second_nearest_dist_b − min_exemplar_dist |
| `margin_min` | float | min(margin_a, margin_b) |
| `margin_gap` | float | Gap from the ConservativeMerger's margin gate (worst_gap) |
| `rank_among_candidates` | int | Rank of this pair by min_exemplar_dist (1 = best) |
| `n_candidates_total` | int | Total candidate pairs in this iteration |

### 4.6 kNN Graph Topology (Group F) — P2

Require exposing the kNN graph to the feature computer.

| Feature | Type | Description |
|---------|------|-------------|
| `knn_edge_count` | int | Number of kNN graph edges between Ci and Cj |
| `knn_edge_density` | float | knn_edge_count / (size_a * size_b) |
| `bidirectional_edges` | int | Edges that are mutual (A→B and B→A) |
| `bidirectional_density` | float | bidirectional_edges / (size_a * size_b) |
| `max_degree_into_other` | int | Max kNN edges any single node has into the other cluster |
| `hub_node_count` | int | Nodes with >= 2 edges into the other cluster |

### 4.7 Quality & Pose Distribution (Group G) — P2

| Feature | Type | Description |
|---------|------|-------------|
| `mean_blur_a` | float | Mean blur score in A |
| `mean_blur_b` | float | Mean blur score in B |
| `blur_min_a` | float | Worst (lowest) blur in A |
| `blur_min_b` | float | Worst (lowest) blur in B |
| `frontal_frac_a` | float | Fraction of frontal faces in A |
| `frontal_frac_b` | float | Fraction of frontal faces in B |
| `frontal_frac_min` | float | min(frontal_frac_a, frontal_frac_b) |
| `pose_diff` | float | Euclidean distance of mean (yaw, pitch) between clusters |
| `yaw_std_a` | float | Yaw variance in A (high = multi-pose) |
| `yaw_std_b` | float | Yaw variance in B |
| `mean_area_a` | float | Mean face area in A |
| `mean_area_b` | float | Mean face area in B |
| `area_ratio` | float | max(mean_area) / min(mean_area) |

### 4.8 Interaction / Composite Features (Group H) — P3

| Feature | Type | Description |
|---------|------|-------------|
| `min_exemplar_dist_x_pose` | float | min_exemplar_dist * (1 + pose_diff/90) |
| `p50_cross_dist_x_pose` | float | p50_cross_dist * (1 + pose_diff/90) |
| `dist_x_blur_penalty` | float | min_exemplar_dist * (1 + 1/min(mean_blur_a, mean_blur_b)) |
| `support_x_margin` | float | support_fraction * margin_min |

### 4.9 Global Context (Group I) — P3

Computed after all cluster-pair features are ready (two-pass).

| Feature | Type | Description |
|---------|------|-------------|
| `n_total_clusters` | int | Total clusters in this run |
| `global_dist_p10` | float | P10 of all inter-cluster min_exemplar_dists |
| `global_dist_p50` | float | P50 across all candidate pairs |
| `dist_zscore` | float | (min_exemplar_dist − mean) / std |

---

## 5. Feature Collection Spec

### 5.1 When to Collect

Features are computed for every candidate pair presented to the user in the **Interactive Merge Approval UI**. This includes both pairs the merger would have merged and pairs it rejected.

### 5.2 Data Schema

Each labeled sample is stored as one row with:

| Column | Type | Source |
|--------|------|--------|
| `run_id` | str | Pipeline run identifier |
| `album_id` | str | Album/directory name |
| `cluster_a` | int | Cluster ID A |
| `cluster_b` | int | Cluster ID B |
| `label` | int | 1 = approve (merge), 0 = reject |
| `labeler` | str | Who labeled (for multi-annotator) |
| `timestamp` | str | ISO 8601 |
| All features from Section 4 | float/int | FeatureComputer |
| `feature_version` | int | Feature schema version |

### 5.3 Storage Format

- **File**: `<output_dir>/merge_training_data.parquet` (per-run)
- **Aggregated**: `~/.sim_bench/merge_training_data.parquet` (all runs)
- Parquet chosen for typed columns, efficient append, and pandas compatibility.

### 5.4 Feature Computation Pipeline

```
PipelineResult
    ├── cluster_result (ClusterResult)
    ├── graph_result (GraphResult)         ← needed for Group F
    ├── faces (List[FaceRecord])           ← needed for Groups D, G
    └── distance_matrix (np.ndarray)       ← needed for Groups A, B, C, E
              │
              v
    FeatureComputer(version=3)
         .compute_all_candidate_features(
              cluster_result,
              graph_result,
              faces,
              distance_matrix
         )
              │
              v
    Dict[Tuple[int,int], Dict[str, float]]
    (cluster_pair → feature dict)
```

### 5.5 Implementation Checklist

1. **Extend `ClusterPairFeatures`** — add all P1/P2 fields (Groups A–G).
2. **Extend `FeatureComputer.compute_cluster_stats()`** — add `n_images`, `blur_min`, `yaw_std`, `mean_area`.
3. **Extend `FeatureComputer.compute_pair_features()`** — add cross-cluster percentiles, post-merge diameter, shared source images.
4. **New: `FeatureComputer.compute_margin_features()`** — second-nearest cluster, margin_a/b.
5. **New: `FeatureComputer.compute_graph_features()`** — kNN edge count, density (accepts `GraphResult`).
6. **New: `FeatureComputer.compute_global_features()`** — two-pass over all pairs for z-score, global percentiles.
7. **Bump `VERSION = 3`**; keep V1/V2 feature sets for backward compat.
8. **Save features to parquet** when user submits merge decisions in the UI.
9. **Contract test**: writer (feature computer) → reader (training script) round-trip.

---

## 6. Classifier Design

### 6.1 Model Candidates

| Model | Pros | Cons |
|-------|------|------|
| **Logistic Regression** | Interpretable coefficients, fast, good baseline | Can't capture feature interactions |
| **Gradient-Boosted Trees** (XGBoost/LightGBM) | Handles interactions, feature importance, robust to scale | Harder to interpret individual trees |
| **Decision Tree** (depth ≤ 5) | Fully interpretable, exportable as rules | Lower capacity, prone to overfitting |
| **Small NN** (2 hidden layers, 32 units) | Learns complex boundaries | Needs more data, less interpretable |

**Recommendation**: Start with gradient-boosted trees (XGBoost). They handle mixed feature types, produce feature importance rankings, and work well with 100–1000 samples.

### 6.2 Training Protocol

1. Accumulate labeled data across multiple albums.
2. Stratified train/test split (80/20) preserving album balance.
3. Class weighting to handle merge/reject imbalance.
4. 5-fold cross-validation on training set for hyperparameter tuning.
5. Evaluate: precision, recall, F1, AUC-ROC.
6. Save model + feature metadata + threshold.

### 6.3 Integration Modes

| Mode | Description |
|------|-------------|
| **Suggest** | Classifier scores all candidates; UI pre-sorts by merge probability. Human still decides. |
| **Auto + Review** | Classifier auto-merges high-confidence pairs (p > 0.9), flags low-confidence for review. |
| **Full Auto** | Classifier replaces ConservativeMerger entirely. Use only with high AUC (> 0.95). |

---

## 7. Key Design Decisions

### 7.1 Why Cluster-Level, Not Face-Level?

Face-level linkage prediction (GCN-style) requires training on millions of face pairs. We have ~50–500 labeled decisions per album. Cluster-level features compress the information: a single cluster pair produces one training sample with ~50 rich features.

### 7.2 Why Not Just Tune Thresholds?

With 4 gates × 6+ threshold knobs, the search space is combinatorial. A classifier learns the decision boundary from data, automatically weighting features and discovering interactions (e.g., "low exemplar distance compensates for borderline support if both clusters are large").

### 7.3 The Shared-Source-Image Feature

This is our strongest novel feature relative to the academic literature. Face-pair methods have no natural way to express "these two clusters both contain faces from the same photograph." For album-scale clustering this is an extremely strong anti-merge signal.

---

## 8. Open Questions

1. **Cross-album generalization**: Can a model trained on Album A transfer to Album B with different lighting/pose distributions? May need album-level normalization of distance features.
2. **Active learning**: Can we select the most informative pairs for the user to label rather than presenting all candidates?
3. **Calibration**: Merge probability must be well-calibrated for the Auto + Review mode threshold to be meaningful. Platt scaling or isotonic regression post-hoc.
4. **Feature stability**: Some features (kNN edge count) depend on the K parameter. If K changes between runs, features aren't comparable. Include K as a feature or normalize.
5. **Minimum training data**: How many labeled decisions are needed before the classifier outperforms the rule-based system? Hypothesis: ~100 decisions with balanced classes.
