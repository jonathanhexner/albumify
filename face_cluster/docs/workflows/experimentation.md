# Face Clustering Experimentation Workflow

**Purpose**: Guide for experimenting with face clustering algorithms and parameters

---

## Overview

The face_cluster module is an **experimentation platform** for testing clustering algorithms before integrating into production. This workflow helps you:

1. Test different algorithms (kNN, HDBSCAN, hybrid)
2. Tune parameters for your dataset
3. Evaluate clustering quality
4. Identify optimal configuration

---

## Workflow Steps

### 1. Prepare Test Dataset

**Option A: Use Test Data** (Quick validation)
```bash
# Built-in test data: 9 images, 3 people
python scripts/run_face_clustering_pipeline.py \
    --album test_data/face_clustering \
    --output results/test_baseline \
    --config configs/face_clustering_experiment.yaml
```

**Expected**: 3 clusters, 0 noise, 9 faces total

**Option B: Use Real Album** (Realistic evaluation)
```bash
# Your photo album
python scripts/run_face_clustering_pipeline.py \
    --album /path/to/your/album \
    --output results/album_baseline \
    --config configs/face_clustering_experiment.yaml
```

### 2. Baseline Run

Run with default configuration to establish baseline:

```yaml
# configs/face_clustering_experiment.yaml (default)
step_configs:
  filter_quality_gate:
    yaw_max: 45.0
    blur_min: 100.0

  build_knn_graph:
    K: 5
    distance_threshold: 0.35

  cluster_connected_components:
    min_cluster_size: 2
```

**Record baseline metrics**:
```
Run: baseline
Date: 2026-03-28
Album: my_photos (427 images)

Results:
  Faces detected: 1,243
  Core faces: 891 (71.7%)
  Clusters: 47
  Noise points: 23 (2.6%)
  Avg cluster size: 18.9

Quality indicators:
  Large clusters (>50 faces): 5
  Small clusters (<5 faces): 12
  Max diameter: 0.68
```

### 3. Quality Gate Experiments

Test different quality thresholds to see impact on clustering:

**Experiment 1: Strict Quality** (fewer faces, higher quality)
```yaml
filter_quality_gate:
  yaw_max: 30.0      # ← Stricter (was 45.0)
  blur_min: 150.0    # ← Stricter (was 100.0)
```

Run and compare:
```bash
python scripts/run_face_clustering_pipeline.py \
    --album /path/to/album \
    --output results/album_strict_quality \
    --config configs/experiment_strict_quality.yaml
```

**Expected changes**:
- Fewer core faces (~50-60% instead of 70%)
- Smaller clusters (fewer faces per cluster)
- Possibly more clusters (less evidence to connect)

**Experiment 2: Lenient Quality** (more faces, tolerate lower quality)
```yaml
filter_quality_gate:
  yaw_max: 60.0      # ← Lenient (was 45.0)
  blur_min: 50.0     # ← Lenient (was 100.0)
```

**Expected changes**:
- More core faces (~85-90%)
- Larger clusters
- Possibly noisier clusters (more mistakes)

### 4. Clustering Threshold Experiments

Test different distance thresholds:

**Experiment 3: Conservative Clustering** (fewer edges, more clusters)
```yaml
build_knn_graph:
  K: 3               # ← Fewer neighbors (was 5)
  distance_threshold: 0.30  # ← Stricter (was 0.35)
```

**Expected changes**:
- More clusters (faces less connected)
- Smaller clusters
- Lower chance of false merges
- Higher chance of false splits

**Experiment 4: Aggressive Clustering** (more edges, fewer clusters)
```yaml
build_knn_graph:
  K: 10              # ← More neighbors (was 5)
  distance_threshold: 0.40  # ← Lenient (was 0.35)
```

**Expected changes**:
- Fewer clusters (faces more connected)
- Larger clusters
- Higher chance of false merges
- Lower chance of false splits

### 5. Evaluate Results

For each experiment, evaluate clustering quality:

**Quantitative Metrics** (if ground truth available):
```python
# Compare against manual labels
from sklearn.metrics import adjusted_rand_score, v_measure_score

ari = adjusted_rand_score(ground_truth, predicted_clusters)
v_measure = v_measure_score(ground_truth, predicted_clusters)

print(f"ARI: {ari:.3f}")  # 0.0 = random, 1.0 = perfect
print(f"V-Measure: {v_measure:.3f}")  # 0.0 = bad, 1.0 = perfect
```

**Qualitative Metrics** (manual inspection):
- **Large cluster purity**: Sample 10 faces from largest cluster, all same person?
- **Small cluster validity**: Are small clusters (<5 faces) real people or noise?
- **Merge candidates**: Do any clusters obviously belong together?
- **Split candidates**: Do any clusters obviously have multiple people?

**Cluster Statistics**:
```python
# Load clustering results
df = pd.read_csv('results/experiment/clusters.csv')

print(f"Total clusters: {len(df)}")
print(f"Avg cluster size: {df['size'].mean():.1f}")
print(f"Max cluster size: {df['size'].max()}")
print(f"Clusters with diameter > 0.50: {sum(df['diameter'] > 0.50)}")
```

### 6. Compare Experiments

Create comparison table:

| Experiment | Core Faces | Clusters | Noise | Avg Size | Max Diameter | Notes |
|------------|-----------|----------|-------|----------|--------------|-------|
| Baseline | 891 (71.7%) | 47 | 23 | 18.9 | 0.68 | Some wide clusters |
| Strict Quality | 623 (50.1%) | 53 | 31 | 11.8 | 0.52 | Cleaner, but missed faces |
| Lenient Quality | 1087 (87.4%) | 42 | 18 | 25.9 | 0.79 | More faces, noisier |
| Conservative | 891 (71.7%) | 62 | 45 | 14.4 | 0.48 | Over-segmented |
| Aggressive | 891 (71.7%) | 35 | 12 | 25.4 | 0.82 | Under-segmented |

**Identify best configuration** based on:
- Lowest max diameter (tightest clusters)
- Reasonable cluster count (not too many, not too few)
- Low noise percentage
- Manual inspection results

### 7. Fine-Tune Winning Configuration

Once you identify the best approach, fine-tune:

```yaml
# Example: Conservative clustering worked best, now fine-tune threshold
build_knn_graph:
  K: 3
  distance_threshold: 0.32  # Try 0.30, 0.32, 0.34
```

Run multiple fine-tuning experiments:
```bash
for thresh in 0.30 0.32 0.34 0.36; do
  # Update config
  sed "s/distance_threshold: .*/distance_threshold: $thresh/" \
    configs/experiment.yaml > configs/experiment_${thresh}.yaml

  # Run
  python scripts/run_face_clustering_pipeline.py \
    --album /path/to/album \
    --output results/album_thresh_${thresh} \
    --config configs/experiment_${thresh}.yaml
done
```

### 8. Manual Labeling & Evaluation

After finding optimal configuration:

1. **Run on labeled dataset**:
   ```bash
   python scripts/run_face_clustering_pipeline.py \
       --album /path/to/labeled/album \
       --output results/labeled_optimal \
       --config configs/optimal.yaml
   ```

2. **Label clusters manually**:
   ```bash
   streamlit run app/face_clustering_labeling.py -- \
       --data-dir results/labeled_optimal
   ```

3. **Compute evaluation metrics**:
   ```python
   # After labeling, compare vs ground truth
   from sklearn.metrics import classification_report

   # Map cluster IDs to person IDs using corrected labels
   y_pred = map_clusters_to_people(clusters, corrected_labels)

   # Compare to ground truth
   report = classification_report(y_true, y_pred)
   print(report)
   ```

---

## Experiment Tracking

### Log Book Template

```markdown
# Experiment Log

## Experiment 1: Baseline
Date: 2026-03-28
Config: Default (K=5, threshold=0.35, yaw_max=45)
Results: 47 clusters, 23 noise, max_diameter=0.68
Notes: Some clusters have high diameter, might contain multiple people

## Experiment 2: Strict Quality
Date: 2026-03-28
Config: yaw_max=30, blur_min=150
Results: 53 clusters, 31 noise, max_diameter=0.52
Notes: Cleaner clusters, but filtered too many faces

## Experiment 3: Conservative Threshold
Date: 2026-03-28
Config: K=3, threshold=0.30
Results: 62 clusters, 45 noise, max_diameter=0.48
Notes: Best max_diameter so far! But too many small clusters

## Experiment 4: Fine-tuned
Date: 2026-03-28
Config: K=3, threshold=0.32, yaw_max=45, blur_min=100
Results: 54 clusters, 28 noise, max_diameter=0.51
Notes: ✅ Best overall! Good balance of cluster count and tightness
```

---

## Parameter Tuning Guidelines

### Quality Gate Parameters

**Impact of `yaw_max`**:
- Lower → Stricter → Fewer faces → Cleaner clusters
- Higher → Lenient → More faces → Noisier clusters
- **Recommended**: 30-45° for most albums

**Impact of `blur_min`**:
- Lower → More faces (tolerate blur)
- Higher → Fewer faces (require sharp)
- **Recommended**: 100-150 (Laplacian variance)

### Clustering Parameters

**Impact of `K` (kNN neighbors)**:
- Lower (K=3) → Fewer edges → More clusters
- Higher (K=10) → More edges → Fewer clusters
- **Recommended**: 3-5 for small albums, 5-10 for large albums

**Impact of `distance_threshold`**:
- Lower (0.30) → Stricter → More clusters
- Higher (0.40) → Lenient → Fewer clusters
- **Recommended**: 0.30-0.35 for most datasets

**Rule of thumb**: Adjust K and threshold together
- Conservative: K=3, threshold=0.30
- Balanced: K=5, threshold=0.35
- Aggressive: K=10, threshold=0.40

---

## Common Patterns

### Pattern 1: Over-segmentation (Too Many Clusters)

**Symptoms**:
- Many small clusters (< 5 faces)
- Same person split across multiple clusters
- Low noise count

**Fix**: Increase connectivity
```yaml
build_knn_graph:
  K: 10              # Increase K
  distance_threshold: 0.38  # Increase threshold
```

### Pattern 2: Under-segmentation (Too Few Clusters)

**Symptoms**:
- Few large clusters (> 50 faces)
- Multiple people in same cluster (high diameter > 0.60)
- Low noise count

**Fix**: Decrease connectivity
```yaml
build_knn_graph:
  K: 3               # Decrease K
  distance_threshold: 0.32  # Decrease threshold
```

### Pattern 3: High Noise (Many Unassigned Faces)

**Symptoms**:
- High noise count (> 10% of faces)
- Many single-face components

**Fix**: Either relax quality gate OR lower `min_cluster_size`
```yaml
filter_quality_gate:
  blur_min: 75.0     # More lenient

cluster_connected_components:
  min_cluster_size: 1  # Allow singletons
```

---

## Advanced: Algorithm Comparison

Compare different clustering algorithms:

**Algorithm 1: Mutual kNN + Connected Components**
```yaml
# configs/experiment_knn.yaml
# (This is the default - no changes needed)
```

**Algorithm 2: HDBSCAN**
```yaml
# Use main app's cluster_people step instead
step_configs:
  cluster_people:
    method: 'hdbscan'
    min_cluster_size: 3
    cluster_selection_epsilon: 0.3
```

**Algorithm 3: Hybrid HDBSCAN + kNN**
```yaml
step_configs:
  cluster_people:
    method: 'hybrid_hdbscan_knn'
    min_cluster_size: 2
    knn_k: 3
    threshold_floor: 0.30
```

Run all three and compare results.

---

**See Also**:
- [Configuration Reference](../pipeline/configuration.md) - All tunable parameters
- [Benchmarking Guide](benchmarking.md) - Quantitative evaluation
- [ML Training Workflow](ml_training_workflow.md) - Training merge classifier
