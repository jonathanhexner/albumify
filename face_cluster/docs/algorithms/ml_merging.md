# ML-Based Cluster Merging

**Purpose**: Use machine learning to learn optimal cluster merge decisions from labeled data.

**Status**: Phase 3 (not yet implemented)

---

## Overview

After initial clustering (kNN graph + connected components), we often get **over-segmentation**:
- Same person split into multiple clusters
- Due to conservative distance thresholds or pose variation

**Two merge strategies**:
1. **Conservative heuristic** (Phase 1-2) - Rule-based merging using exemplar distances
2. **ML-based** (Phase 3) - Learn from labeled data what makes a good merge

This document describes the ML approach (Phase 3).

---

## Motivation

### Why ML Merging?

**Problem with heuristics**:
```python
# Simple heuristic
if min_exemplar_distance < threshold:
    merge(cluster_a, cluster_b)
```

- **Too strict** → misses valid merges (false negatives)
- **Too lenient** → creates bad merges (false positives)
- **Single feature** → ignores cluster shape, size, variance

**ML approach**:
- Learn from **labeled examples** of good/bad merges
- Use **multiple features**: min distance, p10 distance, support, diameter ratio, etc.
- Handle **complex patterns**: "merge if close AND both dense" or "don't merge if one is wide"

---

## Workflow

### Phase 1: Label Generation (Manual)

1. Run initial clustering on album
2. Export clusters to labeling UI
3. User assigns corrected identities:
   ```
   Cluster 0 → "Alice"
   Cluster 3 → "Alice"  ← Should have been merged!
   Cluster 5 → "Alice"  ← Also Alice

   Cluster 1 → "Bob"
   Cluster 2 → "Bob"    ← Should have been merged!
   ```

4. Save corrected labels to `corrected_identities.json`

### Phase 2: Training Data Generation (Automated)

```python
def generate_training_data(
    clusters: ClusterResult,
    corrected_labels: Dict[int, str],  # cluster_id → person_name
    exemplars: Dict[int, List[int]],
    distance_matrix: np.ndarray
) -> DataFrame:
    """
    Generate training examples from corrected labels.

    For each cluster pair (i, j):
    - Label: should_merge = (corrected_labels[i] == corrected_labels[j])
    - Features: compute merge features

    Returns:
        DataFrame with columns: [features..., should_merge]
    """
    training_examples = []

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            # Compute features
            features = compute_merge_features(
                clusters[i], clusters[j],
                exemplars[i], exemplars[j],
                distance_matrix
            )

            # Label: should they have been merged?
            label_i = corrected_labels.get(i)
            label_j = corrected_labels.get(j)

            if label_i is None or label_j is None:
                continue  # Skip unlabeled clusters

            should_merge = (label_i == label_j)

            training_examples.append({
                **features,
                'should_merge': should_merge
            })

    return pd.DataFrame(training_examples)
```

### Phase 3: Model Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load training data
df = pd.read_csv('training_data.csv')
X = df.drop('should_merge', axis=1)
y = df['should_merge']

# Train classifier
clf = LogisticRegression(class_weight='balanced')
clf.fit(X, y)

# Evaluate
scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
print(f"F1 score: {scores.mean():.3f} ± {scores.std():.3f}")

# Save model
import joblib
joblib.dump(clf, 'models/merge_classifier.pkl')
```

### Phase 4: Inference (Use Trained Model)

```python
# Load model
clf = joblib.load('models/merge_classifier.pkl')

# For each cluster pair, predict merge decision
for i, j in cluster_pairs:
    features = compute_merge_features(i, j, ...)
    should_merge = clf.predict([features])[0]
    merge_prob = clf.predict_proba([features])[0, 1]

    if should_merge:
        merge(i, j)
```

---

## Features for ML Model

### Distance Features (3)

1. **min_exemplar_dist**: Minimum distance between any two exemplars
   ```python
   min_dist = min(distance_matrix[ex_a, ex_b]
                  for ex_a in exemplars_a
                  for ex_b in exemplars_b)
   ```

2. **p10_cross_dist**: 10th percentile of all cross-cluster distances
   ```python
   cross_dists = [distance_matrix[a, b]
                  for a in cluster_a
                  for b in cluster_b]
   p10_dist = np.percentile(cross_dists, 10)
   ```

3. **median_cross_dist**: Median of cross-cluster distances
   ```python
   median_dist = np.median(cross_dists)
   ```

### Cluster Shape Features (4)

4. **diameter_ratio**: Ratio of diameters
   ```python
   diameter_a = max distance within cluster A
   diameter_b = max distance within cluster B
   diameter_ratio = max(diameter_a, diameter_b) / min(diameter_a, diameter_b)
   ```

5. **merged_diameter**: Predicted diameter after merge
   ```python
   # Max distance in union of clusters
   merged_diameter = max(distance_matrix[i, j]
                        for i in cluster_a + cluster_b
                        for j in cluster_a + cluster_b)
   ```

6. **diameter_increase**: How much diameter increases after merge
   ```python
   current_max_diameter = max(diameter_a, diameter_b)
   diameter_increase = merged_diameter - current_max_diameter
   ```

7. **intra_vs_cross_ratio**: Ratio of within-cluster to cross-cluster distances
   ```python
   avg_intra_a = mean(distance_matrix[i, j] for i, j in cluster_a)
   avg_intra_b = mean(distance_matrix[i, j] for i, j in cluster_b)
   avg_intra = (avg_intra_a + avg_intra_b) / 2

   avg_cross = mean(cross_dists)

   ratio = avg_cross / avg_intra  # > 1.0 → clusters well-separated
   ```

### Support Features (2)

8. **support_fraction**: Fraction of close face pairs
   ```python
   # How many cross-cluster pairs have distance < threshold?
   close_pairs = sum(1 for d in cross_dists if d < threshold)
   total_pairs = len(cluster_a) * len(cluster_b)
   support_fraction = close_pairs / total_pairs
   ```

9. **size_ratio**: Ratio of cluster sizes
   ```python
   size_ratio = max(len(cluster_a), len(cluster_b)) / min(len(cluster_a), len(cluster_b))
   # Large ratio → unbalanced merge (one tiny cluster)
   ```

### Adaptive Threshold Features (2)

10. **threshold_a**: Adaptive threshold for cluster A
    ```python
    # Based on within-cluster distances
    intra_dists_a = [distance_matrix[i, j] for i, j in pairs_in_cluster_a]
    median_a = np.median(intra_dists_a)
    iqr_a = np.percentile(intra_dists_a, 75) - np.percentile(intra_dists_a, 25)
    threshold_a = median_a + 2 * iqr_a
    ```

11. **threshold_b**: Adaptive threshold for cluster B (same calculation)

### Total: 11 features

---

## Training Data Balance

**Challenge**: Unbalanced classes
- Most cluster pairs should NOT merge (negative class)
- Few pairs should merge (positive class)
- Ratio: ~1:100 (1% positive, 99% negative)

**Solutions**:

1. **Class weighting**:
   ```python
   clf = LogisticRegression(class_weight='balanced')
   # Automatically weights classes by inverse frequency
   ```

2. **Threshold tuning**:
   ```python
   # Adjust decision threshold for desired precision/recall
   merge_prob = clf.predict_proba(features)[0, 1]
   should_merge = (merge_prob > 0.7)  # More conservative
   ```

3. **Stratified sampling**:
   ```python
   from sklearn.model_selection import StratifiedKFold
   cv = StratifiedKFold(n_splits=5)
   ```

---

## Evaluation Metrics

### Classification Metrics

```python
from sklearn.metrics import classification_report, confusion_matrix

# Predict on test set
y_pred = clf.predict(X_test)

# Report
print(classification_report(y_test, y_pred,
                            target_names=['No Merge', 'Merge']))
```

**Example output**:
```
              precision    recall  f1-score   support

    No Merge       0.98      0.99      0.99      1850
       Merge       0.87      0.75      0.81        98

    accuracy                           0.98      1948
   macro avg       0.93      0.87      0.90      1948
weighted avg       0.98      0.98      0.98      1948
```

### Clustering Metrics (End-to-End)

After applying merge model to test album:

```python
from sklearn.metrics import adjusted_rand_score, v_measure_score

# Compare predicted clusters vs ground truth
ari = adjusted_rand_score(y_true, y_pred_clusters)
v_measure = v_measure_score(y_true, y_pred_clusters)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"V-Measure: {v_measure:.3f}")
```

---

## Model Interpretability

### Feature Importance

```python
# For logistic regression
feature_names = X.columns
coefficients = clf.coef_[0]

# Sort by absolute coefficient
importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coef': np.abs(coefficients)
}).sort_values('abs_coef', ascending=False)

print(importance)
```

**Example**:
```
                 feature  coefficient  abs_coef
0     min_exemplar_dist       -3.45      3.45  ← Most important (negative)
7    support_fraction          2.12      2.12  ← Important (positive)
4      merged_diameter         -1.87      1.87
8            size_ratio         -0.95      0.95
...
```

**Interpretation**:
- **min_exemplar_dist** (negative) → Lower distance → MORE likely to merge ✓
- **support_fraction** (positive) → More close pairs → MORE likely to merge ✓
- **merged_diameter** (negative) → Larger merged cluster → LESS likely to merge ✓

---

## Integration with Pipeline

### Conservative Merger (Heuristic - Phase 2)

```python
class ConservativeMerger:
    def merge(self, clusters, exemplars, distance_matrix, threshold):
        # Simple exemplar-based merging
        for i, j in cluster_pairs:
            min_dist = self._min_exemplar_distance(exemplars[i], exemplars[j])
            if min_dist < threshold:
                merge(i, j)
```

### ML Merger (Phase 3)

```python
class MLMerger:
    def __init__(self, model_path):
        self.clf = joblib.load(model_path)

    def merge(self, clusters, exemplars, distance_matrix):
        for i, j in cluster_pairs:
            features = compute_merge_features(i, j, ...)
            should_merge = self.clf.predict([features])[0]

            if should_merge:
                merge(i, j)
```

---

## Training Script Example

```bash
# scripts/train_merge_classifier.py

python scripts/train_merge_classifier.py \
    --data results/face_clustering_training/ \
    --corrected-labels corrected_identities.json \
    --output models/merge_classifier.pkl \
    --eval-splits 5
```

**Output**:
```
Loading training data...
  Clusters: 47
  Corrected labels: 12 unique identities
  Generated examples: 1081 (58 positive, 1023 negative)

Training classifier...
  Model: LogisticRegression (L2, class_weight=balanced)
  Features: 11

Cross-validation (5-fold):
  Fold 1: F1=0.78, Precision=0.82, Recall=0.74
  Fold 2: F1=0.81, Precision=0.85, Recall=0.77
  Fold 3: F1=0.76, Precision=0.79, Recall=0.73
  Fold 4: F1=0.79, Precision=0.81, Recall=0.77
  Fold 5: F1=0.82, Precision=0.86, Recall=0.78

Mean F1: 0.792 ± 0.021

Feature importance:
  1. min_exemplar_dist: -3.45
  2. support_fraction: 2.12
  3. merged_diameter: -1.87
  4. size_ratio: -0.95
  5. ...

Saving model to models/merge_classifier.pkl
Done!
```

---

## Comparison: Heuristic vs ML

| Metric | Heuristic | ML |
|--------|-----------|-----|
| **Precision** | 0.75 | 0.85 |
| **Recall** | 0.68 | 0.77 |
| **F1** | 0.71 | 0.81 |
| **Training required** | No | Yes (manual labeling) |
| **Interpretability** | High | Medium |
| **Adaptability** | Low | High (learns from data) |

**When to use heuristic**:
- No labeled data available
- Need interpretable decisions
- Simple merge criteria sufficient

**When to use ML**:
- Have labeled training data
- Complex merge patterns
- Want to optimize F1 score

---

## Future Enhancements

1. **Deep learning**: Use neural network instead of logistic regression
   ```python
   from tensorflow.keras import Sequential, Dense
   model = Sequential([
       Dense(64, activation='relu', input_dim=11),
       Dense(32, activation='relu'),
       Dense(1, activation='sigmoid')
   ])
   ```

2. **Active learning**: Query user on uncertain pairs
   ```python
   merge_prob = clf.predict_proba(features)[0, 1]
   if 0.4 < merge_prob < 0.6:  # Uncertain
       label = ask_user("Should these clusters merge?")
       # Add to training data
   ```

3. **Multi-task learning**: Jointly learn merge + split decisions

4. **Graph neural networks**: Model cluster relationships as graph

---

## References

- **Cluster ensemble**: [Strehl & Ghosh, 2003](https://dl.acm.org/doi/10.5555/1630659.1630743)
- **Active learning**: [Settles, 2009](https://minds.wisconsin.edu/handle/1793/60660)

---

**See Also**:
- [Exemplar Selection](exemplar_selection.md) - How exemplars are chosen
- [ML Training Workflow](../workflows/ml_training_workflow.md) - Step-by-step training process
- [ML Training Guide](../workflows/ml_training_guide.md) - Complete guide with examples
