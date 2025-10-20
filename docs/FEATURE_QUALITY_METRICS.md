# Feature Quality Metrics for Image Retrieval

Understanding which feature dimensions are useful vs. harmful for image similarity retrieval.

## The Core Question

**For normalized features in retrieval tasks, what makes a feature "good"?**

Unlike classification (where high activation = detected pattern), retrieval uses distance metrics on normalized features. What matters is:
- **Consistency within groups** (similar images should have similar feature values)
- **Separation between groups** (different images should have different feature values)

## Metrics We Provide

### 1. **Within-Group Diversity** (BAD when high)

**Question:** Do images in the same group have consistent feature values?

#### Variance
```python
variance = np.var(group_features, axis=0)
```
- **High variance** = feature varies a lot within the group → BAD
- **Low variance** = feature is consistent for similar images → GOOD
- Sensitive to outliers

#### Range (Your suggestion!)
```python
range = max(group_features) - min(group_features)
```
- **High range** = at least one image is very different → BAD
- **Low range** = all images have similar values → GOOD  
- More robust to outliers than variance
- Useful for identifying extreme inconsistencies

#### Standard Deviation
```python
std = np.std(group_features, axis=0)
```
- Square root of variance
- Same units as features (easier to interpret)

#### IQR (Interquartile Range)
```python
iqr = percentile_75 - percentile_25
```
- **Most robust** to outliers
- Focuses on the middle 50% of data
- Best for noisy datasets

### 2. **Feature Discriminability** (Fisher Criterion)

**Question:** Does this feature help distinguish between groups?

```python
fisher_score = between_group_variance / within_group_variance
```

**High Fisher score** = GOOD feature because:
- Different groups have different values (high between-group variance)
- Same group has similar values (low within-group variance)

**Low Fisher score** = BAD feature because:
- Groups overlap or
- Too much noise within groups

This is **exactly what we want for retrieval!**

## What About "Top Active Features"?

### In Classification:
- High activation = "neuron detected its pattern"
- Makes sense to look at top activations

### In Retrieval (Normalized Features):
- After L2 normalization: `||features|| = 1`
- Distance = `sqrt(2 - 2*dot(a, b))` (for normalized vectors)
- What matters is the **relative pattern**, not magnitude
- A feature with value 0.5 is as important as 0.05

**Therefore:** Top active features (by absolute value) **don't directly tell us about retrieval quality**.

### When They DO Matter:
1. **Grad-CAM attribution** - Shows what image regions contribute to those features
2. **Feature importance for specific images** - Understanding individual examples
3. **Debugging** - "Why did this image get this feature pattern?"

But for **evaluating feature quality for retrieval**, use:
- Within-group diversity (lower is better)
- Fisher discriminability (higher is better)

## Practical Examples

### Example 1: Good Feature (Low Diversity, High Discriminability)

```
Group 1: [0.42, 0.44, 0.43, 0.41]  (consistent)
Group 2: [0.78, 0.81, 0.79, 0.80]  (consistent, but different from Group 1)
Group 3: [0.15, 0.13, 0.14, 0.16]  (consistent, different from both)

Within-group variance: LOW ✓
Between-group variance: HIGH ✓
Fisher score: HIGH ✓
→ EXCELLENT feature for retrieval!
```

### Example 2: Bad Feature (High Diversity)

```
Group 1: [0.10, 0.90, 0.15, 0.85]  (all over the place!)
Group 2: [0.20, 0.80, 0.25, 0.75]  (inconsistent)
Group 3: [0.30, 0.70, 0.35, 0.65]  (inconsistent)

Within-group variance: HIGH ✗
Range: HUGE ✗
Fisher score: LOW ✗
→ TERRIBLE feature - causes retrieval errors!
```

### Example 3: Useless Feature (Low Variance Everywhere)

```
Group 1: [0.50, 0.51, 0.49, 0.50]  (all similar)
Group 2: [0.51, 0.50, 0.52, 0.49]  (also similar, no separation)
Group 3: [0.49, 0.50, 0.51, 0.50]  (overlaps with others)

Within-group variance: LOW ✓
Between-group variance: VERY LOW ✗
Fisher score: LOW ✗
→ Consistent but doesn't discriminate - not helpful
```

## Usage in Code

```python
from sim_bench.analysis import feature_utils

# 1. Within-group diversity (find BAD features)
diversity_results = feature_utils.analyze_within_group_feature_diversity(
    query_indices,
    query_group_ids,
    cache_file,
    metrics=['variance', 'range', 'iqr']  # Choose metrics
)

for gid, result in diversity_results.items():
    print(f"Group {gid}:")
    print(f"  Top diverse (BAD) features by variance: {result['top_diverse_dims_by_metric']['variance'][:5]}")
    print(f"  Top diverse (BAD) features by range: {result['top_diverse_dims_by_metric']['range'][:5]}")

# 2. Feature discriminability (find GOOD features)
discriminability = feature_utils.analyze_feature_discriminability(
    query_indices,
    query_group_ids,
    cache_file
)

print(f"Top discriminative (GOOD) features: {discriminability['top_discriminative_dims'][:10]}")
print(f"Fisher scores: {discriminability['top_fisher_scores'][:10]}")
```

## Key Takeaways

1. **For retrieval, focus on Fisher discriminability**, not top activations
2. **High within-group diversity = BAD** (inconsistent for similar images)
3. **High between-group variance = GOOD** (separates different images)
4. **Range is more robust than variance** for outlier detection
5. **IQR is most robust** when you have noisy data

## Future: Feature Re-weighting

Once you identify good vs bad features, you could:
1. **Remove bad features** - Zero out high-diversity dimensions
2. **Re-weight features** - Weight by Fisher score
3. **Feature selection** - Keep only discriminative features
4. **Learned weighting** - Train weights to optimize retrieval metrics

```python
# Example: Re-weight by Fisher scores
fisher_weights = discriminability['fisher_scores']
fisher_weights = fisher_weights / fisher_weights.sum()  # Normalize

weighted_features = features * fisher_weights
# Now use weighted_features for distance computation
```

This could improve retrieval performance by emphasizing discriminative features and downweighting noisy ones!

