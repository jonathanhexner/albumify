# Merge Criteria Explained

## Quick Reference

| Criterion | Check | Parameter | Default | Recommended |
|-----------|-------|-----------|---------|-------------|
| **Exemplar Distance** | min(exemplar_dist) ≤ T_merge | `merge_threshold_alpha` | 0.7 | Keep default |
| **Support Count** | cross-cluster pairs ≥ threshold | `merge_support_frac`<br>`merge_support_min` | 0.3<br>2 | Keep default |
| **Margin** | Clear closest cluster | `merge_margin` | 0.05 | **Set to 0.0** |
| **Diameter** | Post-merge diameter OK | `merge_diameter_expansion_factor` | 1.5 | Keep default |

**ALL four must pass** for merge to happen.

---

## Overview

Conservative merge uses **4 criteria** that ALL must pass for clusters to merge:

## 1. Exemplar Distance ✓ (Working Well)

**What**: `min(exemplar_distance) ≤ T_merge`

**How T_merge is computed**:
```
T_merge = α × MAX(T_A, T_B) + (1-α) × T_global

where:
  T_A = P90(exemplar pairwise distances within cluster A)
  T_B = P90(exemplar pairwise distances within cluster B)
  T_global = median(T_1, T_2, ..., T_n) across all clusters
  α = merge_threshold_alpha (default 0.7)
```

**Parameter**: `merge_threshold_alpha` (default 0.7) - controls local vs global weighting

**Purpose**: Don't merge if exemplars are too far apart

---

## 2. Support Count ✓ (Working Well)

**What**: Count cross-cluster face pairs with distance ≤ T_merge, require:
```
support ≥ max(merge_support_frac × min(|A|, |B|), merge_support_min)
```

**Parameters**:
- `merge_support_frac` (default 0.3) - fraction of smaller cluster
- `merge_support_min` (default 2) - absolute minimum

**Purpose**: Require enough "evidence" that clusters belong together (not just close exemplars)

---

## 3. Margin to Next Best ⚠️ (VERY CONSERVATIVE - Questionable)

**What**: For EACH exemplar in cluster A, check that cluster B is the **clear closest** cluster:
```python
for exemplar in A:
    dist_to_B = min_distance(exemplar, cluster_B)
    dist_to_other = min_distance(exemplar, any_other_cluster)

    require: dist_to_B + merge_margin < dist_to_other
```

**Parameter**: `merge_margin` (default 0.05)

**Example Why It Fails**:
- Cluster A exemplar is 0.233 away from cluster B
- Same exemplar is 0.240 away from cluster C
- Margin = 0.05
- Check: 0.233 + 0.05 = 0.283 > 0.240? **NO** → FAIL
- Reason: Cluster C is "too close" (within 0.05 margin)

**Purpose**: Prevent ambiguous merges when a cluster is equally close to multiple others

**Problem**:
- **Too strict** - blocks merges even when distances are very small
- If Exemplar Distance and Support already passed, why block on this?
- Creates situations where clusters at 0.233 distance don't merge because another is at 0.240
- The threshold (T_merge) should already handle conservatism

---

## 4. Post-Merge Diameter ✓ (Working Well)

**What**: Check that merged cluster won't be too wide:
```
max(pairwise distances in merged cluster) ≤ current_max_diameter × merge_diameter_expansion_factor
```

**Parameter**: `merge_diameter_expansion_factor` (default 1.5)

**Purpose**: Don't create over-wide clusters

---

## Recommendation: Disable or Reduce Margin Check

### Option 1: Set margin to 0 (effectively disable)
```python
config = PipelineConfig(
    merge_enabled=True,
    merge_margin=0.0,  # Disable margin check
)
```

### Option 2: Make it more lenient
```python
config = PipelineConfig(
    merge_enabled=True,
    merge_margin=0.02,  # More lenient (default 0.05)
)
```

### Reasoning:
- **Exemplar Distance** already controls whether clusters are close enough
- **Support Count** already requires enough evidence
- **Diameter** already prevents over-wide clusters
- **Margin adds extra conservatism** that may be blocking valid merges
- In your data: distances of 0.233-0.280 are small, but margin blocks them

---

## Your Data Analysis

From your merge decisions:
```
C1  C2  Exemplar_Dist  T_merge   Gap    Failed
1   18  0.233         0.339     -0.106  Margin  (passes exemplar by 0.106!)
4   9   0.276         0.318     -0.043  Margin  (passes exemplar by 0.043!)
```

**All passing exemplar distance** (negative Gap) but **all failing margin**.

This suggests: Margin criterion is too strict for your data. Try setting `merge_margin=0.0`.

---

## Understanding Merge Failures

### Failure Patterns and What They Mean

| Failed Criteria | What It Means | Is This a Problem? | Action |
|----------------|---------------|-------------------|--------|
| **Exemplar only** | Distance slightly above threshold | Maybe | Check Gap - if small (<0.05), consider lowering alpha |
| **Support only** | Only exemplars close, not enough other faces | No | Working correctly - prevents outlier-driven merges |
| **Margin only** | Another cluster is equally close | Usually not a real problem | Set `merge_margin=0.0` |
| **Diameter only** | Merged cluster would be too wide | Probably not | Increase `merge_diameter_expansion_factor` if needed |
| **Exemplar + Support + Diameter** | Clusters genuinely different | **No - this is correct!** | Don't merge - likely different people |

### Example: Three Failures (Don't Merge!)

```
C1 C2  Size1 Size2  Exemplar_Dist  T_A   T_B   T_local  T_global  T_merge  Gap   Failed
3  6   6     4      0.413         0.202 0.191  0.202    0.307     0.233   +0.18  Exemplar, Support, Diameter
```

**Analysis**:
1. **Gap = +0.180** (positive = fails exemplar check by a lot)
2. **Exemplar_Dist = 0.413** is **1.8× the threshold** (0.233)
3. **T_A = 0.202, T_B = 0.191** (both clusters are internally tight)
4. Failed **3 out of 4** criteria

**Conclusion**: Clusters 3 and 6 are **genuinely different** (probably different people). Don't merge!

**Would changing alpha help?**

| Alpha | Weight | T_merge Calculation | T_merge | Still Fails? |
|-------|--------|---------------------|---------|--------------|
| 0.7 (current) | 70% local, 30% global | 0.7×0.202 + 0.3×0.307 | 0.233 | Yes (0.413 >> 0.233) |
| 0.5 | 50% local, 50% global | 0.5×0.202 + 0.5×0.307 | 0.255 | Yes (0.413 >> 0.255) |
| 0.3 | 30% local, 70% global | 0.3×0.202 + 0.7×0.307 | 0.276 | Yes (0.413 >> 0.276) |
| 0.0 | 0% local, 100% global | 0.0×0.202 + 1.0×0.307 | 0.307 | Yes (0.413 > 0.307) |

**Answer**: No! Even with 100% global weight (alpha=0), the exemplar distance (0.413) exceeds the threshold (0.307).

**Why this is good**: These clusters have tight internal structure but are far apart from each other. This is exactly what we want to preserve - high precision!

---

## Min_Dist vs Exemplar_Dist

Two different ways to measure cluster proximity:

| Metric | Definition | Used In | What It Tells You |
|--------|------------|---------|-------------------|
| **Min_Dist** | Minimum distance between **any two faces** (one from each cluster) | `get_close_clusters_df()` | Geometric proximity - which clusters have ANY close faces |
| **Exemplar_Dist** | Minimum distance between **exemplars only** | `get_merge_decisions_df()` | Merge decisions - representative faces proximity |

### Why They Differ

**Exemplars** are selected as the densest, most representative faces in each cluster (d10 metric). They may not be the absolute closest two faces between clusters.

**Example**:
```
Cluster pair (4, 23):
  Min_Dist = 0.278       ← Some outlier face in 4 is close to some face in 23
  Exemplar_Dist = 0.520  ← But representative faces are far apart

  Result: Close in Min_Dist, but NOT proposed for merge (exemplar_dist > 0.45)
```

**Example**:
```
Cluster pair (3, 6):
  Min_Dist = 0.379       ← Closest ANY two faces
  Exemplar_Dist = 0.413  ← Closest two EXEMPLARS (different pair of faces!)

  Result: Proposed for merge (< 0.45), but failed due to large exemplar distance
```

### Why Merge Uses Exemplar_Dist

Using `Min_Dist` would be too sensitive to outliers:
- One face could be mis-clustered (outlier)
- Would cause merge based on single outlier pair
- Exemplars represent the cluster "core" → more robust

---

## Troubleshooting Decision Tree

```
Clusters not merging?
│
├─ Check Merge Decisions DataFrame
│  │
│  ├─ Pair not in DataFrame at all?
│  │  └─ Exemplar_Dist > 0.45 (not even proposed)
│  │     → Check Close Clusters DataFrame
│  │     → If Min_Dist << Exemplar_Dist: outliers, correct behavior
│  │
│  ├─ Failed: "Margin" only?
│  │  └─ Set merge_margin=0.0
│  │
│  ├─ Failed: "Exemplar" only, Gap < 0.05?
│  │  └─ Consider lowering alpha (more global weight)
│  │
│  └─ Failed: "Exemplar + Support + Diameter"?
│     └─ Clusters genuinely different - DON'T merge!
│
└─ Too many small clusters overall?
   └─ Lower distance_threshold in initial clustering
```
