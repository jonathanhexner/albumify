# Merge Criteria Reference

All four gates must **PASS** for a cluster pair to merge. If any gate fails, the pair is rejected.

---

## Gate Summary

| Gate | PASS condition | Value column | Threshold column | Config to relax |
|------|---------------|-------------|-----------------|-----------------|
| **Exemplar** | `exemplar_dist <= threshold` | `exemplar_dist` | `threshold` | raise `merge_threshold_alpha` or `merge_threshold_beta` |
| **Support** | `support >= required` | `support` (left of `/`) | `required` (right of `/`) | lower `merge_support_frac` or `merge_support_min` |
| **Margin** | `margin_gap >= merge_margin` | `margin_gap` | `merge_margin` config value | lower `merge_margin` (set to `0.0` to disable entirely) |
| **Diameter** | `post_diam <= max_diam` | `post_diam` | `max_diam` | raise `merge_diameter_expansion_factor` |

---

## Gate 1 — Exemplar Distance

**What it checks:** The closest pair of representative faces (exemplars) between the two clusters must be within a computed threshold.

**PASS when:** `exemplar_dist <= T_merge`

**How `T_merge` is computed (adaptive mode):**

```
T_merge = alpha × max(T_A, T_B) + beta × T_global

T_A      = P{exemplar_percentile} of pairwise distances among A's exemplars
T_B      = P{exemplar_percentile} of pairwise distances among B's exemplars
T_global = P{global_percentile} of all per-cluster thresholds

alpha  = merge_threshold_alpha   (default 0.7)
beta   = merge_threshold_beta    (default 0.3)
```

The `gap` column = `exemplar_dist - threshold`. **Negative gap = passed** (distance is below threshold).

**Table columns:** `exemplar_dist`, `threshold`, `gap`

**To relax:** Raise `merge_threshold_alpha` (weights local cluster spread more) or raise `merge_threshold_beta` (weights global spread more). Alternatively lower `merge_exemplar_percentile` to make per-cluster thresholds smaller — this also tightens the gate, so do the opposite if you want looser.

---

## Gate 2 — Support Count

**What it checks:** Enough face pairs (one from each cluster) must be within `T_merge` of each other — not just the exemplars.

**PASS when:** `support >= required_support`

```
support          = count of (face_A, face_B) pairs where distance <= T_merge
required_support = max(merge_support_frac × min(|A|, |B|), merge_support_min)
```

The `support` column shows `actual/required` (e.g. `3/2` means 3 pairs found, 2 required).

**Table columns:** `support` (format: `actual/required`)

**To relax:** Lower `merge_support_frac` (default 0.3) or lower `merge_support_min` (default 2, minimum 1).

---

## Gate 3 — Margin to Next-Best Cluster

**What it checks:** For every exemplar in cluster A, cluster B must be the **nearest** other cluster by a clear margin. This prevents ambiguous merges when a cluster is equidistant from two candidates.

**PASS when:** `margin_gap >= merge_margin` for **all** exemplars in A

```
margin_gap     = (distance from exemplar_A to nearest competitor) − (distance from exemplar_A to B)
               = competitor_dist − dist_to_b

merge_margin   = config value (default 0.05)
```

The table shows the **worst** (smallest) gap across all exemplars in A.

- `margin_gap > 0`: B is nearer than the competitor by that amount — good.
- `margin_gap = 0`: B and competitor are equidistant — borderline.
- `margin_gap < 0`: competitor is actually closer than B — fails.

**PASS requires `margin_gap >= merge_margin`** (B must be farther from competitors by at least `merge_margin`).

**Table columns:** `margin_gap`, `margin_dist_to_b`, `margin_competitor_dist`

**To relax:** Lower `merge_margin` (default 0.05). Set to `0.0` to **disable this gate entirely** — recommended when Exemplar and Support already pass, because those two gates already enforce closeness conservatively.

**Note:** This gate is one-sided — only A's exemplars are checked, not B's.

---

## Gate 4 — Post-Merge Diameter

**What it checks:** If the two clusters merged, the resulting cluster must not be too wide (spread too far in embedding space).

**PASS when:** `post_diam <= max_diam`

```
post_diam  = max pairwise distance among all faces in A ∪ B
max_diam   = max(diameter_A, diameter_B) × merge_diameter_expansion_factor
```

**Table columns:** `post_diam`, `max_diam`

**To relax:** Raise `merge_diameter_expansion_factor` (default 1.5).

---

## How to Diagnose "Why Isn't This Pair Merging?"

1. Find the pair in the **All Merge Decisions** table (sorted by `exemplar_dist`).
2. Look at the `gates` column (e.g. `2/4`) and the colored PASS/FAIL columns.
3. For each failing gate, compare value vs threshold:

| Failing gate | Look at | How much off? | Action |
|---|---|---|---|
| Exemplar | `gap` (should be negative) | If gap is < 0.05, slightly raise `alpha` | If gap > 0.1, clusters are genuinely different |
| Support | `support` vs `required` | If `support` is 1 below, lower `merge_support_min` to 1 | If far off, clusters lack enough mutual evidence |
| Margin | `margin_gap` vs `merge_margin` (0.05) | If gap is close to 0, set `merge_margin=0.0` | Competing cluster is ambiguously close |
| Diameter | `post_diam` vs `max_diam` | Raise `merge_diameter_expansion_factor` | Merged cluster would be very wide |

**Rule of thumb:** If 3+ gates fail, the clusters are likely different people. If only Margin fails (all others pass), try `merge_margin=0.0`.

---

## How This Compares to Jaccard-Based Merging

**Jaccard index** (used in some graph-based clustering algorithms) measures overlap between neighbor sets in a kNN graph:

```
Jaccard(A, B) = |neighbors(A) ∩ neighbors(B)| / |neighbors(A) ∪ neighbors(B)|
```

A high Jaccard score means A and B share many of the same nearest neighbors — they inhabit the same region of embedding space.

**ConservativeMerger is distance-based, not topology-based:**

| Aspect | Jaccard approach | ConservativeMerger |
|--------|-----------------|-------------------|
| Measurement | Shared neighbors (topology) | Raw embedding distances |
| Exemplar check | Implicit (shared neighbors = nearby) | Explicit: `min(exemplar_dist) <= T_merge` |
| Evidence count | Implicit in intersection size | Explicit: `support` count of pairs below threshold |
| Diameter control | None — can create elongated chains | Explicit: `post_diam <= max_diam × factor` |
| Ambiguity handling | None explicitly | Explicit: `margin_gap >= merge_margin` |
| Adaptive threshold | Fixed Jaccard cutoff | Per-cluster `T_local` based on cluster spread |

**Closest analog:** The **Support gate** is conceptually similar to Jaccard's intersection size — both count how many "connections" exist between two clusters. The key difference is normalization: Jaccard uses the union size, Support uses `fraction × min(|A|, |B|)` which favors merging small clusters into large ones.

**Why not use Jaccard here?**
- kNN graph edges in the face embedding space are already used to form initial clusters. Post-clustering merges operate on cluster-level distances, not individual node neighborhoods.
- Diameter control is critical for face identity: a Jaccard-based merge could chain together transitively similar but ultimately different people. The Diameter gate prevents this.
- Adaptive thresholds (`T_merge` scales with cluster spread) are more principled than a fixed Jaccard cutoff, especially when clusters have different densities (e.g., one person appears in 3 photos vs. 50 photos).
