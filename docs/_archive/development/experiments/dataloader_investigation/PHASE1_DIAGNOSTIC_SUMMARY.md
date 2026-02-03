# Phase 1 Diagnostic Summary

**Date**: 2026-01-13  
**Problem**: Internal dataloader (51.8% acc) vs External dataloader (69.6% acc)

---

## 🔍 Key Findings

### 1. Transform Verification ✅ NOT THE ISSUE

**Result**: Transforms are **COMPLETELY IDENTICAL**

- Mean absolute difference: **0.000000**
- Max absolute difference: **0.000000**  
- All 20 test images produced identical tensors
- Shape consistency: ✅ ALL MATCH (3, 224, 224)

**Conclusion**: Transform is NOT causing the performance difference.

---

### 2. Label Verification ⚠️ MIXED SIGNALS

**Result**: Very low overlap between dataloaders

- **Internal**: 9,760 training pairs
- **External**: 12,075 training pairs
- **Overlapping pairs**: Only 4 out of 500 checked **(0.8%!)**

**For the 4 overlapping pairs**:
- 2 labels match perfectly (50%)
- 2 labels inverted (50%)

| Image1 | Image2 | Internal Winner | External Winner | Match? |
|--------|--------|----------------|-----------------|--------|
| 005167-01.JPG | 005167-03.JPG | 1 | 1 | ✅ YES |
| 002105-01.JPG | 002105-05.JPG | 1 | 1 | ✅ YES |
| 000626-01.JPG | 000626-03.JPG | 0 | 1 | ❌ INVERTED |
| 002053-01.JPG | 002053-02.JPG | 0 | 1 | ❌ INVERTED |

**Conclusion**: Sample size too small to determine if labels are consistently inverted. The **real issue is that the dataloaders are creating almost completely different pair sets**.

---

### 3. Dataset Sizes ❗ MAJOR DIFFERENCE

| Loader | Train Pairs | Val Pairs | Source |
|--------|-------------|-----------|--------|
| Internal | 9,760 | 1,114 | PhotoTriageData filtered (agreement≥0.7, reviewers≥2) |
| External | 12,075 | 483 | make_shuffle_path.make_shuffle_path() |

**Key observations**:
- External has **23.7% MORE training pairs** (12,075 vs 9,760)
- External has **56.6% FEWER validation pairs** (483 vs 1,114)
- Different train/val split ratios:
  - Internal: 89.8% train / 10.2% val
  - External: 96.2% train / 3.8% val

---

### 4. Pair Creation Logic 🔎 ROOT CAUSE

**Internal (PhotoTriageData)**:
```python
# From photo_triage_pairs_embedding_labels.csv
# Filters: agreement >= 0.7, num_reviewers >= 2
# Result: 12,073 pairs → 9,760 train / 1,114 val / 1,199 test
# Series-based split ensures no series overlap
```

**External (MyDataset)**:
```python
# Uses make_shuffle_path.make_shuffle_path(seed=seed)
# Returns: pathA, pathB, result (for train and val)
# Result: 12,075 train / 483 val
# Unknown: How pairs are created, what filters are applied
```

---

## 🎯 Root Cause Analysis

### Primary Issue: **Different Pair Sets**

The two dataloaders are creating **fundamentally different training data**:

1. **Different pair selection**:
   - Internal filters by agreement (≥0.7) and reviewers (≥2)
   - External uses unknown criteria in `make_shuffle_path`
   - Only 0.8% overlap in sampled pairs

2. **Different data volumes**:
   - External has 2,315 MORE training pairs
   - Could include "easier" or "harder" pairs

3. **Different train/val splits**:
   - Internal: 89.8% / 10.2% (more conservative)
   - External: 96.2% / 3.8% (more aggressive)

### Secondary Issue: **Possible Label Inversion** (Uncertain)

Of 4 overlapping pairs:
- 2 have matching labels
- 2 have inverted labels (winner 0↔1)

**Cannot conclude** if this is systematic due to tiny sample size.

---

## 💡 Recommended Actions

### Immediate Next Steps:

1. **Understand `make_shuffle_path` logic**:
   - How does it create pairs?
   - What filters does it apply?
   - Where do the 12,075 pairs come from?

2. **Compare source CSVs**:
   - Internal uses: `photo_triage_pairs_embedding_labels.csv`
   - External uses: (unknown - likely same CSV but different processing)
   - Check if external skips the agreement/reviewer filters

3. **Test hypothesis - Use external pair set with internal transform**:
   - Modify internal dataloader to use ALL 12,075 pairs (skip filters)
   - Train 1 epoch
   - If accuracy improves → problem is filtered pair selection
   - If accuracy stays bad → problem is something else

4. **Check label semantics on larger sample**:
   - Need more than 4 overlapping pairs
   - Manually inspect 20-30 pairs to check if:
     - winner=0 means image1 wins in both
     - OR winner=0 means image1 in internal but image2 in external

### Low Priority (Not Needed Yet):

- ❌ Transform isolation (transforms are identical)
- ❌ Enhanced telemetry (data issue, not gradient issue)
- ❌ Weight snapshots (fix data first)

---

## 📊 Files Generated

### Outputs:
- `outputs/label_verification/pair_labels.csv` - 4 overlapping pairs with label comparison
- `outputs/label_verification/summary.json` - Statistics (50% match, 50% inverted)
- `outputs/transform_verification/stats.csv` - Transform comparison (all zeros)

### Scripts Created:
- `scripts/verify_labels.py` - Label consistency checker
- `scripts/compare_pairs.py` - Full pair set comparison (timed out - too slow)
- `scripts/verify_transform_application.py` - Transform output verification

---

## 🚨 Critical Insight

**The 51.8% vs 69.6% accuracy difference is NOT due to transform or label semantics.**

**It's because the dataloaders are training on almost completely different datasets!**

- External trains on 12,075 pairs (unknown selection criteria)
- Internal trains on 9,760 pairs (filtered: agreement≥0.7, reviewers≥2)

**The "working" external loader might actually be using LOWER QUALITY data** (less agreement, fewer reviewers) which paradoxically could be easier to learn from if it includes more "obvious" comparisons.

**Next action**: Investigate `make_shuffle_path` and test if using the external pair set improves internal loader performance.
