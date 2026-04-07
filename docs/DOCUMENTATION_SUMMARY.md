# Face Clustering Documentation Summary

## Files to Share

### 1. **FACE_CLUSTERING_COMPLETE_GUIDE.md** ⭐ (Primary Document)
**Complete, self-contained guide to the face clustering pipeline.**

**Contents:**
- Quick start code example
- Pipeline overview (6 stages: Quality Gating → Distance Matrix → Mutual kNN Graph → Connected Components → Exemplars → Merge)
- All configuration parameters with defaults and recommendations
- Stage-by-stage detailed explanations
- Merge criteria deep dive (4 criteria with formulas and examples)
- Analysis & debugging methods
- Troubleshooting decision tree
- Performance tuning guide
- Production checklist

**Key Updates:**
- ✅ PRIMARY analysis method: `get_cluster_distances()` - shows why clusters didn't merge
- ✅ Configurable `merge_global_percentile` (25/50/75/90)
- ✅ How to get pairwise distances between clusters
- ✅ How to get exemplar distances between clusters
- ✅ Clear distinction: Exemplar_Dist (for proposals) vs Min_Dist (for outliers)

---

### 2. **README.md** (Directory Index)
**Points to the complete guide and marks legacy docs as consolidated.**

---

## Legacy Files (Now Consolidated)

These files still exist but are **superseded by FACE_CLUSTERING_COMPLETE_GUIDE.md**:
- `KNN_CLUSTERING_PIPELINE.md` - Merged into complete guide
- `MERGE_CRITERIA_EXPLAINED.md` - Merged into complete guide

**Recommendation**: Share only the complete guide, not the legacy files.

---

## Additional Reference (Optional)

### MERGE_ANALYSIS_QUICK_REFERENCE.md
**Quick answers to common questions:**
- Why aren't these merge candidates?
- What criteria failed?
- How to get cluster_struct after merger?
- How to get distance matrix?
- How to experiment with `merge_global_percentile`?

**Use case**: Quick lookup for specific questions during analysis.

---

## What's Complete and Accurate

✅ **Pipeline stages** - All 6 stages documented with code examples
✅ **Configuration** - All parameters with defaults, ranges, and recommendations
✅ **Merge criteria** - All 4 criteria with formulas, examples, and failure patterns
✅ **Analysis methods** - Primary (`get_cluster_distances`) and secondary (`get_merge_decisions_df`)
✅ **Distance metrics** - Clear distinction between Exemplar_Dist, Min_Dist, Mean_Dist
✅ **Troubleshooting** - Decision tree for debugging merge failures
✅ **Code examples** - Working code snippets for all common tasks
✅ **New features** - `merge_global_percentile` parameter documented

---

## What to Tell Recipients

**"Read FACE_CLUSTERING_COMPLETE_GUIDE.md - it's a self-contained guide to the entire pipeline."**

The document is:
- **Complete**: Covers all stages, parameters, and analysis methods
- **Concise**: Tables and code examples, minimal prose
- **Precise**: Exact formulas, thresholds, and parameter names
- **Actionable**: Clear troubleshooting steps and production checklist

---

## Quick Verification

Run this to verify the pipeline works:
```python
from face_cluster import PipelineConfig
config = PipelineConfig(
    K=5,
    distance_threshold=0.35,
    merge_enabled=True,
    merge_margin=0.0,
    merge_global_percentile=50,  # New parameter
)
print(f"Config loaded: {config.merge_global_percentile}")
```

All code examples in the guide use the actual API and should run without modification.
