# Analysis Guide

This guide covers the three analysis notebooks for exploring experimental results.

## Overview

The analysis workflow consists of three notebooks with increasing specificity:

1. **methods_comparison.ipynb** - Compare all methods from an experiment
2. **method_analysis.ipynb** - Analyze a specific method's query performance
3. **feature_exploration.ipynb** - Understand why features succeed/fail

All notebooks support PDF export and save plots to `outputs/<experiment>/analysis_reports/`.

---

## 1. Methods Comparison (methods_comparison.ipynb)

**Purpose**: Compare multiple similarity methods side-by-side.

### What It Does

- **Summary Statistics**: Overall performance for each method
- **Performance Rankings**: Rank methods by each metric
- **Box Plots**: Per-query metric distributions
- **Correlation Analysis**: Do methods agree on query difficulty?

### Configuration

```python
EXPERIMENT_DIR = "outputs/baseline_runs/comprehensive_baseline/2025-10-24_01-10-45"
METHODS = ["emd", "deep", "sift_bovw"]
EXPORT_PDF = True
```

### Outputs

**Tables**:
- Performance comparison table (highlights best values)
- Method rankings (1=best)
- Correlation matrix

**Plots**:
- Bar charts: metrics by method
- Box plots: per-query distributions
- Heatmap: method correlations

**Use Case**: "Which method performs best overall? Do ResNet and SIFT agree on hard queries?"

---

## 2. Method Analysis (method_analysis.ipynb)

**Purpose**: Deep-dive into a single method's performance.

### What It Does

- **Query Selection**: Automatically selects good/intermediate/bad queries
- **Visual Inspection**: Shows query image, top-5 results, ground truth
- **Filenames Displayed**: All images show filenames for clarity

### Configuration

```python
EXPERIMENT_DIR = "outputs/.../2025-10-24_01-10-45"
METHOD = "deep"
RANKING_METRIC = "ap@10"  # Sort queries by this metric
NUM_IMAGES_PER_GROUP = 3  # N queries per category

DATASET_NAME = "holidays"
DATASET_CONFIG = {
    "name": "holidays",
    "root": "D:/Similar Images/DataSets/InriaHolidaysFull",
    "pattern": "*.jpg"
}
```

### Outputs

**For each selected query**:
- **Query image**: With filename, idx, group
- **Top-5 results**: With filenames, distances, group matches (✓)
- **Ground truth**: Expected similar images

**Categories**:
- **GOOD**: Highest metric scores (perfect retrieval)
- **INTERMEDIATE**: Around median (partial success)
- **BAD**: Lowest scores (retrieval failures)

**Use Case**: "Why did ResNet fail on query 745? Let me see what it retrieved vs ground truth."

---

## 3. Feature Exploration (feature_exploration.ipynb)

**Purpose**: Understand why feature representations succeed or fail.

### What It Does

**Discriminability Analysis**:
- **Fisher Criterion**: Between-group variance / within-group variance
- High score = groups well-separated (good feature)
- Low score = groups overlap (bad feature)

**Within-Group Variance Analysis**:
- Which features vary most within similar images?
- High variance = inconsistent feature, may cause failures

**2D Visualization**:
- PCA projection showing group separation
- Colored by group ID

**Feature Attribution** (method-specific):
- **ResNet-50**: Grad-CAM heatmaps (which pixels matter?)
- **SIFT BoVW**: Keypoint visualization (which visual words fire?)

### Configuration

```python
EXPERIMENT_DIR = "outputs/.../2025-10-08_16-25-49"
METHOD_NAME = "sift_bovw"  # or "deep"
DATASET_TYPE = "ukbench"
QUERY_INDICES = list(range(7144, 7152))  # 2 groups
SAVE_PLOTS = True
```

### Outputs

**Discriminability**:
- Top-10 most discriminative features (best separators)
- Top-10 least discriminative features (worst separators)

**Box Plots**:
- Show feature value distributions by group
- Colored by query index
- Identifies unstable features

**PCA 2D**:
- Scatter plot of features in 2D space
- Groups should cluster tightly

**Attribution Maps**:
- **Grad-CAM** (ResNet): Red regions = important pixels
- **Keypoints** (SIFT): Shows which visual words activate

**Use Case**: "Why does SIFT confuse these groups? Let me see which features are unstable and what image regions they focus on."

---

## Analysis Workflow

### Typical Analysis Session

**Step 1**: Run experiment
```bash
python -m sim_bench.cli --methods emd,deep,sift_bovw --datasets holidays
```

**Step 2**: Compare methods
```python
# In methods_comparison.ipynb
EXPERIMENT_DIR = "outputs/.../2025-10-24_01-10-45"
METHODS = ["emd", "deep", "sift_bovw"]
# → Find that ResNet performs best
```

**Step 3**: Analyze best method
```python
# In method_analysis.ipynb
METHOD = "deep"
RANKING_METRIC = "ap@10"
# → Identify specific failure cases (bad queries)
```

**Step 4**: Understand failures
```python
# In feature_exploration.ipynb
METHOD_NAME = "deep"
QUERY_INDICES = [745, 799, 805]  # The bad queries from step 3
# → See which features are unstable, visualize with Grad-CAM
```

---

## Utility Functions

### Loading Data

```python
from sim_bench.analysis.io import (
    load_metrics,           # Overall performance
    load_per_query,         # Per-query metrics
    load_rankings,          # Full rankings
    load_enriched_per_query # With on-demand metrics
)

# Example
df = load_enriched_per_query("deep", k_values=[1,2,3,4,5])
```

### Visualization

```python
from sim_bench.analysis.plotting import plot_query_topn_grid

# Show query with top-5 results and ground truth
plot_query_topn_grid(
    method="deep",
    query_idx=298,
    config=PlotGridConfig(top_n=5),
    dataset_name="holidays",
    dataset_config={...},
    show_ground_truth=True
)
```

### Feature Analysis

```python
from sim_bench.analysis.feature_utils import (
    load_features_from_cache,
    analyze_feature_discriminability,
    analyze_within_group_feature_diversity
)

# Load cached features
features, metadata = load_features_from_cache(cache_file, return_metadata=True)

# Find which features separate groups best
disc_results = analyze_feature_discriminability(
    feature_matrix=features,
    group_ids=group_labels,
    top_k=20
)
```

### Attribution (Deep Learning)

```python
from sim_bench.analysis.attribution.resnet50 import ResNet50AttributionExtractor
from sim_bench.analysis.attribution.visualization import plot_attribution_overlay

# Extract Grad-CAM heatmaps
extractor = ResNet50AttributionExtractor(device='cpu')
heatmap, img = extractor.compute_attribution(image_path, feature_dims=[512, 1024])
overlay = plot_attribution_overlay(heatmap, img, alpha=0.5)
```

---

## Feature Quality Metrics

### Fisher Criterion (Discriminability)

**Formula**: `Between-Group Variance / Within-Group Variance`

**Interpretation**:
- **High (>5)**: Feature clearly separates groups → Keep
- **Medium (1-5)**: Moderate separation → Useful
- **Low (<1)**: Groups overlap → Consider removing

**Use**: Feature selection, understanding which dimensions matter

### Within-Group Variance

**Formula**: `Variance of feature values within same group`

**Interpretation**:
- **High**: Feature is inconsistent for similar images → Problematic
- **Low**: Feature is stable for similar images → Reliable

**Use**: Identify unreliable features causing retrieval failures

---

## Tips

### Performance

- Use `SAVE_PLOTS=True` to save all figures
- Set `EXPORT_PDF=True` for archival reports
- Enriched metrics are cached in `enriched_cache/` (delete to recompute)

### Analysis Best Practices

1. **Start broad** (methods_comparison) → **zoom in** (method_analysis) → **understand deeply** (feature_exploration)
2. **Compare multiple methods** to find consistent patterns vs method-specific issues
3. **Use feature attribution** (Grad-CAM/keypoints) to validate that features make sense
4. **Check Fisher scores** to understand which features are actually useful

### Common Issues

**Q: "Notebook shows map_full=0"**
A: Old experiment results use old metric names. See [METRICS_BUG_FIX.md](METRICS_BUG_FIX.md) for backward compatibility.

**Q: "Grad-CAM not working"**
A: Only available for `method="deep"`. SIFT uses keypoint visualization instead.

**Q: "Feature cache not found"**
A: Run the experiment first with feature caching enabled (`cache_features: true` in config).

---

## Related Documentation

- [DATASETS.md](DATASETS.md) - Dataset details (UKBench, Holidays)
- [PERFORMANCE.md](PERFORMANCE.md) - Feature caching and optimization
- [CACHE_STORAGE.md](CACHE_STORAGE.md) - Cache structure and management

For implementation details, see `sim_bench/analysis/README.md`.
