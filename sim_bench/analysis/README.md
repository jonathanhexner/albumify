# Analysis Utilities

This package provides utilities for analyzing benchmark results.

## Modules

### `config.py`
- `GlobalAnalysisConfig`: Singleton configuration for analysis sessions
- `PlotGridConfig`: Configuration for plotting query results

### `io.py`
- `get_metrics_path()`, `get_per_query_path()`, `get_rankings_path()`: Get paths to result files
- `load_metrics()`, `load_per_query()`, `load_rankings()`: Load result CSVs
- `load_enriched_per_query()`: **Recommended** - Load enriched per-query data with caching
- `build_index_to_path_via_dataset()`: Get image index-to-path mapping using dataset factory

### `metrics.py`
- `compute_recall_at_k()`: Compute recall@k for each query from rankings
- `compute_multiple_recalls()`: Compute recall for multiple k values
- `compute_precision_at_k()`: Compute precision@k for each query
- `get_top_k_results()`: Get top-k results for a specific query
- `compute_enriched_per_query()`: **Main utility** - Load per_query.csv and enrich with additional computed metrics

### `plotting.py`
- `plot_query_topn_grid()`: Visualize a query with its top-N results in a grid layout

### `export.py`
- `export_notebook_to_pdf()`: Export Jupyter notebooks to PDF reports

### `feature_utils.py` 🆕
- `list_feature_caches()`: List all available feature cache files
- `load_features_from_cache()`: Load features from cache with metadata
- `find_cache_for_method()`: Find cache file(s) for a specific method
- `get_features_by_index()`: Extract features by index (e.g., query_idx) **[Recommended]**
- `get_image_path_by_index()`: Get image path by index
- `get_query_feature_matrix()`: Get features as [feature_dim, n_queries] matrix **[New]**
- `analyze_within_group_feature_diversity()`: Find features with high diversity within groups (supports variance, range, std, iqr) **[Updated]**
- `analyze_feature_discriminability()`: Find features that best separate groups (Fisher criterion) **[New]**
- `get_features_for_images()`: Extract features for specific images by filename
- `search_images_by_filename()`: Search for images by filename pattern
- `compute_feature_statistics()`: Comprehensive per-dimension statistics
- `compute_feature_correlations()`: Correlation matrix for feature dimensions
- `compute_sparsity_metrics()`: Analyze feature sparsity
- `compute_distance_statistics()`: Statistics on pairwise distances
- `extract_group_labels_from_paths()`: Extract ground-truth labels from image paths
- `compute_intra_inter_class_distances()`: Compute intra/inter-class distance statistics

### `feature_viz.py` 🆕
- `plot_feature_distributions()`: Plot distributions of feature dimensions
- `plot_feature_correlation_heatmap()`: Correlation matrix heatmap
- `plot_pca_explained_variance()`: PCA scree plot and cumulative variance
- `plot_embedding_2d()`: 2D embedding visualization (t-SNE, UMAP, PCA)
- `plot_distance_distribution()`: Intra vs inter-class distance distributions
- `plot_feature_statistics_summary()`: Summary statistics across dimensions
- `plot_nearest_neighbors_grid()`: Visualize query with nearest neighbors
- `plot_cluster_quality_metrics()`: Silhouette analysis for clustering
- `plot_queries_by_group()`: 2D visualization of queries colored by group (PCA on full dataset)
- `plot_queries_vs_dataset_by_group()`: Queries vs full dataset with group coloring
- `plot_query_feature_analysis_by_group()`: 4-panel detailed query analysis
- `plot_within_group_diversity()`: Visualize feature variance within groups **[New]**

### `attribution/` 🆕 **[Reorganized]**

Feature attribution subpackage for understanding what features focus on.

#### `attribution/base.py`
- `BaseAttributionExtractor`: Abstract base class for attribution methods

#### `attribution/resnet50.py`
- `ResNet50AttributionExtractor`: Grad-CAM attribution for ResNet-50 features
  - `extract_features()`: Extract features (same as sim-bench)
  - `compute_attribution()`: Compute Grad-CAM heatmap
  - `analyze_feature_importance()`: Find most active feature dimensions

#### `attribution/visualization.py` (General utilities)
- `plot_attribution_overlay()`: Overlay heatmap on image
- `plot_attribution_comparison()`: Compare attributions across images
- `plot_feature_importance()`: Visualize feature importance
- `visualize_feature_dimensions()`: Create separate visualizations per feature

**Note:** `feature_attribution.py` is deprecated. Use `from sim_bench.analysis.attribution import ResNet50AttributionExtractor` instead.

## Quick Start

### 1. Set up configuration
```python
from pathlib import Path
from sim_bench.analysis.config import GlobalAnalysisConfig, set_global_config

EXPERIMENT_DIR = Path(r"D:\sim-bench\outputs\baseline_runs\comprehensive_baseline\2025-10-08_16-25-49")
set_global_config(GlobalAnalysisConfig(experiment_dir=EXPERIMENT_DIR).resolve())
```

### 2. Load basic results
```python
from sim_bench.analysis.io import load_per_query, load_rankings

df_per_query = load_per_query("deep")
df_rankings = load_rankings("deep")
```

### 3. Load enriched data (with caching)
```python
from sim_bench.analysis.io import load_enriched_per_query

# Get enriched per-query data with recall@1, recall@2, etc.
# Results are automatically cached in enriched_cache/ to avoid recomputation
df_enriched = load_enriched_per_query("deep", k_values=[1, 2, 3, 4, 5])
print(df_enriched.columns)
# Output: ['query_idx', 'query_path', 'group_id', 'ns_hitcount@4', 'ap@10', 'recall@1', 'recall@2', 'recall@3', 'recall@4', 'recall@5']

# To force recomputation (e.g., if rankings changed):
df_enriched = load_enriched_per_query("deep", k_values=[1, 2, 3, 4, 5], force_recompute=True)
```

### 4. Visualize results
```python
from sim_bench.analysis.plotting import plot_query_topn_grid, PlotGridConfig

config = PlotGridConfig(top_n=6, save=True)
plot_query_topn_grid(
    method="deep",
    query_idx=42,
    config=config,
    dataset_name="ukbench",
    dataset_config={...}
)
```

## Saved Result Files

Each method produces the following files:

- **`metrics.csv`**: Overall aggregate metrics for the method
- **`per_query.csv`**: Per-query metrics (limited to a few pre-computed metrics)
- **`rankings.csv`**: Raw rankings (query_idx, rank, result_idx, distance) for top-k results
- **`manifest.json`**: Method configuration and metadata

## Computing Metrics On-Demand with Caching

The `per_query.csv` file contains only essential pre-computed metrics to keep file sizes manageable:
- **UKBench**: `ns_hitcount@4`, `ap@10`
- **Holidays**: `ap_full`, `ap@10`, `recall@10`

Additional metrics (recall@k, precision@k for various k values) can be computed on-demand from `rankings.csv` using `load_enriched_per_query()`. This approach:
- Keeps saved files lean
- Allows flexible metric computation as needed
- Avoids re-running expensive experiments
- **Caches results** in `enriched_cache/` to avoid recomputation on subsequent runs

**Cache Management:**
- Cache files are stored in `<experiment_dir>/enriched_cache/`
- Cache key includes method name and k_values
- To clear cache: delete the `enriched_cache/` directory or specific files
- To force recomputation: use `force_recompute=True` parameter

## Notebooks

- **`methods_comparison.ipynb`**: 🆕 Compare multiple methods (summary stats, rankings, correlations)
- **`method_analysis.ipynb`**: Deep-dive analysis of ONE method (good/bad/intermediate queries with visualizations)
- **`deep_feature_exploration.ipynb`**: Deep learning feature analysis (PCA, t-SNE, within-group diversity, Grad-CAM)

## Typical Analysis Workflow

1. **Run experiments** to generate metrics and feature caches
2. **Compare methods** using `methods_comparison.ipynb` to see overall winners
3. **Deep-dive** into specific methods using `method_analysis.ipynb` to understand failures
4. **Feature analysis** (for deep learning) using `deep_feature_exploration.ipynb` to understand what the model focuses on

## Tips

1. **Always set global config first** - Most functions use it as a default
2. **Use absolute paths** in notebook configs for clarity
3. **Compute metrics on-demand** rather than saving everything upfront
4. **Export notebooks to PDF** for sharing results
5. **Feature analysis**: Run `feature_exploration.ipynb` to understand WHY a method performs well/poorly

