# Feature Exploration Guide

This guide explains how to explore and analyze feature representations after running experiments.

## Overview

After running an experiment, you can access and analyze:
- ✅ **Features**: Stored in `artifacts/feature_cache/` as pickle files
- ❌ **Full distance matrices**: NOT stored (too large, ~400MB for 10k images)
- ✅ **Top-k distances**: Stored in `outputs/.../rankings.csv` (default k=10)

## What's New

We've added comprehensive feature exploration capabilities:

1. **`sim_bench/analysis/feature_utils.py`** - Feature loading and analysis utilities
2. **`sim_bench/analysis/feature_viz.py`** - Feature space visualization functions
3. **`sim_bench/analysis/feature_exploration.ipynb`** - Ready-to-use Jupyter notebook

## Quick Start

### 1. Run an Experiment (Generate Features)

```bash
# Features are automatically cached when you run experiments
python -m sim_bench.cli --methods deep --datasets ukbench

# Check your feature caches
ls artifacts/feature_cache/
# You'll see files like: deep_6d1e6b231e77eb73.pkl
```

### 2. Open the Feature Exploration Notebook

```bash
jupyter notebook sim_bench/analysis/feature_exploration.ipynb
```

### 3. Configure and Run

In the notebook configuration cell:

```python
METHOD_NAME = "deep"  # Change to your method
DATASET_TYPE = "ukbench"  # Options: 'ukbench', 'holidays'
OUTPUT_DIR = Path("sim_bench/analysis/outputs/feature_exploration")
```

Then **Run All Cells**! The notebook will:
- Load features from cache
- Compute statistics and distributions
- Generate PCA analysis and visualizations
- Create t-SNE embeddings
- Perform clustering analysis
- Analyze class separability
- Save all plots to the output directory

## What the Notebook Provides

### 1. Feature Statistics
- Per-dimension statistics (mean, std, min, max, median, IQR)
- Distribution plots for top dimensions
- Sparsity analysis
- Correlation matrices

### 2. Dimensionality Reduction

**PCA (Principal Component Analysis):**
- Scree plot showing explained variance
- Cumulative variance plot
- 2D projection colored by ground-truth labels

**t-SNE (t-Distributed Stochastic Neighbor Embedding):**
- 2D embedding for visualization
- Colored by ground-truth groups
- Reveals cluster structure

### 3. Cluster Analysis
- K-means clustering
- Silhouette scores (quality metrics)
- Cluster distribution visualization

### 4. Separability Analysis
- Intra-class vs inter-class distances
- Separability ratio (higher = better)
- Distance distribution plots

### 5. Nearest Neighbor Exploration
- Find nearest neighbors in feature space
- Visual comparison with ground-truth
- Identify confusing examples

## Loading Features Programmatically

If you want to write custom analysis code:

### Method 1: Using feature_utils (Recommended)

```python
from sim_bench.analysis import feature_utils
from pathlib import Path

# Find cache for a method
cache_file = feature_utils.find_cache_for_method("deep")

# Load features with metadata
features, metadata = feature_utils.load_features_from_cache(
    cache_file, 
    return_metadata=True
)

print(f"Feature shape: {features.shape}")
print(f"Number of images: {metadata['n_images']}")
print(f"Feature dimension: {metadata['feature_dim']}")
print(f"Image paths: {len(metadata['image_paths'])}")

# Now do custom analysis
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embedding = tsne.fit_transform(features[:500])  # Sample for speed
```

### Method 2: Direct pickle loading

```python
import pickle
from pathlib import Path

cache_file = Path("artifacts/feature_cache/deep_6d1e6b231e77eb73.pkl")

with open(cache_file, 'rb') as f:
    cached_data = pickle.load(f)

features = cached_data['features']  # np.ndarray [n_images, feature_dim]
image_paths = cached_data['image_paths']
method_config = cached_data['method_config']
```

### Method 3: Using the framework (ensures cache is used)

```python
from sim_bench.feature_extraction import load_method
from sim_bench.datasets import load_dataset
import yaml

# Load dataset
dataset_config = yaml.safe_load(open('configs/dataset.ukbench.yaml'))
dataset = load_dataset('ukbench', dataset_config)
dataset.load_data()
image_paths = dataset.get_images()

# Load method
method_config = yaml.safe_load(open('configs/methods/deep.yaml'))
method = load_method('deep', method_config)

# Extract features (loads from cache if available!)
features = method.extract_features(image_paths)
```

## Computing Custom Distances

If you need the full distance matrix for custom analysis:

```python
from sim_bench.analysis import feature_utils
from scipy.spatial.distance import cdist

# Load features
cache_file = feature_utils.find_cache_for_method("deep")
features, _ = feature_utils.load_features_from_cache(cache_file, return_metadata=True)

# Compute custom distance matrix
distance_matrix = cdist(features, features, metric='euclidean')
# or 'cosine', 'cityblock', 'chebyshev', etc.

print(f"Distance matrix shape: {distance_matrix.shape}")

# Find nearest neighbors for query 0
query_idx = 0
distances = distance_matrix[query_idx]
nearest_indices = np.argsort(distances)[:10]  # Top 10
```

## Available Utility Functions

### Feature Loading (`feature_utils.py`)

| Function | Purpose |
|----------|---------|
| `list_feature_caches()` | List all cached feature files |
| `load_features_from_cache()` | Load features with metadata |
| `find_cache_for_method()` | Find cache file for specific method |
| `compute_feature_statistics()` | Comprehensive per-dimension stats |
| `compute_feature_correlations()` | Correlation matrix |
| `compute_sparsity_metrics()` | Sparsity analysis |
| `compute_distance_statistics()` | Pairwise distance stats |
| `extract_group_labels_from_paths()` | Get ground-truth labels |
| `compute_intra_inter_class_distances()` | Separability analysis |

### Visualization (`feature_viz.py`)

| Function | Purpose |
|----------|---------|
| `plot_feature_distributions()` | Histogram of feature values |
| `plot_feature_correlation_heatmap()` | Correlation matrix heatmap |
| `plot_pca_explained_variance()` | PCA scree plot |
| `plot_embedding_2d()` | 2D embedding visualization |
| `plot_distance_distribution()` | Intra vs inter-class distances |
| `plot_feature_statistics_summary()` | Statistics across dimensions |
| `plot_nearest_neighbors_grid()` | Visual nearest neighbor comparison |
| `plot_cluster_quality_metrics()` | Silhouette analysis |

## Example: Compare Multiple Methods

```python
from sim_bench.analysis import feature_utils
import pandas as pd

methods = ['deep', 'chi_square', 'sift_bovw']
results = []

for method in methods:
    cache_file = feature_utils.find_cache_for_method(method)
    if cache_file is None:
        continue
    
    features, metadata = feature_utils.load_features_from_cache(cache_file, return_metadata=True)
    labels = feature_utils.extract_group_labels_from_paths(metadata['image_paths'], 'ukbench')
    
    # Compute separability
    stats = feature_utils.compute_intra_inter_class_distances(features, labels)
    
    results.append({
        'method': method,
        'feature_dim': metadata['feature_dim'],
        'intra_mean': stats['intra_mean'],
        'inter_mean': stats['inter_mean'],
        'separability_ratio': stats['separability_ratio']
    })

df = pd.DataFrame(results)
print(df.sort_values('separability_ratio', ascending=False))
```

## Interpreting Results

### PCA Analysis
- **High variance in few components** → Features are redundant, can be compressed
- **Flat variance distribution** → Features capture diverse information

### t-SNE Visualization
- **Clear clusters by ground-truth** → Good feature separability
- **Mixed/overlapping clusters** → Features struggle to distinguish classes

### Separability Ratio (Inter/Intra)
- **> 2.0** → Excellent class separation
- **1.5-2.0** → Good separation
- **< 1.5** → Poor separation, features need improvement

### Silhouette Score
- **> 0.5** → Well-defined clusters
- **0.3-0.5** → Moderate cluster structure
- **< 0.3** → No clear clusters

## Tips

1. **Always use cached features** - No need to re-extract
2. **Sample for t-SNE** - It's slow on large datasets (use N_TSNE_SAMPLES=500)
3. **Compare methods** - Run notebook for each method to see which has better feature spaces
4. **Check separability** - High ratio indicates features work well for similarity search
5. **Use nearest neighbors** - Visually inspect to understand failures

## Troubleshooting

### "No cache found for method"
Run an experiment first:
```bash
python -m sim_bench.cli --methods deep --datasets ukbench
```

### "Could not extract labels"
Check that:
- Image paths follow standard naming (ukbench00000.jpg or 100000.jpg)
- DATASET_TYPE matches your actual dataset

### t-SNE is too slow
Reduce N_TSNE_SAMPLES in the configuration cell:
```python
N_TSNE_SAMPLES = 200  # Faster but less representative
```

### Out of memory
Sample the dataset before analysis:
```python
sample_size = 1000
indices = np.random.choice(len(features), size=sample_size, replace=False)
features_sample = features[indices]
```

## See Also

- **`docs/PERFORMANCE.md`**: Feature caching details
- **`sim_bench/analysis/README.md`**: Analysis package overview
- **`docs/ANALYSIS_NOTEBOOK_GUIDE.md`**: Result analysis guide


