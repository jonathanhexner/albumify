# Clustering Feature Documentation

## Overview

The clustering feature allows you to run a single similarity method and cluster the resulting feature vectors. This is useful for:
- Grouping visually similar images
- Finding duplicate images
- Organizing image collections
- Exploratory data analysis

## Implementation

### Architecture

The clustering feature follows the same design patterns as the rest of sim-bench:
- **Factory Pattern**: `load_clustering_method()` creates clusterers by algorithm name
- **Strategy Pattern**: `ClusteringMethod` abstract base class with concrete implementations
- **Subpackage Structure**: Organized like `feature_extraction/`

### Core Components

1. **sim_bench/clustering/** - Clustering subpackage
   - `base.py` - Abstract `ClusteringMethod` class and factory function
   - `dbscan.py` - `DBSCANClusterer` implementation
   - `kmeans.py` - `KMeansClusterer` implementation
   - `__init__.py` - Public API exports

2. **ExperimentRunner.run_clustering()** - Integration with experiment runner
   - Reuses existing feature extraction with caching
   - Uses factory to load clustering method
   - Delegates to clusterer's `cluster()` and `save_results()` methods

3. **CLI integration** - Automatic detection of clustering mode
   - Detects `clustering.enabled: true` in config
   - Routes to clustering workflow

### Supported Algorithms

#### DBSCAN (Density-Based)
- **Pros**: Finds clusters of arbitrary shape, identifies noise/outliers
- **Cons**: Requires parameter tuning (eps), can be O(n²) for large datasets
- **Parameters**:
  - `metric`: Distance metric ('cosine' or 'euclidean')
  - `eps`: Maximum distance for neighborhood (tune based on data)
  - `min_samples`: Minimum samples for core point
- **Output**: Cluster labels (0, 1, 2, ...) and -1 for noise

#### HDBSCAN (Hierarchical Density-Based) **[NEW]**
- **Pros**: Automatically determines number of clusters, more robust than DBSCAN, handles varying densities
- **Cons**: Slightly slower than DBSCAN, requires hdbscan package
- **Parameters**:
  - `metric`: Distance metric ('cosine' or 'euclidean')
  - `min_cluster_size`: Minimum cluster size (main parameter to tune)
  - `min_samples`: Minimum samples (defaults to min_cluster_size)
  - `cluster_selection_epsilon`: Optional DBSCAN-like threshold
  - `cluster_selection_method`: 'eom' (excess of mass) or 'leaf'
- **Output**: Cluster labels (0, 1, 2, ...) and -1 for noise, plus persistence scores
- **Installation**: `pip install hdbscan`

#### KMeans (Centroid-Based)
- **Pros**: Fast, scalable, guaranteed clusters
- **Cons**: Requires pre-specifying number of clusters, assumes spherical clusters
- **Parameters**:
  - `n_clusters`: Number of clusters to create
  - `n_init`: Number of random initializations
  - `random_state`: Random seed for reproducibility
- **Output**: Cluster labels (0, 1, 2, ..., n_clusters-1)

## Configuration

### Example: DBSCAN Clustering

```yaml
# configs/run.cluster.yaml

experiment:
  name: ukbench_sift_dbscan
  description: Cluster UKBench images using SIFT BoVW features with DBSCAN

# Single dataset and method
dataset: ukbench
method: sift_bovw

# Optional sampling
sampling:
  max_groups: 200
  max_queries: null

# Clustering configuration
clustering:
  enabled: true
  algorithm: dbscan
  
  params:
    metric: cosine          # 'cosine' or 'euclidean'
    eps: 0.30               # Maximum distance for neighborhood
    min_samples: 4          # Minimum samples for core point
  
  output:
    save_csv: true          # Save clusters.csv
    save_stats: true        # Save cluster_stats.json
    save_galleries: false   # Future: HTML galleries

logging:
  level: INFO
  detailed: false

cache_features: true
output_dir: outputs/cluster_runs/ukbench_sift_dbscan
random_seed: 42
```

### Example: HDBSCAN Clustering

```yaml
# configs/run.cluster_budapest_hdbscan.yaml

experiment:
  name: budapest_hdbscan
  description: HDBSCAN clustering (auto-determines # clusters)

dataset: budapest
method: dinov2

clustering:
  enabled: true
  algorithm: hdbscan
  
  params:
    metric: cosine
    min_cluster_size: 5     # Main parameter to tune
    min_samples: None       # Defaults to min_cluster_size
    cluster_selection_method: eom  # 'eom' or 'leaf'
  
  output:
    save_csv: true
    save_stats: true
    save_galleries: true

cache_features: true
output_dir: outputs/cluster_runs/budapest_hdbscan
```

### Example: KMeans Clustering

```yaml
# configs/run.cluster_kmeans.yaml

experiment:
  name: ukbench_sift_kmeans
  description: Cluster UKBench images using SIFT BoVW features with KMeans

dataset: ukbench
method: sift_bovw

clustering:
  enabled: true
  algorithm: kmeans
  
  params:
    n_clusters: 10          # Number of clusters
    n_init: 10              # Number of initializations
    random_state: 42        # For reproducibility
  
  output:
    save_csv: true
    save_stats: true

cache_features: true
output_dir: outputs/cluster_runs/ukbench_sift_kmeans
random_seed: 42
```

## Usage

### Command Line

```bash
# Run clustering with full dataset
python -m sim_bench.cli --run-config configs/run.cluster.yaml

# Quick mode for testing (small subset)
python -m sim_bench.cli --run-config configs/run.cluster.yaml --quick --quick-size 40

# Disable caching
python -m sim_bench.cli --run-config configs/run.cluster.yaml --no-cache
```

### Output Files

The clustering experiment creates the following outputs:

```
outputs/cluster_runs/experiment_name/YYYY-MM-DD_HH-MM-SS/
├── clusters.csv           # Image paths and cluster assignments
├── cluster_stats.json     # Clustering statistics
└── experiment.log         # Execution log
```

#### clusters.csv Format

```csv
image_path,cluster_id
/path/to/image1.jpg,0
/path/to/image2.jpg,0
/path/to/image3.jpg,1
/path/to/image4.jpg,-1
...
```

- `cluster_id >= 0`: Cluster membership
- `cluster_id = -1`: Noise/outliers (DBSCAN only)

#### cluster_stats.json Format

**DBSCAN:**
```json
{
  "algorithm": "dbscan",
  "n_clusters": 5,
  "n_noise": 12,
  "noise_ratio": 0.15,
  "cluster_sizes": {
    "0": 23,
    "1": 18,
    "2": 15,
    "3": 12,
    "4": 10
  },
  "params": {
    "metric": "cosine",
    "eps": 0.3,
    "min_samples": 4
  }
}
```

**KMeans:**
```json
{
  "algorithm": "kmeans",
  "n_clusters": 10,
  "cluster_sizes": {
    "0": 8,
    "1": 9,
    "2": 7,
    ...
  },
  "inertia": 45.67,
  "params": {
    "n_clusters": 10,
    "n_init": 10,
    "random_state": 42
  }
}
```

## Parameter Tuning Guidelines

### DBSCAN

**eps (epsilon)**: Maximum distance for neighborhood
- Start with 0.3-0.5 for cosine distance on normalized features
- Smaller eps → more, tighter clusters + more noise
- Larger eps → fewer, looser clusters + less noise
- Use k-distance plot to find optimal value

**min_samples**: Minimum samples for core point
- Rule of thumb: `2 * dimensionality` or higher
- For sparse data: use smaller values (3-5)
- For dense data: use larger values (10-20)

**metric**: Distance function
- `cosine`: Best for high-dimensional embeddings (DINOv2, OpenCLIP)
- `euclidean`: Best for normalized features or traditional descriptors

### KMeans

**n_clusters**: Number of clusters
- Use domain knowledge if available
- Try elbow method or silhouette analysis
- For exploration: sqrt(n_samples / 2)

**n_init**: Number of random initializations
- Default: 10 (good balance)
- Increase for better convergence (slower)
- Decrease for speed (less stable)

## Examples

### Test with Quick Mode

```bash
# Test DBSCAN with 40 images
python -m sim_bench.cli --run-config configs/run.cluster.yaml --quick --quick-size 40

# Test KMeans with 40 images
python -m sim_bench.cli --run-config configs/run.cluster_kmeans.yaml --quick --quick-size 40
```

### Real Experiments

```bash
# Cluster full UKBench with SIFT BoVW
python -m sim_bench.cli --run-config configs/run.cluster.yaml

# Cluster with DINOv2 (requires PyTorch)
# Edit config: method: dinov2
python -m sim_bench.cli --run-config configs/run.cluster.yaml
```

## Validation Results

Tested with UKBench (40 images, 10 groups):

**DBSCAN (eps=0.3, min_samples=4, metric=cosine):**
- Result: 100% noise (too strict parameters)
- Recommendation: Increase eps to 0.5-0.7 or decrease min_samples

**KMeans (n_clusters=10):**
- Result: 10 clusters, 4 images each
- Perfect balance (expected for UKBench structure)
- Cluster assignments align with ground truth groups

## Future Enhancements

1. **HTML Galleries** - Generate visual galleries per cluster
2. **Clustering Metrics** - Silhouette score, Davies-Bouldin index
3. **Ground Truth Comparison** - ARI/NMI if labels available
4. **Hierarchical Clustering** - Dendrogram visualization
5. **Interactive Visualization** - t-SNE/UMAP plots with cluster colors

## Architecture Notes

### Design Principles

1. **Container Pattern** - `ClusterConfig` encapsulates all parameters
2. **Reuse** - Leverages existing feature extraction with caching
3. **Minimal API** - Single `cluster_features()` entry point
4. **Standard Patterns** - Strategy pattern for algorithm selection

### Dependencies

- scikit-learn: Clustering algorithms
- numpy: Array operations
- Existing sim_bench infrastructure: Feature extraction, caching, logging

### Effort

Total implementation: ~1 day
- Clustering module: 0.5 day
- Integration: 0.3 day
- Testing & documentation: 0.2 day

