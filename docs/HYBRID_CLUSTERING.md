# Hybrid HDBSCAN+kNN Face Clustering

## Overview

The Hybrid HDBSCAN+kNN clustering method combines the strengths of density-based clustering (HDBSCAN) with graph-based cluster merging (kNN) to improve face identity recognition.

**Problem it solves**: HDBSCAN can over-segment faces of the same person into multiple clusters, especially when there are variations in pose, lighting, or image quality. The hybrid approach merges these over-segmented clusters using proximity analysis.

## Algorithm

### 4-Stage Process

```
Stage 1: HDBSCAN → Dense Identity Cores
         ↓
Stage 2: Build Cluster-Level kNN Graph
         ↓
Stage 3: Merge Clusters (mutual links + distance checks)
         ↓
Stage 4: Attach Singletons
```

### Stage 1: HDBSCAN Core Detection

Run HDBSCAN to identify dense clusters of faces:
- **Input**: Normalized face embeddings (512-dim ArcFace vectors)
- **Output**: Initial cluster assignments + noise points (label = -1)
- **Parameters**: 
  - `min_cluster_size`: Minimum faces to form a cluster (default: 2)
  - `min_samples`: Core point definition (default: 2)
  - `cluster_selection_epsilon`: Merge threshold within HDBSCAN (default: 0.3)

### Stage 2: Build Cluster-Level kNN Graph

Create a graph connecting similar clusters:
- Compute centroid for each cluster
- For each cluster, find k nearest neighbor clusters
- Store mutual nearest neighbor relationships
- **Parameter**: `knn_k` (default: 5)

### Stage 3: Merge Clusters

Merge clusters that satisfy ALL conditions:
1. **Mutual kNN links**: Clusters must be in each other's k-nearest neighbors
2. **Cross-links**: At least `merge_min_links` face pairs below distance ceiling (default: 2)
3. **Distance ceiling**: All pairwise distances < `merge_distance_ceiling` (default: 0.45)

### Stage 4: Attach Singletons

Process noise points from HDBSCAN:
- Find k nearest cluster centroids for each singleton
- If distance to nearest < `singleton_attach_threshold` (default: 0.38), attach to that cluster
- Otherwise, create a singleton cluster
- **Parameter**: `singleton_knn_k` (default: 3)

## Configuration

### Default Parameters

```yaml
hybrid_knn:
  algorithm: hybrid_hdbscan_knn
  params:
    # HDBSCAN parameters
    min_cluster_size: 2
    min_samples: 2
    cluster_selection_epsilon: 0.3
    
    # kNN graph parameters
    knn_k: 5                      # Neighbors for cluster graph
    merge_min_links: 2            # Min cross-cluster links
    merge_distance_ceiling: 0.45  # Max distance for merge
    
    # Singleton attachment
    singleton_attach_threshold: 0.38
    singleton_knn_k: 3
```

### Parameter Tuning Guidelines

**To reduce clusters (more aggressive merging)**:
- Increase `merge_distance_ceiling` (0.45 → 0.50)
- Decrease `merge_min_links` (2 → 1)
- Increase `singleton_attach_threshold` (0.38 → 0.42)

**To increase precision (less merging)**:
- Decrease `merge_distance_ceiling` (0.45 → 0.40)
- Increase `merge_min_links` (2 → 3)
- Decrease `singleton_attach_threshold` (0.38 → 0.35)

## Usage

### 1. Run Benchmark

Compare HDBSCAN vs Hybrid on your photos:

```bash
python scripts/benchmark_face_clustering.py --album-path D:\Budapest2025_Google
```

This will:
1. Run the full pipeline to extract face embeddings (with filtering)
2. Apply both clustering methods
3. Save face crops for visualization
4. Output results to `results/face_clustering_benchmark/`

### 2. View Results

Launch the Streamlit comparison app:

```bash
streamlit run app/face_clustering_comparison.py
```

Features:
- Side-by-side cluster comparison
- Metrics table (clusters, noise, sizes)
- Face thumbnails in grid layout
- Merge decision details
- Filtering and sorting options

### 3. Integrate into Pipeline (Optional)

To use hybrid clustering in the main pipeline, modify `configs/pipeline.yaml`:

```yaml
cluster_people:
  method: hybrid_knn  # Change from 'hdbscan'
  
  # Hybrid parameters (add these)
  knn_k: 5
  merge_min_links: 2
  merge_distance_ceiling: 0.45
  singleton_attach_threshold: 0.38
  singleton_knn_k: 3
  
  # Keep existing HDBSCAN parameters
  min_cluster_size: 2
  min_samples: 2
  cluster_selection_epsilon: 0.3
```

Then update `sim_bench/pipeline/steps/cluster_people.py` to use `load_clustering_method()` from the factory.

## Output Format

### Benchmark JSON

```json
{
  "timestamp": "2026-02-15T...",
  "album_path": "D:\\Budapest2025_Google",
  "total_faces": 189,
  "face_metadata": [...],
  "methods": {
    "hdbscan": {
      "labels": [0, 0, 1, 1, -1, ...],
      "stats": {
        "n_clusters": 12,
        "n_noise": 15,
        "cluster_sizes": {...}
      }
    },
    "hybrid_knn": {
      "labels": [0, 0, 0, 0, 1, ...],
      "stats": {
        "n_clusters": 8,
        "merges": {
          "n_merges": 4,
          "decisions": [...]
        },
        "singletons": {
          "n_attached": 7,
          "n_singletons": 8
        },
        "cluster_sizes": {...}
      }
    }
  }
}
```

### Directory Structure

```
results/face_clustering_benchmark/
├── benchmark_2026-02-15_14-30-45.json  # Full results
├── latest_summary.json                  # Quick summary
└── face_crops/                          # 112x112 face thumbnails
    ├── face_0000.jpg
    ├── face_0001.jpg
    └── ...
```

## Performance Characteristics

### Expected Improvements

**Compared to HDBSCAN alone:**
- **Fewer clusters**: 20-40% reduction in cluster count
- **Larger clusters**: Better consolidation of same-person faces
- **Fewer singletons**: Noise points attached to nearest identities

### Computational Cost

- **Stage 1 (HDBSCAN)**: O(n log n) - same as baseline
- **Stage 2 (kNN graph)**: O(k²) where k = number of clusters
- **Stage 3 (Merging)**: O(k² × m²) where m = avg cluster size
- **Stage 4 (Singletons)**: O(n_noise × k)

**Total overhead**: ~5-10% slower than HDBSCAN alone for typical albums

### Memory

- Additional storage: O(k²) for cluster graph
- Peak memory: Same as HDBSCAN (dominated by embedding storage)

## Troubleshooting

### Too Many Clusters

**Symptom**: Hybrid produces nearly same number of clusters as HDBSCAN

**Solutions**:
1. Increase `merge_distance_ceiling` to 0.50
2. Check merge decisions in Streamlit app - are merges being rejected?
3. Verify face filtering isn't removing too many faces
4. Lower `merge_min_links` to 1

### Too Few Clusters (Over-Merging)

**Symptom**: Different people grouped together

**Solutions**:
1. Decrease `merge_distance_ceiling` to 0.40
2. Increase `merge_min_links` to 3
3. Check face quality - low-quality faces may have noisy embeddings

### No Merges Happening

**Symptom**: `n_merges: 0` in results

**Causes**:
1. Clusters too far apart (no mutual kNN links)
2. Distance ceiling too strict
3. Insufficient cross-links

**Solutions**:
1. Increase `knn_k` to 7 (consider more neighbors)
2. Relax `merge_distance_ceiling`
3. Check HDBSCAN parameters - may need looser initial clustering

## Comparison with Other Methods

### vs. HDBSCAN Alone

**Advantages**:
- Corrects over-segmentation automatically
- Attaches singletons intelligently
- Preserves HDBSCAN's noise handling

**Disadvantages**:
- Slightly slower (~5-10%)
- More parameters to tune

### vs. Agglomerative Clustering

**Advantages**:
- Better handles noise and outliers
- More robust to density variations
- Automatic cluster number selection

**Disadvantages**:
- More complex algorithm
- Requires tuning multiple parameters

### vs. Identity Refinement Step

The pipeline already has `identity_refinement` which does exemplar-based attachment. Key differences:

**Identity Refinement**:
- Attaches singletons to existing clusters
- Uses exemplar selection (top-k faces per cluster)
- Post-processing step

**Hybrid kNN**:
- Merges entire clusters
- Uses centroid-based kNN graph
- Integrated clustering approach

**Recommendation**: Use both! They solve different problems:
- Hybrid kNN: Fixes HDBSCAN over-segmentation
- Identity Refinement: Fine-tunes cluster membership

## Future Enhancements

Potential improvements:

1. **Adaptive thresholds**: Learn merge parameters from data
2. **Face quality weighting**: Give higher weight to frontal, high-confidence faces
3. **Temporal grouping**: Consider image timestamps for same-event faces
4. **Multi-stage refinement**: Iterative merge-and-refine cycles
5. **Confidence scores**: Provide per-cluster confidence metrics

## References

- **HDBSCAN**: McInnes et al. (2017) - Accelerated Hierarchical Density Based Clustering
- **ArcFace**: Deng et al. (2019) - ArcFace: Additive Angular Margin Loss for Deep Face Recognition
- **Face Clustering**: Otto et al. (2018) - Clustering Faces by Graph Connectivity

## Support

For issues or questions:
1. Check benchmark results in Streamlit app
2. Review merge decisions to understand algorithm behavior
3. Adjust parameters based on visual inspection
4. Compare with HDBSCAN baseline to validate improvements
