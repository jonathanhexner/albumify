# Face Clustering Benchmark - Quick Start Guide

## What This Does

Compares two face clustering methods on your photos:
1. **HDBSCAN** (current default)
2. **Hybrid HDBSCAN+kNN** (new, improved merging)

The hybrid method reduces over-segmentation by intelligently merging clusters that belong to the same person.

## Installation

All dependencies are already installed if you have the main sim-bench environment set up.

## Running the Benchmark

### Step 1: Run Benchmark Script

```bash
# Windows
.venv\Scripts\python.exe scripts/benchmark_face_clustering.py --album-path "D:\Budapest2025_Google"

# Linux/Mac
python scripts/benchmark_face_clustering.py --album-path "/path/to/your/photos"
```

**What it does:**
- Discovers images in your album
- Detects and filters faces (using existing pipeline)
- Extracts face embeddings (512-dim ArcFace vectors)
- Runs both clustering methods
- Saves face crops (112x112) for visualization
- Outputs JSON results

**Time estimate:** 2-5 minutes for 100-200 photos (depends on face count)

**Output location:** `results/face_clustering_benchmark/`

### Step 2: View Results in Streamlit

```bash
streamlit run app/face_clustering_comparison.py
```

**Features:**
- **Metrics Table**: Compare cluster counts, noise, sizes
- **Side-by-Side Gallery**: See faces in each cluster
- **Merge Details**: Understand why clusters were merged
- **Filters**: Min cluster size, show/hide singletons
- **Sorting**: By cluster size or ID

## Understanding the Results

### Metrics to Look At

1. **Number of Clusters**
   - HDBSCAN: Usually higher (over-segments)
   - Hybrid: Usually lower (merges related clusters)

2. **Noise/Singletons**
   - HDBSCAN: Faces marked as noise (-1 label)
   - Hybrid: Attempts to attach to nearest cluster

3. **Cluster Sizes**
   - Hybrid typically creates larger, more consolidated clusters
   - Check if multiple small HDBSCAN clusters became one Hybrid cluster

### Visual Inspection

**Look for:**
- ✅ **Good merge**: Multiple HDBSCAN clusters of same person → 1 Hybrid cluster
- ❌ **Bad merge**: Different people in same Hybrid cluster
- ✅ **Good attachment**: Singleton faces correctly attached to their person
- ❌ **Over-merging**: Too few clusters, different people mixed

## Tuning Parameters

If results aren't optimal, adjust parameters in `configs/clustering_benchmark.yaml`:

### Make MORE Clusters (Less Merging)

```yaml
hybrid_knn:
  params:
    merge_distance_ceiling: 0.40     # Lower (was 0.45)
    merge_min_links: 3               # Higher (was 2)
    singleton_attach_threshold: 0.35 # Lower (was 0.38)
```

### Make FEWER Clusters (More Merging)

```yaml
hybrid_knn:
  params:
    merge_distance_ceiling: 0.50     # Higher (was 0.45)
    merge_min_links: 1               # Lower (was 2)
    singleton_attach_threshold: 0.42 # Higher (was 0.38)
```

## Output Files

```
results/face_clustering_benchmark/
├── benchmark_2026-02-15_14-30-45.json  # Full results (open in Streamlit)
├── latest_summary.json                  # Quick metrics
└── face_crops/                          # Face thumbnails
    ├── face_0000.jpg
    ├── face_0001.jpg
    └── ...
```

### JSON Structure

```json
{
  "timestamp": "2026-02-15T...",
  "album_path": "D:\\Budapest2025_Google",
  "total_faces": 189,
  "methods": {
    "hdbscan": {
      "labels": [0, 0, 1, 1, -1, ...],  # Cluster assignments
      "stats": {"n_clusters": 12, "n_noise": 15}
    },
    "hybrid_knn": {
      "labels": [0, 0, 0, 0, 1, ...],
      "stats": {
        "n_clusters": 8,
        "merges": {"n_merges": 4},      # Number of cluster merges
        "singletons": {
          "n_attached": 7,               # Singletons attached
          "n_singletons": 8              # Remaining singletons
        }
      }
    }
  }
}
```

## Troubleshooting

### "No benchmark results found"

**Cause**: Benchmark hasn't been run yet or results directory doesn't exist

**Solution**: Run `python scripts/benchmark_face_clustering.py --album-path "YOUR_PATH"` first

### "Face crops directory not found"

**Cause**: Benchmark ran but `save_face_crops` was disabled

**Solution**: Check `configs/clustering_benchmark.yaml` has `save_face_crops: true`

### "Pipeline failed"

**Common causes:**
1. Album path doesn't exist
2. No images found in album
3. No faces detected

**Solution**: Check the error message and verify album path contains images

### Very Slow Performance

**If benchmark takes >10 minutes:**
1. Check how many images are in the album (use `discover_images` step logs)
2. Consider running on a subset first
3. GPU acceleration: Change `device: cpu` to `device: cuda` in config

## Next Steps

### Use Hybrid in Main Pipeline

Once you're happy with the results, integrate into the main pipeline:

1. Edit `configs/pipeline.yaml`:
   ```yaml
   cluster_people:
     method: hybrid_knn  # Change from 'hdbscan'
     # Add hybrid parameters here
   ```

2. Update `sim_bench/pipeline/steps/cluster_people.py` to use the factory:
   ```python
   from sim_bench.clustering.base import load_clustering_method
   
   # In process() method:
   config_dict = {
       'algorithm': method,
       'params': {
           'min_cluster_size': config.get('min_cluster_size', 2),
           # ... other params
       }
   }
   clustering_method = load_clustering_method(config_dict)
   labels, stats = clustering_method.cluster(embeddings_normalized)
   ```

## Files Created

- `sim_bench/clustering/hybrid_hdbscan_knn.py` - Algorithm implementation
- `scripts/benchmark_face_clustering.py` - Benchmark runner
- `app/face_clustering_comparison.py` - Streamlit visualization app
- `configs/clustering_benchmark.yaml` - Configuration
- `docs/HYBRID_CLUSTERING.md` - Detailed documentation

## Support

For detailed algorithm explanation and advanced usage, see `docs/HYBRID_CLUSTERING.md`.

For parameter tuning guidelines, see the "Parameter Tuning" section in the detailed docs.
