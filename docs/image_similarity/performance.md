# Performance Optimization Guide

This document explains the performance optimizations in sim-bench and how to use them effectively.

## Overview of Optimizations

The benchmark system now includes several performance optimizations:

1. **Feature Caching** - Avoid recomputing features
2. **Progress Monitoring** - Real-time feedback on long-running operations
3. **Quick Test Mode** - Fast iteration during development
4. **Chunked Distance Computation** - Memory-efficient processing
5. **Batch Processing** - Optimized feature extraction

## Feature Caching

### What is Cached?

Features are cached **per dataset + method configuration**. The cache key includes:
- Method name
- Feature extraction parameters (bins, backbone, etc.)
- Image paths (sorted for consistency)

### How It Works

```yaml
# In configs/run.yaml
cache_features: true  # Enable caching (default)
```

**First Run:**
```bash
$ python -m sim_bench.cli --methods chi_square --datasets ukbench

[1/4] 🎨 Feature Extraction
------------------------------------------------------------
Extracting HSV histograms ((16, 16, 16) bins) from 10200 images...
HSV histograms: 100%|██████████| 10200/10200 [00:15<00:00]
✓ Cached features to chi_square_a1b2c3d4.pkl
```

**Second Run (with cache):**
```bash
$ python -m sim_bench.cli --methods chi_square --datasets ukbench

[1/4] 🎨 Feature Extraction
------------------------------------------------------------
✓ Loaded cached features from chi_square_a1b2c3d4.pkl
✓ Feature matrix shape: (10200, 4096)
```

### Cache Location

Features are cached in `artifacts/feature_cache/`:
```
artifacts/
  feature_cache/
    chi_square_a1b2c3d4.pkl
    resnet50_e5f6g7h8.pkl
    ...
```

### Disabling Cache

```bash
# Disable for a single run
python -m sim_bench.cli --no-cache

# Or in config
cache_features: false
```

### When to Clear Cache

Clear cache when:
- You modify feature extraction code
- Dataset images change
- You want to force recomputation

```python
from sim_bench.feature_cache import FeatureCache

cache = FeatureCache()
cache.clear('chi_square')  # Clear specific method
cache.clear()              # Clear all caches
```

## Quick Test Mode

Perfect for development and debugging!

### Usage

```bash
# Test with 100 images (default)
python -m sim_bench.cli --quick

# Custom size
python -m sim_bench.cli --quick --quick-size 200

# Works with any method/dataset combination
python -m sim_bench.cli --quick --methods chi_square,resnet50 --datasets ukbench
```

### What Quick Mode Does

1. **Limits dataset size** to specified number of images
2. **Disables caching** (no point caching small subsets)
3. **Maintains same code path** (tests full pipeline)

### Example Output

```bash
$ python -m sim_bench.cli --quick --quick-size 50

⚡ QUICK MODE: Using small subset for fast testing
   • Limited to 50 images
   • Feature caching disabled

============================================================
📊 Loading dataset: ukbench
============================================================
📉 Applying sampling: {'max_queries': 50, 'random_seed': 42}
✓ Total images: 50
✓ Query images: 50

============================================================
🔧 Method: chi_square
============================================================
[1/4] 🎨 Feature Extraction
------------------------------------------------------------
Extracting HSV histograms ((16, 16, 16) bins) from 50 images...
HSV histograms: 100%|██████████| 50/50 [00:01<00:00]
✓ Feature matrix shape: (50, 4096)

[2/4] 📏 Distance Computation
------------------------------------------------------------
Computing 50 x 50 distance matrix...
Chi-square distances: 100%|██████████| 1/1 [00:00<00:00]
✓ Distance matrix computed: (50, 50)

[3/4] 🔢 Ranking Computation
------------------------------------------------------------
✓ Rankings computed for 50 queries

[4/4] 📊 Metric Evaluation
------------------------------------------------------------

============================================================
📈 RESULTS
============================================================
  ns_score            : 2.4200
  recall@1            : 0.7400
  recall@4            : 0.9200
  map@10              : 0.7823
============================================================
```

## Progress Monitoring

All long-running operations now show progress bars:

### Feature Extraction

```
HSV histograms: 100%|████████████| 10200/10200 [00:15<00:00, 682.1 img/s]
resnet50 features: 100%|████████| 10200/10200 [02:45<00:00, 61.5 img/s]
```

### Distance Computation

```
Chi-square distances: 100%|██████| 10/10 [00:02<00:00, 4.2 chunk/s]
Wasserstein distances: 100%|████| 200/200 [05:30<00:00, 1.65s/query]
```

### Status Messages

Each pipeline stage shows clear status:

```
[1/4] 🎨 Feature Extraction
[2/4] 📏 Distance Computation
[3/4] 🔢 Ranking Computation
[4/4] 📊 Metric Evaluation
```

## Sampling Configuration

### Understanding max_queries

⚠️ **Important:** `max_queries` is misleadingly named - it limits the **total number of images**, not just queries!

```yaml
sampling:
  max_queries: 200     # Limits to first 200 images (queries + gallery)
  max_gallery: 2000    # Reserved for future optimization
  random_seed: 42      # For reproducible sampling
```

### Recommended Values

**Development/Testing:**
```yaml
sampling:
  max_queries: 100   # Very fast, ~5-10 seconds
```

**Validation:**
```yaml
sampling:
  max_queries: 500   # Good balance, ~1-2 minutes
```

**Full Benchmark:**
```yaml
sampling:
  max_queries: null  # Use all images
```

## Performance Tips

### 1. Use Caching for Repeated Experiments

When comparing distance measures on the same features:

```bash
# First run extracts features and caches
python -m sim_bench.cli --methods chi_square --datasets ukbench

# Second run with different distance is instant
python -m sim_bench.cli --methods emd --datasets ukbench
```

Both `chi_square` and `emd` use HSV histograms, so the second run reuses cached features!

### 2. Quick Mode for Development

Always test with `--quick` first:

```bash
# Develop your feature extraction
python -m sim_bench.cli --quick --methods my_new_method

# Once it works, run full benchmark
python -m sim_bench.cli --methods my_new_method
```

### 3. Batch Size Tuning (ResNet50)

```yaml
# configs/methods/deep.yaml
method: deep
backbone: resnet50
batch_size: 32      # Increase for faster GPUs
normalize: true
distance: cosine
```

### 4. Distance Measure Performance

**Fastest → Slowest:**
1. Cosine (matrix multiplication)
2. Euclidean (numpy operations)
3. Chi-square (chunked computation)
4. Wasserstein/EMD (O(n²) scipy calls)

For quick tests, prefer cosine or chi-square.

## Computational Complexity

### Feature Extraction

| Method | Complexity | Time (10k images) |
|--------|-----------|-------------------|
| HSV Histogram | O(n × pixels) | ~15 seconds |
| ResNet50 | O(n × forward_pass) | ~3 minutes (CPU) |
| SIFT BoVW | O(n × keypoints) | ~5 minutes |

### Distance Computation

| Distance | Complexity | Time (10k × 10k) |
|----------|-----------|------------------|
| Cosine | O(n² × d) | ~2 seconds |
| Chi-square | O(n² × d) | ~5 seconds |
| Wasserstein | O(n² × d²) | ~30 minutes |

*Where n = number of images, d = feature dimension*

## Memory Considerations

### Full Distance Matrix

For 10,000 images with float32:
- Distance matrix: 10k × 10k × 4 bytes = **400 MB**
- Feature matrix: 10k × 4096 × 4 bytes = **164 MB**

Total RAM needed: ~600 MB + overhead

### For Large Datasets

If you have >20k images:
1. Use sampling for development
2. Consider feature-only evaluation (skip full matrix)
3. Implement query-based evaluation (compute one row at a time)

## Profiling Your Run

To understand where time is spent:

```bash
# Run with time profiling
time python -m sim_bench.cli --methods chi_square --datasets ukbench

# Check cache efficiency
ls -lh artifacts/feature_cache/

# Monitor system resources
# (use htop, Activity Monitor, Task Manager, etc.)
```

## Summary: Recommended Workflow

```bash
# 1. Quick test during development
python -m sim_bench.cli --quick --quick-size 50 --methods my_method

# 2. Validation on subset
python -m sim_bench.cli --methods my_method --datasets ukbench
# (with max_queries: 500 in config)

# 3. Full benchmark (with caching)
python -m sim_bench.cli --methods my_method --datasets ukbench,holidays
# (with max_queries: null in config)

# 4. Compare methods (reuses cached features!)
python -m sim_bench.cli --methods chi_square,emd,resnet50 --datasets ukbench
```

This workflow minimizes computation time while ensuring correctness!

