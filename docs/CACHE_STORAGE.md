# Feature Cache Storage Documentation

## Overview

The feature cache is a persistent storage system that saves extracted image features to disk, eliminating the need to recompute them on subsequent runs. This provides 10-300x speedups for repeated experiments.

## Storage Architecture

### Is it a Database?

**Sort of, but simpler!** The cache uses **pickle files as a key-value store**:
- **Key**: Hash of (method + config + images)
- **Value**: Pickle file containing features + metadata
- **No SQL**: Direct file I/O (faster for our use case)
- **No Schema**: Python object serialization

Think of it as a **file-based dictionary** where each pickle file is one cache entry.

### Why Not a Real Database?

We chose pickle files over databases (SQLite, Redis, etc.) because:

| Aspect | Pickle Files | Database |
|--------|-------------|----------|
| **Setup** | Zero (built-in) | Requires installation/server |
| **Speed** | Very fast for large arrays | Overhead for large blobs |
| **Simplicity** | One file per cache | Schema, queries, connections |
| **Portability** | Copy files = copy cache | Export/import needed |
| **NumPy arrays** | Native support | Requires serialization |

For our use case (large NumPy arrays, simple lookup), pickle files are ideal.

## Storage Location

```
project_root/
  artifacts/
    feature_cache/              ← Cache root directory
      chi_square_a1b2c3d4.pkl   ← HSV histogram features
      emd_a1b2c3d4e5f6.pkl      ← Same features, different method name
      resnet50_7g8h9i0j.pkl     ← ResNet50 CNN features
      sift_bovw_3m4n5o6p.pkl    ← SIFT BoVW features
```

**Default**: `artifacts/feature_cache/`  
**Configurable**: Yes (in code, not yet in config file)

### Why `artifacts/` and not `outputs/`?

| Directory | Purpose | Lifetime |
|-----------|---------|----------|
| `artifacts/` | Build artifacts, caches, models | **Persistent** across experiments |
| `outputs/` | Experiment results (timestamped) | **Per-experiment**, can be deleted |

Features are **reusable artifacts**, not experiment-specific outputs.

## File Naming Convention

### Format
```
{method_name}_{hash}.pkl
```

### Components

1. **method_name**: Feature extraction method
   - Examples: `chi_square`, `resnet50`, `sift_bovw`
   - Helps humans identify cache files

2. **hash**: 16-character SHA256 hash
   - Computed from: method config + image paths
   - Ensures uniqueness and cache invalidation
   - Example: `a1b2c3d4e5f6g7h8`

### Example
```python
# Method: chi_square
# Config: {'features': {'bins': [16, 16, 16]}}
# Images: [img_0.jpg, img_1.jpg, ..., img_99.jpg]
# 
# Filename: chi_square_94d4d603fde0a080.pkl
```

## Cache File Contents

Each `.pkl` file is a Python dictionary containing:

```python
{
    'image_paths': [
        '/path/to/img_0.jpg',
        '/path/to/img_1.jpg',
        # ... all image paths
    ],
    'features': np.ndarray,  # Shape: [n_images, feature_dim]
    'method_name': 'chi_square',
    'method_config': {
        'method': 'chi_square',
        'features': {'bins': [16, 16, 16]},
        'distance': 'chi_square',
        # ... full config
    }
}
```

### Feature Matrix Details

```python
# Example: HSV histograms from 100 images
features = {
    'dtype': np.float32,
    'shape': (100, 4096),  # 100 images, 4096-dim features (16×16×16)
    'memory': ~1.6 MB
}

# Example: ResNet50 from 10,000 images
features = {
    'dtype': np.float32,
    'shape': (10000, 2048),  # 10k images, 2048-dim features
    'memory': ~82 MB
}
```

## Cache Key Generation

### Hash Computation

The cache key is computed to ensure uniqueness and automatic invalidation.

**IMPORTANT: Full paths are used**, so same filenames in different directories will have different cache keys:

```python
def _compute_cache_key(method_name, method_config, image_paths):
    # 1. Extract relevant config (feature-affecting params only)
    relevant_config = {
        'method': method_config.get('method'),
        'features': method_config.get('features'),      # Bins, etc.
        'backbone': method_config.get('backbone'),      # ResNet50, etc.
        'normalize': method_config.get('normalize'),
        'vocab_size': method_config.get('vocab_size'),  # For BoVW
        # Distance param NOT included (doesn't affect features)
    }
    
    # 2. Create deterministic string
    config_str = json.dumps(relevant_config, sort_keys=True)
    paths_str = '|'.join(sorted(image_paths))  # Sorted for consistency
    combined = f"{method_name}||{config_str}||{paths_str}"
    
    # 3. Hash it
    cache_key = hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    return f"{method_name}_{cache_key}"
```

### What Triggers Cache Invalidation?

Cache is **automatically invalidated** when:

1. **Method name changes**: `chi_square` → `resnet50`
2. **Feature config changes**: `bins: [8,8,8]` → `bins: [16,16,16]`
3. **Image paths change**: Different dataset or sampling
4. **Image order changes**: Due to sorted paths, order matters

Cache is **NOT invalidated** when:
1. **Distance changes**: `chi_square` → `cosine` (features are the same!)
2. **Metrics change**: Different evaluation metrics
3. **Output settings change**: CSV options, etc.

## Storage Format: Pickle Protocol

### What is Pickle?

Pickle is Python's native serialization format:
- **Binary format**: Compact and fast
- **Python-specific**: Not cross-language compatible
- **Arbitrary objects**: Can store any Python object
- **NumPy support**: Efficient array serialization

### Protocol Version

We use `pickle.HIGHEST_PROTOCOL` (currently protocol 5 in Python 3.8+):
- **Faster** than older protocols
- **Smaller files** for large arrays
- **Better compression** for NumPy arrays

### File Size Examples

| Method | Images | Feature Dim | Uncompressed | Pickled | Ratio |
|--------|--------|-------------|--------------|---------|-------|
| HSV | 100 | 4,096 | 1.6 MB | 1.6 MB | 1.0x |
| HSV | 10,000 | 4,096 | 164 MB | 164 MB | 1.0x |
| ResNet50 | 100 | 2,048 | 0.8 MB | 0.8 MB | 1.0x |
| ResNet50 | 10,000 | 2,048 | 82 MB | 82 MB | 1.0x |

*Note: Pickle doesn't compress (use gzip for compression if needed)*

### No Cache Confusion Between Datasets

**Question**: What if I have 2 datasets with the same filenames?

**Answer**: No problem! The cache uses **full paths**, not just filenames:

```python
# Dataset 1
paths1 = ['D:/dataset1/img_0.jpg', 'D:/dataset1/img_1.jpg']
# Cache: chi_square_cef5f1fe071e1b58.pkl

# Dataset 2 (same filenames, different directory)
paths2 = ['D:/dataset2/img_0.jpg', 'D:/dataset2/img_1.jpg']
# Cache: chi_square_fc4467136f6c4bcc.pkl  ← Different hash!
```

**Result**: Different directories → Different cache keys → No confusion!

The hash includes the complete image path: `'D:/dataset1/img_0.jpg'` vs `'D:/dataset2/img_0.jpg'`

## Cache Operations

### 1. Load (Read)

```python
def load(method_name, method_config, image_paths):
    # 1. Compute cache key
    cache_key = compute_cache_key(...)
    cache_path = cache_root / f"{cache_key}.pkl"
    
    # 2. Check if exists
    if not cache_path.exists():
        return None  # Cache miss
    
    # 3. Load pickle file
    with open(cache_path, 'rb') as f:
        cached_data = pickle.load(f)
    
    # 4. Verify integrity
    if cached_data['image_paths'] != image_paths:
        return None  # Cache invalidated
    
    # 5. Return features
    return cached_data['features']  # Cache hit!
```

**Performance**: ~0.1-1 second for large caches (I/O bound)

### 2. Save (Write)

```python
def save(method_name, method_config, image_paths, features):
    # 1. Prepare data
    cached_data = {
        'image_paths': image_paths,
        'features': features,
        'method_name': method_name,
        'method_config': method_config,
    }
    
    # 2. Write pickle file
    cache_path = cache_root / f"{cache_key}.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

**Performance**: ~0.2-2 seconds for large caches (I/O bound)

### 3. Clear (Delete)

```python
def clear(method_name=None):
    if method_name:
        # Clear specific method
        for cache_file in cache_root.glob(f"{method_name}_*.pkl"):
            cache_file.unlink()
    else:
        # Clear all
        for cache_file in cache_root.glob("*.pkl"):
            cache_file.unlink()
```

## Cache Lifetime & Management

### When to Clear Cache

**Automatically cleared**: Never (manual only)

**Manually clear when**:
1. Code changes affect feature extraction
2. Running out of disk space
3. Want to force recomputation
4. Testing cache functionality

### Cache Size Management

```bash
# Check cache size
du -sh artifacts/feature_cache/

# On Windows
dir artifacts\feature_cache /s
```

### Expected Sizes

| Dataset | Method | Features | Disk Space |
|---------|--------|----------|------------|
| UKBench (10k) | HSV | 4096-dim | ~164 MB |
| UKBench (10k) | ResNet50 | 2048-dim | ~82 MB |
| UKBench (10k) | SIFT BoVW | 512-dim | ~20 MB |
| Holidays (1.5k) | HSV | 4096-dim | ~25 MB |
| Holidays (1.5k) | ResNet50 | 2048-dim | ~12 MB |

**Total for full benchmark**: ~300-500 MB

## Thread Safety

### Current Implementation

**NOT thread-safe** - designed for single-process use:
- No file locking
- Race conditions possible on simultaneous writes
- Concurrent reads are safe (read-only files)

### For Parallel Experiments

If running multiple experiments in parallel:
1. **Different methods**: Safe (different cache files)
2. **Same method, different configs**: Safe (different cache keys)
3. **Identical method+config**: Use separate cache directories

```python
# Process 1
cache1 = FeatureCache(Path("artifacts/cache_proc1/"))

# Process 2  
cache2 = FeatureCache(Path("artifacts/cache_proc2/"))
```

## Portability

### Copying Caches

✅ **Can copy cache files** between machines:
```bash
# On machine A
tar -czf feature_cache.tar.gz artifacts/feature_cache/

# On machine B
tar -xzf feature_cache.tar.gz
```

⚠️ **But image paths must match!**
- Cache includes absolute/relative image paths
- If paths differ, cache will be invalidated
- Use consistent path structures across machines

### Cross-Platform

✅ **Pickle files work** across platforms (Windows/Linux/Mac)  
⚠️ **Path separators** may cause issues (`\` vs `/`)  
💡 **Use `Path` objects** for cross-platform compatibility

## Performance Characteristics

### Cache Hit vs. Miss

| Operation | Cold (miss) | Warm (hit) | Speedup |
|-----------|-------------|------------|---------|
| HSV (10k images) | ~15 sec | ~0.5 sec | **30x** |
| ResNet50 (10k images) | ~180 sec | ~1.0 sec | **180x** |
| SIFT BoVW (10k images) | ~300 sec | ~0.8 sec | **375x** |

### I/O Performance

- **Read**: Limited by disk speed (~500 MB/s for SSD)
- **Write**: Limited by disk speed + pickle serialization
- **SSD recommended** for best cache performance
- **HDD acceptable** (still much faster than recomputing)

## Troubleshooting

### Cache Not Loading

**Symptom**: Features recomputed every time

**Causes**:
1. Image paths changed (different order or locations)
2. Config changed (even slightly)
3. File corruption
4. Permissions issue

**Solutions**:
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check what's happening
logger.debug(f"Cache key: {cache_key}")
logger.debug(f"Cache path: {cache_path}")
logger.debug(f"Exists: {cache_path.exists()}")
```

### Disk Space Issues

```bash
# Find large cache files
find artifacts/feature_cache -type f -size +100M

# Remove old caches
python -c "
from sim_bench.feature_cache import FeatureCache
cache = FeatureCache()
cache.clear()  # Remove all caches
"
```

### Corruption

If cache file is corrupted:
```python
try:
    features = cache.load(...)
except Exception as e:
    print(f"Corruption detected: {e}")
    # Cache will be recomputed automatically
```

## Summary

- **Storage**: Pickle files in `artifacts/feature_cache/`
- **Format**: `{method}_{hash}.pkl` containing features + metadata
- **Key**: SHA256 hash of method + config + images
- **Size**: ~20-200 MB per cache file
- **Lifetime**: Persistent until manually cleared
- **Speedup**: 10-300x for repeated experiments
- **Portability**: Copy-friendly but path-dependent

The cache is **not a database** but a simple, efficient file-based key-value store optimized for NumPy arrays and single-process access.

