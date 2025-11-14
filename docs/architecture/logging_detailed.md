# Detailed Logging Guide

## Overview

Sim-bench now provides **two levels of logging**:

1. **Standard Logging** (`experiment.log`) - High-level experiment tracking
2. **Detailed Logging** (`detailed.log`) - Verbose debugging information

## Standard Logging (Always On)

**File**: `outputs/<timestamp>/experiment.log`

**Contains**:
- Experiment configuration
- Dataset loading
- Method execution
- Results
- Errors and warnings

**Example**:
```
2025-10-08 10:30:45 - sim_bench - INFO - EXPERIMENT START
2025-10-08 10:30:45 - sim_bench - INFO - Dataset: ukbench
2025-10-08 10:30:52 - sim_bench - INFO - RESULTS for chi_square:
2025-10-08 10:30:52 - sim_bench - INFO -   ns_score: 2.450000
```

## Detailed Logging (Optional)

**File**: `outputs/<timestamp>/detailed.log`

**Enable in** `configs/run.yaml`:
```yaml
logging:
  level: "INFO"
  detailed: true  # ← Enable detailed logging
```

**Contains**:
- Sampling details (which groups selected, image paths)
- Feature extraction details (per-image, statistics)
- Cache operations (hit/miss, file paths)
- Distance computation (statistics, distribution)
- Ranking details (top-k for sample queries)

## What Gets Logged

### 1. Sampling Details

```
2025-10-08 10:30:46 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:46 - sim_bench.detailed - INFO - SAMPLING DETAILS
2025-10-08 10:30:46 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Sampling config: {'max_groups': 20, 'random_seed': 42}
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Total images after sampling: 80
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Total groups: 20
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Group distribution:
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   Group 0: 4 images
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   Group 1: 4 images
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   ...
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Selected groups: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...]
2025-10-08 10:30:46 - sim_bench.detailed - INFO - Sample image paths (first 5):
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   [0] Group 0: D:/datasets/ukbench/full/ukbench00000.jpg
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   [1] Group 0: D:/datasets/ukbench/full/ukbench00001.jpg
2025-10-08 10:30:46 - sim_bench.detailed - INFO -   [2] Group 0: D:/datasets/ukbench/full/ukbench00002.jpg
```

**Use when**: Debugging sampling issues, verifying group structure

### 2. Feature Extraction Details

```
2025-10-08 10:30:50 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:50 - sim_bench.detailed - INFO - FEATURE EXTRACTION DETAILS: chi_square
2025-10-08 10:30:50 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:50 - sim_bench.detailed - INFO - Number of images: 80
2025-10-08 10:30:50 - sim_bench.detailed - INFO - Feature shape: (80, 4096)
2025-10-08 10:30:50 - sim_bench.detailed - INFO - Feature dtype: float32
2025-10-08 10:30:50 - sim_bench.detailed - INFO - Memory size: 1.25 MB
2025-10-08 10:30:50 - sim_bench.detailed - INFO - Feature statistics:
2025-10-08 10:30:50 - sim_bench.detailed - INFO -   Min: 0.000000
2025-10-08 10:30:50 - sim_bench.detailed - INFO -   Max: 0.125487
2025-10-08 10:30:50 - sim_bench.detailed - INFO -   Mean: 0.000244
2025-10-08 10:30:50 - sim_bench.detailed - INFO -   Std: 0.001432
2025-10-08 10:30:50 - sim_bench.detailed - INFO -   Non-zero ratio: 0.3245
2025-10-08 10:30:50 - sim_bench.detailed - DEBUG - Per-image feature samples (first 3 images):
2025-10-08 10:30:50 - sim_bench.detailed - DEBUG -   Image 0: D:/datasets/ukbench/full/ukbench00000.jpg
2025-10-08 10:30:50 - sim_bench.detailed - DEBUG -     Feature vector shape: (4096,)
2025-10-08 10:30:50 - sim_bench.detailed - DEBUG -     First 10 values: [0.0234 0.0000 0.0156 ...]
```

**Use when**: Debugging feature extraction, verifying feature quality

### 3. Cache Operations

```
2025-10-08 10:30:48 - sim_bench.detailed - INFO - CACHE HIT: chi_square
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Cache file: chi_square_a1b2c3d4e5f6.pkl
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Full path: D:/sim-bench/artifacts/feature_cache/chi_square_a1b2c3d4e5f6.pkl
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Exists: True
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Size: 1.25 MB
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Success: True
```

Or on cache miss:
```
2025-10-08 10:30:48 - sim_bench.detailed - INFO - CACHE MISS: resnet50
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Cache file: resnet50_7g8h9i0j1k2l.pkl
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Full path: D:/sim-bench/artifacts/feature_cache/resnet50_7g8h9i0j1k2l.pkl
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Exists: False
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Success: False
2025-10-08 10:30:48 - sim_bench.detailed - INFO -   Details: Features not in cache, extracting...
```

**Use when**: Debugging caching, understanding which caches are used

### 4. Distance Computation Details

```
2025-10-08 10:30:51 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:51 - sim_bench.detailed - INFO - DISTANCE COMPUTATION DETAILS: chi_square
2025-10-08 10:30:51 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:51 - sim_bench.detailed - INFO - Distance matrix shape: (80, 80)
2025-10-08 10:30:51 - sim_bench.detailed - INFO - Distance matrix dtype: float32
2025-10-08 10:30:51 - sim_bench.detailed - INFO - Memory size: 0.02 MB
2025-10-08 10:30:51 - sim_bench.detailed - INFO - Distance statistics (excluding self-distances):
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   Min: 0.125643
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   Max: 2.456789
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   Mean: 1.234567
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   Median: 1.189432
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   Std: 0.345678
2025-10-08 10:30:51 - sim_bench.detailed - INFO - Distance percentiles:
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   10%: 0.789012
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   25%: 0.912345
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   50%: 1.189432
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   75%: 1.456789
2025-10-08 10:30:51 - sim_bench.detailed - INFO -   90%: 1.789012
```

**Use when**: Analyzing distance distributions, debugging distance measures

### 5. Ranking Details

```
2025-10-08 10:30:52 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:52 - sim_bench.detailed - INFO - RANKING DETAILS (Top-10)
2025-10-08 10:30:52 - sim_bench.detailed - INFO - ================================================================================
2025-10-08 10:30:52 - sim_bench.detailed - INFO - Number of queries: 80
2025-10-08 10:30:52 - sim_bench.detailed - INFO - Number of candidates: 80
2025-10-08 10:30:52 - sim_bench.detailed - INFO - Sample rankings (first 3 queries):
2025-10-08 10:30:52 - sim_bench.detailed - INFO -   Query 0 (Group 0):
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Rank 1: Image 0 (Group 0) [RELEVANT]
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Rank 2: Image 1 (Group 0) [RELEVANT]
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Rank 3: Image 2 (Group 0) [RELEVANT]
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Rank 4: Image 3 (Group 0) [RELEVANT]
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Rank 5: Image 12 (Group 3)
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     ...
2025-10-08 10:30:52 - sim_bench.detailed - INFO -     Relevant in top-10: 4
```

**Use when**: Debugging metric calculations, understanding ranking quality

## Configuration

### Enable Detailed Logging

```yaml
# configs/run.yaml
logging:
  level: "INFO"      # Standard log level
  detailed: true     # Enable detailed.log
```

### Disable (Default)

```yaml
logging:
  level: "INFO"
  detailed: false    # No detailed.log (saves disk space)
```

## File Locations

After running with detailed logging enabled:

```
outputs/
  2025-10-08_10-30-45/
    experiment.log     ← Standard log (always created)
    detailed.log       ← Detailed log (only if enabled)
    chi_square/
      ...
```

## Log Sizes

### Standard Log
- **Typical size**: 10-50 KB per experiment
- **Contains**: ~50-200 lines

### Detailed Log
- **Typical size**: 100-500 KB per experiment (with debug level)
- **Contains**: ~500-5000 lines depending on dataset size
- **Includes**: Per-image details, statistics, samples

## Use Cases

### Development & Debugging

```yaml
logging:
  level: "DEBUG"     # Maximum verbosity
  detailed: true     # All details
```

**When to use**:
- Developing new features
- Debugging errors
- Understanding system behavior

### Production Runs

```yaml
logging:
  level: "INFO"      # Normal verbosity
  detailed: false    # No extra details
```

**When to use**:
- Normal benchmarking
- Published results
- Large-scale experiments

### Performance Analysis

```yaml
logging:
  level: "INFO"
  detailed: true     # Get statistics
```

**When to use**:
- Analyzing feature quality
- Comparing distance measures
- Understanding result patterns

## Performance Impact

### Standard Logging
- **Overhead**: ~0.1% (negligible)
- **Always enabled**: No performance concern

### Detailed Logging
- **Overhead**: ~1-2% (mostly I/O)
- **Recommendation**: Disable for large datasets

## Viewing Logs

### During Experiment
```bash
# Watch standard log in real-time
tail -f outputs/$(ls -t outputs | head -1)/experiment.log

# On Windows
Get-Content outputs\$(Get-ChildItem outputs | Sort-Object LastWriteTime -Descending | Select-Object -First 1)\experiment.log -Wait
```

### After Experiment
```bash
# View standard log
cat outputs/2025-10-08_10-30-45/experiment.log

# View detailed log
cat outputs/2025-10-08_10-30-45/detailed.log

# Search for specific info
grep "CACHE" outputs/2025-10-08_10-30-45/detailed.log
grep "Feature statistics" outputs/2025-10-08_10-30-45/detailed.log
```

## Summary

| Feature | Standard Log | Detailed Log |
|---------|-------------|--------------|
| **Always on** | ✅ Yes | ❌ Optional |
| **File** | `experiment.log` | `detailed.log` |
| **Size** | 10-50 KB | 100-500 KB |
| **Level** | INFO | DEBUG |
| **Contains** | High-level flow | Per-operation details |
| **Console** | No (file only) | No (file only) |
| **Overhead** | ~0.1% | ~1-2% |
| **Use for** | Normal runs | Debugging |

**Recommendation**: Enable detailed logging when developing or debugging, disable for production runs.

