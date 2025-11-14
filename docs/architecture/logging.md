# Logging and Group-Based Sampling

This document explains two important improvements to sim-bench:
1. **Python logging** for detailed experiment tracking
2. **Group-based sampling** for proper relevance structure

## Python Logging

### Overview

All experiments now automatically log to `experiment.log` in the output directory, in addition to console output.

### Log File Location

```
outputs/
  2025-10-08_10-30-45/
    experiment.log          ← Complete experiment log
    chi_square/
      manifest.json
      metrics.csv
      ...
    emd/
      ...
```

### What Gets Logged

**Experiment Start:**
```
2025-10-08 10:30:45 - sim_bench - INFO - ================================================================================
2025-10-08 10:30:45 - sim_bench - INFO - EXPERIMENT START
2025-10-08 10:30:45 - sim_bench - INFO - ================================================================================
2025-10-08 10:30:45 - sim_bench - INFO - Dataset: ukbench
2025-10-08 10:30:45 - sim_bench - INFO - Methods: ['chi_square', 'emd']
2025-10-08 10:30:45 - sim_bench - INFO - Metrics: ['ns', 'recall@1', 'recall@4', 'map@10']
2025-10-08 10:30:45 - sim_bench - INFO - Sampling: {'max_groups': 25}
2025-10-08 10:30:45 - sim_bench - INFO - Caching: True
```

**Dataset Loading:**
```
2025-10-08 10:30:45 - sim_bench - INFO - Loading dataset: ukbench
2025-10-08 10:30:46 - sim_bench - INFO - Total images: 100
2025-10-08 10:30:46 - sim_bench - INFO - Query images: 100
```

**Method Execution:**
```
2025-10-08 10:30:46 - sim_bench - INFO - --------------------------------------------------------------------------------
2025-10-08 10:30:46 - sim_bench - INFO - METHOD: chi_square
2025-10-08 10:30:46 - sim_bench - INFO - Config: {'method': 'chi_square', 'features': {'bins': [16, 16, 16]}, ...}
```

**Results:**
```
2025-10-08 10:30:52 - sim_bench - INFO - RESULTS for chi_square:
2025-10-08 10:30:52 - sim_bench - INFO -   ns_score            : 2.450000
2025-10-08 10:30:52 - sim_bench - INFO -   recall@1            : 0.740000
2025-10-08 10:30:52 - sim_bench - INFO -   recall@4            : 0.920000
2025-10-08 10:30:52 - sim_bench - INFO -   map@10              : 0.782300
2025-10-08 10:30:52 - sim_bench - INFO -   num_queries         : 100
2025-10-08 10:30:52 - sim_bench - INFO -   num_images          : 100
```

### Benefits of Logging

1. **Complete Record**: Full trace of experiment execution
2. **Debugging**: Detailed info when things go wrong
3. **Reproducibility**: Know exactly what config was used
4. **Comparison**: Easy to compare different runs
5. **Auditing**: Track all experiments over time

### Configuration

Control logging level in `configs/run.yaml`:

```yaml
logging:
  level: "INFO"    # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Log Levels:**
- `DEBUG`: Everything (very verbose)
- `INFO`: Normal operation (recommended)
- `WARNING`: Only warnings and errors
- `ERROR`: Only errors

## Group-Based Sampling

### The Problem with Image-Based Sampling

**Old Approach (BAD):**
```yaml
sampling:
  max_queries: 37  # Limit to 37 images
```

**What happens:**
```
UKBench has groups of 4 similar images:
Group 0: [img_0, img_1, img_2, img_3]     ✓ Complete
Group 1: [img_4, img_5, img_6, img_7]     ✓ Complete
...
Group 8: [img_32, img_33, img_34, img_35] ✓ Complete
Group 9: [img_36]                          ✗ BROKEN! Only 1/4 images
```

**Result:** 
- Group 9 has only 1 image instead of 4
- Metrics are meaningless (no relevant images to find!)
- Evaluation is invalid

### The Solution: Group-Based Sampling

**New Approach (GOOD):**
```yaml
sampling:
  max_groups: 10  # Limit to 10 complete groups
```

**What happens:**
```
UKBench:
Group 0: [img_0, img_1, img_2, img_3]     ✓ Complete (4 images)
Group 1: [img_4, img_5, img_6, img_7]     ✓ Complete (4 images)
...
Group 9: [img_36, img_37, img_38, img_39] ✓ Complete (4 images)

Total: 10 groups × 4 images = 40 images
```

**Result:**
- ALL groups are complete
- Relevance structure is maintained
- Metrics are valid and meaningful

### Dataset-Specific Behavior

#### UKBench
- **Group Structure:** Fixed groups of 4 similar images
- **max_groups=25** → 100 images (25 × 4)
- **max_groups=50** → 200 images (50 × 4)

#### Holidays
- **Group Structure:** 1 query + variable number of similar images
- **max_groups=10** → 10 query series (varying total images)
- Each group can have different sizes (1 query + 1-10 similar images)

### Configuration

```yaml
# configs/run.yaml
sampling:
  max_groups: 25      # Approach 1: Sample by complete groups (recommended!)
  max_queries: null   # Approach 2: Limit total images (also supported)
  random_seed: 42

# If both are specified, max_groups takes precedence
```

**Both parameters are fully supported!**
- `max_groups`: Recommended for valid metrics (maintains complete groups)
- `max_queries`: Useful when you need exact image count control

### Quick Mode

Quick mode automatically uses group-based sampling:

```bash
# Quick test with ~100 images (25 groups × 4)
python -m sim_bench.cli --quick --quick-size 100

# Internally converts to:
# max_groups = 100 // 4 = 25 groups = 100 images (UKBench)
```

### Comparison

| Approach | Command | UKBench Result | Metrics Valid? | Use Case |
|----------|---------|----------------|----------------|----------|
| `max_groups: 10` | Recommended | 40 images, all groups complete | ✅ Yes | Valid evaluation |
| `max_queries: 37` | Also supported | 37 images, group 9 incomplete | ⚠️ Partial | Exact count needed |
| `--quick --quick-size 100` | Quick mode | 100 images (25 groups) | ✅ Yes | Fast testing |

**Both approaches are supported!** Choose based on your needs:
- Need valid metrics? → Use `max_groups`
- Need exact image count? → Use `max_queries` (but be aware of potential incomplete groups)

### Usage Guide

**Both parameters are fully supported:**

```yaml
# Option 1: Sample by groups (recommended for valid metrics)
sampling:
  max_groups: 50      # 50 × 4 = 200 images for UKBench

# Option 2: Limit total images (useful for exact count)
sampling:
  max_queries: 200    # Exactly 200 images

# If both specified, max_groups takes precedence
```

**Conversion between approaches:**
- UKBench: `max_groups = max_queries / 4`
- Holidays: `max_groups = max_queries` (1 query = 1 group)

**When to use which:**
- **Use `max_groups`** when you need valid evaluation metrics
- **Use `max_queries`** when you need exact control over image count

### Why This Matters

**Invalid sampling breaks metrics:**

```python
# With max_queries=37 (broken groups):
Group 9 has only 1 image
Query img_36: Looking for 3 similar images, but they don't exist!
Recall@4 = 0/3 = 0.00  ← Wrong! Not the method's fault!

# With max_groups=10 (complete groups):
Group 9 has all 4 images
Query img_36: Looking for 3 similar images, all 3 present
Recall@4 = 3/3 = 1.00  ← Correct!
```

**Valid metrics require complete groups!**

## Best Practices

### For Development

```bash
# Quick test: 25 groups (~100 images)
python -m sim_bench.cli --quick --quick-size 100
```

### For Validation

```yaml
# configs/run.yaml
sampling:
  max_groups: 50  # 200 images (UKBench), good validation set
```

### For Full Benchmark

```yaml
# configs/run.yaml
sampling:
  max_groups: null  # Use all data (10,200 images for UKBench)
```

### Checking Logs

```bash
# After running experiment
cat outputs/2025-10-08_10-30-45/experiment.log

# Or view last experiment
ls -t outputs/ | head -1  # Get latest directory
cat outputs/$(ls -t outputs/ | head -1)/experiment.log
```

## Summary

### Logging ✅
- **Automatic** experiment logging to file
- **Complete** record of all operations
- **Timestamped** entries for debugging
- **Configurable** log levels

### Group-Based Sampling ✅
- **Maintains** relevance structure
- **Ensures** valid metrics
- **Prevents** broken groups
- **Backward compatible** (legacy `max_queries` still works)

Both improvements make sim-bench more robust and user-friendly!

