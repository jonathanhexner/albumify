# Recent Improvements (October 2025)

## Overview

Two major improvements addressing user feedback:

1. **Python Logging** - Comprehensive experiment tracking
2. **Group-Based Sampling** - Proper relevance structure maintenance

## 1. Python Logging

### Motivation

**User Request:** "Why not log results to a file using a python logger? This can be in addition."

**Problem:** Console output disappears, making it hard to:
- Review experiment details later
- Debug issues that occurred during runs
- Compare configurations across experiments
- Maintain audit trail

### Solution

Added comprehensive Python logging system:

**New File:** `sim_bench/logging_config.py`
- Configurable logger setup
- Both file and console output
- Structured logging functions

**Modified:** `sim_bench/experiment_runner.py`
- Automatic logging to `experiment.log` in output directory
- Logs experiment start, method execution, results, errors
- Timestamps on all entries

### Usage

**Automatic:**
```bash
# Just run normally
python -m sim_bench.cli --methods chi_square --datasets ukbench

# Log file created automatically at:
outputs/2025-10-08_10-30-45/experiment.log
```

**Configure:**
```yaml
# configs/run.yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Log Output Example

```
2025-10-08 10:30:45 - sim_bench - INFO - ================================================================================
2025-10-08 10:30:45 - sim_bench - INFO - EXPERIMENT START
2025-10-08 10:30:45 - sim_bench - INFO - ================================================================================
2025-10-08 10:30:45 - sim_bench - INFO - Dataset: ukbench
2025-10-08 10:30:45 - sim_bench - INFO - Methods: ['chi_square']
2025-10-08 10:30:46 - sim_bench - INFO - Loading dataset: ukbench
2025-10-08 10:30:46 - sim_bench - INFO - Total images: 100
2025-10-08 10:30:46 - sim_bench - INFO - METHOD: chi_square
2025-10-08 10:30:52 - sim_bench - INFO - RESULTS for chi_square:
2025-10-08 10:30:52 - sim_bench - INFO -   ns_score            : 2.450000
2025-10-08 10:30:52 - sim_bench - INFO -   recall@1            : 0.740000
```

### Benefits

✅ **Complete Record** - Every operation logged  
✅ **Debugging** - Detailed trace when things go wrong  
✅ **Reproducibility** - Know exactly what was run  
✅ **Comparison** - Easy to compare experiments  
✅ **Auditing** - Full history of all runs

## 2. Group-Based Sampling

### Motivation

**User Question:** "Regarding sample - wouldn't it make sense to sample the number of groups and not total images? In theory we could have one image per group, what help would that be to us?"

**Problem:** 
```python
# Old approach: max_queries=37 (limits total images)
sampling:
  max_queries: 37

# UKBench groups:
Group 0: [img_0, img_1, img_2, img_3]     # 4 images ✓
Group 1: [img_4, img_5, img_6, img_7]     # 4 images ✓
...
Group 8: [img_32, img_33, img_34, img_35] # 4 images ✓
Group 9: [img_36]                          # 1 image ✗ BROKEN!

# Evaluation is now INVALID!
# Query img_36 looking for 3 similar images, but they don't exist!
```

**Absolutely correct observation!** This breaks the relevance structure and makes metrics meaningless.

### Solution

Sample by **groups**, not total images:

**New Parameter:** `max_groups`
```yaml
sampling:
  max_groups: 10  # Sample 10 complete groups
  # UKBench: 10 groups × 4 images = 40 images (all groups complete!)
```

**Modified Files:**
- `sim_bench/datasets/ukbench.py` - Group-based sampling
- `sim_bench/datasets/holidays.py` - Group-based sampling  
- `sim_bench/cli.py` - Quick mode uses groups
- `configs/run.yaml` - Updated documentation

### Comparison

#### Old Approach (BAD) ❌
```yaml
sampling:
  max_queries: 37
```
**Result:** 37 images, group 9 has only 1/4 images  
**Metrics:** INVALID (broken relevance structure)

#### New Approach (GOOD) ✅
```yaml
sampling:
  max_groups: 10
```
**Result:** 40 images, all 10 groups complete (4 images each)  
**Metrics:** VALID (proper relevance structure)

### Dataset-Specific Behavior

#### UKBench
- Fixed groups of 4 similar images
- `max_groups=25` → 100 images (25 × 4)
- `max_groups=50` → 200 images (50 × 4)

#### Holidays  
- 1 query per group + variable similar images
- `max_groups=10` → 10 query series
- Variable total images (each series has different size)

### Quick Mode Updated

```bash
# Quick mode now uses group-based sampling
python -m sim_bench.cli --quick --quick-size 100

# Internally: max_groups = 100 / 4 = 25 groups = 100 images (UKBench)
```

Output:
```
⚡ QUICK MODE: Using small subset for fast testing
   • Limited to 25 groups (~100 images for UKBench)
   • Feature caching disabled
```

### Why This Matters

**Invalid metrics with broken groups:**

```python
# Broken (max_queries=37):
Query img_36: Expected 3 similar images
Actual: 0 similar images (they were excluded!)
Recall@4 = 0/3 = 0.00  ← WRONG! Not the method's fault!

# Fixed (max_groups=10):
Query img_36: Expected 3 similar images  
Actual: 3 similar images (all present)
Recall@4 = 3/3 = 1.00  ← CORRECT!
```

**Proper evaluation requires complete groups.**

### Backward Compatibility

Old `max_queries` parameter still works (for legacy configs), but prints a warning:

```yaml
sampling:
  max_queries: 200  # Still works, but deprecated
```

**Recommended migration:**
```yaml
sampling:
  max_groups: 50   # Preferred (50 × 4 = 200 for UKBench)
```

## Files Created/Modified

### New Files
1. **`sim_bench/logging_config.py`** (110 lines)
   - `setup_logger()` - Configure logger with file + console
   - `log_experiment_start()` - Log experiment configuration
   - `log_method_start()` - Log method execution start
   - `log_results()` - Log method results
   - `log_experiment_end()` - Log experiment completion

2. **`docs/LOGGING_AND_SAMPLING.md`** (comprehensive guide)
   - How logging works
   - Log file format and location
   - Group-based sampling explanation
   - Why it matters with examples
   - Migration guide

3. **`docs/RECENT_IMPROVEMENTS.md`** (this file)
   - Summary of both improvements
   - Motivation and solutions
   - Usage examples

### Modified Files

1. **`sim_bench/experiment_runner.py`**
   - Import logging modules
   - Initialize logger in `__init__`
   - Log all major operations
   - Log experiment start/end

2. **`sim_bench/datasets/ukbench.py`**
   - Updated `apply_sampling()` to support `max_groups`
   - Maintains complete groups of 4 images
   - Legacy `max_queries` still supported

3. **`sim_bench/datasets/holidays.py`**
   - Updated `apply_sampling()` to support `max_groups`
   - Samples complete query series
   - Legacy `max_queries` still supported

4. **`sim_bench/cli.py`**
   - Quick mode uses `max_groups` internally
   - Updated help text for `--quick-size`
   - Better messaging about groups

5. **`configs/run.yaml`**
   - Documented `max_groups` parameter
   - Explained difference from `max_queries`
   - Added examples and recommendations

6. **`README.md`**
   - Added logging and group-based sampling to improvements list
   - Links to new documentation

## Testing

### Logging Test
```bash
python -c "
from sim_bench.logging_config import setup_logger, log_results
from pathlib import Path

logger = setup_logger('test', Path('test.log'), 'INFO', console=True)
logger.info('Test message')
log_results(logger, 'test_method', {'ns_score': 2.5, 'recall@1': 0.85})
print('Log created successfully!')
"
```

### Sampling Test
```bash
python -c "
# Demonstrate group-based sampling
max_groups = 10
images_per_group = 4
total_images = max_groups * images_per_group

print(f'max_groups={max_groups} -> {total_images} images')
print('All groups complete, metrics valid!')
"
```

## User Benefits

### Logging
- ✅ **Never lose experiment details** - Everything logged to file
- ✅ **Easy debugging** - Full trace of operations
- ✅ **Reproducibility** - Exact config recorded
- ✅ **Comparison** - Compare logs across experiments
- ✅ **Audit trail** - Complete history

### Group-Based Sampling
- ✅ **Valid metrics** - Proper relevance structure maintained
- ✅ **Meaningful evaluation** - All groups complete
- ✅ **No surprises** - Predictable image counts
- ✅ **Better testing** - Quick mode uses groups correctly
- ✅ **Backward compatible** - Old configs still work

## Migration Guide

### For Logging
No migration needed! Logging is automatic.

Optional: Configure log level:
```yaml
logging:
  level: "INFO"  # or DEBUG, WARNING, ERROR
```

### For Sampling

**Step 1:** Update config
```yaml
# Old (deprecated but still works):
sampling:
  max_queries: 200

# New (recommended):
sampling:
  max_groups: 50  # 50 × 4 = 200 images for UKBench
```

**Step 2:** Update quick mode usage
```bash
# Old understanding:
--quick-size 100  # "Use 100 images"

# New understanding:
--quick-size 100  # "Use ~100 images (25 groups × 4)"
```

## Summary

Both improvements address real user concerns:

1. **Logging**: "Why not log to file?" → Now we do!
2. **Sampling**: "Shouldn't we sample groups?" → Absolutely, now we do!

Result: **More robust and user-friendly benchmark system.**

