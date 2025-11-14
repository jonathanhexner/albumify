# Quality Assessment Output Format

Complete guide to understanding benchmark output files.

## Output Directory Structure

After running a benchmark, results are saved to:
```
outputs/quality_benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS/
```

## CSV Files (Primary Format)

All results are saved as CSV files for easy analysis in Excel, pandas, etc.

### 1. methods_summary.csv

**Purpose:** Overall performance comparison across all methods

**Columns:**
- `method`: Method name
- `avg_top1_accuracy`: Average Top-1 accuracy across all datasets
- `avg_top2_accuracy`: Average Top-2 accuracy across all datasets
- `avg_mrr`: Average Mean Reciprocal Rank
- `avg_time_ms`: Average processing time per image (milliseconds)
- `datasets_tested`: Number of datasets this method was tested on

**Example:**
```csv
method,avg_top1_accuracy,avg_top2_accuracy,avg_mrr,avg_time_ms,datasets_tested
sharpness_only,0.6495,0.7757,0.7891,60.15,1
contrast_only,0.4837,0.8323,0.7021,45.60,1
composite_balanced,0.4205,0.8481,0.6789,45.87,1
```

**Use case:** Quick comparison of which method performs best overall

### 2. detailed_results.csv

**Purpose:** Per-dataset, per-method detailed results

**Columns:**
- `dataset`: Dataset name (e.g., "phototriage")
- `method`: Method name
- `top1_accuracy`: Top-1 accuracy on this dataset
- `top2_accuracy`: Top-2 accuracy on this dataset
- `mrr`: Mean Reciprocal Rank on this dataset
- `avg_time_ms`: Average time per image (milliseconds)
- `throughput`: Images processed per second

**Example:**
```csv
dataset,method,top1_accuracy,top2_accuracy,mrr,avg_time_ms,throughput
phototriage,sharpness_only,0.6495,0.7757,0.7891,60.15,16.63
phototriage,contrast_only,0.4837,0.8323,0.7021,45.60,21.93
phototriage,composite_balanced,0.4205,0.8481,0.6789,45.87,21.80
```

**Use case:** Compare methods on specific datasets, analyze performance differences

### 3. accuracy_ranking.csv

**Purpose:** Methods ranked by accuracy (best to worst)

**Columns:**
- `method`: Method name
- `accuracy`: Average Top-1 accuracy

**Example:**
```csv
method,accuracy
sharpness_only,0.6495
contrast_only,0.4837
composite_balanced,0.4205
```

**Use case:** Quick answer to "which method is most accurate?"

### 4. speed_ranking.csv

**Purpose:** Methods ranked by speed (fastest to slowest)

**Columns:**
- `method`: Method name
- `time_ms`: Average time per image (milliseconds)

**Example:**
```csv
method,time_ms
exposure_only,42.70
colorfulness_only,43.99
contrast_only,45.60
```

**Use case:** Quick answer to "which method is fastest?"

### 5. efficiency_ranking.csv

**Purpose:** Methods ranked by efficiency (accuracy/time ratio)

**Columns:**
- `method`: Method name
- `accuracy`: Average Top-1 accuracy
- `time_ms`: Average time per image (milliseconds)
- `efficiency`: Accuracy divided by time (higher is better)

**Example:**
```csv
method,accuracy,time_ms,efficiency
sharpness_only,0.6495,60.15,0.0108
contrast_only,0.4837,45.60,0.0106
exposure_only,0.4299,42.70,0.0101
```

**Use case:** Find methods that balance accuracy and speed

### 6. [dataset]_results.csv

**Purpose:** All methods tested on a specific dataset

**Columns:**
- `method`: Method name
- `method_type`: Type of method (rule_based, nima, musiq, etc.)
- `top1_accuracy`: Top-1 accuracy
- `top2_accuracy`: Top-2 accuracy
- `top3_accuracy`: Top-3 accuracy (if available)
- `mrr`: Mean Reciprocal Rank
- `mean_rank`: Average rank of best image
- `num_series`: Number of series evaluated
- `avg_time_ms`: Average time per image
- `throughput`: Images per second
- `status`: "success" or "failed"
- `error`: Error message (if failed)

**Example:**
```csv
method,method_type,top1_accuracy,top2_accuracy,top3_accuracy,mrr,mean_rank,num_series,avg_time_ms,throughput,status
sharpness_only,rule_based,0.6495,0.7757,0.8456,0.7891,1.27,100,60.15,16.63,success
contrast_only,rule_based,0.4837,0.8323,0.9123,0.7021,1.45,100,45.60,21.93,success
```

**Use case:** Complete results for one dataset, see all metrics at once

### 7. [dataset]_[method]_series.csv

**Purpose:** Per-series detailed results for one method (FULL RAW DATA)

**Columns:**
- `group_id`: Series/group identifier
- `num_images`: Number of images in this series
- `predicted_idx`: Index (0-based) of image your algorithm ranked #1
- `ground_truth_idx`: Index (0-based) of human-labeled best image
- `correct`: 1 if predicted_idx == ground_truth_idx, else 0
- `scores`: Comma-separated quality scores for all images in series (in order: img0, img1, img2, ...)
- `ranking`: Comma-separated indices showing full ranking (best to worst). Example: "2,0,1" means img2 is best, img0 is 2nd, img1 is worst
- `image_ranks`: Comma-separated ranks for each image (1-indexed). Example: "2,3,1" means img0 is rank 2, img1 is rank 3, img2 is rank 1
- `image_paths`: Comma-separated paths to all images in series

**Example:**
```csv
group_id,num_images,predicted_idx,ground_truth_idx,correct,scores,ranking,image_ranks,image_paths
000001,4,2,1,0,"0.45,0.78,0.92,0.61","2,1,3,0","3,2,1,4","/path/img0.jpg,/path/img1.jpg,/path/img2.jpg,/path/img3.jpg"
000002,8,3,3,1,"0.52,0.48,0.89,0.67,0.55,0.71,0.38,0.49","3,5,4,1,0,2,6,7","5,4,6,1,3,2,7,8","/path/img0.jpg,..."
000010,2,0,0,1,"0.87,0.79","0,1","1,2","/path/img0.jpg,/path/img1.jpg"
```

**Understanding the columns:**
- `scores`: "0.45,0.78,0.92,0.61" means img0=0.45, img1=0.78, img2=0.92, img3=0.61
- `ranking`: "2,1,3,0" means img2 is best (rank 1), img1 is 2nd (rank 2), img3 is 3rd (rank 3), img0 is worst (rank 4)
- `image_ranks`: "3,2,1,4" means img0 is rank 3, img1 is rank 2, img2 is rank 1, img3 is rank 4

**Use case:** 
- Debug why a method failed on specific series
- Analyze which series sizes are hardest
- Understand score distributions
- Find failure cases

**Note:** All list columns contain comma-separated values. To parse:
```python
import pandas as pd
df = pd.read_csv('phototriage_sharpness_only_series.csv')

# Parse scores
df['scores'] = df['scores'].str.split(',').apply(lambda x: [float(v) for v in x])

# Parse ranking (indices)
df['ranking'] = df['ranking'].str.split(',').apply(lambda x: [int(v) for v in x])

# Parse image_ranks (1-indexed ranks)
df['image_ranks'] = df['image_ranks'].str.split(',').apply(lambda x: [int(v) for v in x])

# Parse image paths
df['image_paths'] = df['image_paths'].str.split(',').apply(lambda x: x)
```

**Why this is useful:**
- Full ranking allows you to compute any metric later (Top-3, Top-5, etc.)
- Image paths let you analyze specific images
- Image ranks let you see where each image was ranked
- All raw data is preserved for post-processing

## Understanding the Metrics

### Top-1 Accuracy
- **Range:** 0.0 to 1.0
- **Meaning:** Percentage of series where your #1 pick matches human's #1 pick
- **Example:** 0.6495 = 64.95% of series, you picked the same best image as humans

### Top-2 Accuracy
- **Range:** 0.0 to 1.0
- **Meaning:** Percentage of series where human's best image is in your top 2
- **Example:** 0.7757 = 77.57% of series, human's pick was in your top 2

### Mean Reciprocal Rank (MRR)
- **Range:** 0.0 to 1.0
- **Meaning:** Average of 1/rank across all series
- **Formula:** For each series, if best image is at rank R, add 1/R. Average all.
- **Example:** 0.7891 means on average, best image is at rank ~1.27

### Mean Rank
- **Range:** 1.0 to N (where N = max series size)
- **Meaning:** Average position of the best image in your rankings
- **Example:** 1.27 means on average, best image is ranked 1.27 (very close to #1)

### Throughput
- **Unit:** Images per second
- **Meaning:** How many images can be processed per second
- **Example:** 16.63 means ~17 images per second

## File Naming Convention

All files follow this pattern:
- `[dataset]_[method]_series.csv` - Per-series results
- `[dataset]_results.csv` - All methods on one dataset
- `methods_summary.csv` - Overall comparison
- `detailed_results.csv` - Per-dataset, per-method
- `accuracy_ranking.csv` - Accuracy ranking
- `speed_ranking.csv` - Speed ranking
- `efficiency_ranking.csv` - Efficiency ranking

## Example Workflow

### 1. Quick Overview
```python
import pandas as pd

# See which method is best
df = pd.read_csv('methods_summary.csv')
print(df.sort_values('avg_top1_accuracy', ascending=False))
```

### 2. Detailed Analysis
```python
# Compare methods on PhotoTriage
df = pd.read_csv('detailed_results.csv')
phototriage = df[df['dataset'] == 'phototriage']
print(phototriage.sort_values('top1_accuracy', ascending=False))
```

### 3. Debug Failures
```python
# Find series where method failed
series_df = pd.read_csv('phototriage_sharpness_only_series.csv')
failures = series_df[series_df['correct'] == 0]
print(f"Failed on {len(failures)} out of {len(series_df)} series")
print(failures[['group_id', 'num_images', 'predicted_idx', 'ground_truth_idx']])
```

### 4. Analyze by Series Size
```python
# See if method works better on larger series
series_df = pd.read_csv('phototriage_sharpness_only_series.csv')
for size in sorted(series_df['num_images'].unique()):
    subset = series_df[series_df['num_images'] == size]
    accuracy = subset['correct'].mean()
    print(f"Series size {size}: {accuracy*100:.1f}% accuracy ({len(subset)} series)")
```

## Logging

All benchmark runs create a log file: `benchmark.log`

**Log file location:**
```
outputs/quality_benchmarks/benchmark_YYYY-MM-DD_HH-MM-SS/benchmark.log
```

**What's logged:**
- Benchmark start/end
- Method evaluation progress
- Cache statistics (how many images were cached)
- Errors with full stack traces
- Timing information

**Example log entries:**
```
2025-01-15 10:30:00 - sim_bench.quality_benchmark - INFO - QUALITY BENCHMARK START
2025-01-15 10:30:01 - sim_bench.quality_benchmark - INFO - Starting evaluation: nima_mobilenet on phototriage
2025-01-15 10:30:05 - sim_bench.quality_benchmark - INFO - Cache statistics: {'cache_size': 150, 'cache_enabled': True}
2025-01-15 10:30:10 - sim_bench.quality_benchmark - INFO - Evaluation complete: nima_mobilenet on phototriage (9.23s)
```

## Caching

Quality scores are automatically cached to avoid re-processing the same image.

**How it works:**
- Each image is processed once per method
- If the same image appears in multiple series, its score is reused from cache
- Cache is per-method (each method has its own cache)
- Cache is cleared when method is recreated

**Cache statistics:**
- Reported in log file
- Included in metrics as `cache_stats`
- Shows `cache_size` (number of cached images) and `cache_enabled` flag

**Benefits:**
- Faster evaluation (especially for large datasets with overlapping series)
- Consistent scores (same image always gets same score)
- Lower computational cost

**Disable caching:**
```python
method = RuleBasedQuality(enable_cache=False)
```

## Related Documentation

- [Understanding Evaluation](evaluation_explained.html) - How metrics are calculated
- [Benchmark Guide](benchmark.md) - How to run benchmarks
- [Quick Start](quickstart.md) - Getting started
