# Analysis Notebook Guide

## Overview

The `analyze_results.ipynb` notebook provides comprehensive analysis of image similarity benchmark results with a focus on understanding per-query performance, method correlation, and visual inspection of results.

## Features

### 1. Configuration & Setup
- **Configurable Parameters**:
  - `experiment_dir`: Path to the experiment results directory
  - `max_k`: Maximum k for recall@k computation (default: 3)
  - `num_examples_per_category`: Number of example images to show per difficulty category (default: 2)
  - `methods`: List of methods to analyze (None = auto-detect)
  - `image_size`: Size of displayed images
  - `methods_per_row`: Grid layout parameter

### 2. Data Loading
- Automatically loads `rankings.csv` and `manifest.json` from each method directory
- Extracts image paths, group information, and dataset metadata
- Validates data consistency across methods

### 3. Per-Query Recall Analysis
Creates a comprehensive DataFrame with:
- `query_idx`: Query image index
- `recall@k_{method}`: Recall@k for each method (k=1 to max_k)
- `top1_idx_{method}`: Index of the closest retrieved image
- `top1_score_{method}`: Distance/similarity score
- `top1_correct_{method}`: Whether top-1 is correct
- `mean_recall@k`, `min_recall@k`, `max_recall@k`: Aggregate statistics

### 4. Image Difficulty Classification
Classifies each query into three categories:
- **Easy**: All methods succeed (min_recall@1 == 1.0)
- **Hard**: All methods fail (max_recall@1 == 0.0)
- **Mixed**: Some methods succeed, some fail

Provides:
- Distribution visualization
- Statistical summary
- Random sampling of examples from each category

### 5. Method Correlation Analysis
- Computes correlation matrices for recall@k across all methods
- Visualizes as heatmaps
- Helps identify complementary vs. redundant methods

### 6-8. Visual Inspection
For each difficulty category (easy, hard, mixed), displays:

**Layout per example:**
```
Row 1:  [Query Image]  [Ground Truth 1]  [Ground Truth 2]

Row 2+: Method results (max 4 methods per row)
        ┌────────────┬────────────┬────────────┬────────────┐
        │  Method 1  │  Method 2  │  Method 3  │  Method 4  │
        │  Top 1     │  Top 1     │  Top 1     │  Top 1     │
        │  score: X  │  score: Y  │  score: Z  │  score: W  │
        │  [CORRECT/ │  [CORRECT/ │  [CORRECT/ │  [CORRECT/ │
        │   WRONG]   │   WRONG]   │   WRONG]   │   WRONG]   │
        │  Top 2     │  Top 2     │  Top 2     │  Top 2     │
        │  score: X  │  score: Y  │  score: Z  │  score: W  │
        │  [CORRECT/ │  [CORRECT/ │  [CORRECT/ │  [CORRECT/ │
        │   WRONG]   │   WRONG]   │   WRONG]   │   WRONG]   │
        └────────────┴────────────┴────────────┴────────────┘
```

- Green titles/borders indicate correct retrievals
- Red titles/borders indicate incorrect retrievals
- Scores show the distance/similarity metric value

### 9. Summary Statistics
- Per-method statistics (mean, std, min, max) for each recall@k
- Method ranking by mean recall@1
- Exports detailed results to `detailed_analysis.csv`

## Usage

### Basic Usage
1. Open `analyze_results.ipynb` in Jupyter or Cursor
2. Select the Python (sim-bench) kernel
3. Update `experiment_dir` in the first code cell to point to your results
4. Run all cells

### Advanced Usage

**Analyze a specific subset of methods:**
```python
experiment_dir = 'outputs/Holidays_100_grps'
methods = ['deep', 'chi_square']  # Only analyze these two
```

**Change recall@k range:**
```python
max_k = 5  # Compute recall@1 through recall@5
```

**Show more examples:**
```python
num_examples_per_category = 5  # Show 5 easy, 5 hard, 5 mixed
```

**Adjust visualization:**
```python
image_size = (4, 4)  # Larger images
methods_per_row = 3  # Fewer methods per row
```

## Outputs

### In-Notebook
- Statistical summaries
- Correlation heatmaps
- Difficulty distribution charts
- Visual comparisons of query results

### Exported Files
- `{experiment_dir}/detailed_analysis.csv`: Complete per-query DataFrame with all metrics

## Interpretation

### Easy Queries
- All methods successfully retrieve at least one correct image
- Indicates well-separated, distinctive groups
- Use these to understand what works well across all methods

### Hard Queries
- All methods fail to retrieve any correct images in top-1
- May indicate:
  - High intra-group variability
  - Ambiguous group definitions
  - Systematic limitations in current feature extraction
- Candidates for feature engineering or ensemble approaches

### Mixed Queries
- Some methods succeed, others fail
- **Most valuable for analysis**
- Reveals method-specific strengths and weaknesses
- Opportunities for method fusion

### Correlation Analysis
- **High correlation (>0.8)**: Methods capture similar characteristics
  - Redundant for ensemble
  - Good for confidence when they agree
  
- **Low correlation (<0.5)**: Complementary approaches
  - Prime candidates for ensemble
  - Likely capture different visual aspects

- **Medium correlation (0.5-0.8)**: Partially overlapping
  - May benefit from weighted combination

## Troubleshooting

### "Module not found" errors
Ensure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### "File not found" errors
- Check that `experiment_dir` points to a valid experiment directory
- Verify the directory contains method subdirectories with `rankings.csv` and `manifest.json`

### Images not displaying
- Verify image paths in `manifest.json` are correct and accessible
- Check file permissions

### Kernel issues
Create/select the project kernel:
```bash
python -m ipykernel install --user --name=sim-bench
```

## Next Steps

After running the analysis:

1. **Identify patterns** in hard/mixed queries
2. **Examine correlation** to find complementary methods
3. **Use `detailed_analysis.csv`** for custom analysis in other tools
4. **Iterate on features** based on failure patterns
5. **Consider ensemble methods** for mixed-difficulty queries

## See Also

- `docs/PERFORMANCE.md`: Performance optimization and caching
- `docs/LOGGING_AND_SAMPLING.md`: Dataset sampling strategies
- `docs/EXPLORATORY_DATA_ANALYSIS.md`: Full EDA framework

