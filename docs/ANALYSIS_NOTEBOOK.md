# Analysis Notebook Guide

## Overview

The `analyze_results.ipynb` notebook provides comprehensive analysis tools for sim-bench experiment results.

## Quick Start

### 1. Install Jupyter

```bash
pip install jupyter notebook matplotlib seaborn
```

### 2. Launch Notebook

```bash
jupyter notebook analyze_results.ipynb
```

### 3. Run All Cells

Click "Cell" → "Run All" or press `Shift+Enter` on each cell.

## What the Notebook Does

### Automatic Analysis

The notebook automatically:
1. **Finds the most recent experiment** in `outputs/`
2. **Loads all result files** (metrics.csv, per_query.csv, rankings.csv)
3. **Generates visualizations** comparing methods
4. **Identifies best/worst queries** for debugging
5. **Shows experiment logs** for configuration details

### Sections

| Section | Purpose | Outputs |
|---------|---------|---------|
| **1. Load Results** | Find and select experiments | List of available experiments |
| **2. Summary** | Overall metrics | Dataframe with all method results |
| **3. Compare Methods** | Visual comparison | Bar charts for each metric |
| **4. Per-Query** | Detailed analysis | Distribution histograms |
| **5. Best/Worst** | Find outliers | Top/bottom performing queries |
| **6. Logs** | Understand execution | Experiment log preview |
| **7. Heatmap** | Multi-metric view | Normalized comparison heatmap |
| **8. Custom** | Your analysis | Placeholder for custom code |

## Customization

### Analyze Different Experiment

```python
# In cell 2, change:
selected_exp = Path('outputs/2025-10-08_10-30-45')
```

### Analyze Different Method

```python
# In cell 5, change:
method_to_analyze = 'resnet50'  # Instead of auto-selected
```

### Focus on Specific Metric

```python
# In cell 6, change:
analysis_metric = 'recall@1'  # Or any available metric
```

## Example Outputs

### Summary Comparison

```
📊 Summary Results (3 methods):

   method      ns_score  recall@1  recall@4  map@10
0  chi_square  2.690     0.850     0.920     0.756
1  emd         2.845     0.890     0.950     0.812
2  resnet50    3.120     0.920     0.980     0.865
```

### Visual Comparison

Bar charts comparing each method across all metrics with:
- Clear metric names
- Sorted by performance
- Grid for easy reading

### Per-Query Distribution

Histograms showing:
- Distribution of scores across queries
- Mean score (red line)
- Outliers and patterns

### Method Ranking

```
🏆 Method Ranking:

NS_SCORE:
  resnet50       : 3.1200
  emd            : 2.8450
  chi_square     : 2.6900
```

## Advanced Usage

### Compare Multiple Experiments

```python
# Load results from different runs
exp1 = pd.read_csv('outputs/2025-10-08_10-30-45/summary.csv')
exp2 = pd.read_csv('outputs/2025-10-07_14-22-30/summary.csv')

# Compare configurations
# ... your analysis code
```

### Export Publication Figures

```python
# High-resolution export
plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('method_comparison.pdf', bbox_inches='tight')
```

### Analyze Specific Failure Cases

```python
# Find queries where method performs poorly
poor_queries = per_query_df[per_query_df['ap@10'] < 0.3]

# Inspect their rankings
for query_idx in poor_queries['query_idx']:
    rankings = rankings_df[rankings_df['query_idx'] == query_idx]
    # Analyze why it failed
```

## Tips

### Quick Iteration

1. Run experiment: `python -m sim_bench.cli --quick`
2. Open notebook: `jupyter notebook analyze_results.ipynb`
3. Run all cells
4. Modify method code
5. Repeat

### Debugging

Enable detailed logging before running:
```yaml
# configs/run.yaml
logging:
  detailed: true
```

Then in notebook:
```python
# Load detailed log
with open(selected_exp / 'detailed.log') as f:
    detailed = f.read()
print(detailed)
```

### Performance Comparison

```python
# Compare distance measures on same features
chi_square_df = pd.read_csv('outputs/.../chi_square/metrics.csv')
emd_df = pd.read_csv('outputs/.../emd/metrics.csv')

# Both use HSV histograms, so difference is just the distance measure
print(f"Chi-square: {chi_square_df['ns_score'].values[0]:.4f}")
print(f"EMD:        {emd_df['ns_score'].values[0]:.4f}")
```

## Extending the Notebook

### Add New Visualizations

```python
# Cell: Precision-Recall Curve
if per_query_df is not None:
    plt.figure(figsize=(8, 8))
    # Your P-R curve code
    plt.show()
```

### Add Statistical Tests

```python
# Cell: Statistical Significance
from scipy import stats

method1_scores = per_query_df1['ap@10']
method2_scores = per_query_df2['ap@10']

t_stat, p_value = stats.ttest_rel(method1_scores, method2_scores)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
```

### Load Custom Data

```python
# Cell: Load Feature Cache for Analysis
import pickle

cache_file = Path('artifacts/feature_cache/chi_square_abc123.pkl')
if cache_file.exists():
    with open(cache_file, 'rb') as f:
        cached = pickle.load(f)
    
    features = cached['features']  # np.ndarray
    # Analyze feature distribution, PCA, t-SNE, etc.
```

## Troubleshooting

### "No experiments found"

Run a benchmark first:
```bash
python -m sim_bench.cli --quick --methods chi_square
```

### "No module named 'seaborn'"

Install visualization dependencies:
```bash
pip install matplotlib seaborn jupyter
```

### Cell execution errors

Cells depend on previous cells. Run them in order:
- Click "Kernel" → "Restart & Run All"

## Summary

The analysis notebook provides:
- ✅ Automatic result loading
- ✅ Multi-metric comparisons
- ✅ Statistical analysis
- ✅ Visual comparisons
- ✅ Best/worst query identification
- ✅ Experiment log inspection
- ✅ Extensible for custom analysis

Perfect for:
- Understanding method performance
- Debugging failures
- Comparing configurations
- Publication-ready figures
- Statistical validation

