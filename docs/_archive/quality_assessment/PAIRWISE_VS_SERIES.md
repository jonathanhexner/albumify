# Pairwise Classification vs Series Selection

This document explains the two different evaluation approaches for quality assessment methods on the PhotoTriage dataset.

## Background

PhotoTriage contains user preference data where people chose between pairs of photos from the same series. The dataset provides:
- Pairwise comparisons (Image A vs Image B, which was chosen)
- User reasons for their choices
- Multiple pairs from each photo series

## Two Evaluation Approaches

### 1. Series Selection (Previous Approach - INCORRECT for PhotoTriage)

**Task**: Given all images in a series, rank them and predict which is the best.

**Evaluation**:
- Top-1 Accuracy: Is the highest-scoring image the best?
- Top-3 Accuracy: Is the best image in the top 3?
- Ranking metrics: Spearman correlation, Kendall's tau

**Problem**: PhotoTriage doesn't provide ground truth rankings for series! It only provides pairwise preferences. We **assumed** the first image in each series was the best (line 86 in `evaluator.py`), which is incorrect.

**Code**:
- `sim_bench/quality_assessment/evaluator.py` (QualityEvaluator)
- `sim_bench/quality_assessment/benchmark.py` (QualityBenchmark)
- `run_quality_benchmark.py`

### 2. Pairwise Classification (New Approach - CORRECT for PhotoTriage)

**Task**: Given two images, predict which one was preferred by users.

**Evaluation**:
- Pairwise Accuracy: P(predict correct winner)
- Per-attribute accuracy: How well does the method predict winners for specific attributes?
- Preference strength analysis: Accuracy on strong vs weak preferences

**Why this is correct**: PhotoTriage provides actual pairwise preference labels from users. We can directly evaluate if a quality method predicts the same winner.

**Code**:
- `sim_bench/quality_assessment/pairwise_evaluator.py` (PairwiseEvaluator)
- `sim_bench/quality_assessment/pairwise_benchmark.py` (PairwiseBenchmark)
- `run_pairwise_benchmark.py`

## Key Differences

| Aspect | Series Selection | Pairwise Classification |
|--------|-----------------|------------------------|
| Input | All images in series | Two images (A, B) |
| Output | Ranking of all images | Binary choice (A or B) |
| Ground Truth | **None** (assumed first is best) | User preference label |
| Metric | Top-K accuracy | Binary accuracy |
| PhotoTriage Fit | **Poor** (wrong task) | **Perfect** (native task) |

## Example

Consider a series with 4 images: [img1, img2, img3, img4]

### Series Selection Approach (Wrong)
```python
# Score all images
scores = [0.7, 0.9, 0.5, 0.8]  # img2 has highest score

# Ground truth (WRONG - we just assumed first is best!)
ground_truth_best = img1

# Check if correct
is_correct = (argmax(scores) == 0)  # False
top1_accuracy = 0.0
```

### Pairwise Classification Approach (Correct)
```python
# PhotoTriage provides actual pairs with user choices:
pairs = [
    (img1, img2, chosen=img2),  # User preferred img2
    (img1, img3, chosen=img1),  # User preferred img1
    (img2, img4, chosen=img2),  # User preferred img2
]

# For each pair, predict winner
for img_a, img_b, chosen in pairs:
    score_a = assess(img_a)  # 0.7
    score_b = assess(img_b)  # 0.9
    predicted = img_b if score_b > score_a else img_a
    correct = (predicted == chosen)  # True in this case

# Pairwise accuracy = % correct predictions
```

## Running the Experiments

### Pairwise Classification (Recommended)

```bash
# Quick test with rule-based methods
python run_pairwise_benchmark.py --config configs/pairwise_benchmark.quick_test.yaml

# Full benchmark with all methods
python run_pairwise_benchmark.py --config configs/pairwise_benchmark.phototriage.yaml
```

### Series Selection (Legacy, for comparison only)

```bash
# Original (incorrect) benchmark
python run_quality_benchmark.py --config configs/quality_benchmark.phototriage.yaml
```

## Results Interpretation

### Pairwise Results (Meaningful)
- **65% pairwise accuracy** = Method agrees with users 65% of the time
- **High strong-preference accuracy** = Method is good at obvious cases
- **Low weak-preference accuracy** = Method struggles with subtle differences

### Series Results (Not Meaningful for PhotoTriage)
- **64% top-1 accuracy** = Method's top choice matches assumed best 64% of time
  - ⚠️ BUT we don't know the true best, so this is meaningless!

## Recommendations

1. **Use pairwise evaluation** for PhotoTriage dataset
2. **Keep series selection code** only for datasets with true rankings (e.g., AVA, KONIQ)
3. **Report pairwise accuracy** as the primary metric for PhotoTriage experiments

## Files Created

### Pairwise Evaluation
- `sim_bench/quality_assessment/pairwise_evaluator.py` - Core evaluator
- `sim_bench/quality_assessment/pairwise_benchmark.py` - Benchmark runner
- `run_pairwise_benchmark.py` - Main script
- `configs/pairwise_benchmark.phototriage.yaml` - Full config
- `configs/pairwise_benchmark.quick_test.yaml` - Quick test config

### Series Selection (Legacy)
- `sim_bench/quality_assessment/evaluator.py` - Core evaluator
- `sim_bench/quality_assessment/benchmark.py` - Benchmark runner
- `run_quality_benchmark.py` - Main script
- `configs/quality_benchmark.*.yaml` - Various configs

## References

- PhotoTriage dataset: Provides pairwise comparisons
- Bradley-Terry model: Used to generate preference strengths from pairwise data
- Attribute extraction: Maps user reasons to quality attributes
