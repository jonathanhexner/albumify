# Bradley-Terry Model for Pairwise Evaluation

## Overview

The Bradley-Terry (BT) model is a probabilistic framework for analyzing pairwise comparison data. Unlike binary accuracy (correct/incorrect), BT provides:

1. **Probabilistic predictions**: How confident is the model in each comparison?
2. **Log-likelihood metric**: How well do predicted scores explain observed preferences?
3. **Calibration assessment**: Are predicted probabilities well-calibrated?

## Why Bradley-Terry?

### Binary Accuracy Limitations

Binary accuracy treats all errors equally:
- Predicting 0.51 vs 0.49 when user chose first image: **Correct** ✓
- Predicting 0.99 vs 0.01 when user chose second image: **Wrong** ✗

But the second prediction is *much worse* - the model was very confident but completely wrong!

### Bradley-Terry Advantages

BT considers confidence through log-likelihood:
- **Rewards confident correct predictions**: High probability to actual winner
- **Severely penalizes confident mistakes**: High probability to actual loser
- **Robust to close calls**: Small penalty when scores are similar

## The Bradley-Terry Model

### Core Formula

Given quality scores q_i and q_j for images i and j:

```
P(i > j) = exp(q_i) / (exp(q_i) + exp(q_j))
         = 1 / (1 + exp(q_j - q_i))
         = sigmoid(q_i - q_j)
```

### Log-Likelihood

For a set of pairwise comparisons, the log-likelihood is:

```
LL = Σ log P(winner > loser)
```

**Higher log-likelihood = better model fit**

### Example Comparison

Consider two methods predicting on 3 pairs:

| Pair | Winner | Method A Scores | Method B Scores | A Prediction | B Prediction |
|------|--------|-----------------|-----------------|--------------|--------------|
| 1    | Image A | (0.6, 0.4) | (0.95, 0.05) | Correct | Correct |
| 2    | Image B | (0.45, 0.55) | (0.1, 0.9) | Correct | Correct |
| 3    | Image A | (0.4, 0.6) | (0.05, 0.95) | **Wrong** | **Wrong** |

**Binary Accuracy:**
- Method A: 2/3 = 67%
- Method B: 2/3 = 67%
- **TIE**

**Bradley-Terry Log-Likelihood:**
- Method A: More modest predictions, smaller penalty on error
- Method B: Very confident predictions, **huge penalty** on error
- **Method A wins** (higher LL)

Method B is overconfident and would be misleading in practice!

## Implementation

### File Structure

```
sim_bench/quality_assessment/
├── bradley_terry.py           # BT model implementation
├── pairwise_evaluator.py      # Integrated BT metrics
└── pairwise_benchmark.py      # BT columns in output
```

### Core Components

#### 1. BradleyTerryModel Class

```python
from sim_bench.quality_assessment.bradley_terry import BradleyTerryModel

# Fit model from pairwise comparison data
bt_model = BradleyTerryModel(regularization=0.01)
quality_params = bt_model.fit(pairs)

# Predict probability
prob_a_wins = bt_model.predict_proba(image_a_id, image_b_id)

# Evaluate method's scores
metrics = bt_model.compute_metrics(pairs, quality_scores)
```

#### 2. Automatic Evaluation

BT metrics are automatically computed in `PairwiseEvaluator`:

```python
from sim_bench.quality_assessment import PairwiseEvaluator

evaluator = PairwiseEvaluator(pairs_file, method)
results = evaluator.evaluate()

# Results include BT metrics
bt_ll = results['global']['bt_log_likelihood']
bt_avg_ll = results['global']['bt_avg_log_likelihood']
bt_cal_error = results['global']['bt_calibration_error']
```

### Output Files

BT metrics appear in all benchmark output files:

#### `overall_comparison.csv`

```csv
Method,Pairwise Accuracy,BT Log-Likelihood,BT Avg LL,BT Calibration Error,...
CLIP_Sharpness,0.6234,-3456.2,-0.2863,0.0421,...
RuleBased_Sharpness,0.6189,-3489.7,-0.2891,0.0512,...
```

**Methods ranked by BT Log-Likelihood** (higher is better)

#### `{method}/results.json`

```json
{
  "global": {
    "accuracy": 0.6234,
    "bt_log_likelihood": -3456.2,
    "bt_avg_log_likelihood": -0.2863,
    "bt_calibration_error": 0.0421
  },
  "bradley_terry": {
    "log_likelihood": -3456.2,
    "avg_log_likelihood": -0.2863,
    "accuracy": 0.6234,
    "calibration_error": 0.0421,
    "num_pairs": 12073
  }
}
```

## Metrics Explained

### 1. Log-Likelihood (LL)

**What**: Total log probability of observed outcomes given predicted scores

**Range**: (-∞, 0]
- 0 = perfect predictions (impossible in practice)
- More negative = worse fit

**Interpretation**:
- Compare methods: **Higher LL is better**
- Absolute value less important than relative comparison

### 2. Average Log-Likelihood (Avg LL)

**What**: LL divided by number of pairs

**Range**: (-∞, 0]
- Typically: -0.5 to -1.0 for reasonable methods
- Below -2.0 = very poor fit

**Interpretation**:
- Normalized metric for comparing across different dataset sizes
- **Higher Avg LL is better**

### 3. Calibration Error

**What**: Expected Calibration Error (ECE) - measures if predicted probabilities match empirical frequencies

**Range**: [0, 1]
- 0 = perfect calibration
- Higher = worse calibration

**Interpretation**:
- **Lower is better**
- Good calibration: if predict 70% confidence, should be right ~70% of time
- Poor calibration: systematic over/under confidence

**Example**:
- Method predicts 80% probability 100 times
- Actually correct 60 times (60% empirical)
- Calibration error contribution: |0.80 - 0.60| = 0.20

## Usage Examples

### Standalone BT Evaluation

```python
from sim_bench.quality_assessment.bradley_terry import evaluate_method_with_bt

# Evaluate method results
bt_metrics = evaluate_method_with_bt(per_pair_results, regularization=0.01)

print(f"Log-Likelihood: {bt_metrics['log_likelihood']:.2f}")
print(f"Avg LL: {bt_metrics['avg_log_likelihood']:.4f}")
print(f"Calibration Error: {bt_metrics['calibration_error']:.4f}")
```

### Fitting BT Parameters

```python
from sim_bench.quality_assessment.bradley_terry import fit_bradley_terry_from_results

# Fit BT model to learn "true" quality parameters
bt_model, quality_params = fit_bradley_terry_from_results(per_pair_results)

# Quality parameters for each image
for image_id, quality in quality_params.items():
    print(f"{image_id}: {quality:.3f}")
```

### Running Benchmark with BT

```bash
# BT metrics automatically included
python run_pairwise_benchmark.py --config configs/pairwise_benchmark.phototriage_final.yaml
```

Check `overall_comparison.csv` - methods ranked by BT Log-Likelihood!

## Regularization

The `regularization` parameter prevents overfitting during BT parameter estimation:

```python
bt_model = BradleyTerryModel(regularization=0.01)  # L2 penalty
```

**Default: 0.01** (works well for most datasets)

- Higher (0.1): More regularization, smoother parameters
- Lower (0.001): Less regularization, may overfit
- Zero: No regularization (not recommended)

## Interpreting Results

### Good vs Bad Methods

| Method | Accuracy | BT Avg LL | Calibration | Assessment |
|--------|----------|-----------|-------------|------------|
| A      | 0.65     | -0.45     | 0.03        | **Excellent**: High accuracy, good calibration |
| B      | 0.65     | -0.65     | 0.15        | Overconfident: Same accuracy, worse LL |
| C      | 0.58     | -0.50     | 0.04        | Well-calibrated: Lower accuracy but honest |

**Prefer Method A**: Best combination of accuracy and calibration

### When BT Disagrees with Accuracy

If Method X has higher accuracy but lower LL than Method Y:
- X makes confident mistakes (overconfident)
- Y is more modest/uncertain when unsure
- **Y is probably better** for real applications

Calibration matters for:
- Confidence-weighted decisions
- Active learning (query uncertain cases)
- Cascaded systems (pass uncertain to human)

## Reference

Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324-345.

## Summary

✅ **Use Bradley-Terry when**:
- Evaluating pairwise comparison models
- Confidence/probability estimates matter
- Want robust metric beyond binary accuracy

✅ **Key benefits**:
- Rewards well-calibrated predictions
- Penalizes overconfidence
- Provides probabilistic interpretations

✅ **Main metrics**:
- **Log-Likelihood**: Higher is better (less negative)
- **Avg Log-Likelihood**: Normalized LL for fair comparison
- **Calibration Error**: Lower is better (closer to 0)

The pairwise benchmark automatically computes all BT metrics - just check the output files!
