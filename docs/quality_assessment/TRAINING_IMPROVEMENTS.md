# PhotoTriage Training Improvements

This document describes the improvements made to the PhotoTriage training pipeline based on best practices for frozen CLIP + learned head architectures.

## Summary of Changes

### 1. CLIP Embedding Normalization ✅

**Problem:** CLIP is trained with L2-normalized embeddings, but our preprocessing wasn't normalizing.

**Fix:** Added L2 normalization in two places:
- `train_binary_classifier.py:268` - During embedding pre-computation
- `phototriage_binary.py:213` - In the model's `encode_image()` method

**Impact:** Ensures embeddings have unit norm, matching CLIP's training setup. This can significantly improve performance.

```python
# Now normalized
embedding = embedding / embedding.norm(dim=-1, keepdim=True)
```

### 2. Linear Head Baseline Support ✅

**Problem:** Using MLP with hidden layers (256→128) might be overkill. If a linear head works well, the bottleneck isn't model capacity.

**Fix:** The config already supports empty `mlp_hidden_dims`, which creates a simple linear classifier.

**Usage:**
```bash
# Linear head only (1024 → 2)
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py --mlp_hidden_dims

# With hidden layers (default)
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py --mlp_hidden_dims 256 128
```

### 3. Random Baseline Reporting ✅

**Problem:** Need to compare model performance to random guessing.

**Fix:** Evaluation now shows:
- Random baseline (50% for pairwise, ~38.5% for series)
- Improvement over random in percentage points

**Output example:**
```
Random Baseline (pairwise): 50.0%
Accuracy: 65.2%
  Improvement over random: 15.2 pp
```

## New: Series-Softmax Training 🎯

**THE BIG CHANGE** - This addresses the core mismatch between training (pairwise) and task (series ranking).

### Why Series-Softmax?

**Current approach (pairwise):**
- Trains on independent pairs: `[img1, img2] → 0 or 1`
- Doesn't match the actual task: "Select best from series"
- Average series has 2.6 images, so pairwise doesn't capture full context

**Series-softmax approach:**
- Trains on full series: `[img1, img2, ..., imgN] → best_idx`
- Uses softmax over all images in series
- Directly optimizes the actual task

### Architecture

Two modes available:

#### Mode 1: Independent MLP (Default)
Each image scored independently, then softmax within series.

```
Series = [img1, img2, img3]
              ↓ (frozen CLIP)
    [emb1, emb2, emb3]
              ↓ (MLP scorer)
    [score1, score2, score3]
              ↓ (softmax)
    [prob1, prob2, prob3]
              ↓ (cross-entropy)
         Loss
```

#### Mode 2: Series Transformer (Advanced)
Images attend to each other before scoring.

```
Series = [img1, img2, img3]
              ↓ (frozen CLIP)
    [emb1, emb2, emb3]
              ↓ (+ positional encoding)
    [emb1', emb2', emb3']
              ↓ (Transformer encoder)
    [context1, context2, context3]
              ↓ (linear scorer)
    [score1, score2, score3]
              ↓ (softmax + cross-entropy)
         Loss
```

### Files Created

1. **`phototriage_series.py`** - Series classifier model and dataset
   - `PhotoTriageSeriesDataset` - Dataset that returns full series
   - `series_collate_fn` - Handles variable-length series with padding
   - `PhotoTriageSeriesClassifier` - Model with two modes (MLP or Transformer)
   - `SeriesClassifierTrainer` - Training loop with series-softmax loss

2. **`train_series_classifier.py`** - Training script
   - Converts pairwise CSV to series-level data
   - Uses pre-computed embeddings (same cache as binary classifier)
   - Evaluates with proper metrics (top-1/2/3 accuracy, MRR)

### Usage

#### Step 1: Pre-compute Embeddings (if not already done)

```bash
# This creates embeddings_cache.pkl
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
    --batch_size 32 \
    --num_epochs 1  # Just to create cache, then cancel
```

#### Step 2: Train Series Classifier

```bash
# Default: Independent MLP (256 → 128 → 1)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --embeddings_cache outputs/phototriage_binary/embeddings_cache.pkl \
    --output_dir outputs/phototriage_series_mlp \
    --batch_size 32 \
    --num_epochs 30

# Linear head only (fastest baseline)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --mlp_hidden_dims \
    --output_dir outputs/phototriage_series_linear

# Transformer (images attend to each other)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --use_transformer \
    --output_dir outputs/phototriage_series_transformer
```

### Evaluation Metrics

The series classifier reports proper metrics for the ranking task:

- **Top-1 Accuracy**: Does argmax match the best image? (main metric)
- **Top-2 Accuracy**: Is best image in top 2?
- **Top-3 Accuracy**: Is best image in top 3?
- **Mean Rank**: Average rank of true best image
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank
- **Random Baseline**: 1/avg_series_length (~38.5% for PhotoTriage)

Example output:
```
TEST SET RESULTS (Series Ranking)
============================================================
Number of series: 500
Average series length: 2.6 images

Random Baseline: 38.5%
Top-1 Accuracy: 68.2%
  Improvement over random: 29.7 pp

Top-2 Accuracy: 85.4%
Top-3 Accuracy: 92.1%
Mean Rank: 1.42
Mean Reciprocal Rank: 0.7812
============================================================
```

## Recommended Training Sequence

Follow these steps in order:

### Phase 1: Quick Wins (2-4 hours)

1. **Test linear head on pairwise task**
   ```bash
   python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
       --mlp_hidden_dims \
       --output_dir outputs/test_linear_pairwise
   ```
   - If this gets ~60%, you know capacity isn't the issue
   - If it's at ~50%, something is wrong with setup

2. **Test linear head on series task**
   ```bash
   python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
       --mlp_hidden_dims \
       --output_dir outputs/test_linear_series
   ```
   - This should beat pairwise approach on series ranking
   - Expected: 50-60% top-1 accuracy

### Phase 2: Main Architecture (4-8 hours)

3. **Train MLP on series**
   ```bash
   python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
       --mlp_hidden_dims 256 128 \
       --output_dir outputs/series_mlp_256_128
   ```
   - Expected: 60-70% top-1 accuracy
   - If not significantly better than linear, try simpler (e.g., just [256])

4. **If still weak, try Transformer**
   ```bash
   python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
       --use_transformer \
       --output_dir outputs/series_transformer
   ```
   - Lets images compare to each other
   - Expected: 65-75% top-1 accuracy

### Phase 3: Advanced (Only If Needed)

5. **Add auxiliary features**
   - Concatenate sharpness/aesthetic/face features to CLIP
   - See ChatGPT recommendations in project docs

6. **Partial CLIP fine-tuning**
   - Unfreeze last CLIP block
   - Use very small LR (1e-6 for CLIP, 1e-3 for head)
   - Only do this if series-Transformer baseline is solid (>65%)

## Expected Performance Targets

Based on ChatGPT recommendations and your current baseline results:

| Approach | Expected Top-1 Accuracy | Notes |
|----------|------------------------|-------|
| Random baseline | ~38.5% | 1 / avg_series_length |
| Rule-based (sharpness) | ~56% | Your current best |
| Frozen CLIP + Linear (pairwise) | ~60% | Good sanity check |
| Frozen CLIP + Linear (series) | 55-60% | Better than pairwise |
| Frozen CLIP + MLP (series) | 60-70% | Recommended starting point |
| Frozen CLIP + Transformer (series) | 65-75% | If MLP not enough |
| Partial CLIP fine-tune | 70-80% | Final optimization |

## Key Insights from ChatGPT Conversation

1. **"Setup issues, not capacity issues"**
   - Missing normalization, wrong loss, wrong metric often explain poor performance
   - Don't jump to "need bigger model" without fixing baseline

2. **"Series-aware is crucial"**
   - Your data is series, train on series
   - Pairwise training doesn't match series ranking task

3. **"Don't immediately fine-tune CLIP"**
   - With 15K images, full fine-tuning will overfit
   - Fix frozen baseline first
   - If needed, partial fine-tuning (last block + LoRA)

4. **"Verify with proper metrics"**
   - Pairwise accuracy ≠ series ranking accuracy
   - Always compare to random baseline
   - Top-1 accuracy on series is the right metric

## Files Modified

1. `train_binary_classifier.py` - Added normalization, random baseline
2. `phototriage_binary.py` - Added normalization in encode_image, clarified linear head

## Files Created

1. `phototriage_series.py` - Series classifier architecture
2. `train_series_classifier.py` - Series training script
3. `docs/quality_assessment/TRAINING_IMPROVEMENTS.md` - This file

## Next Steps

1. Run linear head baseline to verify setup
2. Train series-MLP classifier
3. Compare to rule-based methods (currently at 56%)
4. If series-MLP > 65%, try Transformer
5. If Transformer > 70%, consider partial CLIP fine-tuning

## Questions?

See the original ChatGPT conversation for detailed reasoning and alternative approaches.
