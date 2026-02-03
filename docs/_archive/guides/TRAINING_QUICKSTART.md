# PhotoTriage Training Quick Start

Quick reference for training PhotoTriage quality assessment models.

## Prerequisites

Ensure you have:
- PhotoTriage dataset at: `D:\Similar Images\automatic_triage_photo_series\`
- Python environment with dependencies installed
- ~15-30 GB disk space for embeddings cache

## Training Options

### Option 1: Pairwise Classifier (Original Approach)

Trains on pairs of images, predicts which one wins.

```bash
# Full training with MLP head (256 → 128 → 2)
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
    --batch_size 128 \
    --num_epochs 50 \
    --output_dir outputs/pairwise_mlp

# Linear head only (1024 → 2)
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
    --mlp_hidden_dims \
    --batch_size 128 \
    --num_epochs 50 \
    --output_dir outputs/pairwise_linear
```

**Time:** ~30-60 min for embeddings (one-time), ~1-2 hours for training

**Expected accuracy:** 60-70% on pairwise test set

---

### Option 2: Series Classifier (Recommended) ⭐

Trains on full series, predicts best image from multiple options.

**Step 1: Pre-compute embeddings** (only needed once)

```bash
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py \
    --batch_size 32 \
    --num_epochs 1
# Cancel after "Embeddings saved to cache" message
```

**Step 2: Train series classifier**

```bash
# Independent MLP scorer (recommended starting point)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --embeddings_cache outputs/phototriage_binary/embeddings_cache.pkl \
    --mlp_hidden_dims 256 128 \
    --batch_size 32 \
    --num_epochs 30 \
    --output_dir outputs/series_mlp

# Linear head baseline (fastest)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --embeddings_cache outputs/phototriage_binary/embeddings_cache.pkl \
    --mlp_hidden_dims \
    --batch_size 32 \
    --num_epochs 30 \
    --output_dir outputs/series_linear

# Transformer (images attend to each other - advanced)
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --embeddings_cache outputs/phototriage_binary/embeddings_cache.pkl \
    --use_transformer \
    --batch_size 32 \
    --num_epochs 30 \
    --output_dir outputs/series_transformer
```

**Time:** ~30-60 min for embeddings (one-time), ~1-3 hours for training

**Expected top-1 accuracy:** 60-75% on series test set (vs 38.5% random baseline)

---

## Which Approach to Use?

### Use **Pairwise Classifier** if:
- You want to compare two specific images
- You're building a ranking/comparison tool
- You want to integrate with your pairwise benchmark

### Use **Series Classifier** if: ⭐ RECOMMENDED
- You want to select the best photo from a burst/series
- Your task is "pick the best from N options"
- You want better performance on the actual PhotoTriage task

**Bottom line:** For PhotoTriage, use the **series classifier**. It's designed for the actual task.

---

## Command-Line Arguments

### Common Arguments (Both Scripts)

```
--csv_path PATH           Path to pairs CSV (default: PhotoTriage location)
--images_dir PATH         Path to images directory
--output_dir PATH         Where to save models and logs
--batch_size N           Batch size (default: 32-128)
--num_epochs N           Max training epochs (default: 30-50)
--learning_rate FLOAT    Learning rate (default: 1e-4)
--mlp_hidden_dims N...   Hidden layer sizes, empty [] = linear (default: 256 128)
```

### Series-Specific Arguments

```
--embeddings_cache PATH   Path to pre-computed embeddings
--use_transformer         Use Transformer instead of MLP
```

---

## Understanding the Output

### Pairwise Classifier Output

```
TEST SET RESULTS (Pairwise Comparison)
============================================================
Random Baseline (pairwise): 50.0%
Accuracy: 65.2%
  Improvement over random: 15.2 pp

Class 0 Accuracy: 64.8%
Class 1 Accuracy: 65.6%
Precision: 66.1%
Recall: 65.6%
F1 Score: 65.8%
============================================================
```

**Good performance:** >60% accuracy (>10pp over random)

### Series Classifier Output

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

**Good performance:** >60% top-1 accuracy (>20pp over random)

---

## Troubleshooting

### "Embeddings cache not found"

Run the binary classifier first to create the cache:
```bash
python sim_bench/quality_assessment/trained_models/train_binary_classifier.py --num_epochs 1
```

### "Missing embeddings for N images"

The CSV references images not in the directory. Check:
1. Image directory path is correct
2. Filename normalization (CSV has "1-1.JPG", images are "000001-01.JPG")

### Training is very slow

- Embeddings are pre-computed, so training should be fast
- If first run, embeddings computation takes 30-60 min (one-time)
- Reduce batch size if running out of memory

### Accuracy stuck at ~50% (pairwise) or ~38% (series)

This means the model is performing at random chance:
1. Check CLIP embeddings are normalized (should be fixed now)
2. Try linear head first to verify setup
3. Check data split isn't leaking (we split by series_id, so OK)

---

## Performance Targets

| Method | Pairwise Acc | Series Top-1 | Notes |
|--------|-------------|--------------|-------|
| Random | 50% | 38.5% | Baseline |
| Rule-based (sharpness) | ~56% | ~56% | Current best from benchmarks |
| **Target (linear)** | **60%** | **55-60%** | Good sanity check |
| **Target (MLP)** | **65%** | **60-70%** | Recommended |
| **Target (Transformer)** | - | **65-75%** | Advanced |

---

## Next Steps After Training

1. **Evaluate on your benchmark:**
   ```bash
   python run_pairwise_benchmark.py --config configs/pairwise_benchmark.test.yaml
   ```

2. **Compare to rule-based methods:**
   - Sharpness: 56.4%
   - Combined: 58.4%
   - Your trained model: ???

3. **Integrate into quality assessment:**
   - Register trained model in `quality_assessment/factory.py`
   - Add to benchmark configs
   - Use for actual photo selection

---

## Full Example Workflow

```bash
# 1. Train series classifier with MLP head
python sim_bench/quality_assessment/trained_models/train_series_classifier.py \
    --mlp_hidden_dims 256 128 \
    --batch_size 32 \
    --num_epochs 30 \
    --output_dir outputs/my_series_model

# 2. Check results
cat outputs/my_series_model/test_results.json

# 3. View training curves
# Open outputs/my_series_model/training_curves.png

# 4. Use the trained model
# (Integration code coming soon)
```

---

## More Information

- **Detailed documentation:** [docs/quality_assessment/TRAINING_IMPROVEMENTS.md](docs/quality_assessment/TRAINING_IMPROVEMENTS.md)
- **Architecture details:** `sim_bench/quality_assessment/trained_models/phototriage_series.py`
- **Benchmark results:** `outputs/pairwise_benchmark_test/pairwise_20251125_004908/`
- **ChatGPT recommendations:** See project docs

---

## Summary

**Quick recommendation:** Start with series-MLP, it's the best balance of simplicity and performance.

```bash
# The one command you probably want:
python sim_bench/quality_assessment/trained_models/train_series_classifier.py
```

This uses all the defaults and should give you 60-70% top-1 accuracy, significantly better than the 38.5% random baseline and competitive with rule-based methods (56%).
