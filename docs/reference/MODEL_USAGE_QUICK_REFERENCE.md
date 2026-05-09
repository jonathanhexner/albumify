# Model Usage Quick Reference

## Which Models Does the Album App Actually Use?

### Currently Active Models

| Model | Status | Checkpoint Required? | Where Loaded | Purpose |
|-------|--------|---------------------|--------------|---------|
| **Rule-Based IQA** | ✅ ACTIVE | No (built-in) | `quality_assessment/rule_based.py` | Technical quality (sharpness, exposure) |
| **MediaPipe** | ✅ ACTIVE | No (auto-download) | `portrait_analysis/analyzer.py` | Face detection, eyes, smile |
| **DINOv2** | ✅ ACTIVE | No (auto-download) | `feature_extraction/dinov2.py` | Image embeddings for clustering |
| **HDBSCAN** | ✅ ACTIVE | No (algorithm) | `clustering/hdbscan.py` | Clustering similar images |
| **AVA ResNet** | ⚠️ OPTIONAL | **YES - YOUR TRAINED MODEL** | `image_quality_models/ava_model_wrapper.py` | Aesthetic scoring (1-10) |
| **Siamese CNN** | ❌ NOT USED | Yes - your trained model | `image_quality_models/siamese_model_wrapper.py` | Image comparison (deferred) |

### What This Means

**The app works WITHOUT your trained models**, but uses only rule-based quality assessment.

**To get aesthetic scores (AVA)**, you need to:
1. Locate your trained AVA checkpoint (`best_model.pt`)
2. Configure the path in `configs/global_config.yaml`

---

## Model Files You Trained

Based on your training scripts, you likely have:

### 1. AVA Model (`train_ava_resnet.py`)

**What it does**: Predicts aesthetic score 1-10

**Checkpoint structure**:
```python
{
    'model_state_dict': {...},
    'config': {
        'model': {
            'cnn_backbone': 'resnet50',
            'output_mode': 'distribution'  # or 'regression'
        },
        'transform': {...}
    },
    'epoch': 25,
    'val_spearman': 0.742
}
```

**Where it should be**: Anywhere, but configure path in config

**Example paths to check**:
```
outputs/ava_training_*/best_model.pt
checkpoints/ava/best_model.pt
models/ava_resnet_best.pt
```

### 2. Siamese Model (`train_siamese_e2e.py`)

**What it does**: Compares two images, says which is better

**Currently**: NOT used in album app (future feature for tiebreakers)

---

## How to Configure Your AVA Model

### Step 1: Find Your Checkpoint

```bash
# Search for AVA checkpoints
find . -name "best_model.pt" -o -name "*ava*.pt"

# Or check common locations
ls outputs/ava_training_*/
ls checkpoints/
```

### Step 2: Add to Config

Edit `configs/global_config.yaml`:

```yaml
# Add this section if it doesn't exist:
quality_assessment:
  default_method: clip_aesthetic
  enable_cache: true
  batch_size: 16
  ava_checkpoint: D:\path\to\your\best_model.pt  # ADD THIS LINE
```

### Step 3: Verify Loading

Run app and check logs at `logs/sim-bench.log`:

```
# Should see:
INFO - Loaded AVA model from epoch 25, val_spearman=0.742, mode=distribution

# If missing this line, checkpoint not configured or path wrong
```

### Step 4: Test

Run workflow and check results:
- **Metrics tab**: Should show `ava_score` column
- **Values**: Should be 1-10 range
- **Performance tab**: Should show "AVA Aesthetics" timing

---

## Model Loading Flow

### Without AVA (Current Default)

```
Image → IQA (rule-based) → quality score
     ↘ MediaPipe        → face/eyes/smile
     ↘ DINOv2           → embeddings
       ↓
   Composite score = IQA only (no aesthetics)
```

### With AVA Configured

```
Image → IQA (rule-based) → quality score (0-1)
     ↘ AVA (YOUR MODEL)  → aesthetic score (1-10)
     ↘ MediaPipe        → face/eyes/smile
     ↘ DINOv2           → embeddings
       ↓
   Composite score = 0.5*AVA + 0.2*IQA + 0.3*Portrait
```

---

## Troubleshooting

### "No AVA scores in results"

**Cause**: Checkpoint not configured

**Fix**:
1. Check `configs/global_config.yaml` has `ava_checkpoint` line
2. Verify file exists at that path
3. Check logs for loading errors

### "RuntimeError: Error loading checkpoint"

**Cause**: Incompatible checkpoint format

**Fix**:
- Verify checkpoint was saved with correct format
- Check training script used `torch.save()` with required keys
- Try loading manually:
  ```python
  import torch
  ckpt = torch.load('best_model.pt')
  print(ckpt.keys())  # Should have: model_state_dict, config, epoch
  ```

### "Model output wrong shape"

**Cause**: Config mismatch (distribution vs regression mode)

**Fix**:
- Check checkpoint config: `checkpoint['config']['model']['output_mode']`
- Should match what model expects

---

## Model Checkpoints Directory Structure

### Recommended Organization

```
sim-bench/
├── checkpoints/           # Store all your trained models here
│   ├── ava/
│   │   ├── best_model.pt           # Your best AVA model
│   │   ├── epoch_25.pt             # Specific epoch
│   │   └── config.yaml             # Training config
│   │
│   ├── siamese/
│   │   └── best_model.pt           # Siamese model (future use)
│   │
│   └── README.md                   # What each checkpoint is
│
└── configs/
    └── global_config.yaml          # Points to: ../checkpoints/ava/best_model.pt
```

---

## Quick Test

### Test if AVA is working:

```python
from pathlib import Path
from sim_bench.config import get_global_config
from sim_bench.model_hub import ModelHub

# Load config
config = get_global_config().to_dict()

# Create hub
hub = ModelHub(config)

# Test image
test_img = Path("test_image.jpg")
ava_score = hub.score_aesthetics(test_img)

if ava_score:
    print(f"✅ AVA working: score = {ava_score:.2f}")
else:
    print("❌ AVA not configured - returns None")
```

---

## Summary

1. **App works without your models** (uses rule-based quality + MediaPipe)
2. **To get aesthetic scores**: Configure AVA checkpoint path
3. **Siamese model**: Not used yet (future feature)
4. **All other models**: Auto-download or built-in

**Next action**: Locate your `best_model.pt` from AVA training and configure the path!
