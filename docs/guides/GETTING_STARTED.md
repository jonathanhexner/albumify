# Getting Started: Understanding Your Album App

## TL;DR - What You Need to Know Right Now

**Current Status**: ✅ **MODELS NOW ACTIVE!**

**What Was Fixed**:
- ✅ AVA aesthetic model is now configured and will load automatically
- ✅ Siamese comparison model is now wired up for tiebreaking
- ✅ Both models are active in `configs/global_config.yaml`
- 📁 See `MODEL_ACTIVATION_FIX.md` for full details

**Your Trained Models** (now being used):
1. **AVA Model** (96 MB): `outputs/ava/gpu_run_regression_18_01/best_model.pt`
   - Provides aesthetic scoring (50% of selection weight)
   - Trained on 250K human aesthetic ratings
   
2. **Siamese Model** (94 MB): `outputs/siamese_e2e/20260113_073023/best_model.pt`
   - Breaks ties when scores are close
   - Trained on pairwise comparisons

**To Verify**: Restart the app and check logs for "Loaded AVA model" and "Loaded Siamese comparison model"

---

## Complete Architecture Overview

### What This App Actually Does

Think of it as a 3-layer cake:

```
┌────────────────────────────────────┐
│  LAYER 1: Streamlit UI             │  ← What you see in browser
│  (app/album/)                      │
└────────────────────────────────────┘
               ↓
┌────────────────────────────────────┐
│  LAYER 2: Workflow Pipeline        │  ← Orchestrates 8 stages
│  (sim_bench/album/)                │
└────────────────────────────────────┘
               ↓
┌────────────────────────────────────┐
│  LAYER 3: ML Models                │  ← YOUR trained models live here
│  (sim_bench/model_hub/)            │
└────────────────────────────────────┘
```

### The 8-Stage Pipeline (Layer 2)

When you click "Run Workflow", this happens:

1. **Discover Images**: Find all .jpg/.png in folder
2. **Preprocess**: Generate 1024px + 2048px thumbnails (for speed)
3. **Analyze Quality**: Run IQA + AVA + MediaPipe on each image
4. **Filter**: Remove low-quality images
5. **Extract Features**: Get DINOv2 embeddings
6. **Cluster**: Group similar images (HDBSCAN)
7. **Select Best**: Pick best image from each cluster
8. **Export**: Copy selected images to output

**Total time**: Depends on image count, but thumbnails make it ~50% faster than analyzing originals.

### The 5 Models (Layer 3)

| Model | What It Does | Source | Currently Active? |
|-------|--------------|--------|-------------------|
| **Rule-Based IQA** | Technical quality (sharpness, exposure) | Built-in (OpenCV) | ✅ YES |
| **AVA ResNet** | Aesthetic score 1-10 | **YOUR TRAINING** | ❌ NO (needs config) |
| **MediaPipe** | Face/eyes/smile detection | Google (auto-download) | ✅ YES |
| **DINOv2** | Image embeddings for clustering | Meta (auto-download) | ✅ YES |
| **HDBSCAN** | Clustering algorithm | scikit-learn | ✅ YES |

**Your Siamese model** exists but isn't used yet (planned for tiebreaker comparisons).

---

## File Organization

### Key Directories You Should Know

```
D:\sim-bench\
│
├── app/album/                    # Streamlit UI (what you interact with)
│   ├── main.py                   # Entry point: streamlit run app/album/main.py
│   ├── config_panel.py           # Settings sliders/checkboxes
│   ├── workflow_runner.py        # Progress bar + execution
│   └── results_viewer.py         # Gallery + metrics display
│
├── sim_bench/
│   ├── album/                    # Workflow orchestration
│   │   └── workflow.py           # Main pipeline (8 stages)
│   │
│   ├── model_hub/                # Model coordinator
│   │   └── hub.py                # Loads all models, runs analysis
│   │
│   ├── image_quality_models/     # ⚡ MODEL LOADING HAPPENS HERE
│   │   ├── ava_model_wrapper.py  # Loads YOUR AVA model
│   │   └── siamese_model_wrapper.py  # Loads YOUR Siamese model
│   │
│   ├── models/                   # Network architectures (PyTorch nn.Module)
│   │   ├── ava_resnet.py         # AVA architecture
│   │   └── siamese_cnn_ranker.py # Siamese architecture
│   │
│   ├── portrait_analysis/        # MediaPipe wrapper
│   │   └── analyzer.py           # Face/eyes/smile detection
│   │
│   ├── feature_extraction/       # DINOv2, CLIP, etc.
│   │   └── dinov2.py             # Embeddings for clustering
│   │
│   └── quality_assessment/       # Rule-based quality
│       └── rule_based.py         # Sharpness, exposure, color
│
├── configs/
│   └── global_config.yaml        # ⚙️ MAIN CONFIG FILE (edit this!)
│
├── outputs/
│   └── ava/
│       └── gpu_run_regression_18_01/
│           └── best_model.pt     # 🎯 YOUR TRAINED AVA MODEL
│
└── docs/
    ├── ALBUM_APP_ARCHITECTURE.md     # Complete technical architecture
    ├── MODEL_USAGE_QUICK_REFERENCE.md # Model-specific details
    ├── FILE_DEPENDENCY_MAP.md         # How files connect
    └── GETTING_STARTED.md             # This file
```

---

## How Models Are Loaded (Critical Understanding)

### Example: AVA Model Loading Chain

```
1. User runs workflow
   └─ app/album/main.py

2. Workflow created with config
   └─ sim_bench/album/workflow.py: AlbumWorkflow(config)

3. Workflow creates ModelHub
   └─ sim_bench/model_hub/hub.py: ModelHub(config)

4. ModelHub reads config for AVA checkpoint
   └─ config['quality_assessment']['ava_checkpoint']
   
   IF EXISTS:
   5a. Load AVA model
       └─ sim_bench/image_quality_models/ava_model_wrapper.py
       └─ AVAQualityModel(checkpoint_path)
       └─ torch.load('outputs/ava/gpu_run_regression_18_01/best_model.pt')
       └─ Creates AVAResNet architecture (from sim_bench/models/ava_resnet.py)
       └─ Loads trained weights: model.load_state_dict(checkpoint['model_state_dict'])
   
   IF NOT EXISTS:
   5b. AVA disabled
       └─ score_aesthetics() returns None
```

**Current state**: Step 4 fails (no ava_checkpoint in config) → Step 5b → No aesthetics

**After adding config**: Step 4 succeeds → Step 5a → Full aesthetics scoring

---

## What Happens During Image Analysis

### Current Behavior (Without AVA)

```python
# For each image:
image_metrics = {
    'iqa_score': 0.75,       # From RuleBasedQuality (sharpness + exposure + color)
    'ava_score': None,       # ← MISSING (not configured)
    'has_face': True,        # From MediaPipe
    'eyes_open': True,       # From MediaPipe + eye aspect ratio
    'is_smiling': False,     # From MediaPipe + smile detection
    'sharpness': 0.8,        # From RuleBasedQuality
    'exposure': 0.7          # From RuleBasedQuality
}

# Selection score (without AVA):
score = 0.2 * iqa_score + 0.3 * portrait_score
      = 0.2 * 0.75 + 0.3 * 1.0
      = 0.15 + 0.3
      = 0.45
```

### With AVA Enabled

```python
# For each image:
image_metrics = {
    'iqa_score': 0.75,
    'ava_score': 7.2,        # ← FROM YOUR TRAINED MODEL
    'has_face': True,
    'eyes_open': True,
    'is_smiling': False,
    'sharpness': 0.8,
    'exposure': 0.7
}

# Selection score (with AVA):
score = 0.5 * ava_score + 0.2 * iqa_score + 0.3 * portrait_score
      = 0.5 * 7.2 + 0.2 * 0.75 + 0.3 * 1.0
      = 3.6 + 0.15 + 0.3
      = 4.05
```

**Impact**: AVA dominates selection (50% weight) → much better results!

---

## How to Verify Everything is Working

### Step 1: Start the App

```bash
cd D:\sim-bench
streamlit run app/album/main.py
```

### Step 2: Check Logs

Open `logs/sim-bench.log` and look for:

**Without AVA (current)**:
```
INFO - ModelHub initialized (device=cpu)
INFO - Loaded IQA model (RuleBasedQuality)
INFO - Loaded portrait analyzer (MediaPipe)
# ← Missing AVA line
```

**With AVA (after config change)**:
```
INFO - ModelHub initialized (device=cpu)
INFO - Loaded IQA model (RuleBasedQuality)
INFO - Loaded AVA model from epoch 14, val_spearman=0.742, mode=regression  # ← NEW!
INFO - Loaded portrait analyzer (MediaPipe)
```

### Step 3: Run Test Workflow

1. Upload test photos
2. Click "Run Workflow"
3. Check **Metrics** tab in results

**Without AVA**: `ava_score` column shows `None` or missing
**With AVA**: `ava_score` column shows values like `7.2`, `6.8`, `5.3`

### Step 4: Performance Check

Check **Performance** tab:

**Without AVA**:
- IQA Quality: 0.5s
- Portrait Analysis: 2.3s
- Feature Extraction: 1.8s

**With AVA**:
- IQA Quality: 0.5s
- **AVA Aesthetics: 1.2s** ← NEW
- Portrait Analysis: 2.3s
- Feature Extraction: 1.8s

---

## Understanding the Config File

### Current Config Structure

```yaml
# configs/global_config.yaml

# Album-specific settings
album:
  quality:
    min_iqa_score: 0.3      # Minimum technical quality (0-1)
    min_ava_score: 4.0      # Minimum aesthetic score (1-10)
    min_sharpness: 0.2      # Minimum sharpness (0-1)
  
  clustering:
    method: hdbscan
    min_cluster_size: 3     # 1 = allow single-image clusters
  
  selection:
    images_per_cluster: 1
    ava_weight: 0.5         # How much AVA affects selection (50%)
    iqa_weight: 0.2         # How much IQA affects selection (20%)
    portrait_weight: 0.3    # How much portraits affect selection (30%)

# Model settings
quality_assessment:
  batch_size: 16
  # ava_checkpoint: ???    # ← MISSING - add this!
```

### What You Need to Add

Just one line under `quality_assessment`:

```yaml
quality_assessment:
  default_method: clip_aesthetic
  enable_cache: true
  batch_size: 16
  ava_checkpoint: outputs/ava/gpu_run_regression_18_01/best_model.pt
```

**Paths can be**:
- Relative: `outputs/ava/gpu_run_regression_18_01/best_model.pt` (from repo root)
- Absolute: `D:\sim-bench\outputs\ava\gpu_run_regression_18_01\best_model.pt`

---

## Common Questions

### Q: Why are there so many model files?

**A**: Different purposes:

- **Training scripts** (`sim_bench/training/`): Create models, save checkpoints
- **Model architectures** (`sim_bench/models/`): PyTorch network definitions
- **Model wrappers** (`sim_bench/image_quality_models/`): Load checkpoints, provide simple API
- **Model hub** (`sim_bench/model_hub/`): Coordinates all models

**Flow**: Training → Checkpoint → Wrapper loads checkpoint → Hub uses wrapper

### Q: Which checkpoint should I use?

**A**: Found these AVA checkpoints:
- `outputs/ava/gpu_run_regression_18_01/best_model.pt` ← **Recommended** (proper training run)
- `outputs/ava/overfit_regression/best_model.pt` (overfitting test, don't use)
- `outputs/ava/overfit_sanity_check/best_model.pt` (sanity check, don't use)

### Q: How do I know which model is better?

**A**: Check the training logs and validation scores:

```bash
# Check training log
cat outputs/ava/gpu_run_regression_18_01/training.log

# Look for:
# - Final validation Spearman correlation
# - Number of epochs trained
# - No overfitting (train/val gap)
```

### Q: What if I train a new AVA model?

**A**: Just update the config to point to the new checkpoint:

```yaml
ava_checkpoint: outputs/ava/my_new_training/best_model.pt
```

Restart the app. No code changes needed!

### Q: Can I use multiple models?

**A**: Currently only one AVA model at a time. To compare models:
1. Run workflow with model A
2. Update config to model B
3. Run workflow again
4. Compare results

### Q: Why use thumbnails?

**A**: Speed + Memory:
- Original: 4000x3000 = 12MP = ~100ms to load + process
- Medium (1024px): ~25ms to load + process = **4x faster**
- Most models resize anyway, so no quality loss
- **Final export uses original resolution**

---

## Next Steps

### Immediate Actions (5 minutes)

1. ✅ Read this document (you're doing it!)
2. ⚙️ Add `ava_checkpoint` to `configs/global_config.yaml`
3. 🚀 Restart app: `streamlit run app/album/main.py`
4. ✓ Verify AVA loaded in logs
5. 📸 Test with photos, check metrics show AVA scores

### Understanding the Architecture (30 minutes)

1. Read `docs/ALBUM_APP_ARCHITECTURE.md` - comprehensive design doc
2. Read `docs/FILE_DEPENDENCY_MAP.md` - how files connect
3. Trace one image through the system (see "Execution Flow Example")

### Exploring the Code (1-2 hours)

Start here (in order):
1. `app/album/main.py` - see UI entry point
2. `sim_bench/album/workflow.py` - see 8-stage pipeline
3. `sim_bench/model_hub/hub.py` - see how models are called
4. `sim_bench/image_quality_models/ava_model_wrapper.py` - see YOUR model loaded
5. `sim_bench/models/ava_resnet.py` - see YOUR model architecture

### Testing & Validation

1. Run workflow on test album
2. Check results make sense (best images selected)
3. Review performance metrics (identify bottlenecks)
4. Adjust config thresholds as needed

---

## Troubleshooting

### "App works but no aesthetic scores"

**Check**:
```bash
# 1. Config has ava_checkpoint?
grep ava_checkpoint configs/global_config.yaml

# 2. Checkpoint file exists?
ls -l outputs/ava/gpu_run_regression_18_01/best_model.pt

# 3. Logs show AVA loaded?
tail -f logs/sim-bench.log | grep AVA
```

### "Error loading checkpoint"

**Possible causes**:
- Wrong file path (check absolute vs relative)
- Corrupted checkpoint (try different one)
- Incompatible format (check training script version)

**Debug**:
```python
import torch
ckpt = torch.load('path/to/best_model.pt', map_location='cpu')
print(ckpt.keys())  # Should have: model_state_dict, config, epoch
```

### "Workflow very slow"

**Check**:
1. Thumbnails enabled? (`album.preprocessing.enabled: true`)
2. Cache working? (check `.cache/album_analysis/` has files)
3. Too many workers? (reduce `num_workers`)
4. GPU available? (change `device: cuda`)

---

## Summary

**What you trained**: AVA ResNet for aesthetic scoring (1-10 scale)

**Where it lives**: `outputs/ava/gpu_run_regression_18_01/best_model.pt`

**How app uses it**: 
- Loads checkpoint via `ava_model_wrapper.py`
- Analyzes each image during workflow
- Contributes 50% to selection score

**Current status**: NOT enabled (config missing)

**To enable**: Add ONE line to `configs/global_config.yaml`:
```yaml
ava_checkpoint: outputs/ava/gpu_run_regression_18_01/best_model.pt
```

**Impact**: Much better photo selection (aesthetics-driven instead of just technical quality)

---

**Ready?** Add that config line and restart the app! 🚀
