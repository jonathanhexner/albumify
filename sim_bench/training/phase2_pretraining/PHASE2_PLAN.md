# Phase 2: Multitask Face Pretraining Plan

## Overview

Phase 2 implements joint multitask pretraining on AffectNet to learn face-relevant features before fine-tuning on CASIA-WebFace for face recognition. This addresses the training failure mode observed in Phase 1 where ArcFace training on 10k+ classes with naive random sampling leads to degenerate solutions.

## Problem Statement

### Phase 1 Training Failure

**Observed symptoms:**
- Gradient norms show strong initial values, then linear decline
- Oscillations at epoch transitions
- Accuracy stuck at ~1-2% (random chance)
- Model outputs low-variance logits across classes

**Root cause:**
- Naive random sampling: batches contain ~64 different identities with 1 image each
- Model rarely sees same identity twice in a batch
- Each identity seen ~1x per epoch on average
- Degenerate solution: model learns to output uniform logits (low loss, no discrimination)

### Solution: Multitask Pretraining

**Strategy:**
1. Pretrain on easier task (AffectNet: 8 expression classes vs 10k identities)
2. Learn face-relevant features via two complementary tasks:
   - Expression classification (semantic understanding)
   - Landmark regression (geometric understanding)
3. Transfer pretrained backbone to CASIA ArcFace training

**Benefits:**
- Easier optimization (8 classes vs 10k)
- Dense gradients (regression + classification)
- Regularization (multitask learning)
- Face-specific features (not generic ImageNet)

## Architecture

### Model Structure

```
Input Image (224x224x3)
    ↓
ResNet-50 Backbone (pretrained ImageNet)
    ↓
2048-dim Features
    ↓
Projection Layer → 512-dim Embeddings
    ↓
    ├─→ Expression Head → 8-way Classification
    └─→ Landmark Head → 5-10 points (x,y) Regression
```

### Key Components

**1. MultitaskFaceModel** (`multitask_model.py`)
- Shared ResNet backbone (ResNet-50 or ResNet-18)
- Two task-specific heads:
  - `ExpressionHead`: Linear layer → 8 classes
  - `LandmarkHead`: Linear layer → `num_landmarks * 2` outputs
- Optional uncertainty weighting for automatic loss balancing

**2. UncertaintyWeighting** (`multitask_model.py`)
- Learnable log variances for each task
- Automatically balances expression vs landmark losses
- Formula: `loss = precision * task_loss + log_var`
- Prevents one task from dominating

**3. AffectNetDataset** (`affectnet_dataset.py`)
- Loads AffectNet directory structure: `{expression_id}/{image.jpg}`
- Supports pre-extracted landmarks from cache
- Handles missing landmarks gracefully (masks loss)

**4. LandmarkExtractor** (`landmark_extractor.py`)
- Uses MediaPipe FaceMesh to extract key landmarks
- Supports 5-point or 10-point landmark sets
- Caches results for efficient training

## Landmark Selection

### 5-Point Set (Default)

Key facial features for basic geometry:
- Left eye center
- Right eye center
- Nose tip
- Left mouth corner
- Right mouth corner

**Use case:** Faster training, sufficient for basic face geometry

### 10-Point Set (Optional)

More detailed geometry:
- Left eye (left, right)
- Right eye (left, right)
- Nose (tip, bottom)
- Mouth (left, right, top, bottom)

**Use case:** Better geometric understanding, slightly slower

## Training Flow

### Step 1: Extract Landmarks

```bash
python -m sim_bench.training.phase2_pretraining.extract_landmarks \
    --data_dir D:/DataSets/AffectNet/train \
    --output_cache cache/affectnet_landmarks_train.json \
    --num_landmarks 5
```

**What it does:**
- Scans AffectNet directory structure
- Extracts landmarks using MediaPipe FaceMesh
- Saves to JSON cache for fast loading

**Output:** JSON mapping `image_path -> [[x1,y1], [x2,y2], ...]` in normalized [0,1] coordinates

### Step 2: Train Multitask Model

```bash
python -m sim_bench.training.phase2_pretraining.train_multitask \
    --config configs/face/multitask_affectnet.yaml
```

**Training process:**
1. Load AffectNet dataset with cached landmarks
2. Initialize ResNet-50 (ImageNet pretrained)
3. Train with two losses:
   - Expression: Cross-entropy (8 classes)
   - Landmarks: MSE (masked for images with valid landmarks)
4. Use uncertainty weighting to balance losses automatically
5. Save best model based on validation expression accuracy

**Loss computation:**
```python
expression_loss = CrossEntropy(expression_logits, expression_labels)
landmark_loss = MSE(landmark_predictions, landmark_targets)  # masked
total_loss = uncertainty_weighting([expression_loss, landmark_loss])
```

### Step 3: Transfer to Face Recognition

After Phase 2 training, the pretrained backbone can be loaded for CASIA ArcFace training:

```python
# Load Phase 2 pretrained model
checkpoint = torch.load('outputs/phase2_multitask/.../best_model.pt')
pretrained_state = checkpoint['model_state_dict']

# Create ArcFace model
arcface_model = FaceResNet(config)

# Transfer backbone weights
arcface_model.backbone.load_state_dict({
    k.replace('backbone.', ''): v 
    for k, v in pretrained_state.items() 
    if 'backbone' in k
})

# Train on CASIA with PK sampling + warmup
```

## Loss Balancing

### Problem

Classification (CE) and regression (MSE) losses have very different scales:
- CE loss: typically 0.5-2.0
- MSE loss: typically 0.001-0.01 (normalized coordinates)

Naive sum: `loss = ce + mse` → CE dominates

### Solution: Uncertainty Weighting

**Kendall et al. 2018** - Learnable loss weighting:

```python
precision = exp(-log_var)  # Higher precision = more weight
weighted_loss = precision * task_loss + log_var
total = sum(weighted_losses)
```

**Benefits:**
- Automatically learns optimal balance
- Adapts during training
- No manual tuning needed

**Alternative (if disabled):**
- Manual weights: `loss = λ_expr * ce + λ_landmark * mse`
- Tune λ values based on initial loss magnitudes

## Configuration

### Key Settings

**Model:**
- `backbone`: "resnet50" or "resnet18"
- `num_landmarks`: 5 or 10
- `use_uncertainty_weighting`: true (recommended)

**Training:**
- `learning_rate`: 0.001 (lower than face recognition)
- `differential_lr`: true (backbone 1x, heads 10x)
- `optimizer`: "adamw" (often better for multitask)
- `warmup_epochs`: 1 (stabilize early training)

**Data:**
- `landmarks_cache`: Path to pre-extracted landmarks JSON
- `num_workers`: 4 (parallel data loading)

## Expected Results

### Training Metrics

**Expression Classification:**
- Target: >70% validation accuracy (8-way)
- Easier than 10k-class face recognition

**Landmark Regression:**
- Target: <0.01 MSE (normalized coordinates)
- ~1% pixel error on 224x224 images

**Loss Balance:**
- Uncertainty weights should stabilize after ~5 epochs
- Both tasks should contribute to gradients

### Transfer Performance

After Phase 2 → Phase 1 (CASIA ArcFace):
- Faster convergence (fewer epochs to good accuracy)
- Higher final accuracy (better initialization)
- More stable training (better feature space)

## File Structure

```
sim_bench/training/phase2_pretraining/
├── __init__.py                 # Module exports
├── PHASE2_PLAN.md              # This document
├── multitask_model.py          # Model architecture (<200 lines)
├── affectnet_dataset.py        # Dataset loader (<200 lines)
├── landmark_extractor.py       # MediaPipe extraction (<200 lines)
├── train_multitask.py          # Training script (<300 lines)
└── extract_landmarks.py        # Landmark extraction CLI (<100 lines)

configs/face/
└── multitask_affectnet.yaml    # Training configuration
```

## Design Principles

### SOLID Compliance

**Single Responsibility:**
- `MultitaskFaceModel`: Model architecture only
- `AffectNetDataset`: Data loading only
- `LandmarkExtractor`: Landmark extraction only
- `train_multitask.py`: Training orchestration only

**Open/Closed:**
- Model supports different backbones (ResNet-18/50) via config
- Dataset supports different landmark counts (5/10) via config
- Easy to add new tasks without modifying existing code

**Dependency Inversion:**
- Training script depends on model interface, not implementation
- Dataset uses abstract transform interface

### Code Quality

- **No try/except**: Explicit error handling via validation
- **Modules <300 lines**: Each file focused and concise
- **Type hints**: All functions have type annotations
- **Logging**: Comprehensive logging for debugging
- **Config-driven**: All behavior controlled via YAML

## Usage Workflow

### Complete Pipeline

```bash
# 1. Extract landmarks (one-time, ~30 min for full dataset)
python -m sim_bench.training.phase2_pretraining.extract_landmarks \
    --data_dir D:/DataSets/AffectNet/train \
    --output_cache cache/affectnet_landmarks_train.json \
    --num_landmarks 5

python -m sim_bench.training.phase2_pretraining.extract_landmarks \
    --data_dir D:/DataSets/AffectNet/val \
    --output_cache cache/affectnet_landmarks_val.json \
    --num_landmarks 5

# 2. Train multitask model (~2-4 hours on GPU)
python -m sim_bench.training.phase2_pretraining.train_multitask \
    --config configs/face/multitask_affectnet.yaml

# 3. Transfer to Phase 1 (CASIA ArcFace)
# Edit Phase 1 config to load Phase 2 checkpoint
# Run Phase 1 training with PK sampling + warmup
```

## Next Steps

After Phase 2 completion:

1. **Evaluate pretrained model:**
   - Check expression accuracy (should be >70%)
   - Check landmark error (should be <0.01)
   - Visualize learned features (t-SNE)

2. **Transfer to Phase 1:**
   - Load Phase 2 backbone weights
   - Initialize ArcFace head randomly
   - Train with PK sampling (P=16, K=4)
   - Use 1-epoch warmup

3. **Compare results:**
   - Phase 1 (naive): ~1-2% accuracy
   - Phase 1 (PK+warmup): Expected improvement
   - Phase 1 (Phase 2 pretrained + PK+warmup): Best expected

## Troubleshooting

### Low Expression Accuracy

- Check data loading (verify labels correct)
- Increase model capacity (ResNet-50 vs ResNet-18)
- Adjust learning rate (try 0.0005 or 0.002)
- Check for class imbalance

### High Landmark Error

- Verify landmarks extracted correctly (visualize)
- Check normalization (should be [0,1])
- Increase landmark weight (if not using uncertainty weighting)
- Verify MediaPipe detection rate (>80% should have landmarks)

### Loss Imbalance

- If using uncertainty weighting: monitor log_vars (should stabilize)
- If manual weights: tune based on initial loss magnitudes
- Check gradient magnitudes per task (should be similar order)

### Out of Memory

- Reduce batch size (64 → 32)
- Use ResNet-18 instead of ResNet-50
- Reduce num_workers (4 → 2)

## References

- **Uncertainty Weighting**: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
- **AffectNet Dataset**: Mollahosseini et al. "AffectNet: A Database for Facial Expression" (ICCV 2017)
- **MediaPipe FaceMesh**: Lugaresi et al. "MediaPipe: A Framework for Building Perception Pipelines" (2019)
