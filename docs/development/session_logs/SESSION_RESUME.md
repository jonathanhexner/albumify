# Session Resume: Siamese CNN Training Debug

## Current Status: Silent Crash During Training

### Problem
Training exits silently after first backbone forward pass completes successfully. No Python exception is raised.

**Crash Point:**
```
2025-12-14 08:18:35,944 - INFO -   Forward: Got feat1 with shape torch.Size([8, 2048])
2025-12-14 08:18:35,944 - INFO -   Forward: Starting backbone for img2...
[Process exits with no error message]
```

### What's Been Fixed (52% → 70% accuracy goal)

All changes made to match reference implementation: `D:\Projects\Series-Photo-Selection\train_resnet.py`

#### 1. Critical Bugs Fixed ✓

**Label Inversion Bug** (CRITICAL)
- **Files**: [sim_bench/training/train_siamese_e2e.py:130](sim_bench/training/train_siamese_e2e.py#L130), [sim_bench/training/train_frozen.py:80](sim_bench/training/train_frozen.py#L80)
- **Was**: `loss = F.nll_loss(log_probs, 1 - winners)`
- **Now**: `loss = F.nll_loss(log_probs, winners)`
- **Impact**: Model was learning inverse relationship (worse images instead of better)

**Accuracy Double-Inversion** (CRITICAL)
- **File**: [sim_bench/quality_assessment/trained_models/phototriage_multifeature.py:782](sim_bench/quality_assessment/trained_models/phototriage_multifeature.py#L782)
- **Was**: `pred_winners = 1 - pred_winners`
- **Now**: Direct comparison without inversion
- **Impact**: Evaluation metrics were inverted

**Unicode Encoding**
- **File**: [sim_bench/training/train_siamese_e2e.py:277](sim_bench/training/train_siamese_e2e.py#L277)
- **Fix**: Added `encoding='utf-8'` to FileHandler
- **Also**: Changed all `→` arrows to `->` in logging

#### 2. Architecture Changes to Match Reference ✓

**Config File**: [configs/siamese_e2e/resnet50.yaml](configs/siamese_e2e/resnet50.yaml)

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `mlp_hidden_dims` | `[256, 128]` | `[]` | Match ref: direct 2048 → 2 linear layer |
| `dropout` | `0.3` | `0.0` | Match ref: no dropout |
| `use_paper_preprocessing` | `false` | `true` | Match ref: aspect-ratio preserving resize |
| `max_epochs` | `10` | `100` | Match ref: longer training |
| `learning_rate` | (varied) | `0.00001` | Match ref: 1e-5 for backbone, 1e-4 for head |

**Weight Initialization**
- **File**: [sim_bench/models/siamese_cnn_ranker.py:118-125](sim_bench/models/siamese_cnn_ranker.py#L118-L125)
- **Added**: Kaiming normal initialization for MLP weights
- **Matches**: Reference implementation pattern

#### 3. Code Quality ✓

- **Removed**: All try-except blocks (per user requirement: "Usage of try and except is strictly prohibited")
- **Added**: Extensive debug logging throughout training pipeline to track execution

### Files Changed

1. **sim_bench/training/train_siamese_e2e.py**
   - Fixed label inversion in loss (line 130)
   - Fixed logging encoding (line 277)
   - Added extensive debug logging in train_epoch (lines 104-146)
   - Removed try-except blocks

2. **sim_bench/training/train_frozen.py**
   - Fixed label inversion in loss (lines 80, 105)

3. **sim_bench/models/siamese_cnn_ranker.py**
   - Added `_initialize_mlp_weights()` method (lines 118-125)
   - Added extensive logging in `forward()` method (lines 164-184)

4. **sim_bench/quality_assessment/trained_models/phototriage_multifeature.py**
   - Fixed accuracy calculation (removed double inversion at line 782)

5. **sim_bench/datasets/phototriage_data.py**
   - Fixed Unicode arrow in logging (line 171)

6. **configs/siamese_e2e/resnet50.yaml**
   - Updated all model and training parameters to match reference

### Next Steps After Restart

#### Immediate Action
1. **Check Windows Event Viewer**
   - Look in Application logs for crash details
   - Search for Python.exe crashes with error codes

2. **Try Reduced Batch Size**
   ```yaml
   # In configs/siamese_e2e/resnet50.yaml
   batch_size: 1  # Reduce from 8
   ```

3. **Run Training**
   ```bash
   python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml --quick-experiment 0.1
   ```

#### If Crash Persists
- May need to investigate PyTorch CPU backend issues on Windows
- Could be stack overflow or threading issue in ResNet50 backbone
- The fact that first forward pass succeeds but second fails is unusual

#### Expected Outcome
Once crash is resolved, the fixes should give **68-75% validation accuracy** (matching reference's 70%).

### Key Technical Details

**Architecture**:
- Siamese CNN with shared ResNet50 backbone
- Direct linear layer: 2048 → 2 (no hidden layers)
- LogSoftmax output for NLLLoss

**Label Encoding**:
- `winner=0`: img1 is better
- `winner=1`: img2 is better
- Output index 0: P(img2 > img1)
- Output index 1: P(img1 > img2)

**Training Setup**:
- Differential learning rates: 1x (1e-5) for backbone, 10x (1e-4) for head
- SGD optimizer with momentum=0.9, weight_decay=0.0005
- Batch size: 8
- Max epochs: 100, early stopping patience: 10

**Preprocessing**:
- Aspect-ratio preserving resize (larger dimension → 224)
- Center padding with ImageNet mean color
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### User Requirements
- No try-except blocks allowed
- Want exact match to reference implementation
- Want to understand root causes, not just apply fixes
- Need evidence before making assumptions (don't guess at memory issues, etc.)

### Last Conversation Point
User asked: "How do you know it had a memory issue?" when I suggested memory as crash cause without evidence.
User then requested restart and this summary document.
