# Plan: AVA Score Prediction with ResNet (Updated)

## Overview
Create a minimal training script for predicting AVA aesthetic scores using a ResNet backbone with a small MLP head. Supports both **distribution prediction** (10 bins) and **pure regression** (mean score).

## AVA Dataset Format
The AVA.txt file has 15 space-separated columns:
- Column 0: Image ID
- Column 1: Challenge ID
- **Columns 2-11: Vote counts for ratings 1-10** (target)
- Columns 12-14: Semantic tags and challenge reference

Images are stored as `{image_id}.jpg`.

## Architecture
```
Image -> ResNet50 (pretrained, without final FC) -> 2048-dim features -> MLP -> output
```

**Output modes:**
- `distribution`: 10 bins (softmax) - use KL Divergence loss
- `regression`: single scalar (mean score 1-10) - use MSE/L1 loss

---

## Files to Create

### 1. `sim_bench/models/ava_resnet.py` ✅ (in progress)

```python
class AVAResNet(nn.Module):
    """
    Config keys:
        - backbone: 'resnet50'
        - pretrained: bool
        - mlp_hidden_dims: list (e.g., [256] or [])
        - dropout: float
        - activation: 'relu' or 'tanh'
        - output_mode: 'distribution' or 'regression'
    """
    - forward(images) -> logits (B, 10) or (B, 1)
    - get_1x_lr_params(): backbone params
    - get_10x_lr_params(): head params

def create_transform(config, is_train=False):
    """
    Config keys:
        - resize_size: int (default 256)
        - crop_size: int (default 224)
        - normalize_mean/std: list
        - augmentation: dict (only applied when is_train=True)
            - horizontal_flip: float (probability)
            - random_crop: bool
            - color_jitter: {brightness, contrast, saturation, hue}
    """
```

### 2. `sim_bench/datasets/ava_dataset.py`

```python
class AVADataset(Dataset):
    """
    Args:
        ava_txt_path: Path to AVA.txt
        image_dir: Directory containing {image_id}.jpg files
        transform: torchvision transform
        indices: Optional list of indices to use (for train/val/test splits)
        output_mode: 'distribution' or 'regression'

    Returns dict:
        - 'image': tensor (3, H, W)
        - 'scores': tensor (10,) normalized distribution OR scalar mean
        - 'image_id': str
    """

def load_ava_labels(ava_txt_path):
    """Load AVA.txt and return DataFrame with image_id and vote counts."""

def create_splits(df, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Return (train_indices, val_indices, test_indices)."""
```

### 3. `sim_bench/training/train_ava_resnet.py`

```python
def load_config(path): ...
def set_random_seeds(seed): ...
def create_model(config): ...
def create_optimizer(model, config): ...
def create_dataloaders(config): ...

def compute_loss(output, target, output_mode, loss_type):
    """
    output_mode='distribution': KL divergence or cross-entropy
    output_mode='regression': MSE or L1
    """

def compute_mean_score(output, output_mode):
    """Convert model output to mean score (1-10)."""

def train_epoch(model, loader, optimizer, device, config): ...

def evaluate(model, loader, device, config, output_dir, epoch):
    """Returns loss, spearman_corr. Saves predictions to parquet."""

def save_val_predictions(predictions, targets, image_ids, output_dir, epoch):
    """Save validation predictions as parquet file."""

def train_model(model, train_loader, val_loader, optimizer, config, output_dir):
    """Training loop with early stopping based on Spearman correlation."""

def plot_training_curves(history, output_dir): ...

def main(): ...
```

### 4. `configs/ava/resnet50_cpu.yaml` (for CPU / no GPU)

```yaml
name: "ava_resnet50_cpu"

data:
  ava_txt: "D:/datasets/AVA/AVA.txt"           # UPDATE THIS
  image_dir: "D:/datasets/AVA/images"          # UPDATE THIS
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  num_workers: 0                               # CPU: use 0 workers

model:
  backbone: "resnet50"
  pretrained: true
  mlp_hidden_dims: [256]
  dropout: 0.2
  activation: "relu"
  output_mode: "distribution"                  # or "regression"

transform:
  resize_size: 256
  crop_size: 224
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augmentation:
    horizontal_flip: 0.5
    random_crop: true
    color_jitter:
      brightness: 0.1
      contrast: 0.1
      saturation: 0.1
      hue: 0.05

training:
  batch_size: 8                                # Smaller for CPU
  learning_rate: 0.0001
  differential_lr: true
  optimizer: "adamw"
  weight_decay: 0.0001
  max_epochs: 30
  early_stop_patience: 5
  loss_type: "kl_div"

output_dir: null                               # auto-generated
device: "cpu"                                  # CPU mode
seed: 42
log_interval: 10                               # Log every 10 batches
save_val_predictions: true                     # Save val predictions each epoch
```

### 5. `configs/ava/resnet50_gpu.yaml` (for GPU)

```yaml
name: "ava_resnet50_gpu"

data:
  ava_txt: "D:/datasets/AVA/AVA.txt"           # UPDATE THIS
  image_dir: "D:/datasets/AVA/images"          # UPDATE THIS
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  num_workers: 4                               # GPU: use multiple workers

model:
  backbone: "resnet50"
  pretrained: true
  mlp_hidden_dims: [256]
  dropout: 0.2
  activation: "relu"
  output_mode: "distribution"                  # or "regression"

transform:
  resize_size: 256
  crop_size: 224
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augmentation:
    horizontal_flip: 0.5
    random_crop: true
    color_jitter:
      brightness: 0.1
      contrast: 0.1
      saturation: 0.1
      hue: 0.05

training:
  batch_size: 64                               # Larger for GPU
  learning_rate: 0.0001
  differential_lr: true
  optimizer: "adamw"
  weight_decay: 0.0001
  max_epochs: 30
  early_stop_patience: 5
  loss_type: "kl_div"

output_dir: null                               # auto-generated
device: "cuda"                                 # GPU mode
seed: 42
log_interval: 50                               # Log every 50 batches
save_val_predictions: true                     # Save val predictions each epoch
```

---

## Key Design Decisions

1. **No try/except** - as requested
2. **Logging every N batches** - configurable via `log_interval`
3. **Two output modes**:
   - `distribution`: predict 10-bin probability, loss = KL divergence
   - `regression`: predict mean score directly, loss = MSE or L1
4. **Spearman correlation** - primary metric for early stopping (works for both modes)
5. **Configurable transforms** - separate train/val transforms, augmentation only on train
6. **Differential learning rates** - 1x backbone, 10x head
7. **Device from config** - `device: "cpu"` or `device: "cuda"`
8. **Dataset paths from config** - `data.ava_txt` and `data.image_dir`
9. **Validation predictions saved** - parquet file each epoch with predictions
10. **Training curves plotted** - loss and Spearman correlation plots

---

## Metrics

| Mode | Loss | Primary Metric |
|------|------|----------------|
| distribution | KL Divergence | Spearman correlation (predicted mean vs GT mean) |
| regression | MSE or L1 | Spearman correlation (predicted vs GT mean) |

**Mean score calculation:**
- Distribution mode: `sum(score * softmax(logits)[score-1] for score in 1..10)`
- Regression mode: direct output (clamped to 1-10)

---

## Output Structure
```
outputs/ava/{timestamp}/
├── training.log
├── config.yaml
├── training_history.json
├── training_curves.png              # Loss + Spearman plots
├── results.json
├── best_model.pt
└── predictions/
    ├── val_epoch_000.parquet        # Validation predictions each epoch
    ├── val_epoch_001.parquet
    └── ...
```

**Prediction parquet columns:**
- `image_id`: str
- `pred_mean`: float (predicted mean score)
- `gt_mean`: float (ground truth mean score)
- `pred_dist_1` ... `pred_dist_10`: float (predicted distribution, if distribution mode)
- `gt_dist_1` ... `gt_dist_10`: float (ground truth distribution)

---

## Implementation Order
1. ✅ `ava_resnet.py` - model + transform factory (in progress)
2. `ava_dataset.py` - dataset class
3. `train_ava_resnet.py` - training script
4. `configs/ava/resnet50_cpu.yaml` - CPU config
5. `configs/ava/resnet50_gpu.yaml` - GPU config

---

## Verification
1. Run with synthetic/dummy data to verify pipeline
2. Check pretrained weights load correctly
3. Verify loss decreases during training
4. Confirm Spearman correlation computed correctly
5. Test both `distribution` and `regression` modes
6. Verify parquet files are saved correctly each epoch
7. Check training curves are generated
