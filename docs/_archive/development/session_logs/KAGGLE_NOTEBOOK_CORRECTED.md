# Corrected Kaggle Notebook - Copy This Content

This is the complete, corrected notebook using command-line overrides.

---

## Cell 1: Markdown

```markdown
# Siamese CNN Training for PhotoTriage
End-to-end training of VGG16/ResNet50 for pairwise image quality ranking

## Setup
This notebook:
1. Uses the existing PhotoTriage dataset on Kaggle (https://www.kaggle.com/datasets/ericwolter/triage)
2. Clones and installs sim_bench package
3. **Uses YAML configs directly with command-line overrides**
4. Trains Siamese CNN + MLP end-to-end with **differential learning rates**
5. Saves results and plots

## Before Running
1. **Add the dataset**: Click "+ Add Data" → Search "triage" → Add the dataset by ericwolter
2. **Enable GPU**: Settings → Accelerator → GPU T4 x2
3. **Enable Internet**: Settings → Internet → On (to clone GitHub repo)
```

---

## Cell 2: Code (Python)

```python
# Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## Cell 3: Markdown

```markdown
## 1. Download PhotoTriage Dataset
```

---

## Cell 4: Code (Python)

```python
# Use existing PhotoTriage dataset from Kaggle
# Dataset: https://www.kaggle.com/datasets/ericwolter/triage
# Add it to your notebook: Click "+ Add Data" → Search "triage" → Add

from pathlib import Path
import os

# Check for the dataset
dataset_path = Path('/kaggle/input/triage')
if dataset_path.exists():
    print(f"✓ Dataset found: {dataset_path}")
    print("\nDataset contents:")
    !ls -lh {dataset_path}
    
    # Check subdirectories
    for subdir in ['train_val', 'test']:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            img_count = len(list(subdir_path.rglob('*.JPG'))) + len(list(subdir_path.rglob('*.jpg')))
            print(f"\n{subdir}: {img_count} images")
else:
    print("❌ Dataset not found!")
    print("\nTo add the dataset:")
    print("1. Click '+ Add Data' in the right sidebar")
    print("2. Search for 'triage' or 'ericwolter/triage'")
    print("3. Click 'Add' on the PhotoTriage dataset")
    print("4. Re-run this cell")
```

---

## Cell 5: Markdown

```markdown
## 2. Clone and Install sim_bench
```

---

## Cell 6: Code (Python)

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/sim-bench.git /kaggle/working/sim-bench
%cd /kaggle/working/sim-bench
```

---

## Cell 7: Code (Python)

```python
# Install dependencies
%pip install -e .
%pip install pandas pillow pyyaml matplotlib seaborn
```

---

## Cell 8: Markdown

```markdown
## 3. Verify Configuration Files

We use the YAML configs directly from the repo with command-line overrides.
No copying or modification needed - keeps configs in sync!
```

---

## Cell 9: Code (Python)

```python
import yaml
from pathlib import Path

# Check available configs
configs_dir = Path('/kaggle/working/sim-bench/configs/siamese_e2e')

print("Available configs from repo:\n")
for config_file in sorted(configs_dir.glob('*.yaml')):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ {config_file.name}")
    print(f"  Model: {config['model']['cnn_backbone']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Base LR: {config['training']['learning_rate']} (backbone)")
    print(f"  Head LR: {config['training']['learning_rate'] * 10} (10x)")
    print(f"  Differential LR: {config['training'].get('differential_lr', False)}")
    print(f"  Max epochs: {config['training']['max_epochs']}")
    print()

print("="*70)
print("We'll override only platform-specific settings via command-line:")
print("  --data-dir /kaggle/input/triage")
print("  --device cuda")
print("  --output-dir /kaggle/working/outputs/...")
print("="*70)
```

---

## Cell 10: Markdown

```markdown
## 4. Quick Test (Optional)

Test with 10% of data to verify everything works.
```

---

## Cell 11: Code (Python)

```python
# Run quick test with 10% of data to verify everything works
!python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/vgg16.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda \
    --output-dir /kaggle/working/outputs/quick_test \
    --quick-experiment 0.1
```

---

## Cell 12: Markdown

```markdown
## 5. Full Training

Choose either VGG16 or ResNet50 (or train both for comparison).

**Expected Results (with differential LR):**
- Epoch 1: ~62% train accuracy, ~70% validation accuracy
- Epoch 10+: ~70%+ train/validation accuracy
```

---

## Cell 13: Code (Python)

```python
# Train VGG16 (paper replication)
!python -m sim_bench.training.train_siamese_e2e \
    --config configs/siamese_e2e/vgg16.yaml \
    --data-dir /kaggle/input/triage \
    --device cuda \
    --output-dir /kaggle/working/outputs/siamese_e2e_vgg16

# Or train ResNet50 (uncomment to use)
# !python -m sim_bench.training.train_siamese_e2e \
#     --config configs/siamese_e2e/resnet50.yaml \
#     --data-dir /kaggle/input/triage \
#     --device cuda \
#     --output-dir /kaggle/working/outputs/siamese_e2e_resnet50
```

---

## Cell 14: Markdown

```markdown
## 6. Load and Visualize Results
```

---

## Cell 15: Code (Python)

```python
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Load results
output_dir = Path('/kaggle/working/outputs/siamese_e2e_vgg16')
results_file = output_dir / 'results.json'
history_file = output_dir / 'training_history.json'

if results_file.exists():
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Test Accuracy:  {results['test_acc']:.3f}")
    print(f"Test Loss:      {results['test_loss']:.4f}")
    
    # Check model checkpoint
    model_file = output_dir / 'best_model.pt'
    if model_file.exists():
        checkpoint = torch.load(model_file, map_location='cpu')
        print(f"\nBest model from epoch: {checkpoint['epoch'] + 1}")
        print(f"Validation accuracy: {checkpoint['val_acc']:.3f}")
    
    # Display training curves
    curves_file = output_dir / 'training_curves.png'
    if curves_file.exists():
        print("\n" + "="*50)
        print("TRAINING CURVES")
        print("="*50)
        from IPython.display import Image, display
        display(Image(filename=str(curves_file)))
    
    # Show training history
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print("\n" + "="*50)
        print("TRAINING HISTORY (Last 5 Epochs)")
        print("="*50)
        print("Epoch | Train Loss | Train Acc | Val Loss | Val Acc")
        print("-" * 50)
        for i in range(max(0, len(history['train_loss']) - 5), len(history['train_loss'])):
            print(f"{i+1:5d} | {history['train_loss'][i]:10.4f} | {history['train_acc'][i]:9.3f} | "
                  f"{history['val_loss'][i]:8.4f} | {history['val_acc'][i]:7.3f}")
else:
    print(f"❌ Results not found: {results_file}")
    print("\nMake sure training completed successfully.")
```

---

## Cell 16: Markdown

```markdown
## 7. Package and Download Results

Creates a zip file with all training outputs for download.
```

---

## Cell 17: Code (Python)

```python
import shutil
import os

# Create a zip file of all results
output_zip = '/kaggle/working/siamese_training_results'
shutil.make_archive(output_zip, 'zip', '/kaggle/working/outputs')

zip_file = output_zip + '.zip'
print(f"✓ Results packaged: {zip_file}")
print(f"  File size: {os.path.getsize(zip_file) / 1024 / 1024:.2f} MB")
print("\n" + "="*60)
print("Download this file from the Kaggle output section")
print("="*60)
print("\nContents include:")
print("  - best_model.pt (trained model weights)")
print("  - config.yaml (training configuration)")
print("  - results.json (final test results)")
print("  - training_history.json (per-epoch metrics)")
print("  - training_curves.png (loss/accuracy plots)")
print("  - training.log (complete training logs)")
```

---

## Summary

**Key Changes from Original:**

1. ✅ **Cell 9**: Shows configs from repo, no duplication
2. ✅ **Cell 11**: Uses `--data-dir`, `--device`, `--output-dir` flags
3. ✅ **Cell 13**: Uses command-line overrides, supports both VGG16 and ResNet50
4. ✅ **Cell 15**: Enhanced results display with training curves and history
5. ✅ **Cell 17**: Better output summary

**Total Cells:** 17 (vs original 14)
- 7 Markdown cells (documentation)
- 10 Code cells (Python)

**What Gets Used:**
- YAML configs: `configs/siamese_e2e/vgg16.yaml` and `resnet50.yaml` (from repo)
- Command-line overrides: `--data-dir`, `--device`, `--output-dir`
- All training parameters (batch size, LR, differential_lr) from YAML automatically

**Expected Performance:**
- Epoch 1: ~70% validation accuracy (vs 52% before fix)
- Final: 70%+ test accuracy


















