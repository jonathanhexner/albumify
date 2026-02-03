# Determinism Debugging Guide

When comparing two training runs that should be identical but aren't, even with the same seed, dataloader, and model, here's how to debug.

## Common Sources of Non-Determinism

1. **DataLoader shuffling**: Even with same seed, worker processes may differ
2. **Image transforms**: Random augmentations (crops, flips, color jitter)
3. **CUDA operations**: GPU operations aren't always deterministic by default
4. **Model initialization**: Different initial weights
5. **Batch iteration**: Different order of samples
6. **Import order**: Module loading affecting random state
7. **Worker processes**: Multiple workers have different random states

## Quick Diagnostic Script

Run the determinism checker:

```bash
# Check your current setup
python -m examples.debug_determinism --config configs/example_external.yaml --output-dir check1

# Run again to verify determinism
python -m examples.debug_determinism --config configs/example_external.yaml --output-dir check2

# Compare the two runs
python -m examples.debug_determinism --compare check1 check2
```

This will tell you EXACTLY what differs between runs.

## What to Check

### 1. DataLoader Configuration

**Check shuffle and workers:**
```python
# In your dataloader factory or loader creation
logger.info(f"DataLoader shuffle: {loader.shuffle}")
logger.info(f"DataLoader num_workers: {loader.num_workers}")
logger.info(f"DataLoader seed: {seed}")
```

**Key insight**: Even with `num_workers=0` and same seed, external dataloaders might have internal randomness.

### 2. Transform Pipeline

**Check what transforms are applied:**
```python
from sim_bench.datasets.transform_factory import create_transform

transform = create_transform(config)
logger.info(f"Transform: {transform}")
```

**Look for random transforms:**
- `RandomResizedCrop` - Different crops each time
- `RandomHorizontalFlip` - Random flipping
- `ColorJitter` - Random color changes
- `RandomRotation` - Random rotations

**Fix**: Use deterministic transforms or disable randomness:
```python
# Instead of RandomResizedCrop
transforms.Resize((224, 224))

# Instead of RandomHorizontalFlip
# Just don't flip (or always flip)
```

### 3. Dataset Internal State

**External dataloader may have internal randomness:**
```python
# In the external MyDataset class, check:
# - How pairs are sampled
# - If there's any shuffling
# - Random number generation

# Add this to check:
import random
import numpy as np

logger.info(f"Python random state: {random.getstate()[1][:5]}")  # First 5 numbers
logger.info(f"NumPy random state: {np.random.get_state()[1][:5]}")
logger.info(f"PyTorch random state hash: {torch.get_rng_state()[:10]}")
```

### 4. CUDA Determinism

**If using GPU, enable deterministic mode:**
```python
import torch

# Add this BEFORE creating model/data
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# For PyTorch 1.7+, even more strict:
torch.use_deterministic_algorithms(True)
```

**Warning**: This will make training slower but ensures determinism.

### 5. File Order from Filesystem

**File listing order may differ:**
```python
# If the dataloader reads files from disk
# The order might depend on filesystem (especially on different OS)

# Fix: Sort files explicitly
files = sorted(glob.glob(pattern))  # Always sort!
```

### 6. Model Weight Initialization

**Check initial weights are the same:**
```python
from sim_bench.utils.model_inspection import inspect_model_weights

# Right after creating model
inspect_model_weights(model, output_file='initial_weights.txt')
```

Compare the initial weights file between runs - they should be identical.

## Inline Checks to Add to train_siamese_e2e.py

Add these checks right before your `inspect_model_output` call:

```python
# Around line 586, before inspect_model_output
logger.info("\n=== DETERMINISM CHECKS ===")

# 1. Check random state
logger.info(f"PyTorch RNG state (first 10 bytes): {torch.get_rng_state()[:10]}")
logger.info(f"NumPy RNG state: {np.random.get_state()[1][:3]}")

# 2. Check first batch filenames
batch = next(iter(train_loader))
if 'image1' in batch:
    logger.info(f"First batch first file: {batch['image1'][0]}")
    logger.info(f"First batch size: {len(batch['image1'])}")

# 3. Check model first layer weights
first_param = next(model.parameters())
logger.info(f"First param mean: {first_param.mean().item():.8f}")
logger.info(f"First param std: {first_param.std().item():.8f}")

# 4. Check dataloader config
logger.info(f"Train loader batch size: {train_loader.batch_size}")
logger.info(f"Train loader num workers: {train_loader.num_workers}")

logger.info("=== END DETERMINISM CHECKS ===\n")
```

## Specific Issue: External DataLoader

The external `MyDataset` from Series-Photo-Selection might have:

1. **Internal shuffling** that doesn't respect your seed
2. **Random sampling** of pairs
3. **Different file reading order**

**How to check:**
```python
# Look at the external MyDataset.__init__ and __getitem__
# Check if it:
# 1. Uses random without setting seed
# 2. Shuffles data internally
# 3. Samples randomly

# You can verify by creating the dataset twice and checking:
dataset1 = MyDataset(train=True, image_root=path, seed=42)
dataset2 = MyDataset(train=True, image_root=path, seed=42)

# Check if they return same data
sample1 = dataset1[0]
sample2 = dataset2[0]

logger.info(f"Same image1? {sample1['image1'] == sample2['image1']}")
```

## Expected Behavior

If everything is deterministic:

1. ✅ Same batch files (in same order) every run
2. ✅ Same tensor values (pixel values) for each image
3. ✅ Same model weights after initialization
4. ✅ Same predictions on first batch
5. ✅ `model_inspection.csv` files are IDENTICAL

If files differ, it's likely the **dataloader** or **transform**.
If files match but tensor values differ, it's the **transform** (random augmentation).
If tensors match but predictions differ, it's the **model initialization**.

## Force Determinism Everywhere

Add this at the very start of your script:

```python
def set_deterministic(seed=42):
    """Force deterministic behavior everywhere."""
    import random
    import os

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Environment variables (for some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch deterministic algorithms (PyTorch 1.7+)
    # Note: This may cause errors with some operations
    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        pass  # Older PyTorch version

    logger.info(f"✓ Deterministic mode enabled with seed={seed}")

# Call this FIRST in main()
set_deterministic(config['seed'])
```

## Gotcha: Transform Type Auto-Detection

Check this in your config:

```yaml
transform_type: auto  # This might detect differently!
```

The `auto` detection might give different transforms depending on model type. Use explicit transform type:

```yaml
transform_type: external  # Or 'default', be explicit
```

## Next Steps

1. **Run the debug script** to see what's different
2. **Check the batch_info.json** to see if file order differs
3. **Check the model_weights.json** to see if initialization differs
4. **Add inline checks** to your training script
5. **Enable full determinism** with the `set_deterministic()` function
6. **Check the external dataloader code** for internal randomness

The most likely culprit is the **external dataloader** having internal randomness that doesn't respect your seed!
