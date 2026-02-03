# Checking for Same File Names

If the file names differ between runs, nothing else will match. Start here.

## Method 1: Standalone Script (Recommended)

Run this twice and compare:

```bash
# First run
python check_batch_files.py --config configs/example_external.yaml --save run1.json

# Second run (restart Python, same config)
python check_batch_files.py --config configs/example_external.yaml --save run2.json

# Compare
python check_batch_files.py --compare run1.json run2.json
```

This will tell you EXACTLY if files differ and where.

## Method 2: Quick Inline Check

Add this to your `train_siamese_e2e.py` right before `inspect_model_output` (around line 586):

```python
# Quick file check
print("\n" + "="*70)
print("FIRST BATCH FILE CHECK")
print("="*70)
batch = next(iter(train_loader))
if 'image1' in batch:
    print(f"Batch size: {len(batch['image1'])}")
    print(f"\nFirst 5 files:")
    for i in range(min(5, len(batch['image1']))):
        print(f"  [{i}] winner={batch['winner'][i].item()}")
        print(f"      img1: {batch['image1'][i]}")
        print(f"      img2: {batch['image2'][i]}")
    print(f"\nLast file:")
    i = len(batch['image1']) - 1
    print(f"  [{i}] winner={batch['winner'][i].item()}")
    print(f"      img1: {batch['image1'][i]}")
    print(f"      img2: {batch['image2'][i]}")
else:
    print("ERROR: No image1 field in batch!")
print("="*70 + "\n")
```

Run twice, copy the output, and compare manually.

## Method 3: Hash Comparison

If you just want a quick yes/no answer:

```python
import hashlib

batch = next(iter(train_loader))
files1 = batch['image1']
files2 = batch['image2']

# Create hash of file order
all_files = "|".join(files1) + "|" + "|".join(files2)
file_hash = hashlib.md5(all_files.encode()).hexdigest()

print(f"First batch file hash: {file_hash}")
# Compare this hash between runs - should be identical!
```

## What to Look For

### ✅ Files MATCH
If files are identical:
```
✓ image1 files MATCH
✓ image2 files MATCH
✓ winners MATCH
```

Then the issue is NOT the dataloader file order. Check:
- Image transforms (random crops/flips)
- Model initialization
- CUDA operations

### ❌ Files DIFFER
If files differ:
```
❌ image1 files DIFFER
  First difference at index 2:
    Run1: series_123/img_005.jpg
    Run2: series_456/img_012.jpg
```

Then the dataloader is non-deterministic. Causes:
1. **DataLoader shuffle=True** - Check your factory
2. **External dataset shuffles internally** - Check MyDataset code
3. **Seed not set correctly** - Verify seed is passed
4. **Worker processes** - Check num_workers (should be 0 for determinism)

## Fixing File Order Issues

### Issue 1: DataLoader Shuffling

Check your [dataloader_factory.py](d:\sim-bench\sim_bench\datasets\dataloader_factory.py):

```python
# Should be:
DataLoader(dataset, batch_size=..., shuffle=False, ...)  # For train too!

# NOT:
DataLoader(dataset, batch_size=..., shuffle=True, ...)
```

Even for training, if you want determinism, set `shuffle=False`.

### Issue 2: External Dataset Internal Shuffling

Check the external `MyDataset` class in `D:\Projects\Series-Photo-Selection\data\dataloader.py`:

Look for:
- `random.shuffle()`
- `np.random.permutation()`
- Any random sampling

If found, make sure they use the seed you pass.

### Issue 3: Dataset Not Using Seed

Check how MyDataset uses the seed:

```python
# In MyDataset.__init__:
def __init__(self, train=True, image_root=None, seed=42):
    # MUST set seed BEFORE any random operations
    random.seed(seed)
    np.random.seed(seed)

    # Then do any shuffling/sampling
    # ...
```

### Issue 4: Worker Processes

Workers have separate random states. Force single-threaded:

```python
# In DataLoaderFactory or wherever you create the loader:
DataLoader(..., num_workers=0, ...)  # Single process = deterministic
```

## Expected Output (Files Match)

```
BATCH 1
Size: 8
First 5 pairs:
  [0] winner=0
      img1: series_001/img_003.jpg
      img2: series_001/img_007.jpg
  [1] winner=1
      img1: series_001/img_002.jpg
      img2: series_001/img_009.jpg
  ...
```

Run this twice - should be **EXACTLY** the same!

## Next Steps

1. **Run the check** - Use Method 1 or 2
2. **Compare outputs** - Are files identical?
3. **If files match** → Issue is transforms/model, see [DETERMINISM_DEBUG_GUIDE.md](DETERMINISM_DEBUG_GUIDE.md)
4. **If files differ** → Issue is dataloader, check the fixes above

## Quick Test Right Now

Just run this in a Python shell:

```python
import yaml
import torch
import numpy as np
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.transform_factory import create_transform

# Load your config
with open('configs/example_external.yaml') as f:
    config = yaml.safe_load(f)

# Set seed
seed = config.get('seed', 42)
torch.manual_seed(seed)
np.random.seed(seed)

# Create loader (same way as your training script)
# ... your dataloader creation code ...

# Check first batch
batch = next(iter(train_loader))
print("First file:", batch['image1'][0])

# NOW RUN THIS AGAIN - should print SAME file!
```

If it prints a different file the second time, the dataloader is non-deterministic!
