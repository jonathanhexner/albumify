# Bug Fixes: Accuracy Calculation and Image Paths

## Issue #1: Incorrect Accuracy Calculation with Variable Batch Sizes

### Problem
The training code was averaging per-batch accuracies instead of computing total correct / total samples. This is incorrect when batches have different sizes (which happens with series-based batching where each batch contains pairs from the same number of series, but series can have different numbers of pairs).

**Previous (Incorrect) Code:**
```python
total_acc = 0.0
for batch in loader:
    batch_acc = compute_accuracy(...)
    total_acc += batch_acc
return total_acc / len(loader)  # WRONG: averages per-batch accuracy
```

This fails when:
- Batch 1 has 10 pairs, 9 correct → 90% accuracy
- Batch 2 has 5 pairs, 5 correct → 100% accuracy
- Average: (90% + 100%) / 2 = 95%
- **But actual**: (9+5) / (10+5) = 14/15 = 93.3%

### Solution
Created a common `compute_batch_metrics()` function and track total correct predictions across all samples:

```python
def compute_batch_metrics(log_probs, winners):
    """
    Compute loss and accuracy for a batch.

    Returns:
        loss: Cross-entropy loss
        accuracy: Accuracy (fraction correct)
        num_correct: Number of correct predictions
        batch_size: Total number of samples
    """
    loss = F.nll_loss(log_probs, winners)
    preds = log_probs.argmax(dim=-1)
    num_correct = (preds == winners).sum().item()
    batch_size = len(winners)
    accuracy = num_correct / batch_size
    return loss, accuracy, num_correct, batch_size
```

**Updated Training Code:**
```python
total_correct = 0
total_samples = 0
for batch in loader:
    loss, batch_acc, num_correct, batch_size = compute_batch_metrics(log_probs, winners)
    total_correct += num_correct
    total_samples += batch_size

avg_acc = total_correct / total_samples  # CORRECT: total correct / total samples
```

### Files Modified
- [sim_bench/training/train_siamese_e2e.py](sim_bench/training/train_siamese_e2e.py)
  - Added `compute_batch_metrics()` function (lines 38-53)
  - Updated `train_epoch()` to track total_correct and total_samples (lines 112-188)
  - Updated `evaluate()` to use common metrics function (lines 232-234)
  - Removed unused `compute_pairwise_accuracy` import

## Issue #2: Missing Image Paths for External Dataloader

### Problem
When using the external dataloader, the diagnostic functions tried to load images using fake placeholder paths like `image_0_a` instead of real filenames. This caused errors:

```
WARNING - Failed to inspect pair image_5_a, image_5_b: [Errno 2] No such file or directory:
'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs\image_5_a'
```

The `ExternalDatasetAdapter` was creating a default DataFrame with fake filenames:
```python
'image1': [f'image_{i}_a' for i in range(n)]  # Fake path!
'image2': [f'image_{i}_b' for i in range(n)]  # Fake path!
```

### Solution
Updated `ExternalDatasetAdapter._create_default_dataframe()` to extract real image paths from the external dataset's `pathA` and `pathB` attributes:

```python
def _create_default_dataframe(self) -> pd.DataFrame:
    """Create minimal DataFrame for external datasets without metadata."""
    n = len(self.external_dataset)

    # Try to get real image paths from external dataset
    pathA = getattr(self.external_dataset, 'pathA', None)
    pathB = getattr(self.external_dataset, 'pathB', None)
    result = getattr(self.external_dataset, 'result', None)

    if pathA is not None and pathB is not None:
        # Use real paths from external dataset
        image1_list = list(pathA)
        image2_list = list(pathB)
        winner_list = list(result) if result is not None else [0] * n
    else:
        # Fallback to placeholder names
        image1_list = [f'image_{i}_a' for i in range(n)]
        image2_list = [f'image_{i}_b' for i in range(n)]
        winner_list = [0] * n

    return pd.DataFrame({
        'image1': image1_list,
        'image2': image2_list,
        'winner': winner_list,
        'series_id': ['unknown'] * n,
        'agreement': [1.0] * n,
        'num_reviewers': [1] * n
    })
```

Now the adapter uses the real filenames from the external dataset (e.g., `train_5_a.jpg`, `train_5_b.jpg`) which can be properly loaded for diagnostics.

### Files Modified
- [sim_bench/datasets/siamese_dataloaders.py](sim_bench/datasets/siamese_dataloaders.py)
  - Updated `_create_default_dataframe()` method (lines 102-129)

## Impact

### Accuracy Calculation
- **Before**: Incorrect accuracy when batch sizes vary
- **After**: Correct weighted accuracy across all samples
- **Code Reuse**: Both training and evaluation now use the same `compute_batch_metrics()` function

### Image Paths
- **Before**: Diagnostic functions failed with "file not found" errors
- **After**: Real image filenames from external dataset, diagnostics work correctly
- **Backward Compatibility**: Falls back to placeholder names if external dataset doesn't provide paths

## Testing

All files compile successfully:
```bash
python -m py_compile sim_bench/training/train_siamese_e2e.py
python -m py_compile sim_bench/datasets/siamese_dataloaders.py
```

To test with external dataloader:
```bash
python -m sim_bench.training.train_siamese_e2e --config configs/siamese_e2e/resnet50.yaml
```
(Make sure `use_external_dataloader: true` and `image_root` is set in config)
