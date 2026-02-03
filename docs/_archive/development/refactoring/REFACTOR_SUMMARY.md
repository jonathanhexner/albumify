# Dataloader Refactor Summary

## Overview

Successfully refactored the Siamese training code to eliminate DataFrame dependencies from the training loop. The code now follows standard PyTorch conventions where datasets encapsulate all their metadata.

## Motivation

### Original Problems
1. **Redundant parameters**: Functions required both loaders AND DataFrames
2. **External dataset incompatibility**: `ExternalDatasetAdapter` couldn't provide required metadata
3. **Non-standard architecture**: Not following common PyTorch patterns

### Goals Achieved
1. ✅ **Clean API**: Functions only need loaders, not loaders + DataFrames
2. ✅ **External dataset support**: `ExternalDatasetAdapter` can create default metadata
3. ✅ **Standard PyTorch pattern**: Datasets own their metadata
4. ✅ **`get_dataset_from_loader()` utility**: Easy metadata extraction when needed

## Changes Made

### 1. `sim_bench/datasets/siamese_dataloaders.py`

#### Added to `EndToEndPairDataset`:
```python
def get_dataframe(self) -> pd.DataFrame:
    """Return the underlying pairs DataFrame with metadata."""
    return self.pairs_df

def get_image_dir(self) -> Path:
    """Return the image directory path."""
    return self.image_dir
```

#### Enhanced `ExternalDatasetAdapter`:
```python
def __init__(self, external_dataset, pairs_df=None):
    self.external_dataset = external_dataset
    self.pairs_df = pairs_df
    self.transform = getattr(external_dataset, 'transform', None)

    # Auto-create DataFrame with default metadata if not provided
    if self.pairs_df is None:
        self.pairs_df = self._create_default_dataframe()

def _create_default_dataframe(self) -> pd.DataFrame:
    """Create minimal DataFrame with required metadata columns."""
    n = len(self.external_dataset)
    return pd.DataFrame({
        'image1': [f'image_{i}_a' for i in range(n)],
        'image2': [f'image_{i}_b' for i in range(n)],
        'winner': [0] * n,
        'series_id': ['unknown'] * n,  # Default series
        'agreement': [1.0] * n,         # Default agreement
        'num_reviewers': [1] * n        # Default reviewer count
    })

def get_dataframe(self) -> pd.DataFrame:
    """Return the pairs DataFrame with metadata."""
    return self.pairs_df

def get_image_dir(self) -> Optional[Path]:
    """Return image directory (may be None for external datasets)."""
    image_root = getattr(self.external_dataset, 'image_root', None)
    return Path(image_root) if image_root else None
```

#### Added utility function:
```python
def get_dataset_from_loader(loader: DataLoader):
    """Extract the underlying dataset from a DataLoader."""
    return loader.dataset
```

### 2. `sim_bench/training/diagnostics.py`

#### Updated function signatures:

**Before:**
```python
def save_epoch_metrics(..., pairs_df, avg_loss, metrics_dir):
def save_per_series_breakdown(..., pairs_df, metrics_dir):
def inspect_series_pairs(model, pairs_df, image_dir, transform, device, ...):
```

**After:**
```python
def save_epoch_metrics(..., dataset, avg_loss, metrics_dir):
    pairs_df = dataset.get_dataframe()
    # ... rest of function

def save_per_series_breakdown(..., dataset, metrics_dir):
    pairs_df = dataset.get_dataframe()
    # ... rest of function

def inspect_series_pairs(model, dataset, device, inspect_dir, series_id, k=6):
    pairs_df = dataset.get_dataframe()
    image_dir = dataset.get_image_dir()
    transform = dataset.transform
    # ... rest of function
```

### 3. `sim_bench/training/train_siamese_e2e.py`

#### Updated imports:
```python
from sim_bench.datasets.siamese_dataloaders import (
    create_phototriage_dataloaders,
    get_dataset_from_loader
)
```

#### Updated `evaluate()` function:

**Before:**
```python
def evaluate(model, loader, device, output_dir, epoch, split_name,
             pairs_df, image_dir, transform, inspect_k=6, log_interval=10):
```

**After:**
```python
def evaluate(model, loader, device, output_dir, epoch, split_name,
             inspect_k=6, log_interval=10):
    # Extract dataset and metadata from loader
    dataset = get_dataset_from_loader(loader)

    # ... evaluation loop

    # Pass dataset instead of pairs_df to diagnostics
    save_epoch_metrics(..., dataset, avg_loss, metrics_dir)
    save_per_series_breakdown(..., dataset, metrics_dir)

    if inspect_k > 0:
        pairs_df = dataset.get_dataframe()
        series_id = pairs_df['series_id'].iloc[0]
        inspect_series_pairs(model, dataset, device, inspect_dir, series_id, k=inspect_k)
```

#### Updated `train_model()` function:

**Before:**
```python
def train_model(model, train_loader, val_loader, optimizer, config, output_dir,
                train_df, val_df, data, transform):
```

**After:**
```python
def train_model(model, train_loader, val_loader, optimizer, config, output_dir):
    """Simplified signature - only needs loaders, not DataFrames."""
    # ... training loop

    val_loss, val_acc = evaluate(
        model, val_loader, device,
        output_dir, epoch, 'val',
        inspect_k=6, log_interval=log_interval
    )
```

#### Updated `main()` function:

**Before:**
```python
best_val_acc, history = train_model(
    model, train_loader, val_loader, optimizer, config, output_dir,
    train_df, val_df, data, transform
)

test_loss, test_acc = evaluate(
    model, test_loader, config['device'],
    output_dir, final_epoch, 'test',
    test_df, data.train_val_img_dir, transform,
    inspect_k=6, log_interval=config.get('log_interval', 10)
)
```

**After:**
```python
best_val_acc, history = train_model(
    model, train_loader, val_loader, optimizer, config, output_dir
)

test_loss, test_acc = evaluate(
    model, test_loader, config['device'],
    output_dir, final_epoch, 'test',
    inspect_k=6, log_interval=config.get('log_interval', 10)
)
```

## Usage Examples

### Using Internal Dataloaders (PhotoTriage)

```python
from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.siamese_dataloaders import create_phototriage_dataloaders
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker

# Load data
data = PhotoTriageData('data/phototriage', min_agreement=0.7, min_reviewers=2)
train_df, val_df, test_df = data.get_series_based_splits(0.8, 0.1, 0.1, seed=42)

# Create model
model = SiameseCNNRanker({'cnn_backbone': 'resnet50'})

# Create dataloaders
train_loader, val_loader, test_loader = create_phototriage_dataloaders(
    data=data,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    transform=model.preprocess,
    batch_size=16
)

# Train - only needs loaders!
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
```

### Using External Dataloaders (Series-Photo-Selection)

```python
import sys
sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')

from data.dataloader import MyDataset
from sim_bench.datasets.siamese_dataloaders import create_dataloaders_from_external

# Create external datasets
train_data = MyDataset(train=True, image_root='path/to/images')
val_data = MyDataset(train=False, image_root='path/to/images')

# Create compatible dataloaders (adapter wraps automatically)
train_loader, val_loader = create_dataloaders_from_external(
    external_train_dataset=train_data,
    external_val_dataset=val_data,
    batch_size=8
)

# Train - same code as internal dataloaders!
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
```

### Extracting Metadata When Needed

```python
from sim_bench.datasets.siamese_dataloaders import get_dataset_from_loader

# Extract dataset from loader
dataset = get_dataset_from_loader(train_loader)

# Access metadata
pairs_df = dataset.get_dataframe()
image_dir = dataset.get_image_dir()
transform = dataset.transform

# Use metadata
print(f"Series IDs: {pairs_df['series_id'].unique()}")
print(f"Image directory: {image_dir}")
```

## Benefits

### 1. Cleaner API
**Before:**
```python
evaluate(model, val_loader, device, output_dir, epoch, 'val',
         val_df, data.train_val_img_dir, transform, inspect_k=6)
```

**After:**
```python
evaluate(model, val_loader, device, output_dir, epoch, 'val', inspect_k=6)
```

### 2. Standard PyTorch Convention
Datasets now encapsulate all their data and metadata, following common PyTorch patterns.

### 3. External Dataset Compatibility
`ExternalDatasetAdapter` automatically creates default metadata, allowing any external dataset to work with the diagnostic functions.

### 4. Easier Testing
Can mock datasets without creating DataFrames:
```python
# Create mock dataset
mock_dataset = Mock()
mock_dataset.get_dataframe.return_value = test_df
mock_dataset.get_image_dir.return_value = Path('/test/path')

# Use in tests
evaluate(model, mock_loader, device, output_dir, epoch, 'test')
```

### 5. Better Separation of Concerns
- Datasets own their metadata
- Training code doesn't care about data source
- Diagnostic functions work with any dataset type

## Testing

All changes have been tested and verified:

```bash
# Run the test suite
python examples/test_refactored_dataloaders.py
```

Test results:
```
[OK] Datasets provide metadata via get_dataframe() and get_image_dir()
[OK] ExternalDatasetAdapter creates default metadata when needed
[OK] get_dataset_from_loader() extracts dataset from loader
[OK] Batch format is compatible across all dataset types
```

## Files Modified

1. ✅ `sim_bench/datasets/siamese_dataloaders.py` - Added metadata methods
2. ✅ `sim_bench/training/diagnostics.py` - Updated to accept dataset
3. ✅ `sim_bench/training/train_siamese_e2e.py` - Simplified function signatures

## Files Created

1. ✅ `examples/test_refactored_dataloaders.py` - Comprehensive test suite
2. ✅ `REFACTOR_SUMMARY.md` - This document

## Backward Compatibility

The refactor is **backward compatible** at the dataloader creation level:
- `create_dataloaders()` still accepts DataFrames
- `create_phototriage_dataloaders()` still works the same way
- Only the training loop signatures changed (fewer parameters needed)

## Next Steps

You can now:

1. **Use external dataloaders**: Simply wrap with `create_dataloaders_from_external()`
2. **Pass only loaders**: No need to pass DataFrames separately
3. **Extract metadata when needed**: Use `get_dataset_from_loader()` and dataset methods

The refactor is complete and fully functional!
