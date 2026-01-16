# DataLoader Factory - Complete Guide

## Overview

The **DataLoaderFactory** provides a clean, unified interface for creating dataloaders from any source:
- PhotoTriage DataFrames
- External datasets (Series-Photo-Selection)
- Any custom dataset implementation

**Key Principle**: One factory, multiple sources → all produce interchangeable loaders.

## Quick Start

### Option 1: PhotoTriage Data (Most Common)

```python
from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.models.siamese_cnn_ranker import SiameseCNNRanker

# Load data
data = PhotoTriageData('data/phototriage', min_agreement=0.7, min_reviewers=2)
train_df, val_df, test_df = data.get_series_based_splits(0.8, 0.1, 0.1, seed=42)

# Create model (provides transform)
model = SiameseCNNRanker({'cnn_backbone': 'resnet50'})

# Create factory
factory = DataLoaderFactory(batch_size=16, num_workers=0)

# Create loaders
train_loader, val_loader, test_loader = factory.create_from_phototriage(
    data=data,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    transform=model.preprocess
)

# Train (same code regardless of source!)
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
```

### Option 2: External Datasets (Series-Photo-Selection)

```python
import sys
sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')

from data.dataloader import MyDataset
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

# Create external datasets
train_data = MyDataset(train=True, image_root='path/to/images')
val_data = MyDataset(train=False, image_root='path/to/images')

# Create factory
factory = DataLoaderFactory(batch_size=8, num_workers=0)

# Create loaders (adapter wraps automatically!)
train_loader, val_loader, test_loader = factory.create_from_external(
    train_dataset=train_data,
    val_dataset=val_data,
    test_dataset=None  # Optional
)

# Train (EXACT SAME CODE!)
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
```

### Option 3: Unified Interface

```python
from sim_bench.datasets.dataloader_factory import create_dataloaders_unified

# PhotoTriage
loaders = create_dataloaders_unified(
    source='phototriage',
    data=phototriage_data,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    transform=transform,
    batch_size=16
)

# OR External
loaders = create_dataloaders_unified(
    source='external',
    train_dataset=ext_train,
    val_dataset=ext_val,
    test_dataset=ext_test,
    batch_size=8
)

# OR Plain DataFrames
loaders = create_dataloaders_unified(
    source='dataframes',
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    image_dir='/path/to/images',
    transform=transform,
    batch_size=16
)
```

## Complete Training Example

Here's how to modify your training script to use the factory:

### Before (train_siamese_e2e.py)
```python
def create_dataloaders(train_df, val_df, test_df, data, transform, batch_size):
    """Old way - manual dataset creation."""
    train_dataset = EndToEndPairDataset(train_df, data.train_val_img_dir, transform)
    val_dataset = EndToEndPairDataset(val_df, data.train_val_img_dir, transform)
    test_dataset = EndToEndPairDataset(test_df, data.train_val_img_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
```

### After (with Factory)
```python
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

def create_dataloaders(train_df, val_df, test_df, data, transform, batch_size):
    """New way - factory pattern."""
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=0)
    return factory.create_from_phototriage(
        data=data,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        transform=transform
    )
```

Now you can **swap sources** without changing the training code!

## Replacing PhotoTriage with External Dataset

To use external datasets instead of PhotoTriage:

```python
def main():
    # ... config loading ...

    # OPTION A: PhotoTriage (original)
    if use_phototriage:
        data = PhotoTriageData(config['data']['root_dir'], ...)
        train_df, val_df, test_df = data.get_series_based_splits(...)

        factory = DataLoaderFactory(batch_size=config['training']['batch_size'])
        train_loader, val_loader, test_loader = factory.create_from_phototriage(
            data, train_df, val_df, test_df, model.preprocess
        )

    # OPTION B: External (Series-Photo-Selection)
    else:
        import sys
        sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')
        from data.dataloader import MyDataset

        train_data = MyDataset(train=True)
        val_data = MyDataset(train=False)

        factory = DataLoaderFactory(batch_size=config['training']['batch_size'])
        train_loader, val_loader, test_loader = factory.create_from_external(
            train_data, val_data, None
        )

    # REST OF TRAINING IS IDENTICAL!
    optimizer = create_optimizer(model, config)
    train_model(model, train_loader, val_loader, optimizer, config, output_dir)
    # ... etc ...
```

## Key Benefits

### 1. True Interchangeability

All loaders produce the same batch format:

```python
batch = {
    'img1': torch.Tensor,      # [B, 3, H, W]
    'img2': torch.Tensor,      # [B, 3, H, W]
    'winner': torch.Tensor,    # [B]
    'image1': List[str],       # Filenames
    'image2': List[str]        # Filenames
}
```

### 2. Metadata Always Accessible

```python
from sim_bench.datasets.siamese_dataloaders import get_dataset_from_loader

# Works with ANY loader!
dataset = get_dataset_from_loader(train_loader)
pairs_df = dataset.get_dataframe()
image_dir = dataset.get_image_dir()
transform = dataset.transform
```

### 3. Clean Architecture

```
DataLoaderFactory
├── create_from_phototriage()    → PhotoTriage data
├── create_from_dataframes()     → Raw DataFrames
├── create_from_external()       → External datasets
└── _create_loader()             → Internal helper

All return: (train_loader, val_loader, test_loader)
```

### 4. Automatic Metadata Handling

**PhotoTriage data** → Uses real metadata (series_id, agreement, num_reviewers)

**External data** → Creates default metadata:
- `series_id`: 'unknown'
- `agreement`: 1.0
- `num_reviewers`: 1

## API Reference

### DataLoaderFactory Class

```python
class DataLoaderFactory:
    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle_train: bool = True
    ):
        """Initialize factory with common parameters."""
```

#### Methods

**`create_from_phototriage(data, train_df, val_df, test_df, transform)`**
- For PhotoTriage data
- Returns: `(train_loader, val_loader, test_loader)`

**`create_from_dataframes(train_df, val_df, test_df, image_dir, transform)`**
- For raw DataFrames
- Returns: `(train_loader, val_loader, test_loader or None)`

**`create_from_external(train_dataset, val_dataset, test_dataset=None, ...)`**
- For external datasets
- Automatically wraps with `ExternalDatasetAdapter`
- Returns: `(train_loader, val_loader, test_loader or None)`

### Utility Functions

**`create_dataloaders_unified(source, batch_size, num_workers, **kwargs)`**
- Unified interface for all sources
- `source`: 'phototriage', 'dataframes', or 'external'
- Returns: `(train_loader, val_loader, test_loader or None)`

**`get_dataset_from_loader(loader)`**
- Extract dataset from any loader
- Returns: Dataset instance with `get_dataframe()`, `get_image_dir()`, `transform`

## Testing

Run the comprehensive test suite:

```bash
python examples/test_dataloader_factory.py
```

Tests:
1. ✅ Factory creates loaders from DataFrames
2. ✅ Factory creates loaders from external datasets
3. ✅ All loaders produce identical batch format
4. ✅ Unified interface works for all sources
5. ✅ Loaders are truly interchangeable

**All tests use REAL data structures - NO MOCKS!**

## Migration Path

### Step 1: Use Factory in Existing Code

```python
# In train_siamese_e2e.py
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

def create_dataloaders(...):
    factory = DataLoaderFactory(batch_size=batch_size)
    return factory.create_from_phototriage(...)
```

### Step 2: Add External Dataset Support

```python
# Add a flag to choose source
if config.get('use_external_dataloader'):
    factory = DataLoaderFactory(...)
    loaders = factory.create_from_external(...)
else:
    factory = DataLoaderFactory(...)
    loaders = factory.create_from_phototriage(...)
```

### Step 3: Training Code Stays the Same!

```python
# This code doesn't change at all
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
evaluate(model, test_loader, device, output_dir, epoch, 'test')
```

## Comparison: Old vs New

### Old Approach
```python
# Problem: Different code for different sources
if using_phototriage:
    train_dataset = EndToEndPairDataset(train_df, image_dir, transform)
    train_loader = DataLoader(train_dataset, ...)
elif using_external:
    train_data = MyDataset(train=True)
    adapter = ExternalDatasetAdapter(train_data)
    train_loader = DataLoader(adapter, ...)

# Problem: Need to pass DataFrames separately
evaluate(model, val_loader, device, ..., val_df, image_dir, transform)
```

### New Approach (Factory)
```python
# Solution: Same code for all sources
factory = DataLoaderFactory(batch_size=16)

# Just change which create_from_* method you call
loaders = factory.create_from_phototriage(...)  # OR
loaders = factory.create_from_external(...)

# Solution: Only pass loaders
evaluate(model, val_loader, device, ...)  # Clean!
```

## Summary

✅ **One factory** for all data sources
✅ **Consistent interface** - all return `(train, val, test)` loaders
✅ **Same batch format** regardless of source
✅ **Metadata accessible** via `get_dataset_from_loader()`
✅ **Training code unchanged** - truly interchangeable
✅ **Clean architecture** following strategy/factory pattern
✅ **Fully tested** with real data structures

The factory pattern eliminates the complexity and provides a clean, maintainable solution for working with multiple data sources.
