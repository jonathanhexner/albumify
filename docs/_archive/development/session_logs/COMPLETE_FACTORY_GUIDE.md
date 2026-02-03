# Complete Factory Pattern Guide

## The Two-Factory System

Your codebase now has a clean two-factory system for handling data from any source:

```
DatasetFactory                DataLoaderFactory
     ↓                              ↓
Creates dataset instances    Creates DataLoaders
     ↓                              ↓
         train_model() - Same code for all!
```

## Quick Start

### Method 1: Using Config Files (Recommended)

**Step 1: Create config file**

```yaml
# configs/my_training.yaml
data_source: phototriage  # or 'external'

data:
  root_dir: data/phototriage
  min_agreement: 0.7

model:
  cnn_backbone: resnet50

training:
  batch_size: 16
```

**Step 2: Train**

```python
from sim_bench.datasets.dataset_factory import DatasetFactory
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

# Load config
config = yaml.safe_load(open('configs/my_training.yaml'))

# Create datasets (handles all imports!)
dataset_factory = DatasetFactory(source=config['data_source'], config=config)
datasets = dataset_factory.create_datasets()

# Create loaders
loader_factory = DataLoaderFactory(batch_size=config['training']['batch_size'])

if config['data_source'] == 'phototriage':
    data, train_df, val_df, test_df = datasets
    loaders = loader_factory.create_from_phototriage(
        data, train_df, val_df, test_df, transform
    )
else:
    train_ds, val_ds, test_ds = datasets
    loaders = loader_factory.create_from_external(train_ds, val_ds, test_ds)

train_loader, val_loader, test_loader = loaders

# Train (same code!)
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
```

### Method 2: Direct API

**PhotoTriage:**
```python
from sim_bench.datasets.dataset_factory import create_phototriage_datasets
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

# Create datasets
data, train_df, val_df, test_df = create_phototriage_datasets(
    root_dir='data/phototriage',
    min_agreement=0.7,
    seed=42
)

# Create loaders
factory = DataLoaderFactory(batch_size=16)
train_loader, val_loader, test_loader = factory.create_from_phototriage(
    data, train_df, val_df, test_df, transform
)
```

**External (Series-Photo-Selection):**
```python
from sim_bench.datasets.dataset_factory import create_external_datasets
from sim_bench.datasets.dataloader_factory import DataLoaderFactory

# Create datasets (handles importing!)
train_ds, val_ds, test_ds = create_external_datasets(
    external_path=r'D:\Projects\Series-Photo-Selection',
    image_root=r'D:\path\to\images'
)

# Create loaders
factory = DataLoaderFactory(batch_size=8)
train_loader, val_loader, _ = factory.create_from_external(train_ds, val_ds, test_ds)
```

## Full Example: train_siamese_e2e.py Integration

Here's how to modify your training script:

```python
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-source', choices=['phototriage', 'external'])
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override data source if specified
    if args.data_source:
        config['data_source'] = args.data_source

    # Setup output directory
    output_dir = Path(config.get('output_dir', 'outputs/...'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = create_model(config, output_dir)
    transform = model.preprocess

    # ========== NEW: Use DatasetFactory ==========
    dataset_factory = DatasetFactory(
        source=config.get('data_source', 'phototriage'),
        config=config
    )
    datasets = dataset_factory.create_datasets()

    # ========== NEW: Use DataLoaderFactory ==========
    loader_factory = DataLoaderFactory(
        batch_size=config['training']['batch_size'],
        num_workers=0
    )

    # Create loaders based on source
    if config.get('data_source', 'phototriage') == 'phototriage':
        data, train_df, val_df, test_df = datasets
        train_loader, val_loader, test_loader = loader_factory.create_from_phototriage(
            data, train_df, val_df, test_df, transform
        )
    else:
        train_ds, val_ds, test_ds = datasets
        train_loader, val_loader, test_loader = loader_factory.create_from_external(
            train_ds, val_ds, test_ds
        )

    # ========== REST IS UNCHANGED ==========
    optimizer = create_optimizer(model, config)
    train_model(model, train_loader, val_loader, optimizer, config, output_dir)
    evaluate(model, test_loader, config['device'], output_dir, ...)
```

## Configuration Examples

### PhotoTriage Config

```yaml
# configs/phototriage.yaml
data_source: phototriage

data:
  root_dir: data/phototriage
  min_agreement: 0.7
  min_reviewers: 2
  quick_experiment: 0.1  # Optional: use 10% of data

model:
  cnn_backbone: resnet50
  mlp_hidden_sizes: [512, 256]

training:
  batch_size: 16
  learning_rate: 0.0001
  max_epochs: 50

seed: 42
device: cuda
```

### External Config

```yaml
# configs/external.yaml
data_source: external

data:
  external_path: D:\Projects\Series-Photo-Selection
  image_root: D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs

model:
  cnn_backbone: resnet50
  mlp_hidden_sizes: [512, 256]

training:
  batch_size: 8  # Smaller for external
  learning_rate: 0.0001
  max_epochs: 50

seed: 42
device: cuda
```

## Command Line Usage

```bash
# PhotoTriage
python -m sim_bench.training.train_siamese_e2e --config configs/phototriage.yaml

# External
python -m sim_bench.training.train_siamese_e2e --config configs/external.yaml

# Override data source
python -m sim_bench.training.train_siamese_e2e \
    --config configs/base.yaml \
    --data-source external
```

## Architecture Overview

### DatasetFactory

**Purpose**: Creates dataset instances from any source

**Methods:**
- `create_datasets()` - Main method, returns datasets based on source
- `_create_phototriage_datasets()` - Internal: creates PhotoTriage datasets
- `_create_external_datasets()` - Internal: creates external datasets (handles importing!)

**Returns:**
- PhotoTriage: `(data, train_df, val_df, test_df)`
- External: `(train_dataset, val_dataset, test_dataset)`

**Key Feature**: Handles all `sys.path` manipulation and importing automatically!

### DataLoaderFactory

**Purpose**: Creates DataLoaders from any dataset type

**Methods:**
- `create_from_phototriage(data, train_df, val_df, test_df, transform)`
- `create_from_dataframes(train_df, val_df, test_df, image_dir, transform)`
- `create_from_external(train_ds, val_ds, test_ds, ...)`

**Returns:** Always `(train_loader, val_loader, test_loader)` or `(..., ..., None)`

**Key Feature**: All loaders produce identical batch format!

## Benefits

### 1. No Manual Importing

**Before:**
```python
import sys
sys.path.insert(0, r'D:\Projects\Series-Photo-Selection')
from data.dataloader import MyDataset

train_data = MyDataset(train=True, image_root='...')
val_data = MyDataset(train=False, image_root='...')
```

**After:**
```python
# DatasetFactory handles everything!
dataset_factory = DatasetFactory(source='external', config=config)
train_data, val_data, test_data = dataset_factory.create_datasets()
```

### 2. Config-Driven Source Selection

Just change one line in config:
```yaml
data_source: phototriage  # or 'external'
```

### 3. Same Training Code

```python
# This code works with ANY data source!
train_model(model, train_loader, val_loader, optimizer, config, output_dir)
evaluate(model, test_loader, device, output_dir, epoch, 'test')
```

### 4. Clean Architecture

```
Configuration File
      ↓
DatasetFactory (handles imports, creates datasets)
      ↓
DataLoaderFactory (wraps with adapters, creates loaders)
      ↓
Training Code (source-agnostic!)
```

## API Reference

### DatasetFactory

```python
class DatasetFactory:
    def __init__(self, source: str, config: Dict[str, Any])
    def create_datasets(self) -> Tuple
```

**Convenience functions:**
```python
create_phototriage_datasets(root_dir, min_agreement=0.7, ...)
create_external_datasets(external_path, image_root)
create_datasets_from_config(config)
```

### DataLoaderFactory

```python
class DataLoaderFactory:
    def __init__(self, batch_size=16, num_workers=0, shuffle_train=True)

    def create_from_phototriage(data, train_df, val_df, test_df, transform)
    def create_from_dataframes(train_df, val_df, test_df, image_dir, transform)
    def create_from_external(train_ds, val_ds, test_ds, ...)
```

**Utility:**
```python
get_dataset_from_loader(loader)  # Extract dataset for metadata access
```

## Testing

Run comprehensive tests:

```bash
# Test DataLoaderFactory
python examples/test_dataloader_factory.py

# Test integration examples
python examples/train_with_dataset_factory.py
```

## Migration Checklist

- [ ] Add `data_source` to your config files
- [ ] Import `DatasetFactory` and `DataLoaderFactory`
- [ ] Replace manual dataset creation with `DatasetFactory.create_datasets()`
- [ ] Replace manual loader creation with `DataLoaderFactory.create_from_*()`
- [ ] Remove manual `sys.path` manipulation for external datasets
- [ ] Update command-line arguments to support `--data-source`
- [ ] Test with both sources

## Files Reference

**Core Factories:**
- `sim_bench/datasets/dataset_factory.py` - Dataset creation
- `sim_bench/datasets/dataloader_factory.py` - DataLoader creation

**Supporting:**
- `sim_bench/datasets/siamese_dataloaders.py` - Dataset classes & adapter
- `sim_bench/datasets/phototriage_data.py` - PhotoTriage data handling

**Examples:**
- `examples/train_with_dataset_factory.py` - Usage examples
- `examples/test_dataloader_factory.py` - Tests
- `configs/example_phototriage.yaml` - PhotoTriage config
- `configs/example_external.yaml` - External config

## Summary

✅ **DatasetFactory** - Creates datasets from any source (handles importing!)
✅ **DataLoaderFactory** - Creates loaders from any dataset (handles adapters!)
✅ **Config-driven** - Switch sources with one line
✅ **Clean code** - No manual imports or path manipulation
✅ **Same training code** - True interchangeability
✅ **Fully tested** - Real data, no mocks

The two-factory system provides a clean, maintainable solution for working with multiple data sources!
