# DataLoader Refactoring - Final Solution

## Summary

Successfully refactored the training code to support **easy switching** between PhotoTriage and external dataloaders with a **single config flag**.

## The Solution: ONE Simple Function

```python
def create_dataloaders(config, transform, batch_size):
    """
    Create dataloaders - supports both PhotoTriage and external sources.

    Just set config['use_external_dataloader'] = True to switch sources!
    """
```

## Usage

### PhotoTriage Data (Default)

```yaml
# config.yaml
use_external_dataloader: false  # or omit this line

data:
  root_dir: data/phototriage
  min_agreement: 0.7
  min_reviewers: 2
```

### External Data (Series-Photo-Selection)

```yaml
# config.yaml
use_external_dataloader: true

data:
  # Optional: path to Series-Photo-Selection project (default: D:\Projects\Series-Photo-Selection)
  external_path: D:\Projects\Series-Photo-Selection
  # REQUIRED: path to train_val_imgs directory
  image_root: D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs

  # Or if root_dir points to the base directory, image_root can be omitted and will use root_dir
  # root_dir: D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs
```

## Training Script Changes

### Before (Complex)
```python
# Load data
data, train_df, val_df, test_df = load_data(config)

# Create model and transform
model = create_model(config, output_dir)
transform = model.preprocess

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    train_df, val_df, test_df, data, transform, batch_size
)
```

### After (Simple)
```python
# Create model and transform
model = create_model(config, output_dir)
transform = model.preprocess

# Create dataloaders (handles data loading internally)
train_loader, val_loader, test_loader = create_dataloaders(
    config, transform, config['training']['batch_size']
)
```

## Key Files Modified

1. **[sim_bench/training/train_siamese_e2e.py](sim_bench/training/train_siamese_e2e.py)**
   - Simplified `create_dataloaders()` function (lines 246-299)
   - Removed separate `load_data()` function
   - Updated `main()` to use simplified interface (lines 465-468)

2. **[sim_bench/datasets/siamese_dataloaders.py](sim_bench/datasets/siamese_dataloaders.py)** (Created)
   - `EndToEndPairDataset` class for PhotoTriage
   - `ExternalDatasetAdapter` to wrap external datasets
   - Metadata support via `get_dataframe()`, `get_image_dir()`, `transform` properties

3. **[sim_bench/datasets/dataloader_factory.py](sim_bench/datasets/dataloader_factory.py)** (Created)
   - `DataLoaderFactory` class with methods:
     - `create_from_phototriage()`
     - `create_from_external()`

4. **[sim_bench/training/diagnostics.py](sim_bench/training/diagnostics.py)** (Modified)
   - Updated to accept `dataset` instead of `pairs_df`
   - Extracts DataFrame internally when needed

## Data Format Differences Handled

| Aspect | PhotoTriage | External (Series-Photo-Selection) |
|--------|-------------|-----------------------------------|
| **Return format** | Dict: `{'img1': ..., 'img2': ..., 'winner': ...}` | Tuple: `(imageA, imageB, winner)` |
| **Transform** | Passed externally | Built-in (applied in `__getitem__`) |
| **Metadata** | Full DataFrame with series_id, agreement, etc. | Auto-generated defaults |

The `ExternalDatasetAdapter` handles conversion automatically.

## Benefits

1. **One config flag** to switch data sources
2. **No code changes** in training script
3. **Automatic imports** - no manual sys.path manipulation needed
4. **Clean abstraction** - implementation details hidden
5. **Same training code** for both sources

## Example Configs

See:
- [configs/example_phototriage.yaml](configs/example_phototriage.yaml)
- [configs/example_external.yaml](configs/example_external.yaml)

## Testing

Run tests:
```bash
python examples/test_dataloader_factory.py
```

## Important Notes

- **External dataloader already applies transforms** in its `__getitem__` method
- Transform parameter is only used for PhotoTriage data
- External data gets default metadata: `series_id='unknown'`, `agreement=1.0`, `num_reviewers=1`
