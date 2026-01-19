# AVA ResNet Telemetry

Gradient and learning rate telemetry for AVA aesthetic score prediction training.

## What is Tracked

### Gradient Norms
Tracks gradient L2 norms for:
- **overall**: Full model gradient norm
- **backbone_layer4**: Last CNN layer (ResNet layer4 - 4th residual block)
- **mlp_head**: MLP head (all layers combined)
- **mlp_layer_N**: Per-layer MLP gradients (if multiple MLP layers)

This helps diagnose:
- Vanishing/exploding gradients
- Which parts of model are learning
- Gradient flow issues

### Learning Rates
Tracks actual learning rates per parameter group:
- **lr_group_0**: Backbone learning rate (usually 1x base_lr)
- **lr_group_1**: MLP head learning rate (usually 10x base_lr)

## Configuration

Enable in your config YAML:

```yaml
telemetry:
  enabled: true                # Enable gradient telemetry
  collect_every_n: 10          # Collect metrics every 10 batches
```

## Output

Telemetry data saved to `<output_dir>/telemetry/`:

### `gradient_norms.csv`
```csv
batch_idx,epoch,overall,backbone_layer4,mlp_head,mlp_layer_0,mlp_layer_3
10,0,2.45,0.82,1.63,0.95,0.68
20,0,2.38,0.79,1.59,0.91,0.68
...
```

### `learning_rates.csv`
```csv
batch_idx,epoch,lr_group_0,lr_group_1
10,0,0.0001,0.001
20,0,0.0001,0.001
...
```

## Usage Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load gradient norms
df = pd.read_parquet('outputs/ava/my_run/telemetry/gradient_norms.csv')

# Plot gradient evolution
plt.figure(figsize=(12, 4))
plt.plot(df['batch_idx'], df['overall'], label='Overall')
plt.plot(df['batch_idx'], df['backbone_layer4'], label='Layer4 (last CNN)')
plt.plot(df['batch_idx'], df['mlp_head'], label='MLP Head')
plt.xlabel('Batch')
plt.ylabel('Gradient Norm')
plt.legend()
plt.title('Gradient Norms Over Training')
plt.show()
```

## Interpretation

**Healthy training:**
- Gradients decrease gradually over time
- All parts (backbone, MLP) have non-zero gradients
- No sudden spikes or drops

**Warning signs:**
- `backbone_layer4` near zero → backbone not learning
- `mlp_head` much larger than `backbone_layer4` → head dominating
- Sudden spikes → numerical instability
- All gradients near zero → model converged or vanishing gradients
- Steadily increasing → potential instability

## Validation Predictions

Validation predictions vs ground truth are saved per epoch when `save_val_predictions: true`:

```
outputs/ava/my_run/predictions/
  val_epoch_000.parquet
  val_epoch_001.parquet
  ...
```

Each file contains:
- `image_id`: Image ID
- `pred_mean`: Predicted mean score (1-10)
- `gt_mean`: Ground truth mean score
- `pred_dist_1` through `pred_dist_10`: Predicted distribution
- `gt_dist_1` through `gt_dist_10`: Ground truth distribution

Load and analyze:
```python
df = pd.read_parquet('outputs/ava/my_run/predictions/val_epoch_010.parquet')
print(df[['image_id', 'pred_mean', 'gt_mean']].head())
```

## Comparison with Siamese E2E Telemetry

AVA telemetry is simpler than Siamese E2E:
- **AVA**: Gradient norms, learning rates only (lightweight)
- **Siamese E2E**: Also tracks weight deltas, holdout predictions, batch stats

AVA focuses on gradient flow diagnostics for aesthetic score prediction.
