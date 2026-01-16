# Training Telemetry System - Architecture

## Purpose

Simple, lightweight telemetry system for tracking training metrics to debug model training issues. Specifically designed to help diagnose overfitting (train acc improving, val stuck at 50%).

## Design Principles

- **Simple**: Single-class implementation, ~200 lines of code
- **Lightweight**: Minimal memory overhead, disk-based checkpointing
- **Non-invasive**: Does not modify training sequence or logic
- **Configurable**: Enable/disable via config, adjust collection frequency
- **PyTorch-specific**: Optimized for PyTorch models (not framework-agnostic)

## Tracked Metrics

### 1. Gradient Norm

- **Overall gradient norm**: L2 norm across all parameters
- **Per-group gradient norm**: Separate norms for backbone vs head (differential LR)
- **Purpose**: Detect vanishing/exploding gradients, verify backpropagation

### 2. Weight Delta

- **Overall weight change**: L2 norm of parameter changes since last checkpoint
- **Per-group weight change**: Separate deltas for backbone vs head
- **Purpose**: Verify model is actually learning, detect frozen layers

### 3. Learning Rates

- **Actual LR per parameter group**: Current learning rate values
- **Purpose**: Verify differential LR is correctly applied (1x backbone, 10x head)

### 4. Holdout Logits

- **Model predictions on fixed validation subset**: 50 pairs tracked every N batches
- **Includes**: logits, labels, predictions, accuracy
- **Purpose**: Track how predictions evolve on same examples, detect overfitting patterns

### 5. Batch Statistics

- **Winner distribution**: Percentage of winner=0 vs winner=1 in batch
- **Batch size**: Number of samples
- **Purpose**: Detect data imbalance, verify dataset is properly shuffled

## Architecture Overview

**Single class**: `TrainingTelemetry`

```
TrainingTelemetry
├── Collect metrics every N batches
├── Save to JSONL file (append-only)
├── Use disk for weight checkpointing (no memory overhead)
└── Simple method-based API
```

**No inheritance, no interfaces, no factories** - just one class with helper methods.

## Implementation

### TrainingTelemetry Class

**File**: `telemetry/training_telemetry.py`

Single class that handles everything.

```python
@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection."""
    enabled: bool = True
    collect_every_n: int = 200
    output_dir: Optional[Path] = None

    # Metric-specific settings
    track_gradients: bool = True
    track_weight_delta: bool = True
    track_learning_rates: bool = True
    track_holdout_logits: bool = True
    track_batch_stats: bool = True

    # Holdout settings
    holdout_size: int = 50

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'TelemetryConfig':
        """Create from config dictionary (e.g., from YAML)."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})


class TrainingTelemetry:
    """Simple telemetry collector for PyTorch training."""

    def __init__(self, config: TelemetryConfig):
        """
        Initialize telemetry.

        Args:
            config: TelemetryConfig object
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.holdout_data = None
        self.metrics_file = self.output_dir / 'telemetry.jsonl'
        self.checkpoint_file = self.output_dir / 'last_checkpoint.pt'

    def set_holdout_data(self, holdout_data):
        """Set fixed validation subset for tracking."""
        self.holdout_data = holdout_data

    def on_batch_end(self, model, optimizer, batch_idx, epoch, device, batch=None):
        """
        Collect and save metrics.

        Call this after optimizer.step() in your training loop.
        """
        if not self.config.enabled or batch_idx % self.config.collect_every_n != 0:
            return

        metrics = {
            'batch_idx': batch_idx,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
        }

        # Conditionally add metrics based on config
        if self.config.track_gradients:
            metrics['gradient_norm'] = self._compute_gradient_norm(model)

        if self.config.track_weight_delta:
            metrics['weight_delta'] = self._compute_weight_delta(model)

        if self.config.track_learning_rates:
            metrics['learning_rates'] = self._get_learning_rates(optimizer)

        if self.config.track_holdout_logits and self.holdout_data is not None:
            metrics['holdout_logits'] = self._eval_holdout(model, device)

        if self.config.track_batch_stats and batch is not None:
            metrics['batch_stats'] = self._compute_batch_stats(batch)

        # Append to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def _compute_gradient_norm(self, model) -> Dict[str, float]:
        """Compute gradient norms (overall + per-group)."""
        # Implementation...

    def _compute_weight_delta(self, model) -> Dict[str, float]:
        """Compute weight changes since last checkpoint (disk-based)."""
        # Implementation...

    def _get_learning_rates(self, optimizer) -> Dict[str, float]:
        """Get current learning rates per parameter group."""
        # Implementation...

    def _eval_holdout(self, model, device) -> Dict[str, Any]:
        """Evaluate model on fixed holdout set."""
        # Implementation...

    def _compute_batch_stats(self, batch) -> Dict[str, float]:
        """Compute batch statistics."""
        # Implementation...
```

## Metrics Collected

### 1. Gradient Norm

Computes gradient norms (overall + per parameter group).

**Output**:

```json
{
  "overall": 1.234,
  "backbone": 0.456,
  "head": 0.778
}
```

### 2. Weight Delta

Computes parameter changes since last checkpoint (disk-based).

**Implementation**: Saves model checkpoint every N batches, computes L2 distance from previous checkpoint.

**Output**:

```json
{
  "overall": 0.0123,
  "backbone": 0.0045,
  "head": 0.0078
}
```

**Note**: No memory overhead - checkpoint saved to disk only.

### 3. Learning Rates

Tracks actual learning rates per parameter group.

**Output**:

```json
{
  "group_0": 0.00001,
  "group_1": 0.0001,
  "backbone": 0.00001,
  "head": 0.0001
}
```

### 4. Holdout Logits

Tracks model predictions on fixed validation subset (first 50 validation pairs).

**Output**:

```json
{
  "logits": [[0.3, 0.7], [0.6, 0.4], ...],
  "labels": [1, 0, ...],
  "predictions": [1, 0, ...],
  "accuracy": 0.54
}
```

### 5. Batch Statistics

Tracks batch-level statistics (data balance).

**Output**:

```json
{
  "winner_0_pct": 48.5,
  "winner_1_pct": 51.5,
  "batch_size": 8
}
```

## Storage Format

### JSONL (JSON Lines)

All metrics saved to single file: `telemetry.jsonl`

**Format**:

```jsonl
{"batch_idx": 200, "epoch": 0, "gradient_norm": {...}, "weight_delta": {...}, ...}
{"batch_idx": 400, "epoch": 0, "gradient_norm": {...}, "weight_delta": {...}, ...}
```

**Advantages**:

- Human-readable
- Easy to parse line-by-line
- Append-only (no file locking issues)
- Simple: no HDF5 dependency

## Integration Example

### Setup in Training Script

```python
from sim_bench.telemetry import TrainingTelemetry, TelemetryConfig

# Create config
telemetry_config = TelemetryConfig(
    enabled=True,
    collect_every_n=200,
    output_dir=output_dir / 'telemetry',
    track_gradients=True,
    track_weight_delta=True,
    track_learning_rates=True,
    track_holdout_logits=True,
    track_batch_stats=True,
    holdout_size=50
)

# Or load from YAML config
# telemetry_config = TelemetryConfig.from_dict(config['telemetry'])

# Create telemetry
telemetry = TrainingTelemetry(telemetry_config)

# Setup holdout data (first 50 validation pairs)
holdout_data = []
for i, batch in enumerate(val_loader):
    holdout_data.append(batch)
    if i >= telemetry_config.holdout_size:
        break
telemetry.set_holdout_data(holdout_data)
```

### Hook in Training Loop

```python
def train_epoch(model, optimizer, loader, device, telemetry=None, epoch=0):
    for batch_idx, batch in enumerate(loader, 1):
        # Forward pass
        outputs = model(batch['img1'], batch['img2'])
        loss = compute_loss(outputs, batch['winner'])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TELEMETRY HOOK - Simple one-liner
        if telemetry:
            telemetry.on_batch_end(model, optimizer, batch_idx, epoch, device, batch)
```

## Configuration

### YAML Config

```yaml
telemetry:
  enabled: true
  collect_every_n: 200
  output_dir: null  # Will be set programmatically

  # Enable/disable individual metrics
  track_gradients: true
  track_weight_delta: true
  track_learning_rates: true
  track_holdout_logits: true
  track_batch_stats: true

  # Holdout settings
  holdout_size: 50
```

**Benefits of config object**:

- ✅ Easy to extend (add new fields without breaking constructor)
- ✅ Easy to serialize/deserialize (YAML ↔ dataclass)
- ✅ Type hints and validation
- ✅ Can disable individual metrics without changing code

## Output Structure

```
outputs/siamese_e2e/YYYYMMDD_HHMMSS/
├── telemetry/
│   ├── telemetry.jsonl          # All scalar metrics
│   └── last_checkpoint.pt       # Weight checkpoint for delta computation
└── ... (other training outputs)
```

## Analysis Example

```python
import json
import matplotlib.pyplot as plt

# Load telemetry data
metrics = []
with open('outputs/run_xyz/telemetry/telemetry.jsonl') as f:
    for line in f:
        metrics.append(json.loads(line))

# Extract gradient norms
batch_indices = [m['batch_idx'] for m in metrics]
grad_norms = [m['gradient_norm']['overall'] for m in metrics]

# Plot
plt.plot(batch_indices, grad_norms)
plt.xlabel('Batch Index')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm Over Training')
plt.savefig('grad_norm_plot.png')
```

## Extension Points

### Adding New Metrics

Simply add a new method to `TrainingTelemetry`:

```python
def _compute_my_metric(self, model) -> Dict[str, Any]:
    """Compute my custom metric."""
    return {'value': 42}
```

And add it to the metrics dict in `on_batch_end()`:

```python
metrics = {
    ...
    'my_metric': self._compute_my_metric(model)
}
```

## Performance Considerations

- **Collection frequency**: Every 200 batches (configurable)
- **Storage overhead**: ~5KB per checkpoint for scalars in JSONL
- **Memory overhead**: Zero (weight checkpoints saved to disk only)
- **Computation overhead**: Minimal - O(num_parameters) only every 200 batches
- **Disk overhead**: ~100MB for model checkpoint file (updated every 200 batches)
- **Logits storage**: Stored as JSON arrays, ~1KB per checkpoint (50 pairs)

## Design Benefits

✅ **Simple**: Single class, easy to understand and debug
✅ **Lightweight**: No memory overhead, minimal disk I/O
✅ **Non-invasive**: One-line hook in training loop
✅ **Easy to extend**: Just add methods
✅ **Easy to analyze**: JSONL format, simple Python parsing

## Troubleshooting

### Metrics not being collected

- Check `telemetry.enabled = True`
- Verify `collect_every_n` alignment with batch count
- Check logs for errors in metric computation

### Storage file not created

- Verify output directory exists and is writable
- Check for exceptions in file writing

### High memory usage

- Reduce `collect_every_n` value
- Disable `holdout_logits` if not needed

## Future Enhancements

- [ ] Plotting utilities (built-in visualization scripts)
- [ ] Comparison between multiple runs
- [ ] Export to pandas DataFrame helper
- [ ] Optional TensorBoard logging
