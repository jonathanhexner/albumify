"""Configuration for telemetry system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection.

    Attributes:
        enabled: Whether telemetry collection is enabled
        collect_every_n: Collect metrics every N batches
        output_dir: Directory to save telemetry data (None = auto-determined)

        # Metric-specific settings
        track_gradients: Track gradient norms (overall + per-group)
        track_weight_delta: Track weight changes since last checkpoint
        track_learning_rates: Track actual learning rates per param group
        track_holdout_logits: Track predictions on fixed validation subset
        track_batch_stats: Track batch-level statistics (data balance)

        # Holdout settings
        holdout_size: Number of validation pairs to track
    """
    enabled: bool = True
    collect_every_n: int = 10
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TelemetryConfig':
        """Create TelemetryConfig from dictionary (e.g., from YAML).

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            TelemetryConfig instance
        """
        # Filter to only include valid fields
        valid_fields = {k: v for k, v in config_dict.items() if k in cls.__annotations__}

        # Convert output_dir to Path if provided as string
        if 'output_dir' in valid_fields and valid_fields['output_dir'] is not None:
            valid_fields['output_dir'] = Path(valid_fields['output_dir'])

        return cls(**valid_fields)
