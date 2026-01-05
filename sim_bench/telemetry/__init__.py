"""Training Telemetry System - Module-level singleton."""

from sim_bench.telemetry.config import TelemetryConfig
from sim_bench.telemetry.training_telemetry import TrainingTelemetry

__all__ = ['TelemetryConfig', 'TrainingTelemetry', 'init', 'record']

# Module-level singleton
_telemetry_instance = None


def init(config, val_loader):
    """Initialize telemetry from config.

    Args:
        config: Full config dict (must include output_dir)
        val_loader: Validation dataloader for holdout data
    """
    global _telemetry_instance

    if not config.get('telemetry', {}).get('enabled', False):
        _telemetry_instance = None
        return

    telemetry_dict = config['telemetry'].copy()
    telemetry_dict['output_dir'] = config['output_dir'] / 'telemetry'
    telemetry_config = TelemetryConfig.from_dict(telemetry_dict)

    _telemetry_instance = TrainingTelemetry(telemetry_config)

    # Setup holdout data
    holdout_data = []
    holdout_iter = iter(val_loader)
    for i in range(telemetry_config.holdout_size):
        batch_data = next(holdout_iter, None)
        if batch_data is None:
            break
        holdout_data.append(batch_data)
    _telemetry_instance.set_holdout_data(holdout_data)


def record(model, optimizer, batch_idx, epoch, device, batch):
    """Record metrics for current training step.

    No-op if telemetry not initialized or disabled.
    """
    if _telemetry_instance is not None:
        _telemetry_instance.on_batch_end(model, optimizer, batch_idx, epoch, device, batch)
