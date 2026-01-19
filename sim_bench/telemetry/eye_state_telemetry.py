"""Telemetry collection for eye state classifier training."""

import csv
import math
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EyeStateTelemetryCollector:
    """Collects gradient and weight telemetry for eye state classifier models."""

    def __init__(self, output_dir: Path, collect_every_n: int = 10):
        """
        Initialize telemetry collector.

        Args:
            output_dir: Directory to save telemetry CSVs
            collect_every_n: Collect metrics every N batches
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._collect_every_n = collect_every_n
        logger.info(
            f"Eye state telemetry: output_dir={output_dir}, collect_every_n={collect_every_n}"
        )

    def collect(
        self,
        model: nn.Module,
        optimizer,
        batch_idx: int,
        epoch: int,
        device: str,
    ):
        """
        Collect gradient norms and learning rates after optimizer.step().

        Args:
            model: EyeStateClassifier model
            optimizer: PyTorch optimizer
            batch_idx: Current batch index (1-indexed)
            epoch: Current epoch number
            device: Device string
        """
        if batch_idx % self._collect_every_n != 0:
            return

        metadata = {'batch_idx': batch_idx, 'epoch': epoch}

        grad_norms = self._compute_gradient_norms(model)
        self._append_to_csv('gradient_norms.csv', {**metadata, **grad_norms})

        lr_info = self._get_learning_rates(optimizer)
        self._append_to_csv('learning_rates.csv', {**metadata, **lr_info})

        weight_stats = self._compute_weight_stats(model)
        self._append_to_csv('weight_stats.csv', {**metadata, **weight_stats})

    def _compute_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient norms for different parts of the model.

        Returns:
            Dictionary with gradient norms for backbone and head
        """
        overall = 0.0
        backbone_norm = 0.0
        head_norm = 0.0
        first_conv_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_l2 = torch.norm(param.grad).item()
            overall += grad_l2 ** 2

            if 'backbone' in name:
                backbone_norm += grad_l2 ** 2
                if 'conv1' in name:
                    first_conv_norm += grad_l2 ** 2

            if 'head' in name:
                head_norm += grad_l2 ** 2

        return {
            'overall': math.sqrt(overall),
            'backbone': math.sqrt(backbone_norm),
            'head': math.sqrt(head_norm),
            'first_conv': math.sqrt(first_conv_norm),
        }

    def _compute_weight_stats(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute weight statistics for monitoring training health.

        Returns:
            Dictionary with weight norms and means
        """
        head_weight_norm = 0.0
        head_weight_mean = 0.0
        head_param_count = 0

        for name, param in model.named_parameters():
            if 'head' in name and 'weight' in name:
                head_weight_norm += torch.norm(param).item() ** 2
                head_weight_mean += param.mean().item()
                head_param_count += 1

        return {
            'head_weight_norm': math.sqrt(head_weight_norm),
            'head_weight_mean': head_weight_mean / max(head_param_count, 1),
        }

    def _get_learning_rates(self, optimizer) -> Dict[str, float]:
        """Extract learning rates from optimizer."""
        result = {}
        for i, param_group in enumerate(optimizer.param_groups):
            result[f'lr_group_{i}'] = param_group['lr']
        return result

    def _append_to_csv(self, filename: str, row_dict: Dict[str, Any]):
        """Append row to CSV, creating with header if needed."""
        filepath = self._output_dir / filename
        file_exists = filepath.exists()

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)


# Module-level singleton
_telemetry_collector = None


def init_telemetry(config: dict):
    """
    Initialize eye state telemetry from config.

    Args:
        config: Full training config dict
    """
    global _telemetry_collector

    telemetry_cfg = config.get('telemetry', {})
    if not telemetry_cfg.get('enabled', False):
        _telemetry_collector = None
        return

    output_dir = Path(config['output_dir']) / 'telemetry'
    collect_every_n = telemetry_cfg.get('collect_every_n', 10)

    _telemetry_collector = EyeStateTelemetryCollector(output_dir, collect_every_n)


def collect_telemetry(
    model,
    optimizer,
    batch_idx: int,
    epoch: int,
    device: str,
    config: dict,
):
    """
    Collect telemetry for current training step.

    No-op if telemetry not initialized.
    """
    if _telemetry_collector is not None:
        _telemetry_collector.collect(model, optimizer, batch_idx, epoch, device)
