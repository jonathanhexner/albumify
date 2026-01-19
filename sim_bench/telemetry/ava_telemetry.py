"""Telemetry collection for AVA ResNet training."""

import csv
import math
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AVATelemetryCollector:
    """Collects gradient and weight telemetry for AVA ResNet models."""

    def __init__(self, output_dir: Path, collect_every_n: int = 10):
        """
        Initialize telemetry collector.

        Args:
            output_dir: Directory to save telemetry CSVs
            collect_every_n: Collect metrics every N batches
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collect_every_n = collect_every_n
        logger.info(f"AVA telemetry: output_dir={output_dir}, collect_every_n={collect_every_n}")

    def collect(self, model: nn.Module, optimizer, batch_idx: int, epoch: int, device: str):
        """
        Collect gradient norms and learning rates after optimizer.step().

        Args:
            model: AVAResNet model
            optimizer: PyTorch optimizer
            batch_idx: Current batch index (1-indexed)
            epoch: Current epoch number
            device: Device string
        """
        if batch_idx % self.collect_every_n != 0:
            return

        metadata = {'batch_idx': batch_idx, 'epoch': epoch}

        # Gradient norms
        grad_norms = self._compute_gradient_norms(model)
        self._append_to_csv('gradient_norms.csv', {**metadata, **grad_norms})

        # Learning rates
        lr_info = self._get_learning_rates(optimizer)
        self._append_to_csv('learning_rates.csv', {**metadata, **lr_info})

    def _compute_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient norms for different parts of the model.

        Returns:
            Dictionary with:
            - overall: Full model gradient norm
            - backbone_layer4: Last CNN layer (ResNet layer4)
            - mlp_head: MLP head layers
            - mlp_layer_N: Per-layer MLP gradients (if multiple layers)
        """
        overall = 0.0
        layer4_norm = 0.0
        mlp_norm = 0.0
        mlp_layers = {}  # Per-layer norms

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_l2 = torch.norm(param.grad).item()
            overall += grad_l2 ** 2

            # Last CNN layer (layer4 in ResNet50)
            if 'backbone' in name and 'layer4' in name:
                layer4_norm += grad_l2 ** 2

            # MLP head (breakdown by layer)
            if 'head' in name:
                mlp_norm += grad_l2 ** 2

                # Extract layer number from name (e.g., 'head.0.weight' -> layer 0)
                if '.' in name:
                    parts = name.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        key = f'mlp_layer_{layer_idx}'
                        if key not in mlp_layers:
                            mlp_layers[key] = 0.0
                        mlp_layers[key] += grad_l2 ** 2

        result = {
            'overall': math.sqrt(overall),
            'backbone_layer4': math.sqrt(layer4_norm),
            'mlp_head': math.sqrt(mlp_norm),
        }

        # Add per-layer MLP norms
        for key, val in mlp_layers.items():
            result[key] = math.sqrt(val)

        return result

    def _get_learning_rates(self, optimizer) -> Dict[str, float]:
        """Extract learning rates from optimizer."""
        result = {}
        for i, param_group in enumerate(optimizer.param_groups):
            result[f'lr_group_{i}'] = param_group['lr']
        return result

    def _append_to_csv(self, filename: str, row_dict: Dict[str, Any]):
        """Append row to CSV, creating with header if needed."""
        filepath = self.output_dir / filename
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
    Initialize AVA telemetry from config.

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

    _telemetry_collector = AVATelemetryCollector(output_dir, collect_every_n)


def collect_telemetry(model, optimizer, batch_idx: int, epoch: int, device: str, config: dict):
    """
    Collect telemetry for current training step.

    No-op if telemetry not initialized.
    """
    if _telemetry_collector is not None:
        _telemetry_collector.collect(model, optimizer, batch_idx, epoch, device)
