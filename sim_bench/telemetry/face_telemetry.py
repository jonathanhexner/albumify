"""Telemetry collection for Face ResNet training with ArcFace."""

import csv
import math
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class FaceTelemetryCollector:
    """Collects gradient, weight, and training telemetry for Face ResNet models."""

    def __init__(self, output_dir: Path, collect_every_n: int = 10,
                 track_embeddings: bool = False, track_arcface_weights: bool = True):
        """
        Initialize telemetry collector.

        Args:
            output_dir: Directory to save telemetry CSVs
            collect_every_n: Collect metrics every N batches
            track_embeddings: Whether to track embedding statistics
            track_arcface_weights: Whether to track ArcFace weight statistics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.collect_every_n = collect_every_n
        self.track_embeddings = track_embeddings
        self.track_arcface_weights = track_arcface_weights

        # Store weight snapshots for delta computation
        self._prev_weights: Optional[Dict[str, torch.Tensor]] = None

        logger.info(f"Face telemetry: output_dir={output_dir}, collect_every_n={collect_every_n}")

    def collect(self, model: nn.Module, optimizer, batch_idx: int, epoch: int,
                device: str, embeddings: torch.Tensor = None, labels: torch.Tensor = None):
        """
        Collect telemetry after optimizer.step().

        Args:
            model: FaceResNet model
            optimizer: PyTorch optimizer
            batch_idx: Current batch index (1-indexed)
            epoch: Current epoch number
            device: Device string
            embeddings: Optional embeddings tensor for statistics
            labels: Optional labels for per-class analysis
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

        # Weight deltas (change since last collection)
        weight_deltas = self._compute_weight_deltas(model)
        if weight_deltas:
            self._append_to_csv('weight_deltas.csv', {**metadata, **weight_deltas})

        # Embedding statistics
        if self.track_embeddings and embeddings is not None:
            embed_stats = self._compute_embedding_stats(embeddings)
            self._append_to_csv('embedding_stats.csv', {**metadata, **embed_stats})

        # ArcFace weight statistics
        if self.track_arcface_weights:
            arcface_stats = self._compute_arcface_stats(model)
            self._append_to_csv('arcface_stats.csv', {**metadata, **arcface_stats})

    def _compute_gradient_norms(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient norms for different parts of the model.

        Returns:
            Dictionary with gradient norms for overall, backbone, embedding, and arcface.
        """
        overall = 0.0
        backbone_norm = 0.0
        embedding_norm = 0.0
        arcface_norm = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_l2 = torch.norm(param.grad).item()
            overall += grad_l2 ** 2

            if 'backbone' in name:
                backbone_norm += grad_l2 ** 2
            elif 'embedding' in name:
                embedding_norm += grad_l2 ** 2
            elif 'arcface' in name:
                arcface_norm += grad_l2 ** 2

        return {
            'overall': math.sqrt(overall),
            'backbone': math.sqrt(backbone_norm),
            'embedding': math.sqrt(embedding_norm),
            'arcface': math.sqrt(arcface_norm),
        }

    def _get_learning_rates(self, optimizer) -> Dict[str, float]:
        """Extract learning rates from optimizer."""
        result = {}
        for i, param_group in enumerate(optimizer.param_groups):
            result[f'lr_group_{i}'] = param_group['lr']
        return result

    def _compute_weight_deltas(self, model: nn.Module) -> Dict[str, float]:
        """Compute L2 norm of weight changes since last collection."""
        current_weights = {}
        for name, param in model.named_parameters():
            current_weights[name] = param.data.clone()

        if self._prev_weights is None:
            self._prev_weights = current_weights
            return {}

        deltas = {}
        backbone_delta = 0.0
        embedding_delta = 0.0
        arcface_delta = 0.0

        for name in current_weights:
            if name in self._prev_weights:
                diff = (current_weights[name] - self._prev_weights[name]).norm().item()
                if 'backbone' in name:
                    backbone_delta += diff ** 2
                elif 'embedding' in name:
                    embedding_delta += diff ** 2
                elif 'arcface' in name:
                    arcface_delta += diff ** 2

        deltas['backbone_delta'] = math.sqrt(backbone_delta)
        deltas['embedding_delta'] = math.sqrt(embedding_delta)
        deltas['arcface_delta'] = math.sqrt(arcface_delta)

        self._prev_weights = current_weights
        return deltas

    def _compute_embedding_stats(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute statistics of embedding vectors."""
        with torch.no_grad():
            # L2 norm statistics
            norms = torch.norm(embeddings, p=2, dim=1)

            # Inter-sample cosine similarity (sample subset for efficiency)
            if embeddings.size(0) > 1:
                emb_norm = embeddings / norms.unsqueeze(1)
                # Sample pairs for large batches
                n = min(embeddings.size(0), 100)
                cos_sim = torch.mm(emb_norm[:n], emb_norm[:n].t())
                # Exclude diagonal (self-similarity)
                mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
                mean_cos_sim = cos_sim[mask].mean().item()
            else:
                mean_cos_sim = 0.0

            return {
                'norm_mean': norms.mean().item(),
                'norm_std': norms.std().item(),
                'norm_min': norms.min().item(),
                'norm_max': norms.max().item(),
                'mean_cos_similarity': mean_cos_sim
            }

    def _compute_arcface_stats(self, model: nn.Module) -> Dict[str, float]:
        """Compute statistics of ArcFace weight matrix."""
        stats = {}

        for name, param in model.named_parameters():
            if 'arcface' in name and 'weight' in name:
                with torch.no_grad():
                    # Weight matrix statistics
                    stats['weight_mean'] = param.mean().item()
                    stats['weight_std'] = param.std().item()

                    # Compute weight norms (per-class)
                    weight_norms = torch.norm(param, p=2, dim=1)
                    stats['weight_norm_mean'] = weight_norms.mean().item()
                    stats['weight_norm_std'] = weight_norms.std().item()

        return stats

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
_telemetry_collector: Optional[FaceTelemetryCollector] = None


def init_telemetry(config: dict):
    """
    Initialize Face telemetry from config.

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
    track_embeddings = telemetry_cfg.get('track_embeddings', False)
    track_arcface_weights = telemetry_cfg.get('track_arcface_weights', True)

    _telemetry_collector = FaceTelemetryCollector(
        output_dir=output_dir,
        collect_every_n=collect_every_n,
        track_embeddings=track_embeddings,
        track_arcface_weights=track_arcface_weights
    )


def collect_telemetry(model, optimizer, batch_idx: int, epoch: int, device: str,
                      config: dict, embeddings: torch.Tensor = None, labels: torch.Tensor = None):
    """
    Collect telemetry for current training step.

    No-op if telemetry not initialized.
    """
    if _telemetry_collector is not None:
        _telemetry_collector.collect(model, optimizer, batch_idx, epoch, device, embeddings, labels)
