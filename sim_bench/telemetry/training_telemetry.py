"""Simple telemetry collector for PyTorch training."""

import csv
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import torch

from sim_bench.telemetry.config import TelemetryConfig

logger = logging.getLogger(__name__)


class TrainingTelemetry:
    """Simple telemetry collector for PyTorch training.

    Collects metrics every N batches and saves to CSV files.
    Uses disk-based checkpointing for weight delta (no memory overhead).
    """

    def __init__(self, config: TelemetryConfig):
        """Initialize telemetry.

        Args:
            config: TelemetryConfig object with all settings
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.holdout_data = None
        self.holdout_filenames = []
        self.checkpoint_file = self.output_dir / 'last_checkpoint.pt'

        logger.info(f"Telemetry initialized: output_dir={self.output_dir}, "
                   f"collect_every_n={config.collect_every_n}")

    def set_holdout_data(self, holdout_data: List[Dict[str, torch.Tensor]]):
        """Set fixed validation subset for tracking.

        Args:
            holdout_data: List of batches with 'image1', 'image2' filename keys
        """
        self.holdout_data = holdout_data
        self.holdout_filenames = []

        # Extract filenames from each batch
        for batch in holdout_data:
            if 'image1' in batch and 'image2' in batch:
                for i in range(len(batch['image1'])):
                    self.holdout_filenames.append({
                        'image1': batch['image1'][i],
                        'image2': batch['image2'][i]
                    })

        logger.info(f"Holdout data set: {len(holdout_data)} batches, "
                   f"{len(self.holdout_filenames)} pairs")

    def on_batch_end(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_idx: int,
        epoch: int,
        device: str,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Collect and save metrics after optimizer.step().

        Call this after optimizer.step() in your training loop.

        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            batch_idx: Current batch index (1-indexed)
            epoch: Current epoch number
            device: Device string ('cuda' or 'cpu')
            batch: Current batch data (for batch stats)
        """
        if not self.config.enabled:
            return

        if batch_idx % self.config.collect_every_n != 0:
            return

        # Common metadata for all CSVs
        metadata = {
            'batch_idx': batch_idx,
            'epoch': epoch,
        }

        # Compute and write each metric to its own CSV
        if self.config.track_gradients:
            gradient_norm = self._compute_gradient_norm(model)
            self._append_to_csv('gradient_norms.csv', {
                **metadata,
                **gradient_norm
            })

        if self.config.track_weight_delta:
            weight_delta = self._compute_weight_delta(model)
            self._append_to_csv('weight_deltas.csv', {
                **metadata,
                **weight_delta
            })

        if self.config.track_learning_rates:
            learning_rates = self._get_learning_rates(optimizer)
            self._append_to_csv('learning_rates.csv', {
                **metadata,
                **learning_rates
            })

        if self.config.track_batch_stats and batch is not None:
            batch_stats = self._compute_batch_stats(batch)
            self._append_to_csv('batch_stats.csv', {
                **metadata,
                **batch_stats
            })

        if self.config.track_holdout_logits and self.holdout_data is not None:
            holdout_results = self._eval_holdout(model, device)
            # Write each prediction as a separate row
            for pred in holdout_results['predictions']:
                self._append_to_csv('holdout_predictions.csv', {
                    **metadata,
                    **pred
                })

        logger.debug(f"Telemetry collected at batch {batch_idx}")

    def _append_to_csv(self, filename: str, row_dict: Dict[str, Any]):
        """Append row to CSV file, creating with header if needed."""
        filepath = self.output_dir / filename
        file_exists = filepath.exists()

        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_dict)

    def _compute_gradient_norm(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute gradient norms (overall + per-group).

        Args:
            model: PyTorch model

        Returns:
            Dictionary with 'overall', 'backbone', 'head' gradient norms
        """
        grad_norm_overall = 0.0
        grad_norm_backbone = 0.0
        grad_norm_head = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norm_overall += grad_norm ** 2

                # Separate by parameter group (backbone vs head)
                if 'backbone' in name:
                    grad_norm_backbone += grad_norm ** 2
                elif 'mlp' in name or 'head' in name:
                    grad_norm_head += grad_norm ** 2

        return {
            'overall': math.sqrt(grad_norm_overall),
            'backbone': math.sqrt(grad_norm_backbone),
            'head': math.sqrt(grad_norm_head)
        }

    def _compute_weight_delta(self, model: torch.nn.Module) -> Dict[str, float]:
        """Compute weight changes since last checkpoint (disk-based).

        Saves current weights to disk and computes L2 distance from previous checkpoint.
        No memory overhead - uses disk for storage.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with 'overall', 'backbone', 'head' weight deltas
        """
        # Load previous checkpoint if it exists
        if self.checkpoint_file.exists():
            prev_state = torch.load(self.checkpoint_file)

            delta_overall = 0.0
            delta_backbone = 0.0
            delta_head = 0.0

            for name, param in model.named_parameters():
                if name in prev_state:
                    prev = prev_state[name]
                    curr = param.detach().cpu()
                    delta = torch.norm(curr - prev).item()

                    delta_overall += delta ** 2

                    # Separate by parameter group
                    if 'backbone' in name:
                        delta_backbone += delta ** 2
                    elif 'mlp' in name or 'head' in name:
                        delta_head += delta ** 2

            result = {
                'overall': math.sqrt(delta_overall),
                'backbone': math.sqrt(delta_backbone),
                'head': math.sqrt(delta_head)
            }
        else:
            # First checkpoint - no previous state to compare
            result = {
                'overall': 0.0,
                'backbone': 0.0,
                'head': 0.0
            }

        # Save current checkpoint for next comparison
        checkpoint = {name: param.detach().cpu().clone()
                     for name, param in model.named_parameters()}
        torch.save(checkpoint, self.checkpoint_file)

        return result

    def _get_learning_rates(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Get current learning rates per parameter group.

        Args:
            optimizer: PyTorch optimizer

        Returns:
            Dictionary with learning rates (group_0, group_1, backbone, head)
        """
        lrs = {}

        for i, param_group in enumerate(optimizer.param_groups):
            lrs[f'group_{i}'] = param_group['lr']

        # Also add named groups if we have exactly 2 groups (backbone + head)
        if len(lrs) == 2:
            lrs['backbone'] = lrs['group_0']
            lrs['head'] = lrs['group_1']

        return lrs

    def _eval_holdout(
        self,
        model: torch.nn.Module,
        device: str
    ) -> Dict[str, Any]:
        """Evaluate model on fixed holdout set.

        Args:
            model: PyTorch model
            device: Device string

        Returns:
            Dictionary with logits, labels, predictions, accuracy
        """
        if self.holdout_data is None:
            return {'error': 'No holdout data provided'}

        # Save training mode and switch to eval
        was_training = model.training
        model.eval()

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for i, batch in enumerate(self.holdout_data):
                if i >= self.config.holdout_size:
                    break

                img1 = batch['img1'].to(device)
                img2 = batch['img2'].to(device)
                labels = batch['winner']

                logits = model(img1, img2)

                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())

        # Restore training mode
        if was_training:
            model.train()

        # Concatenate results
        logits_array = np.concatenate(all_logits, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        predictions = np.argmax(logits_array, axis=1)

        # Build per-row results with filenames
        results = []
        for i in range(len(logits_array)):
            results.append({
                'image1': self.holdout_filenames[i]['image1'],
                'image2': self.holdout_filenames[i]['image2'],
                'label': int(labels_array[i]),
                'prediction': int(predictions[i]),
                'logit_0': float(logits_array[i][0]),
                'logit_1': float(logits_array[i][1]),
                'correct': bool(predictions[i] == labels_array[i])
            })

        return {
            'predictions': results,
            'accuracy': float((predictions == labels_array).mean())
        }

    def _compute_batch_stats(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute batch statistics (data balance).

        Args:
            batch: Batch data with 'winner' key

        Returns:
            Dictionary with winner distribution and batch size
        """
        if 'winner' not in batch:
            return {}

        winners = batch['winner']

        winner_counts = {
            0: (winners == 0).sum().item(),
            1: (winners == 1).sum().item()
        }
        total = len(winners)

        return {
            'winner_0_pct': winner_counts[0] / total * 100,
            'winner_1_pct': winner_counts[1] / total * 100,
            'batch_size': total
        }
