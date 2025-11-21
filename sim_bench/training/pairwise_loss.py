"""
Pairwise ranking loss functions for contrastive learning.

Implements various ranking losses for learning from pairwise preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class PairwiseRankingLoss(nn.Module):
    """
    Logistic pairwise ranking loss.

    Loss = log(1 + exp(-(score_winner - score_loser - margin)))

    This encourages score_winner > score_loser by at least margin.
    Equivalent to binary cross-entropy on P(winner) = sigmoid(score_winner - score_loser).
    """

    def __init__(self, margin: float = 0.0, reduction: str = 'mean'):
        """
        Initialize pairwise ranking loss.

        Args:
            margin: Minimum margin between winner and loser scores
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        score_winner: torch.Tensor,
        score_loser: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise ranking loss.

        Args:
            score_winner: Scores for winning images (batch_size,)
            score_loser: Scores for losing images (batch_size,)

        Returns:
            Loss (scalar if reduction='mean' or 'sum', else (batch_size,))
        """
        # Compute loss: log(1 + exp(-(winner - loser - margin)))
        loss = torch.log(1 + torch.exp(-(score_winner - score_loser - self.margin)))

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class HingeLoss(nn.Module):
    """
    Hinge loss for pairwise ranking.

    Loss = max(0, margin - (score_winner - score_loser))

    Linear penalty when winner score is not sufficiently higher than loser.
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize hinge loss.

        Args:
            margin: Desired margin between winner and loser
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        score_winner: torch.Tensor,
        score_loser: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hinge loss.

        Args:
            score_winner: Scores for winning images
            score_loser: Scores for losing images

        Returns:
            Loss
        """
        loss = torch.clamp(self.margin - (score_winner - score_loser), min=0.0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskRankingLoss(nn.Module):
    """
    Multi-task pairwise ranking loss for global + per-attribute learning.

    Combines:
    - Global preference loss (on all pairs)
    - Per-attribute losses (on pairs labeled with each attribute)

    Total loss = w_global * L_global + Σ_k w_k * L_attr_k
    """

    def __init__(
        self,
        attribute_names: List[str],
        global_weight: float = 1.0,
        attribute_weights: Optional[Dict[str, float]] = None,
        loss_type: str = 'logistic',
        margin: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize multi-task ranking loss.

        Args:
            attribute_names: List of attribute names
            global_weight: Weight for global preference loss
            attribute_weights: Dictionary mapping attribute → weight (None = uniform)
            loss_type: 'logistic' or 'hinge'
            margin: Margin for ranking loss
            reduction: 'mean' or 'sum'
        """
        super().__init__()

        self.attribute_names = attribute_names
        self.global_weight = global_weight

        # Set attribute weights
        if attribute_weights is None:
            self.attribute_weights = {attr: 1.0 for attr in attribute_names}
        else:
            self.attribute_weights = attribute_weights

        # Create base loss
        if loss_type == 'logistic':
            self.base_loss = PairwiseRankingLoss(margin=margin, reduction=reduction)
        elif loss_type == 'hinge':
            self.base_loss = HingeLoss(margin=margin, reduction=reduction)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.loss_type = loss_type
        self.margin = margin
        self.reduction = reduction

        logger.info(f"MultiTaskRankingLoss initialized:")
        logger.info(f"  Loss type: {loss_type}")
        logger.info(f"  Global weight: {global_weight}")
        logger.info(f"  Num attributes: {len(attribute_names)}")

    def forward(
        self,
        scores_a: Dict[str, torch.Tensor],
        scores_b: Dict[str, torch.Tensor],
        chosen: torch.Tensor,
        attribute_labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task ranking loss.

        Args:
            scores_a: Scores for images A {
                'global': (batch_size,),
                'sharpness': (batch_size,),
                ...
            }
            scores_b: Scores for images B (same structure)
            chosen: Which image was chosen (batch_size,)
                    0 = A chosen, 1 = B chosen
            attribute_labels: Per-attribute labels {
                'sharpness': (batch_size,),  # 0=A wins, 1=B wins, -1=not labeled
                ...
            } (None = all pairs labeled for all attributes)

        Returns:
            Dictionary of losses:
            {
                'global': global loss,
                'sharpness': sharpness loss,
                ...,
                'total': weighted sum
            }
        """
        losses = {}

        # Global loss (computed on all pairs)
        global_winner = torch.where(
            chosen == 0,
            scores_a['global'],
            scores_b['global']
        )
        global_loser = torch.where(
            chosen == 0,
            scores_b['global'],
            scores_a['global']
        )

        losses['global'] = self.base_loss(global_winner, global_loser)

        # Per-attribute losses
        total_loss = self.global_weight * losses['global']

        for attr_name in self.attribute_names:
            if attr_name not in scores_a:
                continue

            # Get attribute labels
            if attribute_labels is not None and attr_name in attribute_labels:
                attr_chosen = attribute_labels[attr_name]

                # Filter to labeled pairs (-1 = unlabeled)
                labeled_mask = (attr_chosen != -1)

                if labeled_mask.sum() == 0:
                    # No labeled pairs for this attribute
                    losses[attr_name] = torch.tensor(0.0, device=chosen.device)
                    continue

                # Get scores for labeled pairs only
                attr_winner = torch.where(
                    attr_chosen == 0,
                    scores_a[attr_name],
                    scores_b[attr_name]
                )[labeled_mask]

                attr_loser = torch.where(
                    attr_chosen == 0,
                    scores_b[attr_name],
                    scores_a[attr_name]
                )[labeled_mask]

            else:
                # All pairs labeled (use same choice as global)
                attr_winner = torch.where(
                    chosen == 0,
                    scores_a[attr_name],
                    scores_b[attr_name]
                )
                attr_loser = torch.where(
                    chosen == 0,
                    scores_b[attr_name],
                    scores_a[attr_name]
                )

            # Compute attribute loss
            losses[attr_name] = self.base_loss(attr_winner, attr_loser)

            # Add to total loss
            weight = self.attribute_weights.get(attr_name, 1.0)
            total_loss += weight * losses[attr_name]

        losses['total'] = total_loss

        return losses

    def get_attribute_weights(self) -> Dict[str, float]:
        """Get attribute weights."""
        return self.attribute_weights

    def set_attribute_weights(self, weights: Dict[str, float]):
        """Set attribute weights."""
        self.attribute_weights.update(weights)


def create_loss_from_config(config: Dict, attribute_names: List[str]) -> MultiTaskRankingLoss:
    """
    Create multi-task ranking loss from configuration.

    Args:
        config: Loss configuration dictionary
        attribute_names: List of attribute names

    Returns:
        MultiTaskRankingLoss instance

    Example config:
        {
            'type': 'pairwise_ranking',
            'loss_type': 'logistic',  # or 'hinge'
            'margin': 0.0,
            'global_weight': 1.0,
            'attribute_weighting': 'uniform',  # or 'inverse_frequency', 'manual'
            'manual_weights': {  # if attribute_weighting='manual'
                'sharpness': 1.5,
                'exposure': 1.0
            }
        }
    """
    loss_type = config.get('loss_type', 'logistic')
    margin = config.get('margin', 0.0)
    global_weight = config.get('global_weight', 1.0)

    # Determine attribute weights
    weighting = config.get('attribute_weighting', 'uniform')

    if weighting == 'uniform':
        attribute_weights = None  # Will use 1.0 for all
    elif weighting == 'manual':
        attribute_weights = config.get('manual_weights', {})
    else:
        # For inverse_frequency, will need dataset statistics
        # For now, default to uniform
        logger.warning(f"Weighting '{weighting}' not implemented, using uniform")
        attribute_weights = None

    return MultiTaskRankingLoss(
        attribute_names=attribute_names,
        global_weight=global_weight,
        attribute_weights=attribute_weights,
        loss_type=loss_type,
        margin=margin,
        reduction='mean'
    )
