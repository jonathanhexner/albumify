"""
Attribute scoring heads for CLIP-based contrastive learning.

Provides linear and MLP head architectures for converting CLIP embeddings
to scalar preference scores (global + per-attribute).
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LinearHead(nn.Module):
    """
    Simple linear head: embedding → scalar score.

    Fast and parameter-efficient, suitable for most attributes.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.0):
        """
        Initialize linear head.

        Args:
            embed_dim: CLIP embedding dimension (512 for ViT-B/32, 768 for ViT-L/14)
            dropout: Dropout probability (0.0 = no dropout)
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(embed_dim, 1)

        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: CLIP embeddings (batch_size, embed_dim)

        Returns:
            Scores (batch_size, 1)
        """
        x = self.dropout(x)
        return self.linear(x)


class MLPHead(nn.Module):
    """
    Multi-layer perceptron head for non-linear attribute scoring.

    Use when attributes require more complex decision boundaries.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3
    ):
        """
        Initialize MLP head.

        Args:
            embed_dim: CLIP embedding dimension
            hidden_dims: List of hidden layer sizes (e.g., [128, 64])
            dropout: Dropout probability for regularization
        """
        super().__init__()

        layers = []
        in_dim = embed_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: CLIP embeddings (batch_size, embed_dim)

        Returns:
            Scores (batch_size, 1)
        """
        return self.mlp(x)


class AttributeHeads(nn.Module):
    """
    Collection of heads for global + per-attribute scoring.

    Manages multiple heads and provides unified forward pass.
    """

    def __init__(
        self,
        embed_dim: int,
        attribute_names: List[str],
        head_type: str = 'linear',
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3
    ):
        """
        Initialize attribute heads.

        Args:
            embed_dim: CLIP embedding dimension
            attribute_names: List of attribute names (e.g., ['sharpness', 'exposure'])
            head_type: 'linear' or 'mlp'
            hidden_dims: Hidden dimensions for MLP (if head_type='mlp')
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.attribute_names = attribute_names
        self.head_type = head_type

        # Create head factory
        def create_head():
            if head_type == 'linear':
                return LinearHead(embed_dim, dropout=dropout)
            elif head_type == 'mlp':
                if hidden_dims is None:
                    hidden_dims_default = [128, 64]
                else:
                    hidden_dims_default = hidden_dims
                return MLPHead(embed_dim, hidden_dims_default, dropout=dropout)
            else:
                raise ValueError(f"Unknown head_type: {head_type}")

        # Global preference head
        self.global_head = create_head()

        # Per-attribute heads
        self.attribute_heads = nn.ModuleDict({
            attr: create_head() for attr in attribute_names
        })

        logger.info(f"Created AttributeHeads with {len(attribute_names)} attributes")
        logger.info(f"Head type: {head_type}, embed_dim: {embed_dim}")

    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all heads.

        Args:
            embeddings: CLIP embeddings (batch_size, embed_dim)

        Returns:
            Dictionary mapping head names to scores:
            {
                'global': (batch_size, 1),
                'sharpness': (batch_size, 1),
                'exposure': (batch_size, 1),
                ...
            }
        """
        scores = {}

        # Global score
        scores['global'] = self.global_head(embeddings).squeeze(-1)  # (batch_size,)

        # Per-attribute scores
        for attr_name, head in self.attribute_heads.items():
            scores[attr_name] = head(embeddings).squeeze(-1)  # (batch_size,)

        return scores

    def get_attribute_names(self) -> List[str]:
        """Get list of attribute names."""
        return self.attribute_names

    def num_attributes(self) -> int:
        """Get number of attributes."""
        return len(self.attribute_names)

    def num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_attribute_heads(
    embed_dim: int,
    attribute_names: List[str],
    head_config: Dict
) -> AttributeHeads:
    """
    Factory function to create AttributeHeads from config.

    Args:
        embed_dim: CLIP embedding dimension
        attribute_names: List of attribute names
        head_config: Configuration dictionary with keys:
            - architecture: 'linear' or 'mlp'
            - hidden_dims: List[int] (for MLP)
            - dropout: float

    Returns:
        AttributeHeads instance

    Example:
        >>> config = {
        ...     'architecture': 'linear',
        ...     'dropout': 0.3
        ... }
        >>> heads = create_attribute_heads(512, ['sharpness', 'exposure'], config)
    """
    head_type = head_config.get('architecture', 'linear')
    hidden_dims = head_config.get('hidden_dims', [128, 64])
    dropout = head_config.get('dropout', 0.3)

    return AttributeHeads(
        embed_dim=embed_dim,
        attribute_names=attribute_names,
        head_type=head_type,
        hidden_dims=hidden_dims if head_type == 'mlp' else None,
        dropout=dropout
    )
