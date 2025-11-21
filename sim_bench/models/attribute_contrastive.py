"""
CLIP-based contrastive model for attribute-aware photo ranking.

Combines pre-trained CLIP image encoder with per-attribute scoring heads
for multi-task preference learning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

from ..vision_language.clip import CLIPModel
from .clip_heads import AttributeHeads, create_attribute_heads

logger = logging.getLogger(__name__)


class AttributeContrastiveModel(nn.Module):
    """
    CLIP-based contrastive model with attribute-specific heads.

    Architecture:
        Input Image → CLIP Encoder → Embedding → [Global Head, Attr Head 1, ..., Attr Head N]
                                                        ↓           ↓              ↓
                                                   s_global    s_attr_1  ...  s_attr_n
    """

    def __init__(
        self,
        attribute_names: List[str],
        clip_model_name: str = 'ViT-B-32',
        clip_checkpoint: str = 'laion2b_s34b_b79k',
        freeze_backbone: bool = True,
        head_config: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize attribute contrastive model.

        Args:
            attribute_names: List of attribute names (e.g., ['sharpness', 'exposure'])
            clip_model_name: CLIP architecture (ViT-B-32, ViT-L-14, etc.)
            clip_checkpoint: Pre-trained checkpoint name
            freeze_backbone: If True, freeze CLIP encoder (train heads only)
            head_config: Configuration for attribute heads
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        super().__init__()

        self.attribute_names = attribute_names
        self.clip_model_name = clip_model_name
        self.freeze_backbone = freeze_backbone
        self.device = device

        # Initialize CLIP model
        logger.info(f"Loading CLIP model: {clip_model_name} ({clip_checkpoint})")
        self.clip = CLIPModel(
            architecture=clip_model_name,
            checkpoint=clip_checkpoint,
            device=device
        )

        # Get embedding dimension
        self.embed_dim = self.clip.get_embedding_dim()
        logger.info(f"CLIP embedding dimension: {self.embed_dim}")

        # Freeze CLIP if requested
        if freeze_backbone:
            logger.info("Freezing CLIP backbone")
            for param in self.clip.model.visual.parameters():
                param.requires_grad = False
        else:
            logger.info("CLIP backbone will be fine-tuned")

        # Create attribute heads
        if head_config is None:
            head_config = {'architecture': 'linear', 'dropout': 0.3}

        self.heads = create_attribute_heads(
            embed_dim=self.embed_dim,
            attribute_names=attribute_names,
            head_config=head_config
        )

        # Move to device
        self.to(device)

        # Log model info
        self._log_model_info()

    def forward(
        self,
        images: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[Dict[str, torch.Tensor], tuple]:
        """
        Forward pass through model.

        Args:
            images: Batch of images (batch_size, 3, 224, 224)
            return_embeddings: If True, also return CLIP embeddings

        Returns:
            If return_embeddings=False:
                Dictionary of scores: {
                    'global': (batch_size,),
                    'sharpness': (batch_size,),
                    ...
                }
            If return_embeddings=True:
                (scores_dict, embeddings)
        """
        # Get CLIP embeddings
        embeddings = self.clip.encode_image(images)

        # Get scores from all heads
        scores = self.heads(embeddings)

        if return_embeddings:
            return scores, embeddings
        else:
            return scores

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to CLIP embeddings.

        Args:
            images: Batch of images (batch_size, 3, 224, 224)

        Returns:
            Embeddings (batch_size, embed_dim)
        """
        return self.clip.encode_image(images)

    def score_images(
        self,
        images: torch.Tensor,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Score images for specified attributes.

        Args:
            images: Batch of images
            attributes: List of attributes to score (None = all)

        Returns:
            Dictionary of scores for requested attributes
        """
        scores = self.forward(images)

        if attributes is None:
            return scores

        # Filter to requested attributes
        filtered_scores = {}
        if 'global' in attributes or 'global' not in scores:
            filtered_scores['global'] = scores['global']

        for attr in attributes:
            if attr in scores:
                filtered_scores[attr] = scores[attr]

        return filtered_scores

    def get_parameter_groups(
        self,
        backbone_lr: float = 1e-6,
        head_lr: float = 1e-4
    ) -> List[Dict]:
        """
        Get parameter groups for optimizer with different learning rates.

        Args:
            backbone_lr: Learning rate for CLIP backbone (if not frozen)
            head_lr: Learning rate for attribute heads

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Backbone parameters (if not frozen)
        if not self.freeze_backbone:
            param_groups.append({
                'params': self.clip.model.visual.parameters(),
                'lr': backbone_lr,
                'name': 'backbone'
            })

        # Head parameters
        param_groups.append({
            'params': self.heads.parameters(),
            'lr': head_lr,
            'name': 'heads'
        })

        return param_groups

    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def _log_model_info(self):
        """Log model information."""
        trainable = self.num_trainable_parameters()
        total = self.num_total_parameters()

        logger.info("="*60)
        logger.info("Attribute Contrastive Model")
        logger.info("="*60)
        logger.info(f"CLIP Model: {self.clip_model_name}")
        logger.info(f"Embedding Dim: {self.embed_dim}")
        logger.info(f"Num Attributes: {len(self.attribute_names)}")
        logger.info(f"Freeze Backbone: {self.freeze_backbone}")
        logger.info(f"Total Parameters: {total:,}")
        logger.info(f"Trainable Parameters: {trainable:,}")
        logger.info(f"Frozen Parameters: {total - trainable:,}")
        logger.info("="*60)

    def save(self, path: Union[str, Path]):
        """
        Save model state.

        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'attribute_names': self.attribute_names,
            'clip_model_name': self.clip_model_name,
            'embed_dim': self.embed_dim,
            'freeze_backbone': self.freeze_backbone,
            'head_config': {
                'architecture': self.heads.head_type,
                'num_attributes': len(self.attribute_names)
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'cpu') -> 'AttributeContrastiveModel':
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model on

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            attribute_names=checkpoint['attribute_names'],
            clip_model_name=checkpoint['clip_model_name'],
            freeze_backbone=checkpoint['freeze_backbone'],
            device=device
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")

        return model


def create_model_from_config(config: Dict, device: str = 'cpu') -> AttributeContrastiveModel:
    """
    Create model from configuration dictionary.

    Args:
        config: Configuration dictionary with keys:
            - attributes: List of attribute names
            - backbone:
                - type: 'clip'
                - architecture: CLIP model name
                - checkpoint: Checkpoint name
                - freeze: bool
            - heads:
                - architecture: 'linear' or 'mlp'
                - hidden_dims: List[int] (for MLP)
                - dropout: float
        device: Device to use

    Returns:
        AttributeContrastiveModel instance

    Example config:
        {
            'attributes': ['sharpness', 'exposure', 'composition'],
            'backbone': {
                'type': 'clip',
                'architecture': 'ViT-B-32',
                'checkpoint': 'laion2b_s34b_b79k',
                'freeze': True
            },
            'heads': {
                'architecture': 'linear',
                'dropout': 0.3
            }
        }
    """
    # Extract configuration
    attribute_names = config['attributes']

    backbone_config = config.get('backbone', {})
    clip_model_name = backbone_config.get('architecture', 'ViT-B-32')
    clip_checkpoint = backbone_config.get('checkpoint', 'laion2b_s34b_b79k')
    freeze_backbone = backbone_config.get('freeze', True)

    head_config = config.get('heads', {'architecture': 'linear', 'dropout': 0.3})

    # Create model
    model = AttributeContrastiveModel(
        attribute_names=attribute_names,
        clip_model_name=clip_model_name,
        clip_checkpoint=clip_checkpoint,
        freeze_backbone=freeze_backbone,
        head_config=head_config,
        device=device
    )

    return model
