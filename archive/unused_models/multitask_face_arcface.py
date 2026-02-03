"""Multitask face model using ArcFace pretrained backbone.

Architecture:
    Image (112x112) -> ArcFace Backbone (ResNet50) -> 2048-dim features
                                                            |
                                      +---------------------+---------------------+
                                      |                                           |
                               Landmark Head                              Expression Head
                               (2048 -> 68*2)                             (2048 -> 8 classes)

The backbone is loaded from a trained ArcFace checkpoint and can be frozen
for transfer learning or unfrozen for fine-tuning.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LandmarkHead(nn.Module):
    """Head for predicting 68 facial landmarks (x, y coordinates)."""

    def __init__(self, in_features: int, hidden_dim: int = 512, num_landmarks: int = 68):
        super().__init__()
        self.num_landmarks = num_landmarks

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_landmarks * 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, in_features)

        Returns:
            Landmarks (B, 68, 2) normalized to [0, 1]
        """
        out = self.fc(x)
        out = out.view(-1, self.num_landmarks, 2)
        return torch.sigmoid(out)


class ExpressionHead(nn.Module):
    """Head for predicting facial expression class."""

    def __init__(self, in_features: int, hidden_dim: int = 512, num_classes: int = 8):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, in_features)

        Returns:
            Expression logits (B, num_classes)
        """
        return self.fc(x)


class MultitaskFaceModelArcFace(nn.Module):
    """
    Multitask model using ArcFace pretrained backbone.

    Uses the ResNet50 backbone from a trained ArcFace model to extract
    2048-dim features, then applies separate heads for landmarks and expression.

    Config keys:
        - arcface_checkpoint: Path to trained ArcFace model
        - freeze_backbone: Whether to freeze backbone weights (default: True)
        - landmark_hidden_dim: Hidden dim for landmark head (default: 512)
        - expression_hidden_dim: Hidden dim for expression head (default: 512)
        - num_expression_classes: Number of expression classes (default: 8)
        - num_landmarks: Number of landmarks (default: 68)
    """

    def __init__(self, config: dict):
        super().__init__()

        self.feature_dim = 2048  # ResNet50 features before embedding layer

        # Head configuration
        landmark_hidden = config.get('landmark_hidden_dim', 512)
        expression_hidden = config.get('expression_hidden_dim', 512)
        num_classes = config.get('num_expression_classes', 8)
        num_landmarks = config.get('num_landmarks', 68)

        # Backbone will be set by from_arcface_checkpoint or manually
        self.backbone = None

        # Task heads
        self.landmark_head = LandmarkHead(
            in_features=self.feature_dim,
            hidden_dim=landmark_hidden,
            num_landmarks=num_landmarks
        )
        self.expression_head = ExpressionHead(
            in_features=self.feature_dim,
            hidden_dim=expression_hidden,
            num_classes=num_classes
        )

        # Uncertainty weighting parameters (learned log variances)
        self.log_var_landmark = nn.Parameter(torch.zeros(1))
        self.log_var_expression = nn.Parameter(torch.zeros(1))

        self._backbone_frozen = False

    @classmethod
    def from_arcface_checkpoint(cls, checkpoint_path: str, config: dict) -> 'MultitaskFaceModelArcFace':
        """
        Create model with backbone loaded from ArcFace checkpoint.

        Args:
            checkpoint_path: Path to trained ArcFace model (.pt file)
            config: Model configuration dict

        Returns:
            MultitaskFaceModelArcFace with pretrained backbone
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ArcFace checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading ArcFace backbone from {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        arcface_config = checkpoint['config']['model']
        state_dict = checkpoint['model_state_dict']

        # Create our multitask model
        model = cls(config)

        # Extract backbone weights from ArcFace state dict
        # ArcFace model has: backbone.*, embedding.*, arcface.*
        # We only want the backbone part
        backbone_state = {}
        for key, value in state_dict.items():
            if key.startswith('backbone.'):
                backbone_state[key] = value

        # Create backbone (same architecture as ArcFace)
        import torchvision.models as models

        backbone_name = arcface_config.get('backbone', 'resnet50')
        if backbone_name == 'resnet50':
            resnet = models.resnet50(weights=None)
        elif backbone_name == 'resnet18':
            resnet = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove final fc layer, keep everything up to avgpool
        model.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        # Load the backbone weights
        model.backbone.load_state_dict(backbone_state, strict=True)

        lfw_acc = checkpoint.get('lfw_acc', 'unknown')
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"Loaded ArcFace backbone (epoch={epoch}, LFW acc={lfw_acc}%)")

        # Optionally freeze backbone
        if config.get('freeze_backbone', True):
            model.freeze_backbone()

        return model

    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized")

        for param in self.backbone.parameters():
            param.requires_grad = False
        self._backbone_frozen = True
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized")

        for param in self.backbone.parameters():
            param.requires_grad = True
        self._backbone_frozen = False
        logger.info("Backbone unfrozen")

    @property
    def is_backbone_frozen(self) -> bool:
        """Check if backbone is frozen."""
        return self._backbone_frozen

    def forward(self, images: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            images: Batch of face images (B, 3, 112, 112)

        Returns:
            Dict with:
                - 'landmarks': (B, 68, 2) normalized coordinates
                - 'expression': (B, num_classes) logits
                - 'features': (B, 2048) backbone features
        """
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized. Use from_arcface_checkpoint() to load.")

        # Extract features from backbone
        features = self.backbone(images)  # (B, 2048)

        # Task predictions
        landmarks = self.landmark_head(features)
        expression = self.expression_head(features)

        return {
            'landmarks': landmarks,
            'expression': expression,
            'features': features
        }

    def compute_loss(
        self,
        outputs: dict,
        landmark_targets: torch.Tensor,
        expression_targets: torch.Tensor,
        use_uncertainty_weighting: bool = True
    ) -> dict:
        """
        Compute multi-task loss.

        Args:
            outputs: Model outputs dict
            landmark_targets: Ground truth landmarks (B, 68, 2), normalized
            expression_targets: Ground truth expression labels (B,)
            use_uncertainty_weighting: Use learned task weights

        Returns:
            Dict with 'total', 'landmark', 'expression', 'landmark_weight', 'expression_weight'
        """
        # Landmark loss (MSE)
        landmark_loss = nn.functional.mse_loss(outputs['landmarks'], landmark_targets)

        # Expression loss (Cross Entropy)
        expression_loss = nn.functional.cross_entropy(outputs['expression'], expression_targets)

        if use_uncertainty_weighting:
            # Kendall et al. 2018: Loss = (1/2σ²) * L + log(σ)
            # Using log_var = log(σ²), so: Loss = (1/2) * exp(-log_var) * L + (1/2) * log_var
            precision_landmark = torch.exp(-self.log_var_landmark)
            precision_expression = torch.exp(-self.log_var_expression)

            weighted_landmark = 0.5 * precision_landmark * landmark_loss + 0.5 * self.log_var_landmark
            weighted_expression = 0.5 * precision_expression * expression_loss + 0.5 * self.log_var_expression

            total_loss = weighted_landmark + weighted_expression

            return {
                'total': total_loss,
                'landmark': landmark_loss,
                'expression': expression_loss,
                'weighted_landmark': weighted_landmark,
                'weighted_expression': weighted_expression,
                'landmark_weight': precision_landmark.item(),
                'expression_weight': precision_expression.item()
            }
        else:
            # Simple sum
            total_loss = landmark_loss + expression_loss
            return {
                'total': total_loss,
                'landmark': landmark_loss,
                'expression': expression_loss,
                'landmark_weight': 1.0,
                'expression_weight': 1.0
            }

    def get_backbone_params(self):
        """Get backbone parameters (for lower learning rate)."""
        if self.backbone is None:
            return []
        return list(self.backbone.parameters())

    def get_head_params(self):
        """Get head parameters (for higher learning rate)."""
        params = []
        params.extend(self.landmark_head.parameters())
        params.extend(self.expression_head.parameters())
        params.append(self.log_var_landmark)
        params.append(self.log_var_expression)
        return params

    def get_num_params(self) -> dict:
        """Get parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters()) if self.backbone else 0
        heads = sum(p.numel() for p in self.landmark_head.parameters()) + \
                sum(p.numel() for p in self.expression_head.parameters())

        return {
            'total': total,
            'trainable': trainable,
            'backbone': backbone,
            'heads': heads
        }
