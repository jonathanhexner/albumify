"""
Multitask model for landmark regression and expression classification.

Architecture:
    Image -> ResNet50 (pretrained) -> 2048-dim features -> 
        -> Expression Head (8-way classification)
        -> Landmark Head (5-10 key points regression)

Uses uncertainty weighting (Kendall et al. 2018) for loss balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ExpressionHead(nn.Module):
    """Expression classification head (8 classes from AffectNet)."""

    def __init__(self, embedding_dim: int, num_classes: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(embedding_dim, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        return self.fc(features)


class LandmarkHead(nn.Module):
    """Landmark regression head (5-10 key points)."""

    def __init__(self, embedding_dim: int, num_landmarks: int = 5, dropout: float = 0.0):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.fc = nn.Linear(embedding_dim, num_landmarks * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.dropout(features)
        landmarks = self.fc(features)
        return landmarks.view(-1, self.num_landmarks, 2)


class UncertaintyWeighting(nn.Module):
    """
    Learnable uncertainty weighting for multitask losses.

    Reference: Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses"

    Formula: L = sum_i [ 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i ]
    where log_var_i = log(sigma_i^2)

    The 0.5 factor ensures the loss stays positive and well-behaved.
    """

    def __init__(self, num_tasks: int = 2):
        super().__init__()
        # Initialize log_vars to 0, meaning sigma^2 = 1 (equal weighting initially)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: list) -> tuple:
        """
        Weight losses by learned uncertainties.

        Args:
            losses: List of task losses [expression_loss, landmark_loss]

        Returns:
            (weighted_total_loss, individual_weighted_losses)
        """
        weighted_losses = []

        # Clamp log_vars to prevent extreme values (-6 to 6 corresponds to sigma^2 in ~[0.002, 400])
        clamped_log_vars = torch.clamp(self.log_vars, min=-6.0, max=6.0)

        for i, loss in enumerate(losses):
            # Precision = 1 / sigma^2 = exp(-log_var)
            precision = torch.exp(-clamped_log_vars[i])
            # Weighted loss = 0.5 * precision * loss + 0.5 * log_var
            # The 0.5 factor is critical to keep loss positive
            weighted = 0.5 * precision * loss + 0.5 * clamped_log_vars[i]
            weighted_losses.append(weighted)

        total = sum(weighted_losses)
        return total, weighted_losses

    def get_weights(self) -> dict:
        """Get current task weights for logging."""
        with torch.no_grad():
            clamped = torch.clamp(self.log_vars, min=-6.0, max=6.0)
            precisions = torch.exp(-clamped)
            return {
                'expression_weight': precisions[0].item(),
                'landmark_weight': precisions[1].item(),
                'expression_log_var': clamped[0].item(),
                'landmark_log_var': clamped[1].item()
            }


class MultitaskFaceModel(nn.Module):
    """
    Multitask model for face pretraining.
    
    Shared ResNet backbone with two task heads:
    - Expression classification (8 classes)
    - Landmark regression (5-10 key points)
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']
        
        self.backbone_name = model_cfg['backbone']
        self.embedding_dim = model_cfg['embedding_dim']
        self.num_expression_classes = model_cfg.get('num_expression_classes', 8)
        self.num_landmarks = model_cfg.get('num_landmarks', 5)
        self.dropout = model_cfg.get('dropout', 0.0)
        self.use_uncertainty_weighting = model_cfg.get('use_uncertainty_weighting', True)
        
        self._build_backbone()
        self._build_heads()
        
        if self.use_uncertainty_weighting:
            self.uncertainty = UncertaintyWeighting(num_tasks=2)

    def _build_backbone(self):
        """Build ResNet backbone."""
        if self.backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif self.backbone_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        backbone_dim = 2048 if '50' in self.backbone_name else 512
        
        if backbone_dim != self.embedding_dim:
            self.projection = nn.Linear(backbone_dim, self.embedding_dim)
        else:
            self.projection = nn.Identity()

    def _build_heads(self):
        """Build task-specific heads."""
        self.expression_head = ExpressionHead(
            self.embedding_dim,
            self.num_expression_classes,
            self.dropout
        )
        self.landmark_head = LandmarkHead(
            self.embedding_dim,
            self.num_landmarks,
            self.dropout
        )

    def forward(self, images: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            images: Input images (B, 3, H, W)
        
        Returns:
            Dict with 'expression_logits' and 'landmarks'
        """
        features = self.backbone(images)
        features = self.projection(features)
        expression_logits = self.expression_head(features)
        landmarks = self.landmark_head(features)
        
        return {
            'expression_logits': expression_logits,
            'landmarks': landmarks,
            'features': features
        }

    def get_1x_lr_params(self):
        """Parameters for base learning rate (backbone)."""
        for param in self.backbone.parameters():
            yield param
        for param in self.projection.parameters():
            yield param

    def get_10x_lr_params(self):
        """Parameters for 10x learning rate (heads)."""
        for param in self.expression_head.parameters():
            yield param
        for param in self.landmark_head.parameters():
            yield param
        if self.use_uncertainty_weighting:
            for param in self.uncertainty.parameters():
                yield param
