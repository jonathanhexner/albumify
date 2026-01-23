"""
ResNet model for face classification with ArcFace loss.

Architecture:
    Image -> ResNet50 (pretrained) -> 2048-dim features -> embedding layer -> ArcFace

This model produces embeddings that can be used for face verification/identification.
The ArcFace loss adds angular margin for better discriminability.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms


class ArcFaceHead(nn.Module):
    """
    ArcFace loss implementation.

    Adds an angular margin penalty to the softmax loss for better face recognition.
    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 64.0):
        """
        Initialize ArcFace head.

        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of identity classes
            margin: Angular margin in radians (default: 0.5 ~ 28.6 degrees)
            scale: Feature scale (default: 64.0)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Class weight matrix (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin values
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            embeddings: Normalized embeddings (B, embedding_dim)
            labels: Ground truth labels (B,) - required during training

        Returns:
            If training with labels: scaled logits with angular margin (B, num_classes)
            If inference without labels: cosine similarity logits (B, num_classes)
        """
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cos_theta = F.linear(embeddings_norm, weight_norm)
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)  # Numerical stability

        if labels is None:
            # Inference mode: just return scaled cosine similarity
            return cos_theta * self.scale

        # Training mode: apply angular margin
        # We have cos(theta), need cos(theta + m)
        # Using identity: sin(theta) = sqrt(1 - cos^2(theta))
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # Angle addition formula: cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # Handle edge case where theta + m > pi (cos becomes non-monotonic)
        # threshold = cos(pi - m), so if cos(theta) < threshold, theta > pi - m
        # In this case, use linear approximation: cos(theta) - m*sin(pi-m)
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.mm
        )

        # Create one-hot labels to select which logits get the margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Apply margin only to ground truth class, keep others unchanged
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        return output * self.scale


class FaceResNet(nn.Module):
    """
    ResNet + Embedding layer + ArcFace for face classification.

    During training, outputs logits for classification with ArcFace margin.
    During inference, can extract embeddings for face verification.
    """

    def __init__(self, config: dict):
        """
        Initialize model from config dict.

        Args:
            config: Configuration dict with keys:
                - backbone: 'resnet50' or 'resnet18' (default: 'resnet50')
                - pretrained: Load ImageNet weights (default: True)
                - embedding_dim: Embedding dimension (default: 512)
                - num_classes: Number of identity classes
                - dropout: Dropout rate before embedding (default: 0.0)
                - arcface_margin: Angular margin (default: 0.5)
                - arcface_scale: Feature scale (default: 64.0)
        """
        super().__init__()

        backbone_name = config.get('backbone', 'resnet50')
        pretrained = config.get('pretrained', True)
        self.embedding_dim = config.get('embedding_dim', 512)
        num_classes = config['num_classes']
        dropout = config.get('dropout', 0.0)
        arcface_margin = config.get('arcface_margin', 0.5)
        arcface_scale = config.get('arcface_scale', 64.0)

        # Create backbone
        if backbone_name == 'resnet50':
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet50(weights=None)
            self.feature_dim = 2048
        elif backbone_name == 'resnet18':
            if pretrained:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet18(weights=None)
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Remove final fc layer, keep everything up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        # Embedding layer: feature_dim -> embedding_dim
        self.embedding = nn.Sequential(
            nn.Linear(self.feature_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim)
        )
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # ArcFace classification head
        self.arcface = ArcFaceHead(
            embedding_dim=self.embedding_dim,
            num_classes=num_classes,
            margin=arcface_margin,
            scale=arcface_scale
        )

        # Initialize embedding weights
        self._initialize_embedding_weights()

    def _initialize_embedding_weights(self):
        """Initialize embedding layer weights."""
        for m in self.embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embeddings for face verification.

        Args:
            images: Batch of face images (B, 3, H, W)

        Returns:
            L2-normalized embeddings (B, embedding_dim)
        """
        features = self.backbone(images)
        if self.dropout is not None:
            features = self.dropout(features)
        embeddings = self.embedding(features)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Batch of face images (B, 3, H, W)
            labels: Ground truth identity labels (B,) - required for training

        Returns:
            If training with labels: ArcFace logits (B, num_classes)
            If inference without labels: cosine similarity logits (B, num_classes)
        """
        features = self.backbone(images)
        if self.dropout is not None:
            features = self.dropout(features)
        embeddings = self.embedding(features)
        logits = self.arcface(embeddings, labels)
        return logits

    def get_1x_lr_params(self):
        """Get backbone parameters (for lower learning rate)."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    def get_10x_lr_params(self):
        """Get embedding and ArcFace head parameters (for higher learning rate)."""
        for param in self.embedding.parameters():
            if param.requires_grad:
                yield param
        for param in self.arcface.parameters():
            if param.requires_grad:
                yield param


def create_transform(config: dict, is_train: bool = False):
    """
    Create image transform based on config.

    Args:
        config: Transform configuration with keys:
            - input_size: Size for final resize (default: 112 for face recognition)
            - normalize_mean: RGB mean for normalization (default: [0.5, 0.5, 0.5])
            - normalize_std: RGB std for normalization (default: [0.5, 0.5, 0.5])
            - augmentation: Dict with augmentation settings (only applied if is_train=True)
                - horizontal_flip: probability (default: 0.5)
                - random_crop: use random crop instead of center crop (default: False)
                - color_jitter: dict with brightness, contrast, saturation, hue
        is_train: Whether this is for training (enables augmentation)

    Returns:
        torchvision.transforms.Compose transform
    """
    input_size = config.get('input_size', 112)
    # Standard face recognition normalization: map [0,255] to [-1,1]
    normalize_mean = config.get('normalize_mean', [0.5, 0.5, 0.5])
    normalize_std = config.get('normalize_std', [0.5, 0.5, 0.5])
    augmentation = config.get('augmentation', {})

    transform_list = []

    if is_train and augmentation:
        # Training augmentations
        if augmentation.get('random_crop', False):
            # Slightly larger resize, then random crop
            transform_list.append(transforms.Resize(int(input_size * 1.1)))
            transform_list.append(transforms.RandomCrop(input_size))
        else:
            transform_list.append(transforms.Resize((input_size, input_size)))

        if augmentation.get('horizontal_flip', 0.5) > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip']))

        color_jitter = augmentation.get('color_jitter')
        if color_jitter:
            transform_list.append(transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0)
            ))
    else:
        # Validation/test: just resize
        transform_list.append(transforms.Resize((input_size, input_size)))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    return transforms.Compose(transform_list)
