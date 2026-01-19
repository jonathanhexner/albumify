"""
ResNet model for eye open/closed classification.

Architecture:
    Eye crop image -> ResNet (pretrained) -> features -> MLP -> binary output

Trained on MRL Eye Dataset for detecting open vs closed eyes.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class EyeStateClassifier(nn.Module):
    """
    ResNet + MLP for eye open/closed binary classification.

    Input: Eye crop images (grayscale or RGB)
    Output: Binary classification (0=closed, 1=open)
    """

    def __init__(self, config: dict):
        """
        Initialize model from config dict.

        Args:
            config: Configuration dict with keys:
                - backbone: 'resnet18', 'resnet34', 'resnet50' (default: 'resnet18')
                - pretrained: Load ImageNet weights (default: True)
                - mlp_hidden_dims: List of hidden layer dimensions (default: [128])
                - dropout: Dropout rate (default: 0.3)
                - activation: 'relu' or 'tanh' (default: 'relu')
                - input_channels: 1 for grayscale, 3 for RGB (default: 1)
        """
        super().__init__()

        backbone_name = config.get('backbone', 'resnet18')
        pretrained = config.get('pretrained', True)
        mlp_hidden_dims = config.get('mlp_hidden_dims', [128])
        dropout = config.get('dropout', 0.3)
        activation = config.get('activation', 'relu')
        input_channels = config.get('input_channels', 1)

        self.input_channels = input_channels

        # Create backbone
        self.backbone, self.feature_dim = self._create_backbone(
            backbone_name, pretrained, input_channels
        )

        # Build MLP head
        activation_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        layers = []
        in_dim = self.feature_dim
        for hidden_dim in mlp_hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                activation_fn,
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        # Binary output (single logit)
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

        self._initialize_head_weights()

    def _create_backbone(
        self,
        backbone_name: str,
        pretrained: bool,
        input_channels: int,
    ) -> tuple:
        """Create backbone network with optional grayscale input modification."""
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            feature_dim = 512
        elif backbone_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            feature_dim = 512
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Modify first conv layer for grayscale input
        if input_channels == 1:
            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                1, 64,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Initialize by averaging RGB weights
            if pretrained:
                resnet.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)

        # Remove final FC, add flatten
        backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())

        return backbone, feature_dim

    def _initialize_head_weights(self):
        """Initialize MLP head weights with Kaiming normal."""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Batch of eye images (batch_size, C, H, W)
                    C=1 for grayscale, C=3 for RGB

        Returns:
            Logits for binary classification (batch_size, 1)
        """
        features = self.backbone(images)
        output = self.head(features)
        return output

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get probability of eye being open.

        Args:
            images: Batch of eye images

        Returns:
            Probabilities (batch_size,) in range [0, 1]
        """
        logits = self.forward(images)
        return torch.sigmoid(logits).squeeze(-1)

    def get_1x_lr_params(self):
        """Get backbone parameters (for lower learning rate)."""
        for param in self.backbone.parameters():
            if param.requires_grad:
                yield param

    def get_10x_lr_params(self):
        """Get MLP head parameters (for higher learning rate)."""
        for param in self.head.parameters():
            if param.requires_grad:
                yield param


def create_eye_transform(config: dict, is_train: bool = False):
    """
    Create image transform for eye images.

    Args:
        config: Transform configuration with keys:
            - resize_size: Size for initial resize (default: 64)
            - crop_size: Size for center crop (default: 48)
            - grayscale: Convert to grayscale (default: True)
            - normalize_mean: Mean for normalization (default: [0.5])
            - normalize_std: Std for normalization (default: [0.5])
            - augmentation: Dict with augmentation settings
                - horizontal_flip: probability (default: 0.5)
                - rotation: max rotation degrees (default: 10)
                - brightness: brightness jitter (default: 0.2)
                - contrast: contrast jitter (default: 0.2)
        is_train: Whether this is for training (enables augmentation)

    Returns:
        torchvision.transforms.Compose transform
    """
    resize_size = config.get('resize_size', 64)
    crop_size = config.get('crop_size', 48)
    grayscale = config.get('grayscale', True)
    normalize_mean = config.get('normalize_mean', [0.5] if grayscale else [0.485, 0.456, 0.406])
    normalize_std = config.get('normalize_std', [0.5] if grayscale else [0.229, 0.224, 0.225])
    augmentation = config.get('augmentation', {})

    transform_list = []

    # Grayscale conversion first
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))

    transform_list.append(transforms.Resize(resize_size))

    if is_train and augmentation:
        # Training augmentations
        if augmentation.get('random_crop', False):
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.CenterCrop(crop_size))

        if augmentation.get('horizontal_flip', 0.0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=augmentation['horizontal_flip'])
            )

        rotation = augmentation.get('rotation', 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        brightness = augmentation.get('brightness', 0)
        contrast = augmentation.get('contrast', 0)
        if brightness > 0 or contrast > 0:
            transform_list.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
            ))
    else:
        transform_list.append(transforms.CenterCrop(crop_size))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    return transforms.Compose(transform_list)
