"""
ResNet model for AVA aesthetic score prediction.

Architecture:
    Image -> ResNet50 (pretrained) -> 2048-dim features -> MLP -> output

Supports two output modes:
- 'distribution': 10 bins for score distribution (1-10), use with KL divergence loss
- 'regression': Single scalar for mean score prediction, use with MSE/L1 loss
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms


class AVAResNet(nn.Module):
    """
    ResNet + MLP for AVA score prediction.

    Supports distribution prediction (10 bins) or direct regression (single value).
    """

    def __init__(self, config: dict):
        """
        Initialize model from config dict.

        Args:
            config: Configuration dict with keys:
                - backbone: 'resnet50' (default)
                - pretrained: Load ImageNet weights (default: True)
                - mlp_hidden_dims: List of hidden layer dimensions (default: [256])
                - dropout: Dropout rate (default: 0.2)
                - activation: 'relu' or 'tanh' (default: 'relu')
                - output_mode: 'distribution' (10 bins) or 'regression' (single value)
        """
        super().__init__()

        backbone_name = config.get('backbone', 'resnet50')
        pretrained = config.get('pretrained', True)
        mlp_hidden_dims = config.get('mlp_hidden_dims', [256])
        dropout = config.get('dropout', 0.2)
        activation = config.get('activation', 'relu')
        self.output_mode = config.get('output_mode', 'distribution')

        # Create backbone
        if backbone_name == 'resnet50':
            if pretrained:
                resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

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

        # Output dimension depends on mode
        output_dim = 10 if self.output_mode == 'distribution' else 1
        layers.append(nn.Linear(in_dim, output_dim))
        self.head = nn.Sequential(*layers)

        # Initialize head weights
        self._initialize_head_weights()

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
            images: Batch of images (batch_size, 3, H, W)

        Returns:
            If output_mode='distribution': Logits for 10-bin score distribution (batch_size, 10)
            If output_mode='regression': Predicted mean score (batch_size, 1)
        """
        features = self.backbone(images)
        output = self.head(features)
        return output

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


def create_transform(config: dict, is_train: bool = False):
    """
    Create image transform based on config.

    Args:
        config: Transform configuration with keys:
            - resize_size: Size for initial resize (default: 256)
            - crop_size: Size for center crop (default: 224)
            - normalize_mean: RGB mean for normalization (default: ImageNet)
            - normalize_std: RGB std for normalization (default: ImageNet)
            - augmentation: Dict with augmentation settings (only applied if is_train=True)
                - horizontal_flip: probability (default: 0.0)
                - random_crop: use random crop instead of center crop (default: False)
                - color_jitter: dict with brightness, contrast, saturation, hue (default: None)
        is_train: Whether this is for training (enables augmentation)

    Returns:
        torchvision.transforms.Compose transform
    """
    resize_size = config.get('resize_size', 256)
    crop_size = config.get('crop_size', 224)
    normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
    normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
    augmentation = config.get('augmentation', {})

    transform_list = [transforms.Resize(resize_size)]

    if is_train and augmentation:
        # Training augmentations
        if augmentation.get('random_crop', False):
            transform_list.append(transforms.RandomCrop(crop_size))
        else:
            transform_list.append(transforms.CenterCrop(crop_size))

        if augmentation.get('horizontal_flip', 0.0) > 0:
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
        # Validation/test: just center crop
        transform_list.append(transforms.CenterCrop(crop_size))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    return transforms.Compose(transform_list)
