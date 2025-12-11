"""
Shared ResNet feature extractor for multi-level visual features.

This module provides a unified interface for extracting features from different
layers of ResNet architectures (ResNet34, ResNet50) at various depths (layer3, layer4).

Usage:
    from sim_bench.feature_extraction.resnet_features import ResNetFeatureExtractor

    extractor = ResNetFeatureExtractor(
        backbone='resnet50',
        layer='layer3',
        device='cuda'
    )

    features = extractor.extract(image_pil)  # Returns tensor of shape (1024,) for ResNet50 layer3
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)


class ResNetFeatureExtractor(nn.Module):
    """
    Extract mid-level features from ResNet architectures.

    Supports configurable backbone (ResNet34/50) and layer extraction (layer3/4).
    All features are frozen and extracted from pre-trained ImageNet models.

    Attributes:
        backbone: ResNet variant ('resnet34' or 'resnet50')
        layer: Layer to extract from ('layer3' or 'layer4')
        device: Device for computation
        feature_dim: Dimension of extracted features
        features: Sequential module for feature extraction
        preprocess: Image preprocessing transforms

    Examples:
        >>> extractor = ResNetFeatureExtractor('resnet50', 'layer3')
        >>> img = Image.open('photo.jpg')
        >>> features = extractor.extract(img)  # Shape: (1024,)

        >>> # Batch processing
        >>> imgs = [Image.open(f'photo{i}.jpg') for i in range(10)]
        >>> features = extractor.extract_batch(imgs)  # Shape: (10, 1024)
    """

    # Feature dimensions for each (backbone, layer) combination
    FEATURE_DIMS = {
        ('resnet34', 'layer3'): 256,
        ('resnet34', 'layer4'): 512,
        ('resnet50', 'layer3'): 1024,
        ('resnet50', 'layer4'): 2048,
    }

    def __init__(
        self,
        backbone: Literal['resnet34', 'resnet50'] = 'resnet50',
        layer: Literal['layer3', 'layer4'] = 'layer3',
        device: Optional[str] = None,
        pretrained: bool = True
    ):
        """
        Initialize ResNet feature extractor.

        Args:
            backbone: ResNet architecture variant
            layer: Layer to extract features from
            device: Device for computation (auto-detected if None)
            pretrained: Whether to use ImageNet pretrained weights

        Raises:
            ValueError: If backbone or layer is invalid
        """
        super().__init__()

        if backbone not in ['resnet34', 'resnet50']:
            raise ValueError(f"Unknown backbone: {backbone}. Must be 'resnet34' or 'resnet50'")

        if layer not in ['layer3', 'layer4']:
            raise ValueError(f"Unknown layer: {layer}. Must be 'layer3' or 'layer4'")

        self.backbone = backbone
        self.layer = layer
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Get feature dimension
        self.feature_dim = self.FEATURE_DIMS[(backbone, layer)]

        logger.info(f"Initializing ResNetFeatureExtractor: {backbone}, {layer} → {self.feature_dim}-dim")

        # Load ResNet model
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        else:  # resnet34
            resnet = models.resnet34(pretrained=pretrained)

        # Freeze all parameters
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.eval()

        # Build feature extractor up to specified layer
        layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        ]

        if layer == 'layer4':
            layers.append(resnet.layer4)

        # Add global average pooling and flatten
        layers.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        ])

        self.features = nn.Sequential(*layers)
        self.features.to(self.device)

        # ImageNet preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"ResNetFeatureExtractor initialized on {self.device}")

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Extract features from a single image.

        Args:
            image: PIL Image in RGB format

        Returns:
            Feature tensor of shape (feature_dim,)

        Examples:
            >>> extractor = ResNetFeatureExtractor('resnet50', 'layer3')
            >>> img = Image.open('photo.jpg')
            >>> features = extractor.extract(img)
            >>> features.shape
            torch.Size([1024])
        """
        self.features.eval()

        # Preprocess and add batch dimension
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Extract features
        features = self.features(img_tensor)

        return features.squeeze(0).cpu()  # Remove batch dim and move to CPU

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Extract features from multiple images in batches.

        Args:
            images: List of PIL Images in RGB format
            batch_size: Batch size for processing

        Returns:
            Feature tensor of shape (num_images, feature_dim)

        Examples:
            >>> extractor = ResNetFeatureExtractor('resnet50', 'layer3')
            >>> imgs = [Image.open(f'photo{i}.jpg') for i in range(100)]
            >>> features = extractor.extract_batch(imgs, batch_size=32)
            >>> features.shape
            torch.Size([100, 1024])
        """
        self.features.eval()

        all_features = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            # Preprocess batch
            batch_tensors = torch.stack([
                self.preprocess(img) for img in batch_images
            ]).to(self.device)

            # Extract features
            batch_features = self.features(batch_tensors)
            all_features.append(batch_features.cpu())

        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for pre-processed tensors.

        Args:
            x: Preprocessed image tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        return self.features(x)

    def __repr__(self) -> str:
        return (f"ResNetFeatureExtractor(backbone='{self.backbone}', "
                f"layer='{self.layer}', feature_dim={self.feature_dim}, "
                f"device='{self.device}')")
