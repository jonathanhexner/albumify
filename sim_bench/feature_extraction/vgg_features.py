"""
VGG feature extraction for PhotoTriage replication.

Implements VGG16 feature extraction matching the original PhotoTriage paper:
- VGG16 architecture (16 layers)
- Removes FC-1000 and softmax layers
- Output: 4096-dimensional features
- Can be frozen or fine-tuned
- Supports paper's aspect-ratio preserving preprocessing

Paper-Accurate Preprocessing:
To use the exact preprocessing method from the paper, set use_paper_preprocessing=True
and provide the training set mean color. To compute the mean color:

    python scripts/phototriage/compute_mean_color.py

This creates data/phototriage/training_mean_color.json with:
    {
      "mean_rgb_normalized": [0.460, 0.450, 0.430],  # For use in config
      "mean_rgb_255": [117.3, 114.7, 109.6],         # For reference
      "num_images": 11716
    }

Then use the normalized values in your config:
    padding_mean_color: [0.460, 0.450, 0.430]

Reference:
Chang, H., Yu, F., Wang, J., Ashley, D., & Finkelstein, A. (2016).
Automatic Triage for a Photo Series. ACM Transactions on Graphics.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AspectRatioPreservingResize:
    """
    Resize image maintaining aspect ratio and pad with mean color.

    Matches PhotoTriage paper preprocessing:
    "we resize each image so that its larger dimension is the required size,
    while maintaining the original aspect ratio and padding with the mean
    pixel color in the training set."

    Args:
        target_size: Target size for larger dimension (default: 256)
        output_size: Final output size after padding (default: 224)
        mean_color: RGB mean color for padding in [0,1] range (default: ImageNet mean)
                    Should be computed from training set for exact paper replication
    """
    def __init__(
        self,
        target_size: int = 256,
        output_size: int = 224,
        mean_color: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    ):
        self.target_size = target_size
        self.output_size = output_size
        self.mean_color = mean_color

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Resize image preserving aspect ratio and pad to square.

        Args:
            img: PIL Image in RGB format

        Returns:
            Resized and padded image (output_size x output_size)
        """
        # Get current size
        w, h = img.size

        # Resize so larger dimension = target_size
        if w > h:
            new_w = self.target_size
            new_h = int(h * (self.target_size / w))
        else:
            new_h = self.target_size
            new_w = int(w * (self.target_size / h))

        # Resize maintaining aspect ratio
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Create square canvas with mean color
        # Convert mean from [0,1] normalized to [0,255] RGB
        mean_rgb = tuple(int(c * 255) for c in self.mean_color)
        canvas = Image.new('RGB', (self.output_size, self.output_size), mean_rgb)

        # Center the resized image on canvas
        offset_x = (self.output_size - new_w) // 2
        offset_y = (self.output_size - new_h) // 2
        canvas.paste(img_resized, (offset_x, offset_y))

        return canvas


class VGGFeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for PhotoTriage.

    Extracts 4096-dimensional features from VGG16's FC7 layer
    (before the final FC-1000 classification layer).

    This matches the original PhotoTriage paper architecture.

    Args:
        pretrained: Whether to use ImageNet pre-trained weights
        freeze: Whether to freeze feature extraction (not trainable)
        device: Device to load model on

    Attributes:
        features: VGG16 convolutional layers
        avgpool: Adaptive average pooling
        fc6: First fully connected layer (4096-dim)
        fc7: Second fully connected layer (4096-dim)
        output_dim: Feature dimension (4096)
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = None,  # Backward compatibility (deprecated, use freeze_until instead)
        freeze_until: str = "all",  # Options: "all", "fc_layers", "none"
        device: Optional[str] = None,
        use_paper_preprocessing: bool = False,  # Aspect-ratio + padding vs resize + crop
        mean_color: Optional[Tuple[float, float, float]] = None  # For padding (paper uses training set mean)
    ):
        super().__init__()

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Handle backward compatibility: if freeze is explicitly set, use it; otherwise use freeze_until
        if freeze is not None:
            self.freeze_until = "all" if freeze else "none"
            self.freeze = freeze
        else:
            self.freeze_until = freeze_until
            self.freeze = (freeze_until == "all")
        self.output_dim = 4096
        self.use_paper_preprocessing = use_paper_preprocessing

        # Load VGG16
        if pretrained:
            logger.info("Loading VGG16 with ImageNet pre-trained weights")
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            logger.info("Loading VGG16 with random initialization")
            vgg16 = models.vgg16(weights=None)

        # Extract layers
        # VGG16 architecture:
        #   - features: Conv layers (output: 512 x 7 x 7 after pooling)
        #   - avgpool: Adaptive average pooling
        #   - classifier: FC layers
        self.features = vgg16.features  # Conv layers
        self.avgpool = vgg16.avgpool    # Adaptive pooling

        # FC layers from VGG16 classifier
        # classifier[0]: Linear(25088, 4096) - FC6
        # classifier[3]: Linear(4096, 4096)  - FC7
        # classifier[6]: Linear(4096, 1000)  - FC8 (removed)
        self.fc6 = vgg16.classifier[0]  # 25088 → 4096
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = vgg16.classifier[3]  # 4096 → 4096
        self.relu7 = nn.ReLU(inplace=True)

        # Note: Original paper removed dropout layers
        # VGG16 normally has dropout after FC6 and FC7
        # We don't include them to match the paper

        # Apply freezing strategy
        if self.freeze_until == "all":
            self._freeze_all()
            logger.info("VGG16 features frozen (not trainable)")
        elif self.freeze_until == "fc_layers":
            self._freeze_conv_layers()
            logger.info("VGG16 conv layers frozen, FC layers trainable (partial fine-tuning)")
        elif self.freeze_until == "none":
            logger.info("VGG16 fully trainable (end-to-end fine-tuning)")
        else:
            raise ValueError(f"Invalid freeze_until value: {self.freeze_until}. Must be 'all', 'fc_layers', or 'none'")

        # Image preprocessing
        if use_paper_preprocessing:
            # Paper preprocessing: aspect-ratio preserving + padding
            padding_mean = mean_color or (0.485, 0.456, 0.406)
            logger.info(f"Using paper preprocessing (aspect-ratio + padding with mean={padding_mean})")
            self.preprocess = transforms.Compose([
                AspectRatioPreservingResize(
                    target_size=256,
                    output_size=224,
                    mean_color=padding_mean
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Standard preprocessing: resize + center crop
            logger.info("Using standard preprocessing (resize + center crop)")
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.to(self.device)
        logger.info(f"VGG16 feature extractor initialized (output_dim={self.output_dim})")

    def _freeze_all(self):
        """Freeze all parameters so they're not trainable."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _freeze_conv_layers(self):
        """Freeze only conv layers, keep FC layers trainable."""
        for param in self.features.parameters():
            param.requires_grad = False
        # FC6 and FC7 remain trainable (requires_grad=True by default)
        self.features.eval()  # Put conv layers in eval mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract VGG16 features.

        Args:
            x: Input images (batch_size, 3, H, W)
               Expected size: 224x224 (VGG16 standard)

        Returns:
            Features of shape (batch_size, 4096)
        """
        # Conv layers
        x = self.features(x)  # (B, 512, 7, 7)

        # Adaptive pooling
        x = self.avgpool(x)   # (B, 512, 7, 7)

        # Flatten
        x = torch.flatten(x, 1)  # (B, 25088)

        # FC6
        x = self.fc6(x)       # (B, 4096)
        x = self.relu6(x)
        # No dropout (removed as per paper)

        # FC7
        x = self.fc7(x)       # (B, 4096)
        x = self.relu7(x)
        # No dropout (removed as per paper)

        return x  # (B, 4096)

    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Extract features from a single image.

        Args:
            image: PIL Image in RGB format

        Returns:
            Feature tensor of shape (4096,)
        """
        self.eval()

        # Preprocess and add batch dimension
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.forward(img_tensor)

        return features.squeeze(0).cpu()  # Remove batch dim and move to CPU


def get_vgg_feature_dim() -> int:
    """Get VGG16 feature dimension (always 4096)."""
    return 4096


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create extractor
    extractor = VGGFeatureExtractor(pretrained=True, freeze=True)

    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)
    features = extractor(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature dim: {extractor.output_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    trainable_params = sum(p.numel() for p in extractor.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
