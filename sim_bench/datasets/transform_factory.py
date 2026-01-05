"""
Transform factory for creating image preprocessing pipelines.

Provides transform functions independent of model implementations.
Supports both our internal transforms and external transforms (e.g., from Series-Photo-Selection).
"""
import logging
from typing import Optional, Tuple
from torchvision import transforms
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class AspectRatioResizeAndPad:
    """
    Resize image preserving aspect ratio, then pad to square.

    Matches PhotoTriage paper preprocessing.
    """

    def __init__(
        self,
        target_size: int = 224,
        mean_color: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.target_size = target_size
        self.mean_color = mean_color
        self.normalize = transforms.Normalize(mean=normalize_mean, std=normalize_std)

    def __call__(self, img):
        """Apply transform to PIL image."""
        # Get original size
        w, h = img.size

        # Resize so larger dimension equals target_size
        if w > h:
            new_w = self.target_size
            new_h = int(h * self.target_size / w)
        else:
            new_h = self.target_size
            new_w = int(w * self.target_size / h)

        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Convert to tensor
        img_tensor = transforms.ToTensor()(img)

        # Pad to square with mean color
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w

        # Create padding tensor with mean color
        padded = torch.ones(3, self.target_size, self.target_size)
        for c in range(3):
            padded[c] = self.mean_color[c]

        # Place resized image in center
        top = pad_h // 2
        left = pad_w // 2
        padded[:, top:top+new_h, left:left+new_w] = img_tensor

        # Normalize
        return self.normalize(padded)


def create_transform(config: dict):
    """
    Create image preprocessing transform from config.

    Supports multiple transform types:
    - 'standard': Standard ImageNet preprocessing (resize + center crop)
    - 'paper': PhotoTriage paper preprocessing (aspect ratio + padding)
    - 'external': Use transform from external module (e.g., Series-Photo-Selection)

    Args:
        config: Configuration dict with transform settings

    Returns:
        Transform function or None (if external transform is used in dataset)

    Example:
        >>> config = {'transform_type': 'standard'}
        >>> transform = create_transform(config)
        >>> config = {'transform_type': 'paper', 'use_paper_preprocessing': True}
        >>> transform = create_transform(config)
    """
    transform_type = config.get('transform_type', 'auto')

    # Auto-detect based on model_type or other config
    if transform_type == 'auto':
        model_type = config.get('model_type', 'siamese_cnn')
        use_external = config.get('use_external_dataloader', False)
        use_paper = config.get('model', {}).get('use_paper_preprocessing', False)

        if model_type == 'reference' or use_external:
            transform_type = 'external'
        elif use_paper:
            transform_type = 'paper'
        else:
            transform_type = 'standard'

        logger.info(f"Auto-detected transform type: {transform_type}")

    if transform_type == 'external':
        # External transform (e.g., from Series-Photo-Selection)
        # Return None - the external dataset will apply its own transform
        logger.info("Using external transform (applied in dataset __getitem__)")
        return None

    elif transform_type == 'paper':
        # PhotoTriage paper preprocessing
        padding_mean_color = config.get('model', {}).get('padding_mean_color')
        if padding_mean_color:
            padding_mean_color = tuple(padding_mean_color)
        else:
            padding_mean_color = (0.485, 0.456, 0.406)

        transform = AspectRatioResizeAndPad(
            target_size=224,
            mean_color=padding_mean_color,
            normalize_mean=(0.485, 0.456, 0.406),
            normalize_std=(0.229, 0.224, 0.225)
        )
        logger.info(f"Created paper-style transform (aspect ratio + padding)")
        return transform

    elif transform_type == 'standard':
        # Standard ImageNet preprocessing
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Created standard ImageNet transform")
        return transform

    else:
        raise ValueError(f"Unknown transform_type: {transform_type}. "
                        f"Must be 'standard', 'paper', 'external', or 'auto'")


def get_external_transform(external_module_path: str = None):
    """
    Import and return transform from external module.

    Args:
        external_module_path: Path to external module (e.g., 'D:\\Projects\\Series-Photo-Selection')

    Returns:
        Transform function from external module

    Example:
        >>> transform = get_external_transform('D:\\Projects\\Series-Photo-Selection')
    """
    import sys
    if external_module_path and external_module_path not in sys.path:
        sys.path.insert(0, external_module_path)

    try:
        from data.dataloader import transform
        logger.info(f"Loaded external transform from {external_module_path}")
        return transform
    except ImportError as e:
        logger.warning(f"Could not import external transform: {e}")
        return None
