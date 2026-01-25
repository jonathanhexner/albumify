"""
AffectNet dataset loader with landmark support.

AffectNet structure:
    train/
        expression_id/
            image.jpg
    val/
        expression_id/
            image.jpg

Expressions: 0=Neutral, 1=Happiness, 2=Sadness, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger, 7=Contempt
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# Landmark swap indices for horizontal flip
# For 5-point: [left_eye, right_eye, nose, mouth_left, mouth_right]
# After flip: left_eye <-> right_eye, mouth_left <-> mouth_right
FLIP_SWAP_5 = [1, 0, 2, 4, 3]

# For 10-point: [left_eye_left, left_eye_right, right_eye_left, right_eye_right,
#                nose_tip, nose_bottom, mouth_left, mouth_right, mouth_top, mouth_bottom]
FLIP_SWAP_10 = [3, 2, 1, 0, 4, 5, 7, 6, 8, 9]

# Expression name to ID mapping (AffectNet uses named folders)
EXPRESSION_NAME_TO_ID = {
    'neutral': 0,
    'happiness': 1, 'happy': 1,
    'sadness': 2, 'sad': 2,
    'surprise': 3, 'surprised': 3,
    'fear': 4, 'fearful': 4,
    'disgust': 5, 'disgusted': 5,
    'anger': 6, 'angry': 6,
    'contempt': 7,
}


class AffectNetDataset(Dataset):
    """
    AffectNet dataset with expression labels and landmarks.

    Supports loading pre-extracted landmarks from cache.
    Handles synchronized augmentation (horizontal flip) for both image and landmarks.
    """

    def __init__(
        self,
        data_dir: Path,
        landmarks_cache: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
        num_landmarks: int = 5,
        horizontal_flip_prob: float = 0.0
    ):
        """
        Initialize AffectNet dataset.

        Args:
            data_dir: Root directory (contains train/ or val/ subdirectories)
            landmarks_cache: Optional path to landmarks cache JSON
            transform: Optional torchvision transforms (should NOT include RandomHorizontalFlip)
            num_landmarks: Number of landmarks per image
            horizontal_flip_prob: Probability of horizontal flip (applied to both image and landmarks)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.num_landmarks = num_landmarks
        self.horizontal_flip_prob = horizontal_flip_prob
        self.flip_swap = FLIP_SWAP_5 if num_landmarks == 5 else FLIP_SWAP_10

        self.samples = self._build_samples()
        self.landmarks = self._load_landmarks(landmarks_cache) if landmarks_cache else {}

        logger.info(f"Loaded {len(self.samples)} samples from {data_dir}")
        if self.landmarks:
            logger.info(f"Loaded landmarks for {len(self.landmarks)} images")

    def _build_samples(self) -> list:
        """Build list of (image_path, expression_label) samples."""
        samples = []

        for expr_dir in sorted(self.data_dir.iterdir()):
            if not expr_dir.is_dir():
                continue

            # Handle both numeric ("0", "1", ...) and named ("anger", "happiness", ...) folders
            dir_name = expr_dir.name.lower()
            if dir_name.isdigit():
                expr_id = int(dir_name)
                if expr_id < 0 or expr_id > 7:
                    continue
            elif dir_name in EXPRESSION_NAME_TO_ID:
                expr_id = EXPRESSION_NAME_TO_ID[dir_name]
            else:
                logger.warning(f"Skipping unknown expression folder: {expr_dir.name}")
                continue

            for img_file in expr_dir.glob("*.jpg"):
                samples.append((img_file, expr_id))

        return samples

    def _load_landmarks(self, cache_path: Path) -> Dict[Path, np.ndarray]:
        """Load landmarks from cache file."""
        if not cache_path.exists():
            logger.warning(f"Landmarks cache not found: {cache_path}")
            return {}
        
        with open(cache_path) as f:
            data = json.load(f)
        
        landmarks = {}
        for path_str, landmarks_list in data.items():
            path = Path(path_str)
            if landmarks_list is not None:
                landmarks[path] = np.array(landmarks_list, dtype=np.float32)
        
        return landmarks

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample.

        Returns:
            Dict with 'image', 'expression', 'landmarks', 'has_landmarks'
        """
        img_path, expr_label = self.samples[idx]

        image = Image.open(img_path).convert('RGB')

        landmarks = self.landmarks.get(img_path)
        has_landmarks = landmarks is not None

        if landmarks is None:
            landmarks = np.zeros((self.num_landmarks, 2), dtype=np.float32)
        else:
            landmarks = landmarks.copy()

        # Apply synchronized horizontal flip to both image and landmarks
        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            image = TF.hflip(image)
            if has_landmarks:
                # Mirror x coordinates: x_new = 1 - x_old
                landmarks[:, 0] = 1.0 - landmarks[:, 0]
                # Swap left/right landmarks
                landmarks = landmarks[self.flip_swap]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'expression': torch.tensor(expr_label, dtype=torch.long),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'has_landmarks': torch.tensor(has_landmarks, dtype=torch.bool)
        }


def create_affectnet_transform(config: dict) -> transforms.Compose:
    """
    Create transforms for AffectNet training.

    Note: Horizontal flip is handled separately in the dataset to synchronize
    with landmark transformations. Do NOT add RandomHorizontalFlip here.
    """
    transform_cfg = config.get('transform', {})
    input_size = transform_cfg.get('input_size', 224)

    mean = transform_cfg.get('normalize_mean', [0.485, 0.456, 0.406])
    std = transform_cfg.get('normalize_std', [0.229, 0.224, 0.225])

    aug_cfg = transform_cfg.get('augmentation', {})

    train_transforms = [
        transforms.Resize((input_size, input_size)),
    ]

    # Note: horizontal_flip is handled in dataset.__getitem__ to sync with landmarks

    if 'color_jitter' in aug_cfg:
        cj = aug_cfg['color_jitter']
        train_transforms.append(transforms.ColorJitter(
            brightness=cj.get('brightness', 0),
            contrast=cj.get('contrast', 0),
            saturation=cj.get('saturation', 0),
            hue=cj.get('hue', 0)
        ))

    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transforms.Compose(train_transforms)


def get_horizontal_flip_prob(config: dict) -> float:
    """Extract horizontal flip probability from config."""
    return config.get('transform', {}).get('augmentation', {}).get('horizontal_flip', 0.0)


def create_val_transform(config: dict) -> transforms.Compose:
    """Create validation transforms (no augmentation)."""
    transform_cfg = config.get('transform', {})
    input_size = transform_cfg.get('input_size', 224)
    mean = transform_cfg.get('normalize_mean', [0.485, 0.456, 0.406])
    std = transform_cfg.get('normalize_std', [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
