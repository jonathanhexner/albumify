"""
Data loader for PhotoTriage attribute-labeled pairs.

Provides PyTorch Dataset and DataLoader for training contrastive models.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class AttributePairDataset(Dataset):
    """
    Dataset of attribute-labeled image pairs.

    Each item is a pair (image_a, image_b) with:
    - Global preference (A or B)
    - Per-attribute preferences (which image wins for each attribute)
    """

    def __init__(
        self,
        pairs_file: Path,
        attribute_names: List[str],
        transform: Optional[transforms.Compose] = None,
        load_all_attributes: bool = True
    ):
        """
        Initialize dataset.

        Args:
            pairs_file: Path to pairs JSONL file (train/val/test_pairs.jsonl)
            attribute_names: List of attribute names the model uses
            transform: Image transformations (None = default CLIP transform)
            load_all_attributes: If True, load all pairs even if no attributes
        """
        self.pairs_file = Path(pairs_file)
        self.attribute_names = attribute_names
        self.transform = transform or self._get_default_transform()
        self.load_all_attributes = load_all_attributes

        # Load pairs
        self.pairs = self._load_pairs()

        logger.info(f"Loaded {len(self.pairs)} pairs from {pairs_file}")

    def _load_pairs(self) -> List[Dict]:
        """Load pairs from JSONL file."""
        pairs = []

        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line)

                # Filter if needed
                if not self.load_all_attributes and len(pair['attributes']) == 0:
                    continue

                pairs.append(pair)

        return pairs

    def _get_default_transform(self) -> transforms.Compose:
        """Get default CLIP image transform."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
                std=[0.26862954, 0.26130258, 0.27577711]   # CLIP std
            )
        ])

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a pair item.

        Returns:
            Dictionary with:
            - image_a: Tensor (3, 224, 224)
            - image_b: Tensor (3, 224, 224)
            - chosen: int (0=A, 1=B)
            - attribute_labels: Dict[str, int] {attr_name: 0=A, 1=B, -1=unlabeled}
            - pair_id: str
            - preference_strength: float
        """
        pair = self.pairs[idx]

        # Load images
        try:
            image_a = Image.open(pair['image_a_path']).convert('RGB')
            image_b = Image.open(pair['image_b_path']).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load images for pair {pair['pair_id']}: {e}")
            # Return a dummy pair (will be filtered in collate_fn)
            return None

        # Apply transforms
        image_a = self.transform(image_a)
        image_b = self.transform(image_b)

        # Convert chosen to index (A=0, B=1)
        chosen = 0 if pair['chosen_image'] == 'A' else 1

        # Parse attribute labels
        attribute_labels = self._parse_attribute_labels(pair['attributes'])

        return {
            'image_a': image_a,
            'image_b': image_b,
            'chosen': chosen,
            'attribute_labels': attribute_labels,
            'pair_id': pair['pair_id'],
            'preference_strength': pair['preference_strength']
        }

    def _parse_attribute_labels(self, attributes: List[Dict]) -> Dict[str, int]:
        """
        Parse attribute labels from pair data.

        Args:
            attributes: List of attribute dicts from pair

        Returns:
            Dictionary mapping attribute name → winner index (0=A, 1=B, -1=unlabeled)
        """
        labels = {attr: -1 for attr in self.attribute_names}  # -1 = unlabeled

        for attr_dict in attributes:
            attr_name = attr_dict['name']

            if attr_name in labels:
                # Convert winner ("A" or "B") to index
                labels[attr_name] = 0 if attr_dict['winner'] == 'A' else 1

        return labels


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Collate function for DataLoader.

    Handles None items (failed image loads) and creates tensors.

    Args:
        batch: List of items from __getitem__

    Returns:
        Batched dictionary or None if all items failed
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    # Stack images
    images_a = torch.stack([item['image_a'] for item in batch])
    images_b = torch.stack([item['image_b'] for item in batch])

    # Stack chosen
    chosen = torch.tensor([item['chosen'] for item in batch], dtype=torch.long)

    # Combine attribute labels
    # Get all attribute names from first item
    attr_names = list(batch[0]['attribute_labels'].keys())

    attribute_labels = {}
    for attr_name in attr_names:
        labels = torch.tensor(
            [item['attribute_labels'][attr_name] for item in batch],
            dtype=torch.long
        )
        attribute_labels[attr_name] = labels

    # Other fields
    pair_ids = [item['pair_id'] for item in batch]
    preference_strengths = torch.tensor(
        [item['preference_strength'] for item in batch],
        dtype=torch.float32
    )

    return {
        'images_a': images_a,
        'images_b': images_b,
        'chosen': chosen,
        'attribute_labels': attribute_labels,
        'pair_ids': pair_ids,
        'preference_strengths': preference_strengths
    }


def create_data_loaders(
    train_file: Path,
    val_file: Path,
    test_file: Path,
    attribute_names: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    train_augment: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, val, and test data loaders.

    Args:
        train_file: Path to train_pairs.jsonl
        val_file: Path to val_pairs.jsonl
        test_file: Path to test_pairs.jsonl
        attribute_names: List of attribute names
        batch_size: Batch size
        num_workers: Number of worker processes
        train_augment: If True, apply data augmentation to training set

    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Define transforms
    if train_augment:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
    else:
        train_transform = None  # Use default

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # Create datasets
    train_dataset = AttributePairDataset(
        pairs_file=train_file,
        attribute_names=attribute_names,
        transform=train_transform,
        load_all_attributes=True
    )

    val_dataset = AttributePairDataset(
        pairs_file=val_file,
        attribute_names=attribute_names,
        transform=val_test_transform,
        load_all_attributes=True
    )

    test_dataset = AttributePairDataset(
        pairs_file=test_file,
        attribute_names=attribute_names,
        transform=val_test_transform,
        load_all_attributes=True
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    logger.info("Data loaders created:")
    logger.info(f"  Train: {len(train_dataset)} pairs, {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_dataset)} pairs, {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_dataset)} pairs, {len(test_loader)} batches")

    return train_loader, val_loader, test_loader
