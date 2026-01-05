"""
Siamese network dataloaders for pairwise image comparison.

This module provides flexible dataloader implementations that can work with both
internal PhotoTriage data and external data formats. The interfaces are designed
to be interchangeable with external dataloader implementations.
"""
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class EndToEndPairDataset(Dataset):
    """
    Dataset that loads raw images for end-to-end Siamese training.

    Compatible interface with external dataloaders - returns pairs of images
    with winner labels.

    Args:
        pairs_df: DataFrame with columns ['image1', 'image2', 'winner']
        image_dir: Directory containing images
        transform: Transform function to apply to images

    Returns:
        Dictionary with:
            - 'img1': First image tensor
            - 'img2': Second image tensor
            - 'winner': Winner label (0 or 1)
            - 'image1': First image filename
            - 'image2': Second image filename
    """

    def __init__(self, pairs_df: pd.DataFrame, image_dir: str, transform):
        self.pairs_df = pairs_df
        self.image_dir = Path(image_dir)
        self.transform = transform
        logger.info(f"Dataset initialized with {len(pairs_df)} pairs from {image_dir}")

    def __len__(self):
        return len(self.pairs_df)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying pairs DataFrame with metadata."""
        return self.pairs_df

    def get_image_dir(self) -> Path:
        """Return the image directory path."""
        return self.image_dir

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]

        img1_path = self.image_dir / row['image1']
        img2_path = self.image_dir / row['image2']

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)

        return {
            'img1': img1_tensor,
            'img2': img2_tensor,
            'winner': torch.tensor(int(row['winner']), dtype=torch.long),
            'image1': row['image1'],
            'image2': row['image2']
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_dir: str,
    transform,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch dataloaders for train/val/test splits.

    This is the standard interface for creating dataloaders from DataFrames.

    Args:
        train_df: Training pairs DataFrame
        val_df: Validation pairs DataFrame
        test_df: Test pairs DataFrame
        image_dir: Directory containing images
        transform: Transform function to apply to images
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = EndToEndPairDataset(train_df, image_dir, transform)
    val_dataset = EndToEndPairDataset(val_df, image_dir, transform)
    test_dataset = EndToEndPairDataset(test_df, image_dir, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    logger.info(f"Created dataloaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches, test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def create_dataloaders_from_external(
    external_train_dataset,
    external_val_dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders from external dataset instances.

    External datasets (like MyDataset) must return dictionaries with keys:
    {'img1', 'img2', 'winner', 'image1', 'image2'}

    Args:
        external_train_dataset: External training dataset instance
        external_val_dataset: External validation dataset instance
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> # Using external dataloader
        >>> from data.dataloader import MyDataset
        >>> train_data = MyDataset(train=True, image_root=image_root)
        >>> val_data = MyDataset(train=False, image_root=image_root)
        >>> train_loader, val_loader = create_dataloaders_from_external(
        ...     train_data, val_data, batch_size=8
        ... )
    """
    # Use external datasets directly - they already return the correct format
    train_loader = DataLoader(
        external_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=False  # Compatible with CPU training
    )
    val_loader = DataLoader(
        external_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    logger.info(f"Created external dataloaders: train={len(train_loader)} batches, "
                f"val={len(val_loader)} batches")

    return train_loader, val_loader


def create_phototriage_dataloaders(
    data,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    transform,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create dataloaders from PhotoTriageData instance.

    This is a convenience wrapper around create_dataloaders that automatically
    uses the correct image directory from the PhotoTriageData instance.

    Args:
        data: PhotoTriageData instance
        train_df: Training pairs DataFrame
        val_df: Validation pairs DataFrame
        test_df: Test pairs DataFrame
        transform: Transform function to apply to images
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    return create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        image_dir=data.train_val_img_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )


def get_dataset_from_loader(loader: DataLoader):
    """
    Extract the underlying dataset from a DataLoader.

    This is the key function that enables the refactor - it allows
    training code to extract metadata from loaders when needed.

    Args:
        loader: PyTorch DataLoader

    Returns:
        The underlying dataset instance (EndToEndPairDataset or ExternalDatasetAdapter)

    Example:
        >>> dataset = get_dataset_from_loader(train_loader)
        >>> pairs_df = dataset.get_dataframe()
        >>> image_dir = dataset.get_image_dir()
    """
    return loader.dataset
