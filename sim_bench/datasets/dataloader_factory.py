"""
DataLoader Factory - Unified interface for creating dataloaders from any source.

This module provides a clean factory pattern for creating dataloaders from:
- PhotoTriage DataFrames
- External datasets (Series-Photo-Selection)
- Any custom dataset implementation

Usage:
    factory = DataLoaderFactory(seed=42)  # For reproducible shuffling
    train_loader, val_loader, test_loader = factory.create_from_phototriage(...)
    # OR
    train_loader, val_loader, test_loader = factory.create_from_external(...)
"""
from pathlib import Path
from typing import Tuple, Optional, Union
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sim_bench.datasets.siamese_dataloaders import EndToEndPairDataset

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory for creating dataloaders from different sources.

    This provides a unified interface - all methods return the same format:
    (train_loader, val_loader, test_loader) or (train_loader, val_loader)
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 0,
        shuffle_train: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize factory with common dataloader parameters.

        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            shuffle_train: Whether to shuffle training data
            seed: Random seed (kept for compatibility, but not used -
                  DataLoader uses Python's random module instead)
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.seed = seed

        # Note: We don't create a PyTorch Generator anymore
        # DataLoader will use Python's random module for shuffling
        # This matches the reference implementation's behavior
        self.generator = None

    def create_from_dataframes(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame],
        image_dir: Union[str, Path],
        transform
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create dataloaders from pandas DataFrames.

        This is the standard method for PhotoTriage data.

        Args:
            train_df: Training pairs DataFrame
            val_df: Validation pairs DataFrame
            test_df: Test pairs DataFrame (optional)
            image_dir: Directory containing images
            transform: Transform function to apply

        Returns:
            (train_loader, val_loader, test_loader) if test_df provided
            (train_loader, val_loader, None) otherwise
        """
        logger.info(f"Creating dataloaders from DataFrames: "
                   f"train={len(train_df)}, val={len(val_df)}, "
                   f"test={len(test_df) if test_df is not None else 0}")

        # Create datasets
        train_dataset = EndToEndPairDataset(train_df, image_dir, transform)
        val_dataset = EndToEndPairDataset(val_df, image_dir, transform)
        test_dataset = EndToEndPairDataset(test_df, image_dir, transform) if test_df is not None else None

        # Create loaders
        train_loader = self._create_loader(train_dataset, shuffle=self.shuffle_train)
        val_loader = self._create_loader(val_dataset, shuffle=False)
        test_loader = self._create_loader(test_dataset, shuffle=False) if test_dataset else None

        return train_loader, val_loader, test_loader

    def create_from_phototriage(
        self,
        data,  # PhotoTriageData instance
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        transform
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create dataloaders from PhotoTriageData instance.

        Convenience method that automatically uses the correct image directory.

        Args:
            data: PhotoTriageData instance
            train_df: Training pairs DataFrame
            val_df: Validation pairs DataFrame
            test_df: Test pairs DataFrame
            transform: Transform function

        Returns:
            (train_loader, val_loader, test_loader)
        """
        return self.create_from_dataframes(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            image_dir=data.train_val_img_dir,
            transform=transform
        )

    def create_from_external(
        self,
        train_dataset,
        val_dataset,
        test_dataset=None
    ) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Create dataloaders from external dataset instances.

        External datasets (like MyDataset) must return dictionaries with keys:
        {'img1', 'img2', 'winner', 'image1', 'image2'}

        Args:
            train_dataset: External training dataset instance
            val_dataset: External validation dataset instance
            test_dataset: External test dataset instance (optional)

        Returns:
            (train_loader, val_loader, test_loader) if test_dataset provided
            (train_loader, val_loader, None) otherwise

        Example:
            >>> from data.dataloader import MyDataset
            >>> train_data = MyDataset(train=True, image_root=image_root)
            >>> val_data = MyDataset(train=False, image_root=image_root)
            >>> factory = DataLoaderFactory(batch_size=8, seed=42)
            >>> train_loader, val_loader, _ = factory.create_from_external(
            ...     train_data, val_data
            ... )
        """
        logger.info(f"Creating dataloaders from external datasets: "
                   f"train={len(train_dataset)}, val={len(val_dataset)}, "
                   f"test={len(test_dataset) if test_dataset else 0}")

        # Use external datasets directly - they already return the correct format
        train_loader = self._create_loader(train_dataset, shuffle=self.shuffle_train)
        val_loader = self._create_loader(val_dataset, shuffle=False)
        test_loader = self._create_loader(test_dataset, shuffle=False) if test_dataset else None

        return train_loader, val_loader, test_loader

    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        """
        Create a DataLoader from a dataset.

        Note: Does NOT use PyTorch Generator - relies on Python's random module
        for shuffling (same as reference implementation). This ensures
        deterministic behavior when random.seed() is set globally.
        """
        if dataset is None:
            return None

        loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers,
            'pin_memory': False  # Compatible with CPU training
        }

        # Don't use generator - let DataLoader use Python's random module
        # This matches the reference implementation's behavior

        return DataLoader(dataset, **loader_kwargs)


# Convenience functions for backward compatibility

def create_dataloaders_unified(
    source: str,
    batch_size: int = 16,
    num_workers: int = 0,
    **kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Unified function to create dataloaders from any source.

    Args:
        source: Source type - 'phototriage', 'dataframes', or 'external'
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Source-specific parameters

    Returns:
        (train_loader, val_loader, test_loader)

    Example:
        >>> # PhotoTriage source
        >>> loaders = create_dataloaders_unified(
        ...     source='phototriage',
        ...     data=phototriage_data,
        ...     train_df=train_df,
        ...     val_df=val_df,
        ...     test_df=test_df,
        ...     transform=transform
        ... )
        >>>
        >>> # External source
        >>> loaders = create_dataloaders_unified(
        ...     source='external',
        ...     train_dataset=external_train,
        ...     val_dataset=external_val,
        ...     test_dataset=external_test
        ... )
    """
    factory = DataLoaderFactory(batch_size=batch_size, num_workers=num_workers)

    if source == 'phototriage':
        return factory.create_from_phototriage(**kwargs)
    elif source == 'dataframes':
        return factory.create_from_dataframes(**kwargs)
    elif source == 'external':
        return factory.create_from_external(**kwargs)
    else:
        raise ValueError(f"Unknown source: {source}. Must be 'phototriage', 'dataframes', or 'external'")
