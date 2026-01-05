"""
Dataset Factory - Creates dataset instances from different sources.

This factory handles importing and instantiating datasets from:
- PhotoTriage (internal)
- Series-Photo-Selection (external)
- Any custom source

Usage:
    factory = DatasetFactory(source='phototriage', config=config)
    train_data, val_data, test_data = factory.create_datasets()

    # OR
    factory = DatasetFactory(source='external', config=config)
    train_data, val_data, test_data = factory.create_datasets()
"""
import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DatasetFactory:
    """
    Factory for creating dataset instances from different sources.

    This handles all the importing, path manipulation, and instantiation
    so you don't have to worry about it in your training code.
    """

    def __init__(self, source: str, config: Dict[str, Any]):
        """
        Initialize dataset factory.

        Args:
            source: Dataset source - 'phototriage', 'external', or custom
            config: Configuration dictionary with data parameters

        Supported sources:
            - 'phototriage': PhotoTriageData with DataFrames
            - 'external': Series-Photo-Selection MyDataset
            - Custom: Provide your own in config['custom_dataset_class']
        """
        self.source = source.lower()
        self.config = config

    def create_datasets(self) -> Tuple:
        """
        Create dataset instances based on source.

        Returns:
            For 'phototriage': (data, train_df, val_df, test_df)
            For 'external': (train_dataset, val_dataset, test_dataset)

        The return format varies by source, but downstream code
        can handle both via the DataLoaderFactory.
        """
        if self.source == 'phototriage':
            return self._create_phototriage_datasets()
        elif self.source == 'external':
            return self._create_external_datasets()
        else:
            raise ValueError(f"Unknown source: {self.source}. Use 'phototriage' or 'external'")

    def _create_phototriage_datasets(self) -> Tuple:
        """
        Create PhotoTriage datasets and DataFrames.

        Returns:
            (data, train_df, val_df, test_df)
        """
        from sim_bench.datasets.phototriage_data import PhotoTriageData

        logger.info("Creating PhotoTriage datasets")

        # Create PhotoTriageData instance
        data = PhotoTriageData(
            root_dir=self.config['data']['root_dir'],
            min_agreement=self.config['data'].get('min_agreement', 0.7),
            min_reviewers=self.config['data'].get('min_reviewers', 2)
        )

        # Get splits
        train_df, val_df, test_df = data.get_series_based_splits(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=self.config.get('seed', 42),
            quick_experiment=self.config['data'].get('quick_experiment')
        )

        logger.info(f"PhotoTriage: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

        return data, train_df, val_df, test_df

    def _create_external_datasets(self) -> Tuple:
        """
        Create external datasets (Series-Photo-Selection).

        Handles importing from external path automatically.

        Returns:
            (train_dataset, val_dataset, test_dataset or None)
        """
        # Get external path from config
        external_path = self.config['data'].get(
            'external_path',
            r'D:\Projects\Series-Photo-Selection'
        )

        # Add to path if not already there
        if external_path not in sys.path:
            logger.info(f"Adding external path to sys.path: {external_path}")
            sys.path.insert(0, external_path)

        # Import external dataset
        try:
            from data.dataloader import MyDataset
            logger.info("Successfully imported MyDataset from external source")
        except ImportError as e:
            raise ImportError(
                f"Could not import external dataset from {external_path}. "
                f"Make sure the path is correct. Error: {e}"
            )

        # Get image root
        image_root = self.config['data'].get(
            'image_root',
            r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs'
        )

        logger.info(f"Creating external datasets from {image_root}")

        # Create dataset instances
        train_dataset = MyDataset(train=True, image_root=image_root)
        val_dataset = MyDataset(train=False, image_root=image_root)

        # External dataset doesn't have separate test, return None
        test_dataset = None

        logger.info(f"External: {len(train_dataset)} train, {len(val_dataset)} val")

        return train_dataset, val_dataset, test_dataset


def create_datasets_from_config(config: Dict[str, Any]) -> Tuple:
    """
    Convenience function to create datasets directly from config.

    Args:
        config: Configuration dictionary with:
            - 'data_source': 'phototriage' or 'external'
            - 'data': data configuration parameters

    Returns:
        Dataset instances (format depends on source)

    Example:
        >>> config = {
        ...     'data_source': 'phototriage',
        ...     'data': {'root_dir': 'data/phototriage'},
        ...     'seed': 42
        ... }
        >>> data, train_df, val_df, test_df = create_datasets_from_config(config)

        >>> config = {
        ...     'data_source': 'external',
        ...     'data': {
        ...         'external_path': r'D:\\Projects\\Series-Photo-Selection',
        ...         'image_root': 'path/to/images'
        ...     }
        ... }
        >>> train_data, val_data, test_data = create_datasets_from_config(config)
    """
    source = config.get('data_source', 'phototriage')
    factory = DatasetFactory(source=source, config=config)
    return factory.create_datasets()


# Convenience functions for direct usage

def create_phototriage_datasets(
    root_dir: str,
    min_agreement: float = 0.7,
    min_reviewers: int = 2,
    seed: int = 42,
    quick_experiment: Optional[float] = None
):
    """
    Create PhotoTriage datasets with minimal configuration.

    Args:
        root_dir: Path to PhotoTriage data
        min_agreement: Minimum agreement threshold
        min_reviewers: Minimum number of reviewers
        seed: Random seed
        quick_experiment: Optional fraction for quick testing

    Returns:
        (data, train_df, val_df, test_df)
    """
    config = {
        'data': {
            'root_dir': root_dir,
            'min_agreement': min_agreement,
            'min_reviewers': min_reviewers,
            'quick_experiment': quick_experiment
        },
        'seed': seed
    }
    factory = DatasetFactory(source='phototriage', config=config)
    return factory.create_datasets()


def create_external_datasets(
    external_path: str = r'D:\Projects\Series-Photo-Selection',
    image_root: str = r'D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs'
):
    """
    Create external datasets (Series-Photo-Selection) with minimal configuration.

    Args:
        external_path: Path to Series-Photo-Selection project
        image_root: Path to images

    Returns:
        (train_dataset, val_dataset, None)
    """
    config = {
        'data': {
            'external_path': external_path,
            'image_root': image_root
        }
    }
    factory = DatasetFactory(source='external', config=config)
    return factory.create_datasets()
