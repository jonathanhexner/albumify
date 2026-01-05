"""
Unified PhotoTriage dataset management and preprocessing.

This module consolidates all PhotoTriage data operations:
- Loading train/val/test splits
- Processing pairwise comparisons
- Normalizing filenames
- Filtering by agreement/reviewers
- Creating datasets for training

Directory structure:
    D:/Similar Images/automatic_triage_photo_series/
    ├── train_val/
    │   ├── train_val_imgs/         # Training + validation images
    │   ├── reviews_trainval/       # Raw review JSONs
    │   ├── train_pairlist.txt
    │   └── val_pairlist.txt
    ├── test/
    │   ├── test_imgs/              # Held-out test images
    │   └── test_pairlist.txt
    ├── photo_triage_pairs_embedding_labels.csv   # Main pairwise CSV
    └── reviews_df.csv              # Aggregated reviews
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PhotoTriageData:
    """
    Unified interface for PhotoTriage dataset.

    Handles:
    - Train/val/test splits (official dataset splits)
    - Pairwise comparison loading
    - Filename normalization
    - Data filtering by agreement/reviewers

    Usage:
        data = PhotoTriageData(
            root_dir="D:/Similar Images/automatic_triage_photo_series",
            min_agreement=0.7,
            min_reviewers=2
        )

        train_df, val_df, test_df = data.get_splits()
    """

    def __init__(
        self,
        root_dir: str,
        min_agreement: float = 0.7,
        min_reviewers: int = 2
    ):
        """
        Initialize PhotoTriage data loader.

        Args:
            root_dir: Root directory containing train_val/ and test/
            csv_filename: Name of pairwise comparisons CSV
            min_agreement: Minimum agreement threshold (0-1)
            min_reviewers: Minimum number of reviewers
        """
        csv_filename: str = "photo_triage_pairs_embedding_labels.csv"
        self.root_dir = Path(root_dir)
        self.min_agreement = min_agreement
        self.min_reviewers = min_reviewers
        
        # Try to find CSV in root_dir first, then fall back to package
        self.csv_path = self.root_dir / csv_filename
        if not self.csv_path.exists():
            # Fall back to package data directory
            package_csv = Path(__file__).parent.parent.parent / "data" / "phototriage" / csv_filename
            if package_csv.exists():
                self.csv_path = package_csv
                logger.info(f"Using CSV from package: {package_csv}")
            else:
                raise FileNotFoundError(
                    f"CSV file not found in:\n"
                    f"  1. Dataset: {self.root_dir / csv_filename}\n"
                    f"  2. Package: {package_csv}"
                )

        # Image directories
        self.train_val_img_dir = self.root_dir / "train_val" / "train_val_imgs"
        self.test_img_dir = self.root_dir / "test" / "test_imgs"

        # Validate paths
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        if not self.train_val_img_dir.exists():
            raise FileNotFoundError(f"Train/val images not found: {self.train_val_img_dir}")
        if not self.test_img_dir.exists():
            raise FileNotFoundError(f"Test images not found: {self.test_img_dir}")

        logger.info(f"Initialized PhotoTriageData from {root_dir}")
        logger.info(f"  Train/val images: {len(list(self.train_val_img_dir.glob('*.JPG')))} files")
        logger.info(f"  Test images: {len(list(self.test_img_dir.glob('*.JPG')))} files")

    @staticmethod
    def normalize_filename(filename: str) -> str:
        """
        Convert CSV filename format to actual image filename format.

        CSV has: "1-1.JPG", "12-3.JPG"
        Images are: "000001-01.JPG", "000012-03.JPG"

        Args:
            filename: Filename from CSV (e.g., "1-1.JPG")

        Returns:
            Normalized filename (e.g., "000001-01.JPG")
        """
        # Split on '-' and '.'
        parts = filename.replace('.JPG', '').replace('.jpg', '').split('-')
        if len(parts) == 2:
            series_num, image_num = parts
            # Pad to 6 digits for series, 2 digits for image number
            normalized = f"{int(series_num):06d}-{int(image_num):02d}.JPG"
            return normalized
        else:
            # Return as-is if format doesn't match expected pattern
            return filename

    def load_raw_csv(self) -> pd.DataFrame:
        """
        Load raw pairwise comparisons CSV.

        Returns:
            Raw DataFrame with all pairs
        """
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} pairs from {self.csv_path}")
        return df

    def load_filtered_pairs(self) -> pd.DataFrame:
        """
        Load and filter pairwise comparisons.

        Returns:
            Filtered DataFrame with columns:
                - image1, image2: Normalized filenames
                - winner: 0 if image1 wins, 1 if image2 wins
                - agreement: Human agreement score (0-1)
                - num_reviewers: Number of reviewers
                - series_id: Series identifier
                - label_*: Quality attribute labels
        """
        df = self.load_raw_csv()

        logger.info(f"Columns: {list(df.columns)}")

        # Filter by agreement and reviewers
        df_filtered = df[
            (df['Agreement'] >= self.min_agreement) &
            (df['num_reviewers'] >= self.min_reviewers)
        ].copy()

        logger.info(f"After filtering (agreement>={self.min_agreement}, reviewers>={self.min_reviewers}): {len(df_filtered)} pairs")

        # Normalize filenames to match actual image files
        df_filtered['compareFile1'] = df_filtered['compareFile1'].apply(self.normalize_filename)
        df_filtered['compareFile2'] = df_filtered['compareFile2'].apply(self.normalize_filename)

        logger.info(f"Normalized filenames (e.g., '1-1.JPG' -> '{df_filtered.iloc[0]['compareFile1']}')")

        # Create winner label: 1 if MaxVote == compareID2 (image2 wins), else 0 (image1 wins)
        df_filtered['winner'] = (df_filtered['MaxVote'] == df_filtered['compareID2']).astype(int)

        # Rename columns for consistency
        df_filtered = df_filtered.rename(columns={
            'compareFile1': 'image1',
            'compareFile2': 'image2',
            'Agreement': 'agreement'
        })

        # Check winner distribution
        winner_dist = df_filtered['winner'].value_counts().to_dict()
        logger.info(f"Winner distribution: {winner_dist}")

        if len(winner_dist) == 1:
            logger.warning("WARNING: Only one winner class! Data may have labeling issues.")

        return df_filtered

    def get_official_splits(
        self,
        use_official_test: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Get train/val/test splits based on official dataset structure.

        The PhotoTriage dataset has TWO separate image directories:
        - train_val/train_val_imgs: For training and validation
        - test/test_imgs: Held-out test set

        Args:
            use_official_test: If True, use official test set from test/test_imgs
                             If False, split train_val into train/val/test

        Returns:
            (train_df, val_df, test_df) or (train_df, val_df, None)
        """
        df = self.load_filtered_pairs()

        if use_official_test:
            # Check which images are in test directory
            test_images = set(f.name for f in self.test_img_dir.glob('*.JPG'))
            train_val_images = set(f.name for f in self.train_val_img_dir.glob('*.JPG'))

            logger.info(f"Official test images: {len(test_images)}")
            logger.info(f"Train/val images: {len(train_val_images)}")

            # Split pairs based on which directory their images are in
            # A pair is "test" if BOTH images are in test directory
            def is_test_pair(row):
                return row['image1'] in test_images and row['image2'] in test_images

            def is_train_val_pair(row):
                return row['image1'] in train_val_images and row['image2'] in train_val_images

            test_mask = df.apply(is_test_pair, axis=1)
            train_val_mask = df.apply(is_train_val_pair, axis=1)

            test_df = df[test_mask].copy()
            train_val_df = df[train_val_mask].copy()

            # Further split train_val into train and val (80/20)
            n_train = int(0.8 * len(train_val_df))
            train_val_df = train_val_df.sample(frac=1, random_state=42).reset_index(drop=True)

            train_df = train_val_df[:n_train]
            val_df = train_val_df[n_train:]

            logger.info(f"Official splits:")
            logger.info(f"  Train: {len(train_df)} pairs")
            logger.info(f"  Val: {len(val_df)} pairs")
            logger.info(f"  Test: {len(test_df)} pairs")

            return train_df, val_df, test_df

        else:
            # Simple random split (old method)
            return self.get_random_splits(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    def get_series_based_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        quick_experiment: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get train/val/test splits based on series_id to prevent data leakage.

        This ensures that all pairs from a given photo series are in the same split,
        preventing the model from learning series-specific patterns rather than
        generalizable quality assessment.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_seed: Random seed for reproducibility
            quick_experiment: If set, subsample to this fraction of series
                            (e.g., 0.1 for 10% of series for quick testing)

        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        df = self.load_filtered_pairs()

        # Get unique series_ids and shuffle them
        unique_series = df['series_id'].unique()
        np.random.seed(random_seed)
        np.random.shuffle(unique_series)

        # Split series_ids
        n_series = len(unique_series)
        n_train = int(n_series * train_ratio)
        n_val = int(n_series * val_ratio)

        train_series = set(unique_series[:n_train])
        val_series = set(unique_series[n_train:n_train + n_val])
        test_series = set(unique_series[n_train + n_val:])

        # Split DataFrame based on series_id
        train_df = df[df['series_id'].isin(train_series)].reset_index(drop=True)
        val_df = df[df['series_id'].isin(val_series)].reset_index(drop=True)
        test_df = df[df['series_id'].isin(test_series)].reset_index(drop=True)

        n_total = len(df)
        logger.info(f"Series-based splits:")
        logger.info(f"  Train: {len(train_series)} series, {len(train_df)} pairs ({100*len(train_df)/n_total:.1f}%)")
        logger.info(f"  Val: {len(val_series)} series, {len(val_df)} pairs ({100*len(val_df)/n_total:.1f}%)")
        logger.info(f"  Test: {len(test_series)} series, {len(test_df)} pairs ({100*len(test_df)/n_total:.1f}%)")

        # Verify no series overlap
        assert len(train_series & val_series) == 0, "Train and val series overlap!"
        assert len(train_series & test_series) == 0, "Train and test series overlap!"
        assert len(val_series & test_series) == 0, "Val and test series overlap!"
        logger.info("  Verified: No series overlap between splits")

        # Quick experiment mode: subsample series
        if quick_experiment is not None:
            logger.info(f"\nQuick experiment mode: using {quick_experiment*100:.0f}% of series")
            train_df = self._subsample_series(train_df, quick_experiment, random_seed)
            val_df = self._subsample_series(val_df, quick_experiment, random_seed + 1)
            test_df = self._subsample_series(test_df, quick_experiment, random_seed + 2)

        return train_df, val_df, test_df

    def _subsample_series(self, df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
        """
        Subsample DataFrame to a fraction of series.

        Args:
            df: DataFrame with 'series_id' column
            fraction: Fraction of series to keep (e.g., 0.1 for 10%)
            seed: Random seed

        Returns:
            Subsampled DataFrame
        """
        if fraction <= 0 or fraction >= 1:
            raise ValueError(f"fraction must be between 0 and 1, got {fraction}")

        np.random.seed(seed)

        # Get unique series
        unique_series = df['series_id'].unique()
        n_original_series = len(unique_series)
        n_original_pairs = len(df)

        # Subsample series
        n_keep = max(1, int(n_original_series * fraction))
        selected_series = np.random.choice(unique_series, n_keep, replace=False)

        # Filter DataFrame
        df_subsampled = df[df['series_id'].isin(selected_series)].reset_index(drop=True)

        logger.info(f"  Subsampled: {n_original_series} series ({n_original_pairs} pairs) "
                   f"-> {n_keep} series ({len(df_subsampled)} pairs)")

        return df_subsampled

    def get_random_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get random train/val/test splits (ignores series_id - may cause data leakage!).

        WARNING: This method does NOT respect series boundaries and may result in
        images from the same series appearing in both train and test sets, leading
        to overoptimistic accuracy estimates. Use get_series_based_splits() instead
        for proper evaluation.

        Args:
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_seed: Random seed for reproducibility

        Returns:
            (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        df = self.load_filtered_pairs()

        # Shuffle
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = df[:n_train]
        val_df = df[n_train:n_train + n_val]
        test_df = df[n_train + n_val:]

        logger.info(f"Random splits (WARNING: May have series leakage!):")
        logger.info(f"  Train: {len(train_df)} pairs ({100*len(train_df)/n:.1f}%)")
        logger.info(f"  Val: {len(val_df)} pairs ({100*len(val_df)/n:.1f}%)")
        logger.info(f"  Test: {len(test_df)} pairs ({100*len(test_df)/n:.1f}%)")

        return train_df, val_df, test_df

    def get_image_path(self, image_name: str, split: str = 'train_val') -> Path:
        """
        Get full path to an image file.

        Args:
            image_name: Normalized filename (e.g., "000001-01.JPG")
            split: 'train_val' or 'test'

        Returns:
            Full path to image
        """
        if split == 'train_val':
            return self.train_val_img_dir / image_name
        elif split == 'test':
            return self.test_img_dir / image_name
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train_val' or 'test'")

    def verify_image_exists(self, df: pd.DataFrame, split: str = 'train_val') -> Dict[str, int]:
        """
        Verify that all images in DataFrame exist on disk.

        Args:
            df: DataFrame with 'image1' and 'image2' columns
            split: 'train_val' or 'test'

        Returns:
            Dictionary with counts:
                {'found': N, 'missing': M, 'missing_list': [...]}
        """
        img_dir = self.train_val_img_dir if split == 'train_val' else self.test_img_dir

        unique_images = set(df['image1'].unique()) | set(df['image2'].unique())

        missing = []
        for img_name in unique_images:
            img_path = img_dir / img_name
            if not img_path.exists():
                missing.append(img_name)

        result = {
            'total': len(unique_images),
            'found': len(unique_images) - len(missing),
            'missing': len(missing),
            'missing_list': missing
        }

        logger.info(f"Image verification for {split}:")
        logger.info(f"  Total unique images: {result['total']}")
        logger.info(f"  Found: {result['found']}")
        logger.info(f"  Missing: {result['missing']}")

        if missing:
            logger.warning(f"  First 10 missing: {missing[:10]}")

        return result

    def get_series_info(self) -> pd.DataFrame:
        """
        Get information about photo series.

        Returns:
            DataFrame with series statistics
        """
        df = self.load_filtered_pairs()

        # Extract series ID from image names (first 6 digits)
        df['series1'] = df['image1'].str[:6]
        df['series2'] = df['image2'].str[:6]

        # Count images per series
        all_series_images = pd.concat([
            df[['series1', 'image1']].rename(columns={'series1': 'series_id', 'image1': 'image'}),
            df[['series2', 'image2']].rename(columns={'series2': 'series_id', 'image2': 'image'})
        ])

        series_counts = all_series_images.groupby('series_id')['image'].nunique().reset_index()
        series_counts.columns = ['series_id', 'num_images']

        logger.info(f"Series statistics:")
        logger.info(f"  Total series: {len(series_counts)}")
        logger.info(f"  Avg images per series: {series_counts['num_images'].mean():.1f}")
        logger.info(f"  Min images per series: {series_counts['num_images'].min()}")
        logger.info(f"  Max images per series: {series_counts['num_images'].max()}")

        return series_counts

    def precompute_features(
        self,
        pairs_df: pd.DataFrame,
        feature_extractor,
        cache_dir: str,
        use_clip: bool = True,
        use_cnn: bool = True,
        use_iqa: bool = True
    ) -> Dict[str, 'torch.Tensor']:
        """
        Precompute and cache features for all images in pairs_df.

        Handles separate cache files for CLIP, CNN, and IQA features to allow
        extending with new features without recalculating existing ones.

        Args:
            pairs_df: DataFrame with 'image1' and 'image2' columns
            feature_extractor: MultiFeatureExtractor instance with extract_all() method
            cache_dir: Directory to store cache files
            use_clip: Whether to use CLIP features
            use_cnn: Whether to use CNN features
            use_iqa: Whether to use IQA features

        Returns:
            Dictionary mapping image_name → concatenated features tensor
        """
        import pickle
        import torch
        from tqdm import tqdm

        # Get unique images
        all_images = set(pairs_df['image1'].unique()) | set(pairs_df['image2'].unique())
        logger.info(f"Precomputing features for {len(all_images)} unique images...")

        # Setup cache paths
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get cache names from feature extractor config
        # These are model-specific to avoid conflicts
        extractor_config = feature_extractor.config if hasattr(feature_extractor, 'config') else {}

        clip_model = extractor_config.get('clip_model', 'openai_clip-vit-large-patch14')
        clip_cache_name = f"clip_{clip_model.replace('/', '_')}_cache.pkl"
        clip_cache_path = cache_dir / clip_cache_name

        cnn_backbone = extractor_config.get('cnn_backbone', 'resnet50')
        cnn_layer = extractor_config.get('cnn_layer', 'layer4')
        cnn_cache_name = f"cnn_{cnn_backbone}_{cnn_layer}_cache.pkl"
        cnn_cache_path = cache_dir / cnn_cache_name

        iqa_cache_path = cache_dir / 'iqa_cache.pkl'

        # Load existing caches
        clip_cache = {}
        cnn_cache = {}
        iqa_cache = {}

        if use_clip and clip_cache_path.exists():
            logger.info(f"Loading CLIP cache from {clip_cache_path}")
            with open(clip_cache_path, 'rb') as f:
                clip_cache = pickle.load(f)
            logger.info(f"  Loaded {len(clip_cache)} CLIP features")

        if use_cnn and cnn_cache_path.exists():
            logger.info(f"Loading CNN cache from {cnn_cache_path}")
            with open(cnn_cache_path, 'rb') as f:
                cnn_cache = pickle.load(f)
            logger.info(f"  Loaded {len(cnn_cache)} CNN features")

        if use_iqa and iqa_cache_path.exists():
            logger.info(f"Loading IQA cache from {iqa_cache_path}")
            with open(iqa_cache_path, 'rb') as f:
                iqa_cache = pickle.load(f)
            logger.info(f"  Loaded {len(iqa_cache)} IQA features")

        # Identify images that need feature extraction
        images_needing_clip = set(all_images) - set(clip_cache.keys()) if use_clip else set()
        images_needing_cnn = set(all_images) - set(cnn_cache.keys()) if use_cnn else set()
        images_needing_iqa = set(all_images) - set(iqa_cache.keys()) if use_iqa else set()

        images_needing_extraction = images_needing_clip | images_needing_cnn | images_needing_iqa

        if len(images_needing_extraction) > 0:
            logger.info(f"Extracting features for {len(images_needing_extraction)} images:")
            if use_clip:
                logger.info(f"  CLIP: {len(images_needing_clip)} images need extraction")
            if use_cnn:
                logger.info(f"  CNN: {len(images_needing_cnn)} images need extraction")
            if use_iqa:
                logger.info(f"  IQA: {len(images_needing_iqa)} images need extraction")

            feature_extractor.eval()

            for img_name in tqdm(images_needing_extraction, desc="Extracting features"):
                # Get image path (assume train_val for now - could be enhanced)
                img_path = self.get_image_path(img_name, split='train_val')

                # Extract all enabled features at once
                all_features = feature_extractor.extract_all(str(img_path)).cpu()

                # Get feature dimensions to split the concatenated tensor
                feature_dims = feature_extractor.get_feature_dims()

                # Split features back into individual caches
                offset = 0
                if use_clip and 'clip' in feature_dims and img_name in images_needing_clip:
                    clip_dim = feature_dims['clip']
                    clip_feat = all_features[offset:offset+clip_dim]
                    clip_cache[img_name] = clip_feat
                    offset += clip_dim
                elif use_clip:
                    offset += feature_dims.get('clip', 0)

                if use_cnn and 'cnn' in feature_dims and img_name in images_needing_cnn:
                    cnn_dim = feature_dims['cnn']
                    cnn_feat = all_features[offset:offset+cnn_dim]
                    cnn_cache[img_name] = cnn_feat
                    offset += cnn_dim
                elif use_cnn:
                    offset += feature_dims.get('cnn', 0)

                if use_iqa and 'iqa' in feature_dims and img_name in images_needing_iqa:
                    iqa_dim = feature_dims['iqa']
                    iqa_feat = all_features[offset:offset+iqa_dim]
                    iqa_cache[img_name] = iqa_feat

            # Save updated caches
            if use_clip and len(images_needing_clip) > 0:
                with open(clip_cache_path, 'wb') as f:
                    pickle.dump(clip_cache, f)
                logger.info(f"Saved CLIP cache to {clip_cache_path} ({len(clip_cache)} images)")

            if use_cnn and len(images_needing_cnn) > 0:
                with open(cnn_cache_path, 'wb') as f:
                    pickle.dump(cnn_cache, f)
                logger.info(f"Saved CNN cache to {cnn_cache_path} ({len(cnn_cache)} images)")

            if use_iqa and len(images_needing_iqa) > 0:
                with open(iqa_cache_path, 'wb') as f:
                    pickle.dump(iqa_cache, f)
                logger.info(f"Saved IQA cache to {iqa_cache_path} ({len(iqa_cache)} images)")
        else:
            logger.info("All features already cached!")

        # Concatenate features for all images
        logger.info("Concatenating features...")
        feature_dims = feature_extractor.get_feature_dims()

        final_cache = {}
        for img_name in all_images:
            features = []
            if use_clip:
                clip_dim = feature_dims.get('clip', 0)
                features.append(clip_cache.get(img_name, torch.zeros(clip_dim)))
            if use_cnn:
                cnn_dim = feature_dims.get('cnn', 0)
                features.append(cnn_cache.get(img_name, torch.zeros(cnn_dim)))
            if use_iqa:
                iqa_dim = feature_dims.get('iqa', 0)
                features.append(iqa_cache.get(img_name, torch.zeros(iqa_dim)))

            if len(features) > 0:
                final_cache[img_name] = torch.cat(features)

        logger.info(f"Successfully prepared features for {len(final_cache)} images")
        return final_cache

    def export_splits_to_files(
        self,
        output_dir: str,
        use_official_test: bool = True
    ):
        """
        Export train/val/test splits to separate CSV files.

        Args:
            output_dir: Directory to save CSV files
            use_official_test: Use official test set or random split
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_df, val_df, test_df = self.get_official_splits(use_official_test=use_official_test)

        train_df.to_csv(output_dir / 'train_pairs.csv', index=False)
        val_df.to_csv(output_dir / 'val_pairs.csv', index=False)

        if test_df is not None:
            test_df.to_csv(output_dir / 'test_pairs.csv', index=False)

        logger.info(f"Exported splits to {output_dir}")
        logger.info(f"  train_pairs.csv: {len(train_df)} pairs")
        logger.info(f"  val_pairs.csv: {len(val_df)} pairs")
        if test_df is not None:
            logger.info(f"  test_pairs.csv: {len(test_df)} pairs")


# Convenience functions for backward compatibility

def load_phototriage_data(
    root_dir: str = r"D:\Similar Images\automatic_triage_photo_series",
    min_agreement: float = 0.7,
    min_reviewers: int = 2,
    use_official_test: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Quick function to load PhotoTriage train/val/test splits.

    Args:
        root_dir: Root directory
        min_agreement: Minimum agreement threshold
        min_reviewers: Minimum reviewers
        use_official_test: Use official test set

    Returns:
        (train_df, val_df, test_df)
    """
    data = PhotoTriageData(root_dir, min_agreement=min_agreement, min_reviewers=min_reviewers)
    return data.get_official_splits(use_official_test=use_official_test)


def normalize_filename(filename: str) -> str:
    """Normalize filename (backward compatibility)."""
    return PhotoTriageData.normalize_filename(filename)
