"""
Create final attribute-labeled dataset with ground truth.

This script:
1. Loads labeled comparisons
2. Loads ground truth from pair lists
3. Maps to actual image file paths
4. Creates train/val/test splits (by series_id)
5. Saves final dataset in training-ready format

Usage:
    python scripts/phototriage/04_create_attribute_dataset.py
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, asdict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AttributePair:
    """Final attribute-labeled pair for training."""
    pair_id: str
    series_id: str
    image_a_id: str
    image_b_id: str
    image_a_path: str
    image_b_path: str
    chosen_image: str  # "A" or "B"
    reason_raw: str
    preference_strength: float  # From Bradley-Terry model (0.0-1.0)
    rank_a: int  # Ground truth rank (1=best)
    rank_b: int  # Ground truth rank (1=best)
    attributes: List[Dict]
    metadata: Dict


class GroundTruthLoader:
    """Load ground truth rankings from pair lists."""

    def __init__(self, pairlist_file: Path):
        """
        Initialize loader.

        Args:
            pairlist_file: Path to train_pairlist.txt or val_pairlist.txt
        """
        self.pairlist_file = Path(pairlist_file)
        self.pair_data: Dict[Tuple[str, int, int], Dict] = {}

    def load(self) -> None:
        """Load pair list data."""
        logger.info(f"Loading ground truth from {self.pairlist_file}")

        with open(self.pairlist_file, 'r') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 6:
                    continue

                series_id = parts[0]
                photo1_idx = int(parts[1])  # 1-based!
                photo2_idx = int(parts[2])  # 1-based!
                preference_ratio = float(parts[3])
                rank1 = int(parts[4])
                rank2 = int(parts[5])

                # Store with 0-based indices as key
                key = (series_id, photo1_idx - 1, photo2_idx - 1)

                self.pair_data[key] = {
                    'preference_ratio': preference_ratio,
                    'rank1': rank1,
                    'rank2': rank2
                }

        logger.info(f"Loaded {len(self.pair_data)} ground truth pairs")

    def get_pair_data(
        self,
        series_id: str,
        compare_id_1: int,  # 0-based
        compare_id_2: int   # 0-based
    ) -> Dict:
        """
        Get ground truth data for a pair.

        Args:
            series_id: Series ID
            compare_id_1: 0-based index
            compare_id_2: 0-based index

        Returns:
            Dictionary with preference_ratio, rank1, rank2
        """
        # Try both orderings
        key1 = (series_id, compare_id_1, compare_id_2)
        key2 = (series_id, compare_id_2, compare_id_1)

        if key1 in self.pair_data:
            return self.pair_data[key1]
        elif key2 in self.pair_data:
            # Swap ranks and invert preference
            data = self.pair_data[key2]
            return {
                'preference_ratio': 1.0 - data['preference_ratio'],
                'rank1': data['rank2'],
                'rank2': data['rank1']
            }
        else:
            # Not found - return defaults
            return {
                'preference_ratio': 0.5,
                'rank1': -1,
                'rank2': -1
            }


class DatasetCreator:
    """Create final attribute dataset."""

    def __init__(
        self,
        labeled_comparisons_file: Path,
        image_dir: Path,
        ground_truth_file: Path,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ):
        """
        Initialize dataset creator.

        Args:
            labeled_comparisons_file: Path to labeled_comparisons.jsonl
            image_dir: Directory containing images
            ground_truth_file: Path to pair list file
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            random_seed: Random seed for reproducibility
        """
        self.labeled_comparisons_file = Path(labeled_comparisons_file)
        self.image_dir = Path(image_dir)
        self.ground_truth_file = Path(ground_truth_file)

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # Data
        self.comparisons: List[Dict] = []
        self.ground_truth_loader = GroundTruthLoader(ground_truth_file)
        self.pairs: List[AttributePair] = []

        # Splits
        self.train_pairs: List[AttributePair] = []
        self.val_pairs: List[AttributePair] = []
        self.test_pairs: List[AttributePair] = []

    def create_dataset(self) -> None:
        """Create complete dataset."""
        logger.info("Creating attribute dataset...")

        # Load data
        self._load_comparisons()
        self.ground_truth_loader.load()

        # Create pairs with ground truth
        self._create_pairs()

        # Split by series
        self._split_dataset()

        logger.info(f"Dataset created:")
        logger.info(f"  Train: {len(self.train_pairs)} pairs")
        logger.info(f"  Val: {len(self.val_pairs)} pairs")
        logger.info(f"  Test: {len(self.test_pairs)} pairs")
        logger.info(f"  Total: {len(self.pairs)} pairs")

    def _load_comparisons(self) -> None:
        """Load labeled comparisons."""
        logger.info(f"Loading comparisons from {self.labeled_comparisons_file}")

        with open(self.labeled_comparisons_file, 'r', encoding='utf-8') as f:
            self.comparisons = [json.loads(line) for line in f]

        logger.info(f"Loaded {len(self.comparisons)} comparisons")

    def _create_pairs(self) -> None:
        """Create pairs with ground truth and image paths."""
        logger.info("Creating pairs with ground truth...")

        for comparison in self.comparisons:
            # Extract comparison metadata
            series_id = comparison['series_id']
            compare_id_1 = comparison['compare_id_1']
            compare_id_2 = comparison['compare_id_2']
            user_choice = comparison['user_choice']

            # Get ground truth
            gt_data = self.ground_truth_loader.get_pair_data(
                series_id, compare_id_1, compare_id_2
            )

            # Map to image paths
            # Format: GGGGGG-II.JPG where GGGGGG is series_id (zero-padded)
            image_a_id = f"{int(series_id):06d}-{compare_id_1 + 1:02d}"
            image_b_id = f"{int(series_id):06d}-{compare_id_2 + 1:02d}"

            image_a_path = str(self.image_dir / f"{image_a_id}.JPG")
            image_b_path = str(self.image_dir / f"{image_b_id}.JPG")

            # Determine chosen image (A or B)
            if user_choice == "LEFT":
                chosen = "A"
            elif user_choice == "RIGHT":
                chosen = "B"
            else:
                chosen = "A"  # Default

            # Create pair
            pair = AttributePair(
                pair_id=f"{series_id}_{compare_id_1}_{compare_id_2}",
                series_id=series_id,
                image_a_id=image_a_id,
                image_b_id=image_b_id,
                image_a_path=image_a_path,
                image_b_path=image_b_path,
                chosen_image=chosen,
                reason_raw=comparison['reason_text'],
                preference_strength=gt_data['preference_ratio'],
                rank_a=gt_data['rank1'],
                rank_b=gt_data['rank2'],
                attributes=comparison['attributes'],
                metadata={
                    'review_file': comparison['review_file'],
                    'review_index': comparison['review_index'],
                    'num_attributes': comparison['num_attributes']
                }
            )

            self.pairs.append(pair)

        logger.info(f"Created {len(self.pairs)} pairs")

    def _split_dataset(self) -> None:
        """Split dataset by series_id to prevent data leakage."""
        logger.info("Splitting dataset by series...")

        # Set random seed
        random.seed(self.random_seed)

        # Get unique series
        series_ids = sorted(set(pair.series_id for pair in self.pairs))
        logger.info(f"Found {len(series_ids)} unique series")

        # Shuffle series
        random.shuffle(series_ids)

        # Calculate split indices
        n_series = len(series_ids)
        n_train = int(n_series * self.train_ratio)
        n_val = int(n_series * self.val_ratio)

        # Split series IDs
        train_series = set(series_ids[:n_train])
        val_series = set(series_ids[n_train:n_train + n_val])
        test_series = set(series_ids[n_train + n_val:])

        logger.info(f"Split series:")
        logger.info(f"  Train: {len(train_series)} series")
        logger.info(f"  Val: {len(val_series)} series")
        logger.info(f"  Test: {len(test_series)} series")

        # Assign pairs to splits
        for pair in self.pairs:
            if pair.series_id in train_series:
                self.train_pairs.append(pair)
            elif pair.series_id in val_series:
                self.val_pairs.append(pair)
            elif pair.series_id in test_series:
                self.test_pairs.append(pair)

    def save_dataset(self, output_dir: Path) -> None:
        """Save dataset splits."""
        logger.info(f"Saving dataset to {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each split
        splits = {
            'train': self.train_pairs,
            'val': self.val_pairs,
            'test': self.test_pairs
        }

        for split_name, pairs in splits.items():
            output_file = output_dir / f"{split_name}_pairs.jsonl"

            with open(output_file, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    json.dump(asdict(pair), f)
                    f.write('\n')

            logger.info(f"Saved {len(pairs)} pairs to {output_file}")

        # Save dataset info
        info = {
            'total_pairs': len(self.pairs),
            'train_pairs': len(self.train_pairs),
            'val_pairs': len(self.val_pairs),
            'test_pairs': len(self.test_pairs),
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_seed': self.random_seed,
            'image_dir': str(self.image_dir),
            'created_from': {
                'labeled_comparisons': str(self.labeled_comparisons_file),
                'ground_truth': str(self.ground_truth_file)
            }
        }

        info_file = output_dir / 'dataset_info.json'
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)

        logger.info(f"Saved dataset info to {info_file}")

    def print_statistics(self) -> None:
        """Print dataset statistics."""
        logger.info("="*70)
        logger.info("DATASET STATISTICS")
        logger.info("="*70)

        for split_name, pairs in [
            ('Train', self.train_pairs),
            ('Val', self.val_pairs),
            ('Test', self.test_pairs),
            ('Total', self.pairs)
        ]:
            logger.info(f"\n{split_name} Split:")
            logger.info(f"  Pairs: {len(pairs)}")

            if pairs:
                # Count attributes
                total_attrs = sum(len(p.attributes) for p in pairs)
                avg_attrs = total_attrs / len(pairs)

                # Count series
                unique_series = len(set(p.series_id for p in pairs))

                logger.info(f"  Unique series: {unique_series}")
                logger.info(f"  Total attribute labels: {total_attrs}")
                logger.info(f"  Avg attributes/pair: {avg_attrs:.2f}")

                # Attribute distribution
                attr_counts = defaultdict(int)
                for pair in pairs:
                    for attr in pair.attributes:
                        attr_counts[attr['name']] += 1

                if attr_counts:
                    logger.info(f"  Top 5 attributes:")
                    for attr, count in sorted(attr_counts.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:5]:
                        logger.info(f"    {attr}: {count}")

        logger.info("="*70)


def main():
    """Main dataset creation pipeline."""
    # Paths
    labeled_comparisons = Path("data/phototriage/labeled_comparisons.jsonl")
    image_dir = Path("D:/Similar Images/automatic_triage_photo_series/train_val/train_val_imgs")
    ground_truth = Path("D:/Similar Images/automatic_triage_photo_series/train_val/train_pairlist.txt")
    output_dir = Path("data/phototriage/dataset")

    # Check inputs
    if not labeled_comparisons.exists():
        logger.error(f"Labeled comparisons not found: {labeled_comparisons}")
        logger.error("Please run 03_map_attributes.py first")
        return

    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.error("Please update the path in this script")
        return

    if not ground_truth.exists():
        logger.error(f"Ground truth file not found: {ground_truth}")
        logger.error("Please update the path in this script")
        return

    # Create dataset
    logger.info("="*70)
    logger.info("PhotoTriage Attribute Dataset Creation")
    logger.info("="*70)

    creator = DatasetCreator(
        labeled_comparisons_file=labeled_comparisons,
        image_dir=image_dir,
        ground_truth_file=ground_truth,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42
    )

    # Create and save dataset
    creator.create_dataset()
    creator.save_dataset(output_dir)

    # Print statistics
    creator.print_statistics()

    logger.info("\n✅ Dataset creation complete!")
    logger.info(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
