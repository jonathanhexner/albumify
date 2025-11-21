"""
Create simple pairs JSONL for pairwise benchmark.

This is a simpler version that just creates pairs with user choices
from the pair list files, without requiring attribute labeling.

Usage:
    python scripts/phototriage/05_create_simple_pairs.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
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
class SimplePair:
    """Simple pair for pairwise evaluation."""
    pair_id: str
    series_id: str
    image_a_path: str
    image_b_path: str
    chosen_image: str  # "A" or "B"
    preference_strength: float  # From Bradley-Terry model (0.0-1.0)


def create_pairs_from_pairlist(
    pairlist_file: Path,
    image_dir: Path,
    output_file: Path
) -> List[SimplePair]:
    """
    Create pairs from pairlist.txt file.

    Pairlist format (space-separated):
    series_id photo1_idx photo2_idx preference_ratio rank1 rank2

    Where:
    - photo indices are 1-based
    - preference_ratio: P(photo1 chosen) from Bradley-Terry model
    - rank: 1=best, 2=second best, etc.
    """
    logger.info(f"Creating pairs from {pairlist_file}")

    pairs = []

    with open(pairlist_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()

            if len(parts) != 6:
                logger.warning(f"Line {line_num}: Invalid format, skipping")
                continue

            series_id = parts[0]
            photo1_idx = int(parts[1])  # 1-based
            photo2_idx = int(parts[2])  # 1-based
            preference_ratio = float(parts[3])  # P(photo1 chosen)
            rank1 = int(parts[4])
            rank2 = int(parts[5])

            # Map to image paths
            # Format: GGGGGG-II.JPG where GGGGGG is series_id (zero-padded), II is photo index
            image_a_id = f"{int(series_id):06d}-{photo1_idx:02d}"
            image_b_id = f"{int(series_id):06d}-{photo2_idx:02d}"

            image_a_path = str(image_dir / f"{image_a_id}.JPG")
            image_b_path = str(image_dir / f"{image_b_id}.JPG")

            # Determine chosen based on preference ratio
            # If P(photo1) > 0.5, photo1 was preferred
            if preference_ratio > 0.5:
                chosen = "A"
                preference_strength = preference_ratio
            else:
                chosen = "B"
                preference_strength = 1.0 - preference_ratio

            # Create pair
            pair = SimplePair(
                pair_id=f"{series_id}_{photo1_idx}_{photo2_idx}",
                series_id=series_id,
                image_a_path=image_a_path,
                image_b_path=image_b_path,
                chosen_image=chosen,
                preference_strength=preference_strength
            )

            pairs.append(pair)

    logger.info(f"Created {len(pairs)} pairs")

    # Save to JSONL
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json.dump(asdict(pair), f)
            f.write('\n')

    logger.info(f"Saved pairs to {output_file}")

    return pairs


def main():
    """Create pairs for train/val/test splits."""
    # Paths
    phototriage_root = Path("D:/Similar Images/automatic_triage_photo_series")
    image_dir = phototriage_root / "train_val" / "train_val_imgs"
    output_dir = Path("data/phototriage")

    # Check if image directory exists
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.error("Please update the path in this script")
        return

    # Create pairs for each split
    splits = {
        'train': phototriage_root / "train_val" / "train_pairlist.txt",
        'val': phototriage_root / "train_val" / "val_pairlist.txt",
        'test': phototriage_root / "test" / "test_pairlist.txt"
    }

    all_stats = {}

    for split_name, pairlist_file in splits.items():
        if not pairlist_file.exists():
            logger.warning(f"Pairlist not found: {pairlist_file}, skipping {split_name}")
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Creating {split_name} pairs")
        logger.info(f"{'='*70}")

        output_file = output_dir / f"pairs_{split_name}.jsonl"

        pairs = create_pairs_from_pairlist(
            pairlist_file=pairlist_file,
            image_dir=image_dir,
            output_file=output_file
        )

        # Compute statistics
        num_series = len(set(p.series_id for p in pairs))
        avg_strength = sum(p.preference_strength for p in pairs) / len(pairs)
        strong_pairs = sum(1 for p in pairs if p.preference_strength >= 0.75)
        weak_pairs = sum(1 for p in pairs if p.preference_strength <= 0.55)

        all_stats[split_name] = {
            'num_pairs': len(pairs),
            'num_series': num_series,
            'avg_preference_strength': avg_strength,
            'strong_pairs': strong_pairs,
            'weak_pairs': weak_pairs
        }

        logger.info(f"  Pairs: {len(pairs)}")
        logger.info(f"  Series: {num_series}")
        logger.info(f"  Avg preference strength: {avg_strength:.3f}")
        logger.info(f"  Strong preferences (≥0.75): {strong_pairs}")
        logger.info(f"  Weak preferences (≤0.55): {weak_pairs}")

    # Save summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")

    summary = {
        'description': 'PhotoTriage pairwise comparison dataset',
        'image_dir': str(image_dir),
        'splits': all_stats
    }

    summary_file = output_dir / 'pairs_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")

    # Print total
    total_pairs = sum(s['num_pairs'] for s in all_stats.values())
    logger.info(f"\nTotal pairs across all splits: {total_pairs}")

    logger.info("\n✅ Pair creation complete!")


if __name__ == "__main__":
    main()
