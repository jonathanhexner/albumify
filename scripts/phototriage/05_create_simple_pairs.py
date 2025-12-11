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
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
import sys
import pandas as pd

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


def load_valid_pairs(reviews_csv: Path) -> Set[Tuple[str, int, int]]:
    """
    Load valid pairs from reviews CSV, filtering out 'no_reason_given'.
    
    Args:
        reviews_csv: Path to reviews_df.csv
        
    Returns:
        Set of (series_id, compareID1, compareID2) tuples for valid pairs
    """
    logger.info(f"Loading valid pairs from {reviews_csv}")
    df = pd.read_csv(reviews_csv)
    
    # Filter out 'no_reason_given'
    df_valid = df[df['label'] != 'no_reason_given'].copy()
    
    logger.info(f"Total reviews: {len(df)}")
    logger.info(f"After filtering 'no_reason_given': {len(df_valid)} ({100*len(df_valid)/len(df):.1f}%)")
    
    # Create set of valid pairs
    valid_pairs = set()
    for _, row in df_valid.iterrows():
        pair_key = (str(row['series_id']), int(row['compareID1']), int(row['compareID2']))
        valid_pairs.add(pair_key)
    
    logger.info(f"Unique valid pairs: {len(valid_pairs)}")
    return valid_pairs


def create_pairs_from_pairlist(
    pairlist_file: Path,
    image_dir: Path,
    output_file: Path,
    valid_pairs: Set[Tuple[str, int, int]] = None
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
    if valid_pairs is not None:
        logger.info(f"Filtering pairs using {len(valid_pairs)} valid pairs from reviews CSV")

    pairs = []
    dropped_count = 0
    empty_lines = 0
    csv_filtered_count = 0

    with open(pairlist_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            
            # Skip empty lines
            if not parts:
                empty_lines += 1
                continue

            if len(parts) != 6:
                dropped_count += 1
                # Only warn for non-empty lines that don't match format
                if line_num <= 5:  # Warn for first few lines to help debug
                    logger.warning(f"Line {line_num}: Invalid format (expected 6 values, got {len(parts)}), skipping")
                continue

            series_id = parts[0]
            photo1_idx = int(parts[1])  # 1-based
            photo2_idx = int(parts[2])  # 1-based
            preference_ratio = float(parts[3])  # P(photo1 chosen)
            rank1 = int(parts[4])
            rank2 = int(parts[5])

            # Filter: skip if this pair is not in valid_pairs
            # Note: pairlist uses 1-based photo indices, CSV uses 0-based compareIDs
            if valid_pairs is not None:
                # Convert 1-based photo indices to 0-based compareIDs
                compare_id1 = photo1_idx - 1
                compare_id2 = photo2_idx - 1
                
                # Check both orderings since pairs can be in either direction
                pair_key1 = (series_id, compare_id1, compare_id2)
                pair_key2 = (series_id, compare_id2, compare_id1)
                
                if pair_key1 not in valid_pairs and pair_key2 not in valid_pairs:
                    csv_filtered_count += 1
                    continue

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
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows with invalid format")
    if empty_lines > 0:
        logger.info(f"Skipped {empty_lines} empty lines")
    if csv_filtered_count > 0:
        logger.info(f"Filtered out {csv_filtered_count} pairs with 'no_reason_given' label (from CSV)")

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
    reviews_csv = phototriage_root / "reviews_df.csv"

    # Check if image directory exists
    if not image_dir.exists():
        logger.error(f"Image directory not found: {image_dir}")
        logger.error("Please update the path in this script")
        return

    # Load valid pairs (filtering out 'no_reason_given')
    if reviews_csv.exists():
        valid_pairs = load_valid_pairs(reviews_csv)
    else:
        logger.warning(f"Reviews CSV not found: {reviews_csv}")
        logger.warning("Proceeding without filtering (all pairs will be included)")
        valid_pairs = None

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
            output_file=output_file,
            valid_pairs=valid_pairs
        )

        # Compute statistics
        if len(pairs) == 0:
            logger.warning(f"No pairs created for {split_name} split")
            all_stats[split_name] = {
                'num_pairs': 0,
                'num_series': 0,
                'avg_preference_strength': 0.0,
                'strong_pairs': 0,
                'weak_pairs': 0
            }
            continue
        
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
