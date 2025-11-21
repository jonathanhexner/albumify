"""
Convert PhotoTriage pairs CSV to JSONL format for pairwise benchmark.

Takes the aggregated pairs CSV with vote counts and labels, filters for
high-agreement pairs, and converts to JSONL format expected by PairwiseEvaluator.

Usage:
    python convert_pairs_csv_to_jsonl.py
    python convert_pairs_csv_to_jsonl.py --input path/to/pairs.csv --output data/pairs.jsonl
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import sys

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_csv_to_jsonl(
    csv_path: Path,
    output_path: Path,
    image_dir: Path,
    min_agreement: float = 0.7,
    min_reviewers: int = 2
) -> int:
    """
    Convert pairs CSV to JSONL format.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output JSONL file
        image_dir: Directory containing image files
        min_agreement: Minimum agreement threshold (0-1)
        min_reviewers: Minimum number of reviewers

    Returns:
        Number of pairs written
    """
    logger.info(f"Loading pairs from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df)} pairs")
    logger.info(f"Filtering: Agreement >= {min_agreement}, num_reviewers >= {min_reviewers}")

    # Filter pairs
    df_filtered = df[
        (df['Agreement'] >= min_agreement) &
        (df['num_reviewers'] >= min_reviewers)
    ].copy()

    logger.info(f"After filtering: {len(df_filtered)} pairs ({100*len(df_filtered)/len(df):.1f}%)")

    # Convert to JSONL
    pairs_written = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df_filtered.iterrows():
            series_id = str(row['series_id'])
            compare_id_1 = int(row['compareID1'])
            compare_id_2 = int(row['compareID2'])
            compare_file_1 = row['compareFile1']
            compare_file_2 = row['compareFile2']
            max_vote = int(row['MaxVote'])
            agreement = float(row['Agreement'])

            # Map compare files to full paths
            # Images are in format: GGGGGG-II.JPG where GGGGGG is zero-padded series_id
            # But compareFile columns have format like "1-1.JPG" (not zero-padded)
            # We need to construct the proper filename

            # Parse compareFile to get photo index
            # Format: "SERIES-INDEX.JPG" where SERIES may or may not be zero-padded
            compare_file_1_parts = compare_file_1.replace('.JPG', '').replace('.jpg', '').split('-')
            compare_file_2_parts = compare_file_2.replace('.JPG', '').replace('.jpg', '').split('-')

            photo_idx_1 = int(compare_file_1_parts[1]) if len(compare_file_1_parts) > 1 else int(compare_id_1) + 1
            photo_idx_2 = int(compare_file_2_parts[1]) if len(compare_file_2_parts) > 1 else int(compare_id_2) + 1

            # Construct proper filenames with zero-padded series_id
            image_a_filename = f"{int(series_id):06d}-{photo_idx_1:02d}.JPG"
            image_b_filename = f"{int(series_id):06d}-{photo_idx_2:02d}.JPG"

            image_a_path = str(image_dir / image_a_filename)
            image_b_path = str(image_dir / image_b_filename)

            # Determine chosen image based on MaxVote
            # MaxVote is the compareID of the winning image
            # compareID1 maps to image A, compareID2 maps to image B
            if max_vote == compare_id_1:
                chosen_image = "A"
            else:
                chosen_image = "B"

            # Create pair object
            pair = {
                'pair_id': f"{series_id}_{compare_id_1}_{compare_id_2}",
                'series_id': series_id,
                'image_a_path': image_a_path,
                'image_b_path': image_b_path,
                'chosen_image': chosen_image,
                'preference_strength': agreement
            }

            # Write to JSONL
            json.dump(pair, f)
            f.write('\n')
            pairs_written += 1

    logger.info(f"Successfully wrote {pairs_written} pairs to {output_path}")

    return pairs_written


def print_statistics(csv_path: Path):
    """Print statistics about the pairs CSV."""
    df = pd.read_csv(csv_path)

    print("\n" + "="*70)
    print("PAIRS CSV STATISTICS")
    print("="*70)
    print(f"Total pairs: {len(df)}")
    print(f"Unique series: {df['series_id'].nunique()}")
    print(f"\nAgreement statistics:")
    print(f"  Mean: {df['Agreement'].mean():.3f}")
    print(f"  Median: {df['Agreement'].median():.3f}")
    print(f"  Min: {df['Agreement'].min():.3f}")
    print(f"  Max: {df['Agreement'].max():.3f}")

    print(f"\nReviewers statistics:")
    print(f"  Mean: {df['num_reviewers'].mean():.2f}")
    print(f"  Median: {df['num_reviewers'].median():.0f}")
    print(f"  Min: {df['num_reviewers'].min():.0f}")
    print(f"  Max: {df['num_reviewers'].max():.0f}")

    # Print filtering impact
    for min_agreement in [0.6, 0.7, 0.8]:
        for min_reviewers in [2, 3]:
            filtered = df[(df['Agreement'] >= min_agreement) & (df['num_reviewers'] >= min_reviewers)]
            pct = 100 * len(filtered) / len(df)
            print(f"\nAgreement >= {min_agreement}, reviewers >= {min_reviewers}:")
            print(f"  {len(filtered)} pairs ({pct:.1f}%)")

    print("="*70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Convert PhotoTriage pairs CSV to JSONL for pairwise benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_pairs_csv_to_jsonl.py
  python convert_pairs_csv_to_jsonl.py --min-agreement 0.7 --min-reviewers 2
  python convert_pairs_csv_to_jsonl.py --stats-only
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_embedding_labels.csv"),
        help='Input CSV file path (default: embedding labels CSV)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(r"data\phototriage\pairs_train_filtered.jsonl"),
        help='Output JSONL file path (default: data/phototriage/pairs_train_filtered.jsonl)'
    )

    parser.add_argument(
        '--image-dir',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series\train_val\train_val_imgs"),
        help='Directory containing image files'
    )

    parser.add_argument(
        '--min-agreement',
        type=float,
        default=0.7,
        help='Minimum agreement threshold (0-1, default: 0.7)'
    )

    parser.add_argument(
        '--min-reviewers',
        type=int,
        default=2,
        help='Minimum number of reviewers (default: 2)'
    )

    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only print statistics, do not convert'
    )

    args = parser.parse_args()

    # Validate input file
    if not args.input.exists():
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)

    # Validate image directory
    if not args.image_dir.exists():
        logger.error(f"Image directory does not exist: {args.image_dir}")
        logger.error("Please update --image-dir to point to train_val_imgs directory")
        sys.exit(1)

    # Print statistics
    print_statistics(args.input)

    if args.stats_only:
        logger.info("Stats-only mode, exiting")
        return

    # Convert
    logger.info(f"\n{'='*70}")
    logger.info("CONVERTING CSV TO JSONL")
    logger.info(f"{'='*70}")

    num_pairs = convert_csv_to_jsonl(
        csv_path=args.input,
        output_path=args.output,
        image_dir=args.image_dir,
        min_agreement=args.min_agreement,
        min_reviewers=args.min_reviewers
    )

    print(f"\n{'='*70}")
    print("CONVERSION COMPLETE")
    print(f"{'='*70}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Pairs:  {num_pairs}")
    print(f"Filter: Agreement >= {args.min_agreement}, Reviewers >= {args.min_reviewers}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
