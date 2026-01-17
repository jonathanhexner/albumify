"""
Preprocess AVA dataset: scan for existing images and create cached parquet file.

This script scans the image directory once and creates a parquet file with:
- image_id
- votes_1 through votes_10
- total_votes
- mean_score
- exists (boolean)

Future dataset loading can use the cached parquet file instead of scanning repeatedly.

Usage:
    python -m sim_bench.datasets.preprocess_ava \
        --ava-txt /path/to/AVA.txt \
        --image-dir /path/to/images \
        --output ava_filtered.parquet
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_ava_dataset(ava_txt_path: str, image_dir: str, output_path: str, 
                           extensions: list = None):
    """
    Preprocess AVA dataset and create filtered parquet file.
    
    Args:
        ava_txt_path: Path to AVA.txt
        image_dir: Path to image directory
        output_path: Path to output parquet file
        extensions: List of image extensions to check (default: ['.jpg'])
    """
    if extensions is None:
        extensions = ['.jpg']
    
    logger.info("=" * 60)
    logger.info("AVA Dataset Preprocessing")
    logger.info("=" * 60)
    
    # Load AVA.txt
    logger.info(f"Loading AVA.txt from: {ava_txt_path}")
    df = pd.read_csv(ava_txt_path, sep=' ', header=None)
    logger.info(f"  Total entries: {len(df)}")
    
    # Extract relevant columns
    # Column 0: Index (ignore), Column 1: Image ID, Columns 2-11: Votes 1-10
    result = pd.DataFrame()
    result['image_id'] = df[1].astype(str)
    
    # Vote counts for scores 1-10 (columns 2-11)
    for i in range(10):
        result[f'votes_{i+1}'] = df[i + 2]
    
    # Compute total votes and mean score
    vote_cols = [f'votes_{i}' for i in range(1, 11)]
    result['total_votes'] = result[vote_cols].sum(axis=1)
    
    # Mean score: weighted average
    scores = np.arange(1, 11)
    vote_matrix = result[vote_cols].values
    result['mean_score'] = (vote_matrix * scores).sum(axis=1) / result['total_votes']
    
    # Check which images exist
    logger.info(f"\nScanning image directory: {image_dir}")
    image_dir = Path(image_dir)
    
    exists_flags = []
    existing_count = 0
    
    for img_id in tqdm(result['image_id'], desc="Checking images"):
        exists = False
        for ext in extensions:
            if (image_dir / f"{img_id}{ext}").exists():
                exists = True
                existing_count += 1
                break
        exists_flags.append(exists)
    
    result['exists'] = exists_flags
    
    # Statistics
    total = len(result)
    missing = total - existing_count
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total entries:    {total}")
    logger.info(f"  Images found:     {existing_count} ({100*existing_count/total:.1f}%)")
    logger.info(f"  Images missing:   {missing} ({100*missing/total:.1f}%)")
    
    # Label statistics (for existing images only)
    existing_df = result[result['exists']]
    logger.info(f"\nLabel Statistics (existing images only):")
    logger.info(f"  Mean score range: [{existing_df['mean_score'].min():.2f}, {existing_df['mean_score'].max():.2f}]")
    logger.info(f"  Mean score avg:   {existing_df['mean_score'].mean():.2f}")
    logger.info(f"  Mean score std:   {existing_df['mean_score'].std():.2f}")
    logger.info(f"  Total votes avg:  {existing_df['total_votes'].mean():.1f}")
    
    # Save full dataset (including exists flag)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving to: {output_path}")
    result.to_parquet(output_path, index=False)
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Also save filtered version (only existing images)
    filtered_path = output_path.parent / (output_path.stem + "_filtered" + output_path.suffix)
    existing_df_no_flag = existing_df.drop(columns=['exists'])
    existing_df_no_flag.to_parquet(filtered_path, index=False)
    logger.info(f"\nSaved filtered version to: {filtered_path}")
    logger.info(f"  Entries: {len(existing_df_no_flag)}")
    logger.info(f"  File size: {filtered_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info("\n✅ Preprocessing complete!")
    logger.info(f"\nUsage in dataset loading:")
    logger.info(f"  df = pd.read_parquet('{filtered_path}')")
    logger.info(f"  # All {existing_count} images are guaranteed to exist")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Preprocess AVA dataset')
    parser.add_argument('--ava-txt', required=True, help='Path to AVA.txt')
    parser.add_argument('--image-dir', required=True, help='Path to image directory')
    parser.add_argument('--output', default='ava_dataset.parquet', 
                       help='Output parquet file path')
    parser.add_argument('--extensions', nargs='+', default=['.jpg'],
                       help='Image file extensions to check (default: .jpg)')
    
    args = parser.parse_args()
    
    preprocess_ava_dataset(
        args.ava_txt,
        args.image_dir,
        args.output,
        args.extensions
    )


if __name__ == '__main__':
    main()
