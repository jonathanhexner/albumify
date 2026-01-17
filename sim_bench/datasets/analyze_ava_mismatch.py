"""
Analyze mismatches between AVA.txt and image directory.

Finds:
1. Images on disk that are NOT in AVA.txt
2. Images in AVA.txt that do NOT exist on disk

Usage:
    python -m sim_bench.datasets.analyze_ava_mismatch \
        --ava-txt /workspace/AVA_Files/AVA.txt \
        --image-dir /workspace/images \
        --num-samples 10
"""
import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_mismatch(ava_txt_path: str, image_dir: str, num_samples: int = 10,
                     output_dir: str = None):
    """Analyze mismatches between AVA.txt and image directory."""
    
    logger.info("=" * 70)
    logger.info("AVA Dataset Mismatch Analysis")
    logger.info("=" * 70)
    
    # Load AVA.txt with all columns
    logger.info(f"\nLoading AVA.txt from: {ava_txt_path}")
    df = pd.read_csv(ava_txt_path, sep=' ', header=None)
    
    # Name the columns for clarity (based on AVA.txt format)
    # Column 0: Index, Column 1: Image ID, Columns 2-11: Votes 1-10,
    # Columns 12-13: Tag IDs, Column 14: Challenge ID
    col_names = ['index', 'image_id'] + \
                [f'votes_{i}' for i in range(1, 11)] + \
                ['tag1', 'tag2', 'challenge_id']
    df.columns = col_names
    df['image_id'] = df['image_id'].astype(str)
    
    ava_ids = set(df['image_id'].tolist())
    logger.info(f"  IDs in AVA.txt: {len(ava_ids)}")
    
    # Get all image files from directory
    logger.info(f"\nScanning image directory: {image_dir}")
    image_dir = Path(image_dir)
    
    jpg_files = list(image_dir.glob("*.jpg"))
    png_files = list(image_dir.glob("*.png"))
    jpeg_files = list(image_dir.glob("*.jpeg"))
    
    all_image_files = jpg_files + png_files + jpeg_files
    logger.info(f"  Images on disk: {len(all_image_files)}")
    logger.info(f"    .jpg:  {len(jpg_files)}")
    logger.info(f"    .png:  {len(png_files)}")
    logger.info(f"    .jpeg: {len(jpeg_files)}")
    
    # Extract IDs from filenames (without extension)
    disk_ids = set(f.stem for f in all_image_files)
    logger.info(f"  Unique IDs on disk: {len(disk_ids)}")
    
    # Find mismatches
    logger.info("\n" + "=" * 70)
    logger.info("ANALYSIS")
    logger.info("=" * 70)
    
    # 1. Images on disk NOT in AVA.txt
    orphan_images = disk_ids - ava_ids
    logger.info(f"\n1. Images on disk NOT in AVA.txt: {len(orphan_images)}")
    
    if orphan_images:
        samples = sorted(list(orphan_images))[:num_samples]
        logger.info(f"\n   Sample {num_samples} orphan images:")
        for img_id in samples:
            # Find the actual file
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = image_dir / f"{img_id}{ext}"
                if img_path.exists():
                    logger.info(f"     - {img_id}{ext}")
                    break
    else:
        logger.info("   ✅ All images on disk have entries in AVA.txt")
    
    # 2. Images in AVA.txt NOT on disk
    missing_images = ava_ids - disk_ids
    logger.info(f"\n2. Images in AVA.txt NOT on disk: {len(missing_images)}")
    
    missing_df = df[df['image_id'].isin(missing_images)].copy()
    
    if missing_images:
        samples = sorted(list(missing_images), key=lambda x: int(x) if x.isdigit() else 0)[:num_samples]
        logger.info(f"\n   Sample {num_samples} missing images (full AVA.txt rows):")
        
        sample_df = missing_df[missing_df['image_id'].isin(samples)]
        for _, row in sample_df.head(num_samples).iterrows():
            logger.info(f"\n     Image ID: {row['image_id']}")
            logger.info(f"       Votes: {row['votes_1']},{row['votes_2']},{row['votes_3']},{row['votes_4']},{row['votes_5']},{row['votes_6']},{row['votes_7']},{row['votes_8']},{row['votes_9']},{row['votes_10']}")
            mean_score = sum((i+1) * row[f'votes_{i+1}'] for i in range(10)) / sum(row[f'votes_{i+1}'] for i in range(10))
            logger.info(f"       Mean score: {mean_score:.2f}")
            
        # Show ID range of missing images
        numeric_missing = [int(x) for x in missing_images if x.isdigit()]
        if numeric_missing:
            logger.info(f"\n   Missing image ID range:")
            logger.info(f"     Min: {min(numeric_missing)}")
            logger.info(f"     Max: {max(numeric_missing)}")
    else:
        logger.info("   ✅ All AVA.txt entries have images on disk")
    
    # 3. Intersection (images that match)
    matching_images = ava_ids & disk_ids
    logger.info(f"\n3. Matching images (in both AVA.txt AND on disk): {len(matching_images)}")
    
    # Coverage statistics
    logger.info("\n" + "=" * 70)
    logger.info("COVERAGE STATISTICS")
    logger.info("=" * 70)
    
    total_ava = len(ava_ids)
    total_disk = len(disk_ids)
    total_match = len(matching_images)
    
    logger.info(f"\nAVA.txt → Disk coverage:")
    logger.info(f"  {total_match}/{total_ava} = {100*total_match/total_ava:.1f}% of AVA.txt IDs have images")
    
    logger.info(f"\nDisk → AVA.txt coverage:")
    logger.info(f"  {total_match}/{total_disk} = {100*total_match/total_disk:.1f}% of disk images have AVA labels")
    
    # Show sample of matching images
    logger.info(f"\n" + "=" * 70)
    logger.info("SAMPLE MATCHING IMAGES (for verification)")
    logger.info("=" * 70)
    
    sample_matching = sorted(list(matching_images), key=lambda x: int(x) if x.isdigit() else 0)[:num_samples]
    logger.info(f"\nSample {num_samples} images that exist in BOTH:")
    for img_id in sample_matching:
        for ext in ['.jpg', '.png', '.jpeg']:
            img_path = image_dir / f"{img_id}{ext}"
            if img_path.exists():
                logger.info(f"  ✅ {img_id}{ext}")
                break
    
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATION")
    logger.info("=" * 70)
    
    if len(orphan_images) > 0:
        logger.info(f"\n⚠️  Found {len(orphan_images)} orphan images (on disk but not in AVA.txt)")
        logger.info("   These images cannot be used for training (no labels)")
        logger.info("   Consider removing them to save disk space")
    
    if len(missing_images) > 0:
        pct_missing = 100 * len(missing_images) / len(ava_ids)
        logger.info(f"\n⚠️  Found {len(missing_images)} missing images ({pct_missing:.1f}% of AVA.txt)")
        if pct_missing > 50:
            logger.info("   This is a significant portion - consider downloading the full dataset")
        else:
            logger.info("   This is expected for partial datasets - training will use available images")
    
    if len(orphan_images) == 0 and len(missing_images) == 0:
        logger.info("\n✅ Perfect match! All images on disk have labels, and all labels have images")
    
    # Save dataframes if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "=" * 70)
        logger.info("SAVING DATAFRAMES")
        logger.info("=" * 70)
        
        # Save orphan images list
        if orphan_images:
            orphan_df = pd.DataFrame({
                'image_id': sorted(list(orphan_images)),
                'note': 'Image exists on disk but has no label in AVA.txt'
            })
            orphan_path = output_path / 'orphan_images.csv'
            orphan_df.to_csv(orphan_path, index=False)
            logger.info(f"\n✅ Saved {len(orphan_df)} orphan images to:")
            logger.info(f"   {orphan_path}")
        
        # Save missing images (full AVA.txt rows)
        if missing_images:
            missing_path = output_path / 'missing_images_full_ava_rows.csv'
            missing_df.to_csv(missing_path, index=False)
            logger.info(f"\n✅ Saved {len(missing_df)} missing images (full AVA.txt rows) to:")
            logger.info(f"   {missing_path}")
            
            # Also save as parquet
            missing_parquet = output_path / 'missing_images_full_ava_rows.parquet'
            missing_df.to_parquet(missing_parquet, index=False)
            logger.info(f"   {missing_parquet}")
        
        # Save matching images (usable dataset)
        matching_df = df[df['image_id'].isin(matching_images)].copy()
        matching_path = output_path / 'usable_images_full_ava_rows.csv'
        matching_df.to_csv(matching_path, index=False)
        logger.info(f"\n✅ Saved {len(matching_df)} usable images (full AVA.txt rows) to:")
        logger.info(f"   {matching_path}")
        
        matching_parquet = output_path / 'usable_images_full_ava_rows.parquet'
        matching_df.to_parquet(matching_parquet, index=False)
        logger.info(f"   {matching_parquet}")
    
    logger.info("")
    
    return {
        'orphan_images': sorted(list(orphan_images)),
        'missing_df': missing_df if missing_images else pd.DataFrame(),
        'matching_df': df[df['image_id'].isin(matching_images)] if matching_images else pd.DataFrame()
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze AVA dataset mismatches')
    parser.add_argument('--ava-txt', required=True, help='Path to AVA.txt')
    parser.add_argument('--image-dir', required=True, help='Path to image directory')
    parser.add_argument('--num-samples', type=int, default=10, 
                       help='Number of sample mismatches to show')
    parser.add_argument('--output-dir', default=None,
                       help='Directory to save CSV/parquet files with full results')
    
    args = parser.parse_args()
    
    result = analyze_mismatch(args.ava_txt, args.image_dir, args.num_samples, args.output_dir)
    
    return result


if __name__ == '__main__':
    main()
