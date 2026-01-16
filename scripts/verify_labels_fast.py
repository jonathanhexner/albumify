"""
Fast label verification - works directly with data sources without loading images.

Compares winner labels between internal and external dataloaders by:
- Internal: Reading DataFrames directly (no image loading)
- External: Calling MyDataset.__getitem__ to get pairs (minimal overhead)
"""
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.phototriage_data import PhotoTriageData


def extract_series_id(filename: str) -> int:
    """Extract series_id from filename like '000123-02.JPG' -> 123."""
    return int(filename.split('-')[0])


def load_internal_pairs(config_path: Path):
    """Load internal pairs directly from DataFrames (fast - no image loading)."""
    config = yaml.safe_load(open(config_path))
    
    data = PhotoTriageData(
        config['data']['root_dir'],
        config['data']['min_agreement'],
        config['data']['min_reviewers']
    )
    
    train_df, val_df, test_df = data.get_series_based_splits(
        0.8, 0.1, 0.1,
        config['seed'],
        config['data'].get('quick_experiment')
    )
    
    # Combine train and val
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    all_pairs = pd.concat([train_df, val_df], ignore_index=True)
    
    # Rename winner column to winner_internal
    all_pairs = all_pairs.rename(columns={'winner': 'winner_internal'})
    
    logger.info(f"Internal: {len(train_df)} train + {len(val_df)} val = {len(all_pairs)} total")
    return all_pairs[['image1', 'image2', 'winner_internal', 'series_id', 'split']]


def load_external_pairs(config_path: Path):
    """Load external pairs by querying MyDataset (no image loading, just metadata)."""
    config = yaml.safe_load(open(config_path))
    
    external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
    if external_path not in sys.path:
        sys.path.insert(0, external_path)
    
    from data.dataloader import MyDataset
    
    image_root = config['data'].get('image_root') or config['data']['root_dir']
    train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed', 42))
    val_data = MyDataset(train=False, image_root=image_root, seed=config.get('seed', 42))
    
    logger.info(f"External: {len(train_data)} train + {len(val_data)} val")
    
    # Extract pairs without loading images (just metadata)
    all_pairs = []
    
    logger.info("  Extracting train pairs...")
    for i in range(len(train_data)):
        all_pairs.append({
            'image1': train_data.pathA[i],
            'image2': train_data.pathB[i],
            'winner_external': int(train_data.result[i]),
            'series_id': extract_series_id(train_data.pathA[i]),
            'split': 'train'
        })
    
    logger.info("  Extracting val pairs...")
    for i in range(len(val_data)):
        all_pairs.append({
            'image1': val_data.pathA[i],
            'image2': val_data.pathB[i],
            'winner_external': int(val_data.result[i]),
            'series_id': extract_series_id(val_data.pathA[i]),
            'split': 'val'
        })
    
    logger.info(f"External: Extracted {len(all_pairs)} total pairs")
    return pd.DataFrame(all_pairs)


def compare_labels(internal_df: pd.DataFrame, external_df: pd.DataFrame):
    """Compare labels between internal and external dataloaders."""
    logger.info("\n" + "="*80)
    logger.info("LABEL COMPARISON ANALYSIS")
    logger.info("="*80)
    
    # Create pair keys for matching
    internal_df['pair_key'] = internal_df['image1'] + '|' + internal_df['image2']
    external_df['pair_key'] = external_df['image1'] + '|' + external_df['image2']
    
    # Find overlapping pairs
    internal_keys = set(internal_df['pair_key'])
    external_keys = set(external_df['pair_key'])
    
    overlap = internal_keys & external_keys
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Internal total: {len(internal_df)} pairs")
    logger.info(f"  External total: {len(external_df)} pairs")
    logger.info(f"  Overlapping: {len(overlap)} pairs ({100*len(overlap)/max(len(internal_df), len(external_df)):.1f}%)")
    
    if len(overlap) == 0:
        logger.error("\n❌ NO OVERLAPPING PAIRS FOUND!")
        return None
    
    # Merge on pair_key to compare labels
    merged = internal_df[['pair_key', 'image1', 'image2', 'series_id', 'winner_internal', 'split']].merge(
        external_df[['pair_key', 'winner_external']],
        on='pair_key',
        how='inner'
    )
    
    # Compare labels
    merged['labels_match'] = merged['winner_internal'] == merged['winner_external']
    merged['labels_inverted'] = merged['winner_internal'] == (1 - merged['winner_external'])
    
    match_count = merged['labels_match'].sum()
    inverted_count = merged['labels_inverted'].sum()
    other_count = len(merged) - match_count - inverted_count
    
    logger.info(f"\nLabel comparison across {len(merged)} overlapping pairs:")
    logger.info(f"  ✅ Matching labels:  {match_count:5d} ({100*match_count/len(merged):5.1f}%)")
    logger.info(f"  ⚠️  Inverted labels:  {inverted_count:5d} ({100*inverted_count/len(merged):5.1f}%)")
    logger.info(f"  ❌ Other mismatches: {other_count:5d} ({100*other_count/len(merged):5.1f}%)")
    
    # Determine issue
    if match_count == len(merged):
        logger.info("\n✅ RESULT: LABELS MATCH PERFECTLY")
        issue_type = 'none'
    elif inverted_count == len(merged):
        logger.error("\n❌ RESULT: LABELS ARE COMPLETELY INVERTED!")
        logger.error("   winner=0 in internal means winner=1 in external (or vice versa)")
        issue_type = 'fully_inverted'
    elif inverted_count > match_count:
        logger.error(f"\n❌ RESULT: LABELS ARE MOSTLY INVERTED ({inverted_count}/{len(merged)})")
        issue_type = 'mostly_inverted'
    elif match_count > inverted_count:
        logger.warning(f"\n⚠️  RESULT: LABELS MOSTLY MATCH ({match_count}/{len(merged)}) but some inversions exist")
        issue_type = 'mostly_match'
    else:
        logger.error("\n❌ RESULT: LABELS ARE INCONSISTENT (no clear pattern)")
        issue_type = 'inconsistent'
    
    # Add issue column
    merged['issue'] = merged.apply(
        lambda row: 'none' if row['labels_match'] 
        else 'inverted' if row['labels_inverted']
        else 'mismatch',
        axis=1
    )
    
    return merged, issue_type


def main():
    # Config paths
    internal_config = Path('outputs/siamese_e2e/20260113_073023/config.yaml')
    external_config = Path('outputs/siamese_e2e/20260111_005327/config.yaml')
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE LABEL VERIFICATION (FAST VERSION)")
    logger.info("Train + Val, All Pairs, No Image Loading")
    logger.info("="*80)
    
    logger.info("\nLoading internal pairs...")
    internal_df = load_internal_pairs(internal_config)
    
    logger.info("\nLoading external pairs...")
    external_df = load_external_pairs(external_config)
    
    logger.info("\nComparing labels...")
    result = compare_labels(internal_df, external_df)
    
    if result is None:
        logger.error("Cannot proceed - no overlapping pairs!")
        return
    
    merged, issue_type = result
    
    # Save results
    output_dir = Path('outputs/label_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full CSV with ALL overlapping pairs
    output_path = output_dir / 'all_overlapping_pairs_comprehensive.csv'
    # Reorder columns for clarity
    output_cols = ['image1', 'image2', 'series_id', 'split', 'winner_internal', 'winner_external', 
                   'labels_match', 'labels_inverted', 'issue']
    merged[output_cols].to_csv(output_path, index=False)
    logger.info(f"\n✅ Full results saved to: {output_path}")
    logger.info(f"   ({len(merged)} overlapping pairs)")
    
    # Save summary
    summary = {
        'total_internal_pairs': len(internal_df),
        'total_external_pairs': len(external_df),
        'total_overlapping_pairs': len(merged),
        'overlap_percentage': float(100 * len(merged) / max(len(internal_df), len(external_df))),
        'matching_labels': int(merged['labels_match'].sum()),
        'inverted_labels': int(merged['labels_inverted'].sum()),
        'other_mismatches': int((~merged['labels_match'] & ~merged['labels_inverted']).sum()),
        'match_percentage': float(100 * merged['labels_match'].sum() / len(merged)),
        'inverted_percentage': float(100 * merged['labels_inverted'].sum() / len(merged)),
        'issue_type': issue_type
    }
    
    import json
    summary_path = output_dir / 'summary_comprehensive.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary saved to: {summary_path}")
    
    # Show sample pairs from each category
    logger.info("\n" + "="*80)
    logger.info("SAMPLE PAIRS")
    logger.info("="*80)
    
    matches = merged[merged['labels_match']].head(5)
    inverted = merged[merged['labels_inverted']].head(5)
    other = merged[~merged['labels_match'] & ~merged['labels_inverted']].head(5)
    
    if len(matches) > 0:
        logger.info("\n✅ Matching labels (first 5):")
        print(matches[['image1', 'image2', 'winner_internal', 'winner_external']].to_string(index=False))
    
    if len(inverted) > 0:
        logger.info("\n⚠️  Inverted labels (first 5):")
        print(inverted[['image1', 'image2', 'winner_internal', 'winner_external']].to_string(index=False))
    
    if len(other) > 0:
        logger.info("\n❌ Other mismatches (first 5):")
        print(other[['image1', 'image2', 'winner_internal', 'winner_external']].to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("Verification complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
