"""
Verify label consistency between internal and external dataloaders.

Compares winner labels for overlapping pairs to identify if:
- Labels are swapped (winner=0 vs winner=1 inverted)
- Image order is different (image1/image2 swapped)
- Labels simply don't match
"""
import sys
from pathlib import Path
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_bench.datasets.dataloader_factory import DataLoaderFactory
from sim_bench.datasets.phototriage_data import PhotoTriageData
from sim_bench.datasets.transform_factory import create_transform


def extract_series_id(filename: str) -> int:
    """Extract series_id from filename like '000123-02.JPG' -> 123."""
    return int(filename.split('-')[0])


def load_all_pairs_from_config(config_path: Path):
    """Load ALL pairs (train + val) from a config file."""
    config = yaml.safe_load(open(config_path))
    
    use_external = config.get('use_external_dataloader', False)
    batch_size = config['training']['batch_size']
    
    all_pairs = []
    
    if use_external:
        # External dataloader
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            sys.path.insert(0, external_path)
        
        from data.dataloader import MyDataset
        
        image_root = config['data'].get('image_root') or config['data']['root_dir']
        train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed', 42))
        val_data = MyDataset(train=False, image_root=image_root, seed=config.get('seed', 42))
        
        factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=config.get('seed', 42))
        train_loader, val_loader, _ = factory.create_from_external(train_data, val_data, None)
        
        logger.info(f"Loading from EXTERNAL dataloader: {len(train_data)} train + {len(val_data)} val")
        loaders = [('train', train_loader), ('val', val_loader)]
        
    else:
        # Internal dataloader
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
        
        transform = create_transform(config)
        factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=config.get('seed', 42))
        train_loader, val_loader, _ = factory.create_from_phototriage(data, train_df, val_df, test_df, transform)
        
        logger.info(f"Loading from INTERNAL dataloader: {len(train_df)} train + {len(val_df)} val")
        loaders = [('train', train_loader), ('val', val_loader)]
    
    # Extract ALL pairs from both train and val
    for split_name, loader in loaders:
        logger.info(f"  Extracting {split_name} pairs...")
        for batch in loader:
            for i in range(len(batch['image1'])):
                all_pairs.append({
                    'image1': batch['image1'][i],
                    'image2': batch['image2'][i],
                    'winner': int(batch['winner'][i]),
                    'series_id': extract_series_id(batch['image1'][i]),
                    'split': split_name
                })
    
    logger.info(f"Extracted {len(all_pairs)} total pairs")
    return pd.DataFrame(all_pairs)


def compare_labels(internal_df: pd.DataFrame, external_df: pd.DataFrame):
    """Compare labels between internal and external dataloaders."""
    logger.info("\n" + "="*80)
    logger.info("LABEL VERIFICATION REPORT")
    logger.info("="*80)
    
    # Create pair keys for matching
    internal_df['pair_key'] = internal_df['image1'] + '|' + internal_df['image2']
    external_df['pair_key'] = external_df['image1'] + '|' + external_df['image2']
    
    # Also check reverse order (image1/image2 swapped)
    internal_df['pair_key_rev'] = internal_df['image2'] + '|' + internal_df['image1']
    
    # Find overlapping pairs
    internal_keys = set(internal_df['pair_key'])
    external_keys = set(external_df['pair_key'])
    
    overlap = internal_keys & external_keys
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Internal: {len(internal_df)} pairs")
    logger.info(f"  External: {len(external_df)} pairs")
    logger.info(f"  Overlapping: {len(overlap)} pairs")
    
    if len(overlap) == 0:
        logger.warning("\n⚠️  NO OVERLAPPING PAIRS FOUND!")
        logger.warning("This means the dataloaders are producing completely different pairs.")
        logger.warning("Checking if image order is swapped...")
        
        # Check for swapped pairs
        internal_keys_rev = set(internal_df['pair_key_rev'])
        overlap_swapped = internal_keys_rev & external_keys
        
        if len(overlap_swapped) > 0:
            logger.warning(f"⚠️  Found {len(overlap_swapped)} pairs with SWAPPED image order!")
            logger.warning("Example: Internal has (A, B) but External has (B, A)")
        
        return None
    
    # Merge on pair_key to compare labels
    merged = internal_df[['pair_key', 'image1', 'image2', 'series_id', 'winner']].merge(
        external_df[['pair_key', 'winner']],
        on='pair_key',
        suffixes=('_internal', '_external')
    )
    
    # Compare labels
    merged['labels_match'] = merged['winner_internal'] == merged['winner_external']
    merged['labels_inverted'] = merged['winner_internal'] == (1 - merged['winner_external'])
    
    match_count = merged['labels_match'].sum()
    inverted_count = merged['labels_inverted'].sum()
    
    logger.info(f"\nLabel comparison:")
    logger.info(f"  Matching labels: {match_count}/{len(merged)} ({100*match_count/len(merged):.1f}%)")
    logger.info(f"  Inverted labels: {inverted_count}/{len(merged)} ({100*inverted_count/len(merged):.1f}%)")
    logger.info(f"  Other mismatches: {len(merged)-match_count-inverted_count}/{len(merged)}")
    
    # Determine issue
    if match_count == len(merged):
        logger.info("\n✅ LABELS MATCH PERFECTLY - No label issue detected")
        issue = 'none'
    elif inverted_count == len(merged):
        logger.error("\n❌ LABELS ARE INVERTED!")
        logger.error("winner=0 in internal means winner=1 in external (or vice versa)")
        issue = 'inverted'
    elif inverted_count > match_count:
        logger.error("\n❌ LABELS ARE MOSTLY INVERTED!")
        issue = 'mostly_inverted'
    else:
        logger.error("\n❌ LABELS DON'T MATCH!")
        logger.error("This suggests different label logic or data corruption")
        issue = 'mismatch'
    
    # Add issue column
    merged['issue'] = merged.apply(
        lambda row: 'none' if row['labels_match'] 
        else 'inverted' if row['labels_inverted']
        else 'mismatch',
        axis=1
    )
    
    return merged


def main():
    # Config paths for the two experiments
    internal_config = Path('outputs/siamese_e2e/20260113_073023/config.yaml')
    external_config = Path('outputs/siamese_e2e/20260111_005327/config.yaml')
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE LABEL VERIFICATION - ALL PAIRS (TRAIN + VAL)")
    logger.info("="*80)
    
    logger.info("\nLoading ALL pairs from internal dataloader...")
    internal_df = load_all_pairs_from_config(internal_config)
    
    logger.info("\nLoading ALL pairs from external dataloader...")
    external_df = load_all_pairs_from_config(external_config)
    
    logger.info("\nComparing labels across all overlapping pairs...")
    result = compare_labels(internal_df, external_df)
    
    # Save results
    output_dir = Path('outputs/label_verification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if result is not None:
        # Save full CSV with all overlapping pairs
        output_path = output_dir / 'all_overlapping_pairs.csv'
        result.to_csv(output_path, index=False)
        logger.info(f"\n✅ Full results saved to: {output_path}")
        logger.info(f"   ({len(result)} overlapping pairs)")
        
        # Save summary statistics
        summary = {
            'total_internal_pairs': len(internal_df),
            'total_external_pairs': len(external_df),
            'total_overlapping_pairs': len(result),
            'overlap_percentage': float(100 * len(result) / max(len(internal_df), len(external_df))),
            'matching_labels': int(result['labels_match'].sum()),
            'inverted_labels': int(result['labels_inverted'].sum()),
            'other_mismatches': int((~result['labels_match'] & ~result['labels_inverted']).sum()),
            'match_percentage': float(100 * result['labels_match'].sum() / len(result)),
            'inverted_percentage': float(100 * result['labels_inverted'].sum() / len(result))
        }
        
        import json
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✅ Summary saved to: {summary_path}")
        
        # Show sample mismatches
        logger.info("\n" + "="*80)
        logger.info("SAMPLE PAIRS")
        logger.info("="*80)
        
        # Show samples of each category
        matches = result[result['labels_match']].head(5)
        inverted = result[result['labels_inverted']].head(5)
        other = result[~result['labels_match'] & ~result['labels_inverted']].head(5)
        
        if len(matches) > 0:
            logger.info("\nMatching labels (first 5):")
            logger.info(matches[['image1', 'image2', 'winner_internal', 'winner_external', 'issue']].to_string(index=False))
        
        if len(inverted) > 0:
            logger.info("\nInverted labels (first 5):")
            logger.info(inverted[['image1', 'image2', 'winner_internal', 'winner_external', 'issue']].to_string(index=False))
        
        if len(other) > 0:
            logger.info("\nOther mismatches (first 5):")
            logger.info(other[['image1', 'image2', 'winner_internal', 'winner_external', 'issue']].to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("Verification complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
