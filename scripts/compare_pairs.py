"""
Compare which pairs exist in internal vs external dataloaders.

Checks:
- Which pairs are only in internal
- Which pairs are only in external
- Which pairs exist in both
- Total pair counts and overlap statistics
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


def load_all_pairs(config_path: Path, source_name: str):
    """Load ALL pairs from a config (not just first 500)."""
    config = yaml.safe_load(open(config_path))
    
    use_external = config.get('use_external_dataloader', False)
    batch_size = config['training']['batch_size']
    
    pairs = []
    
    if use_external:
        external_path = config['data'].get('external_path', r'D:\Projects\Series-Photo-Selection')
        if external_path not in sys.path:
            sys.path.insert(0, external_path)
        
        from data.dataloader import MyDataset
        
        image_root = config['data'].get('image_root') or config['data']['root_dir']
        train_data = MyDataset(train=True, image_root=image_root, seed=config.get('seed', 42))
        
        factory = DataLoaderFactory(batch_size=batch_size, num_workers=0, seed=config.get('seed', 42))
        train_loader, _, _ = factory.create_from_external(train_data, train_data, None)
        
        logger.info(f"{source_name}: Loading from EXTERNAL dataloader...")
        
    else:
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
        train_loader, _, _ = factory.create_from_phototriage(data, train_df, val_df, test_df, transform)
        
        logger.info(f"{source_name}: Loading from INTERNAL dataloader...")
    
    # Extract ALL pairs
    for batch in train_loader:
        for i in range(len(batch['image1'])):
            pairs.append({
                'image1': batch['image1'][i],
                'image2': batch['image2'][i],
                'winner': int(batch['winner'][i]),
                'series_id': extract_series_id(batch['image1'][i]),
                'source': source_name
            })
    
    logger.info(f"{source_name}: Extracted {len(pairs)} pairs")
    return pd.DataFrame(pairs)


def compare_pair_sets(internal_df: pd.DataFrame, external_df: pd.DataFrame):
    """Compare pair sets between dataloaders."""
    logger.info("\n" + "="*80)
    logger.info("PAIR SET COMPARISON REPORT")
    logger.info("="*80)
    
    # Create pair keys
    internal_df['pair_key'] = internal_df['image1'] + '|' + internal_df['image2']
    external_df['pair_key'] = external_df['image1'] + '|' + external_df['image2']
    
    # Get unique pairs
    internal_pairs = set(internal_df['pair_key'])
    external_pairs = set(external_df['pair_key'])
    
    # Compute overlaps
    overlap = internal_pairs & external_pairs
    only_internal = internal_pairs - external_pairs
    only_external = external_pairs - internal_pairs
    
    logger.info(f"\nPair counts:")
    logger.info(f"  Internal total: {len(internal_df)}")
    logger.info(f"  External total: {len(external_df)}")
    logger.info(f"  Internal unique: {len(internal_pairs)}")
    logger.info(f"  External unique: {len(external_pairs)}")
    
    logger.info(f"\nOverlap analysis:")
    logger.info(f"  Pairs in both: {len(overlap)}")
    logger.info(f"  Only in internal: {len(only_internal)}")
    logger.info(f"  Only in external: {len(only_external)}")
    logger.info(f"  Overlap percentage: {100*len(overlap)/max(len(internal_pairs), len(external_pairs)):.1f}%")
    
    # Create output dataframe
    all_pairs = []
    
    # Pairs in both
    for pair_key in overlap:
        img1, img2 = pair_key.split('|')
        internal_row = internal_df[internal_df['pair_key'] == pair_key].iloc[0]
        external_row = external_df[external_df['pair_key'] == pair_key].iloc[0]
        
        all_pairs.append({
            'image1': img1,
            'image2': img2,
            'series_id': internal_row['series_id'],
            'in_internal': True,
            'in_external': True,
            'internal_winner': internal_row['winner'],
            'external_winner': external_row['winner'],
            'labels_match': internal_row['winner'] == external_row['winner']
        })
    
    # Pairs only in internal
    for pair_key in only_internal:
        img1, img2 = pair_key.split('|')
        internal_row = internal_df[internal_df['pair_key'] == pair_key].iloc[0]
        
        all_pairs.append({
            'image1': img1,
            'image2': img2,
            'series_id': internal_row['series_id'],
            'in_internal': True,
            'in_external': False,
            'internal_winner': internal_row['winner'],
            'external_winner': None,
            'labels_match': None
        })
    
    # Pairs only in external
    for pair_key in only_external:
        img1, img2 = pair_key.split('|')
        external_row = external_df[external_df['pair_key'] == pair_key].iloc[0]
        
        all_pairs.append({
            'image1': img1,
            'image2': img2,
            'series_id': external_row['series_id'],
            'in_internal': False,
            'in_external': True,
            'internal_winner': None,
            'external_winner': external_row['winner'],
            'labels_match': None
        })
    
    result_df = pd.DataFrame(all_pairs)
    
    # Label match stats for overlapping pairs
    if len(overlap) > 0:
        overlap_df = result_df[result_df['in_internal'] & result_df['in_external']]
        match_count = overlap_df['labels_match'].sum()
        logger.info(f"\nLabel consistency in overlapping pairs:")
        logger.info(f"  Matching labels: {match_count}/{len(overlap)} ({100*match_count/len(overlap):.1f}%)")
    
    return result_df


def main():
    # Config paths
    internal_config = Path('outputs/siamese_e2e/20260113_073023/config.yaml')
    external_config = Path('outputs/siamese_e2e/20260111_005327/config.yaml')
    
    logger.info("Loading ALL pairs from both dataloaders...")
    logger.info("(This may take a minute...)\n")
    
    internal_df = load_all_pairs(internal_config, 'internal')
    external_df = load_all_pairs(external_config, 'external')
    
    logger.info("\nComparing pair sets...")
    result = compare_pair_sets(internal_df, external_df)
    
    # Save results
    output_dir = Path('outputs/pair_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'overlap_analysis.csv'
    result.to_csv(output_path, index=False)
    logger.info(f"\n✅ Full comparison saved to: {output_path}")
    
    # Save summary statistics
    summary = {
        'internal_total_pairs': len(internal_df),
        'external_total_pairs': len(external_df),
        'internal_unique_pairs': len(set(internal_df['image1'] + '|' + internal_df['image2'])),
        'external_unique_pairs': len(set(external_df['image1'] + '|' + external_df['image2'])),
        'pairs_in_both': int(result['in_internal'].sum() & result['in_external'].sum()),
        'pairs_only_internal': int(result['in_internal'].sum() & ~result['in_external'].sum()),
        'pairs_only_external': int(~result['in_internal'].sum() & result['in_external'].sum()),
    }
    
    import json
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary saved to: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Pair comparison complete!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
