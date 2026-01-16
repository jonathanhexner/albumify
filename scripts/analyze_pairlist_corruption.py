"""
Analyze corruption in train_pairlist.txt and val_pairlist.txt files.

Identifies pairs where the winner is NEITHER of the two images being compared,
which causes make_shuffle_path to incorrectly assign winner=1 to all such pairs.
"""
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def analyze_pairlist_file(filepath: Path):
    """Analyze a pairlist file for data corruption."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Analyzing: {filepath.name}")
    logger.info(f"{'='*80}")
    
    pairs = []
    valid_img1_wins = 0
    valid_img2_wins = 0
    invalid_pairs = 0
    invalid_examples = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split()
            
            if len(parts) < 6:
                continue
            
            series_id = parts[0]
            img1_id = parts[1]
            img2_id = parts[2]
            score = parts[3]
            winner = parts[4]
            loser = parts[5]
            
            # Check if winner matches img1 or img2
            if winner == img1_id:
                valid_img1_wins += 1
                result_label = 0
                status = 'valid_img1_wins'
            elif winner == img2_id:
                valid_img2_wins += 1
                result_label = 1
                status = 'valid_img2_wins'
            else:
                invalid_pairs += 1
                result_label = 1  # This is what make_shuffle_path assigns!
                status = 'INVALID_winner_not_in_pair'
                
                if len(invalid_examples) < 10:
                    invalid_examples.append({
                        'line': line_num,
                        'series': series_id,
                        'img1': img1_id,
                        'img2': img2_id,
                        'winner': winner,
                        'issue': f'Comparing img{img1_id} vs img{img2_id}, but winner=img{winner}'
                    })
            
            pairs.append({
                'series_id': series_id,
                'img1_id': img1_id,
                'img2_id': img2_id,
                'score': float(score),
                'winner_id': winner,
                'loser_id': loser,
                'result_label': result_label,
                'status': status
            })
    
    total = len(pairs)
    
    # Statistics
    logger.info(f"\nTotal pairs: {total}")
    logger.info(f"\nValidity breakdown:")
    logger.info(f"  ✅ Valid - img1 wins (result=0): {valid_img1_wins:5d} ({100*valid_img1_wins/total:5.1f}%)")
    logger.info(f"  ✅ Valid - img2 wins (result=1): {valid_img2_wins:5d} ({100*valid_img2_wins/total:5.1f}%)")
    logger.info(f"  ❌ INVALID - winner not in pair: {invalid_pairs:5d} ({100*invalid_pairs/total:5.1f}%)")
    
    # Result label distribution (what make_shuffle_path actually creates)
    result_0_count = valid_img1_wins
    result_1_count = valid_img2_wins + invalid_pairs
    
    logger.info(f"\nActual result labels created by make_shuffle_path:")
    logger.info(f"  result=0 (img1 wins): {result_0_count:5d} ({100*result_0_count/total:5.1f}%)")
    logger.info(f"  result=1 (img2 wins): {result_1_count:5d} ({100*result_1_count/total:5.1f}%)")
    logger.info(f"  ^ BIAS: {result_1_count - result_0_count:+5d} extra result=1 labels due to invalid pairs")
    
    # Show invalid examples
    if invalid_examples:
        logger.info(f"\n❌ Examples of INVALID pairs (first 10):")
        for ex in invalid_examples:
            logger.info(f"  Line {ex['line']:5d}: {ex['issue']}")
    
    return pd.DataFrame(pairs), {
        'total': total,
        'valid_img1_wins': valid_img1_wins,
        'valid_img2_wins': valid_img2_wins,
        'invalid': invalid_pairs,
        'result_0': result_0_count,
        'result_1': result_1_count,
        'bias': result_1_count - result_0_count
    }


def main():
    logger.info("="*80)
    logger.info("PAIRLIST CORRUPTION ANALYSIS")
    logger.info("="*80)
    
    train_pairlist = Path(r'D:\Projects\Series-Photo-Selection\data\train_pairlist.txt')
    val_pairlist = Path(r'D:\Projects\Series-Photo-Selection\data\val_pairlist.txt')
    
    # Analyze both files
    train_df, train_stats = analyze_pairlist_file(train_pairlist)
    val_df, val_stats = analyze_pairlist_file(val_pairlist)
    
    # Save corrupted pairs to CSV
    output_dir = Path('outputs/pairlist_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train invalid pairs
    train_invalid = train_df[train_df['status'] == 'INVALID_winner_not_in_pair']
    train_invalid.to_csv(output_dir / 'train_invalid_pairs.csv', index=False)
    logger.info(f"\n✅ Train invalid pairs saved to: {output_dir / 'train_invalid_pairs.csv'}")
    logger.info(f"   ({len(train_invalid)} invalid pairs)")
    
    # Save val invalid pairs  
    val_invalid = val_df[val_df['status'] == 'INVALID_winner_not_in_pair']
    val_invalid.to_csv(output_dir / 'val_invalid_pairs.csv', index=False)
    logger.info(f"✅ Val invalid pairs saved to: {output_dir / 'val_invalid_pairs.csv'}")
    logger.info(f"   ({len(val_invalid)} invalid pairs)")
    
    # Save summary
    summary = {
        'train': train_stats,
        'val': val_stats,
        'combined_invalid_rate': 100 * (train_stats['invalid'] + val_stats['invalid']) / (train_stats['total'] + val_stats['total'])
    }
    
    import json
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Summary saved to: {output_dir / 'summary.json'}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    
    total_invalid = train_stats['invalid'] + val_stats['invalid']
    total_pairs = train_stats['total'] + val_stats['total']
    
    logger.info(f"\n🚨 CORRUPTION DETECTED!")
    logger.info(f"  Total pairs: {total_pairs}")
    logger.info(f"  Invalid pairs: {total_invalid} ({100*total_invalid/total_pairs:.1f}%)")
    logger.info(f"  These invalid pairs ALL get assigned winner=1 incorrectly!")
    
    logger.info(f"\n📊 This explains the winner=1 bias:")
    logger.info(f"  Valid pairs: ~50/50 split")
    logger.info(f"  Invalid pairs: 100% assigned winner=1")
    logger.info(f"  Combined: ~{100*train_stats['result_1']/train_stats['total']:.0f}% winner=1")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()
