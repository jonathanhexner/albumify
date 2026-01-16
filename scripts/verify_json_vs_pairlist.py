"""
Compare JSON review votes with train_pairlist.txt preference ratios.

Analyzes discrepancies between:
1. Direct vote counts from JSON reviews
2. Preference ratios in train_pairlist.txt
"""
import json
import logging
from pathlib import Path
from collections import defaultdict
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def parse_json_votes(json_path: Path):
    """Parse JSON file and count votes for each pair."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    series_id = json_path.stem.lstrip('0')  # Remove leading zeros
    reviews = data.get('reviews', [])
    
    # Count votes for each pair (normalized to img1 < img2)
    vote_counts = defaultdict(lambda: {'img1_wins': 0, 'img2_wins': 0, 'total': 0})
    
    for review in reviews:
        cid1 = review['compareID1']
        cid2 = review['compareID2']
        choice = review['userChoice']
        
        # Normalize: always store as (smaller_id, larger_id)
        img1 = min(cid1, cid2)
        img2 = max(cid1, cid2)
        
        # Determine winner based on position
        if cid1 < cid2:
            # img1 is LEFT, img2 is RIGHT
            winner = img1 if choice == 'LEFT' else img2
        else:
            # img2 is LEFT, img1 is RIGHT
            winner = img2 if choice == 'LEFT' else img1
        
        pair_key = (img1, img2)
        vote_counts[pair_key]['total'] += 1
        
        if winner == img1:
            vote_counts[pair_key]['img1_wins'] += 1
        else:
            vote_counts[pair_key]['img2_wins'] += 1
    
    # Convert to list with preference ratios
    results = []
    for (img1, img2), counts in vote_counts.items():
        total = counts['total']
        img1_wins = counts['img1_wins']
        img2_wins = counts['img2_wins']
        
        # Preference ratio: fraction who prefer img1
        pref_ratio = img1_wins / total if total > 0 else 0.5
        
        results.append({
            'series_id': series_id,
            'img1': img1 + 1,  # Convert from 0-indexed to 1-indexed
            'img2': img2 + 1,
            'img1_wins': img1_wins,
            'img2_wins': img2_wins,
            'total_votes': total,
            'json_pref_ratio': round(pref_ratio, 3)
        })
    
    return results


def parse_pairlist_line(line: str):
    """Parse a line from train_pairlist.txt."""
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    
    return {
        'series_id': parts[0],
        'img1': int(parts[1]),
        'img2': int(parts[2]),
        'txt_pref_ratio': float(parts[3]),
        'rank1': int(parts[4]),
        'rank2': int(parts[5])
    }


def compare_series(series_id: str, json_results: list, pairlist_path: Path):
    """Compare JSON votes with pairlist for a specific series."""
    logger.info(f"\n{'='*80}")
    logger.info(f"SERIES {series_id} - Comparison")
    logger.info(f"{'='*80}")
    
    # Load pairlist lines for this series
    pairlist_pairs = {}
    with open(pairlist_path, 'r') as f:
        for line in f:
            parsed = parse_pairlist_line(line)
            if parsed and parsed['series_id'] == series_id:
                key = (parsed['img1'], parsed['img2'])
                pairlist_pairs[key] = parsed
    
    # Create comparison DataFrame
    comparison = []
    
    for json_pair in json_results:
        img1 = json_pair['img1']
        img2 = json_pair['img2']
        key = (img1, img2)
        
        txt_pair = pairlist_pairs.get(key, {})
        
        comparison.append({
            'img1': img1,
            'img2': img2,
            'json_img1_wins': json_pair['img1_wins'],
            'json_img2_wins': json_pair['img2_wins'],
            'json_total': json_pair['total_votes'],
            'json_pref_ratio': json_pair['json_pref_ratio'],
            'txt_pref_ratio': txt_pair.get('txt_pref_ratio', None),
            'txt_rank1': txt_pair.get('rank1', None),
            'txt_rank2': txt_pair.get('rank2', None)
        })
    
    df = pd.DataFrame(comparison)
    
    # Calculate discrepancy
    df['ratio_diff'] = (df['txt_pref_ratio'] - df['json_pref_ratio']).abs()
    df['discrepancy'] = df['ratio_diff'] > 0.1  # Flag if >10% difference
    
    # Display results
    logger.info("\nPair-by-Pair Comparison:")
    logger.info("-" * 80)
    
    for _, row in df.iterrows():
        img1, img2 = row['img1'], row['img2']
        json_ratio = row['json_pref_ratio']
        txt_ratio = row['txt_pref_ratio']
        diff = row['ratio_diff']
        
        flag = "⚠️" if row['discrepancy'] else "✓"
        
        logger.info(f"{flag} Pair {img1} vs {img2}:")
        logger.info(f"    JSON: {row['json_img1_wins']} votes for img{img1}, "
                   f"{row['json_img2_wins']} votes for img{img2} "
                   f"(pref_ratio={json_ratio:.3f})")
        logger.info(f"    TXT:  pref_ratio={txt_ratio:.3f}, "
                   f"rank{img1}={row['txt_rank1']}, rank{img2}={row['txt_rank2']}")
        
        if row['discrepancy']:
            logger.info(f"    ❌ MISMATCH: Difference = {diff:.3f}")
        
        logger.info("")
    
    # Summary
    mismatches = df['discrepancy'].sum()
    logger.info(f"{'='*80}")
    logger.info(f"Summary: {mismatches} out of {len(df)} pairs have >10% discrepancy")
    
    return df


def main():
    logger.info("="*80)
    logger.info("JSON vs PAIRLIST VERIFICATION")
    logger.info("="*80)
    
    # Analyze Series 1
    json_path = Path(r'D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval\000001.json')
    pairlist_path = Path(r'D:\Projects\Series-Photo-Selection\data\train_pairlist.txt')
    
    logger.info(f"\nParsing JSON: {json_path}")
    json_results = parse_json_votes(json_path)
    
    logger.info(f"Found {len(json_results)} pairs in JSON")
    
    # Compare with pairlist
    df = compare_series('1', json_results, pairlist_path)
    
    # Save results
    output_dir = Path('outputs/json_vs_pairlist')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / 'series_1_comparison.csv', index=False)
    logger.info(f"\n✅ Results saved to: {output_dir / 'series_1_comparison.csv'}")
    
    # Key finding: Check img1 vs img3
    logger.info("\n" + "="*80)
    logger.info("KEY FINDING: img1 vs img3")
    logger.info("="*80)
    
    row_1_3 = df[(df['img1'] == 1) & (df['img2'] == 3)]
    if not row_1_3.empty:
        row = row_1_3.iloc[0]
        logger.info(f"\nJSON says:")
        logger.info(f"  img1 wins: {row['json_img1_wins']} votes")
        logger.info(f"  img3 wins: {row['json_img2_wins']} votes")
        logger.info(f"  Preference for img1: {row['json_pref_ratio']:.1%}")
        
        logger.info(f"\nTXT says:")
        logger.info(f"  Preference ratio: {row['txt_pref_ratio']:.1%}")
        logger.info(f"  Rank of img1: {row['txt_rank1']}")
        logger.info(f"  Rank of img3: {row['txt_rank2']}")
        
        logger.info(f"\n❌ Discrepancy: {abs(row['txt_pref_ratio'] - row['json_pref_ratio']):.1%}")
        
        if row['json_img1_wins'] == 0:
            logger.info("\n🚨 CRITICAL: JSON shows img3 won ALL votes, but TXT shows 63.6% prefer img1!")


if __name__ == '__main__':
    main()
