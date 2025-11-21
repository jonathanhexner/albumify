"""
Create labeled pairs CSV from PhotoTriage reviews.

This script post-processes PhotoTriage review JSON files into a consolidated CSV
for benchmarking. It performs the following steps:

1. Loads all review JSON files from the input directory
2. Extracts reason texts and classifies them using keyword-based AttributeMapper
3. Aggregates votes and labels per image pair
4. Outputs a single CSV with vote counts, agreement metrics, and label distributions

The classification uses static keyword matching (no LLMs or embeddings) to map
free-text reasons to predefined quality attributes.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import sys

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.phototriage.attribute_mapper import AttributeMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_all_reviews(root_dir: Path) -> pd.DataFrame:
    """
    Load all review JSON files and extract individual reviews.

    Args:
        root_dir: Directory containing review JSON files (*.json)

    Returns:
        DataFrame with columns: series_id, compareID1, compareID2, userChoice, reason_text
    """
    json_files = list(root_dir.glob('*.json'))

    if not json_files:
        raise ValueError(f"No JSON files found in {root_dir}")

    logger.info(f"Found {len(json_files)} JSON files")

    all_reviews = []

    for json_file in json_files:
        series_id = json_file.stem

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {json_file}: {e}")
            continue

        reviews = data.get('reviews', [])

        for review in reviews:
            compare_id_1 = review.get('compareID1')
            compare_id_2 = review.get('compareID2')
            user_choice = review.get('userChoice', '')

            # Extract reason text - IMPORTANT: reason[1] is the actual text!
            # Format is ["", "actual reason text"]
            reason = review.get('reason', ['', ''])

            if len(reason) < 2:
                logger.debug(f"Series {series_id}: Reason array has < 2 elements")
                reason_text = ''
            else:
                reason_text = reason[1].strip()

            all_reviews.append({
                'series_id': series_id,
                'compareID1': compare_id_1,
                'compareID2': compare_id_2,
                'userChoice': user_choice,
                'reason_text': reason_text
            })

    df = pd.DataFrame(all_reviews)
    logger.info(f"Extracted {len(df)} individual reviews from {len(json_files)} files")

    return df


def classify_reasons(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify reason texts into quality attribute labels.

    Uses AttributeMapper with keyword-based pattern matching. Aggregates
    composition-related attributes (framing, cropping, placement, clutter)
    into a single 'composition' label.

    Args:
        reviews_df: DataFrame with reason_text and userChoice columns

    Returns:
        DataFrame with added 'label' column
    """
    mapper = AttributeMapper()

    # Composition sub-attributes to aggregate
    comp_attrs = {
        'framing',
        'cropping_completeness',
        'subject_placement',
        'background_clutter'
    }

    labels = []

    for _, row in reviews_df.iterrows():
        reason_text = row['reason_text']
        user_choice = row['userChoice']

        if not reason_text:
            label = 'no_reason_given'
        else:
            # Map reason to attributes using keyword matching
            attributes = mapper.map_reason_to_attributes(reason_text, user_choice)

            if len(attributes) == 0:
                label = 'no_specific_attribute'
            else:
                # Take FIRST attribute as primary label
                first_attr = attributes[0]
                attr_name = first_attr.name

                # Aggregate composition sub-attributes
                if attr_name in comp_attrs:
                    label = 'composition'
                else:
                    label = attr_name

        labels.append(label)

    reviews_df['label'] = labels

    logger.info(f"Classified {len(reviews_df)} reviews into labels")
    logger.info(f"Label distribution:\n{pd.Series(labels).value_counts()}")

    return reviews_df


def aggregate_by_pair(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate reviews per (compareID1, compareID2) pair.

    Computes:
    - Vote counts (LEFT, RIGHT, Total)
    - Agreement metrics
    - Majority winner (MaxVote)
    - Per-label counts
    - Majority label

    Args:
        classified_df: DataFrame with classified reviews

    Returns:
        DataFrame with one row per unique pair
    """
    # Step 1: Count votes LEFT vs RIGHT per pair
    # IMPORTANT: Group by series_id to keep each series separate!
    df_votes = classified_df.groupby(
        ['series_id', 'compareID1', 'compareID2', 'userChoice']
    ).size().reset_index(name='count')

    df_votes_pivot = pd.pivot_table(
        df_votes,
        values='count',
        index=['series_id', 'compareID1', 'compareID2'],
        columns='userChoice',
        aggfunc='sum'
    ).reset_index()

    # Ensure LEFT and RIGHT columns exist
    for col in ['LEFT', 'RIGHT']:
        if col not in df_votes_pivot.columns:
            df_votes_pivot[col] = 0
        df_votes_pivot[col] = df_votes_pivot[col].fillna(0).astype(int)

    # Calculate vote aggregates
    df_votes_pivot['Total'] = df_votes_pivot['LEFT'] + df_votes_pivot['RIGHT']
    df_votes_pivot['max_count'] = df_votes_pivot[['LEFT', 'RIGHT']].max(axis=1)
    df_votes_pivot['Agreement'] = df_votes_pivot['max_count'] / df_votes_pivot['Total']

    # MaxVote = winning image ID
    # Default to RIGHT (compareID2), but override if LEFT wins
    df_votes_pivot['MaxVote'] = df_votes_pivot['compareID2']
    left_wins = df_votes_pivot['LEFT'] > df_votes_pivot['RIGHT']
    df_votes_pivot.loc[left_wins, 'MaxVote'] = df_votes_pivot.loc[left_wins, 'compareID1']

    # Step 2: Count labels per pair
    df_labels = classified_df.groupby(
        ['series_id', 'compareID1', 'compareID2', 'label']
    ).size().reset_index(name='count')

    df_labels_pivot = pd.pivot_table(
        df_labels,
        values='count',
        index=['series_id', 'compareID1', 'compareID2'],
        columns='label',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Rename columns to label_* format
    label_cols = [col for col in df_labels_pivot.columns
                  if col not in ['series_id', 'compareID1', 'compareID2']]

    rename_dict = {col: f'label_{col}' for col in label_cols}
    df_labels_pivot.rename(columns=rename_dict, inplace=True)

    # Step 3: Determine majority label (most frequent)
    label_count_cols = [col for col in df_labels_pivot.columns if col.startswith('label_')]

    if label_count_cols:
        df_labels_pivot['majority_label'] = df_labels_pivot[label_count_cols].idxmax(axis=1)
        df_labels_pivot['majority_label'] = df_labels_pivot['majority_label'].str.replace('label_', '')
    else:
        df_labels_pivot['majority_label'] = 'unknown'

    # Step 4: Merge votes + labels
    result = pd.merge(
        df_votes_pivot,
        df_labels_pivot,
        on=['series_id', 'compareID1', 'compareID2'],
        how='left'
    )

    # Add num_reviewers column (same as Total)
    result['num_reviewers'] = result['Total']

    # Fill NaN label counts with 0
    for col in result.columns:
        if col.startswith('label_'):
            result[col] = result[col].fillna(0).astype(int)

    logger.info(f"Aggregated into {len(result)} unique pairs")

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create labeled pairs CSV from PhotoTriage reviews',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_labeled_pairs.py
  python create_labeled_pairs.py --input-dir /path/to/reviews --output results.csv
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval"),
        help='Root directory containing review JSON files (default: PhotoTriage reviews_trainval)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series\photo_triage_pairs_with_labels.csv"),
        help='Output CSV file path (default: photo_triage_pairs_with_labels.csv)'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load all reviews
    logger.info(f"Loading reviews from {args.input_dir}")
    reviews_df = load_all_reviews(args.input_dir)
    logger.info(f"Loaded {len(reviews_df)} individual reviews")

    # Classify reasons to labels
    logger.info("Classifying reason texts using AttributeMapper (keyword matching)")
    classified_df = classify_reasons(reviews_df)

    # Aggregate by pair
    logger.info("Aggregating by image pairs")
    pairs_df = aggregate_by_pair(classified_df)

    # Reorder columns for better readability
    # Series ID and image IDs first, then votes, then labels
    id_cols = ['series_id', 'compareID1', 'compareID2']
    vote_cols = ['LEFT', 'RIGHT', 'Total', 'max_count', 'Agreement', 'MaxVote', 'num_reviewers']
    label_meta_cols = ['majority_label']
    label_count_cols = sorted([col for col in pairs_df.columns if col.startswith('label_')])

    column_order = id_cols + vote_cols + label_meta_cols + label_count_cols
    pairs_df = pairs_df[column_order]

    # Save CSV
    logger.info(f"Saving to {args.output}")
    pairs_df.to_csv(args.output, index=False)
    logger.info(f"Successfully saved {len(pairs_df)} pairs to CSV")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal unique pairs: {len(pairs_df)}")
    print(f"Total individual reviews: {len(reviews_df)}")
    print(f"Average reviews per pair: {len(reviews_df) / len(pairs_df):.2f}")
    print(f"\nAgreement metrics:")
    print(f"  Mean agreement: {pairs_df['Agreement'].mean():.3f}")
    print(f"  Median agreement: {pairs_df['Agreement'].median():.3f}")
    print(f"  Pairs with >=70% agreement: {(pairs_df['Agreement'] >= 0.70).sum()} "
          f"({100 * (pairs_df['Agreement'] >= 0.70).sum() / len(pairs_df):.1f}%)")
    print(f"  Pairs with >=2 reviewers: {(pairs_df['Total'] >= 2).sum()} "
          f"({100 * (pairs_df['Total'] >= 2).sum() / len(pairs_df):.1f}%)")

    print(f"\nMajority label distribution:")
    majority_counts = pairs_df['majority_label'].value_counts()
    for label, count in majority_counts.items():
        pct = 100 * count / len(pairs_df)
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nOutput saved to: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
