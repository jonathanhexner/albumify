"""
Create labeled pairs CSV from PhotoTriage reviews with DUAL labeling methods.

This script post-processes PhotoTriage review JSON files and outputs TWO CSVs:
1. Keyword-based labeling (improved with correct reason extraction)
2. Embedding-based similarity labeling

Both methods:
- Load all review JSON files
- Extract reason texts correctly based on userChoice
- Aggregate votes and labels per image pair
- Output separate CSVs for comparison
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import sys

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sim_bench.phototriage.attribute_mapper import AttributeMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Label descriptions for embedding similarity
LABEL_DESCRIPTIONS = {
    'sharpness': 'The image is sharp, focused, and clear with good definition, not blurry or soft',
    'detail_visibility': 'Image details are visible and clear, you can see important elements, not hazy or obscured',
    'motion_blur': 'The image has motion blur from movement, or is static and still',
    'composition': 'The image has good composition, framing, cropping, subject placement, or clean uncluttered background',
    'exposure_quality': 'The image has good exposure and brightness, not too dark or too bright',
    'lighting_quality': 'The image has good lighting with appropriate shadows and illumination',
    'dynamic_range': 'The image has good contrast and dynamic range, not washed out or flat',
    'field_of_view': 'The image shows an appropriate field of view, not too narrow or too wide',
    'distance_appropriateness': 'The subject distance is appropriate, not too far away or too close',
    'subject_interest': 'The image subject is interesting, engaging, and captivating, not boring or dull',
    'other': 'General image quality or aesthetic preference not covered by specific attributes'
}


def load_all_reviews(root_dir: Path) -> pd.DataFrame:
    """
    Load all review JSON files and extract individual reviews.

    IMPORTANT: Correctly extracts reason based on userChoice:
    - reason[0] = reason for LEFT image
    - reason[1] = reason for RIGHT image

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
            compare_file_1 = review.get('compareFile1', '')
            compare_file_2 = review.get('compareFile2', '')
            user_choice = review.get('userChoice', '')

            # Extract reason text - concatenate both positions
            # reason = [reason_for_left, reason_for_right]
            reason = review.get('reason', ['', ''])

            if len(reason) < 2:
                reason = ['', '']

            # Concatenate both reasons (already fixed by user)
            reason_text = (reason[0].strip() or reason[1].strip())

            all_reviews.append({
                'series_id': series_id,
                'compareID1': compare_id_1,
                'compareID2': compare_id_2,
                'compareFile1': compare_file_1,
                'compareFile2': compare_file_2,
                'userChoice': user_choice,
                'reason_text': reason_text
            })

    df = pd.DataFrame(all_reviews)
    logger.info(f"Extracted {len(df)} individual reviews from {len(json_files)} files")

    return df


def classify_reasons_keyword(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify reason texts using keyword-based AttributeMapper.

    Args:
        reviews_df: DataFrame with reason_text and userChoice columns

    Returns:
        DataFrame with added 'label_keyword' column
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

    reviews_df['label_keyword'] = labels

    logger.info(f"Keyword method - Classified {len(reviews_df)} reviews")
    logger.info(f"Label distribution:\n{pd.Series(labels).value_counts()}")

    return reviews_df


def classify_reasons_embedding(reviews_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Classify reason texts using embedding-based cosine similarity.

    Args:
        reviews_df: DataFrame with reason_text column
        model_name: SentenceTransformer model to use

    Returns:
        DataFrame with added 'label_embedding' and 'label_embedding_score' columns
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Encode label descriptions
    label_names = list(LABEL_DESCRIPTIONS.keys())
    label_descs = list(LABEL_DESCRIPTIONS.values())
    label_embeddings = model.encode(label_descs, show_progress_bar=False)

    labels = []
    scores = []

    # Process in batches for efficiency
    batch_size = 256
    reason_texts = reviews_df['reason_text'].tolist()

    logger.info(f"Encoding {len(reason_texts)} reason texts...")

    for i in range(0, len(reason_texts), batch_size):
        batch = reason_texts[i:i + batch_size]

        for reason_text in batch:
            if not reason_text:
                labels.append('no_reason_given')
                scores.append(0.0)
                continue

            # Encode reason text
            reason_embedding = model.encode([reason_text], show_progress_bar=False)

            # Compute cosine similarity with all label descriptions
            similarities = cosine_similarity(reason_embedding, label_embeddings)[0]

            # Get best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            # Use 'other' if similarity is too low (threshold = 0.3)
            if best_score < 0.3:
                labels.append('other')
                scores.append(best_score)
            else:
                labels.append(label_names[best_idx])
                scores.append(best_score)

        if (i // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(i + batch_size, len(reason_texts))}/{len(reason_texts)} texts")

    reviews_df['label_embedding'] = labels
    reviews_df['label_embedding_score'] = scores

    logger.info(f"Embedding method - Classified {len(reviews_df)} reviews")
    logger.info(f"Label distribution:\n{pd.Series(labels).value_counts()}")

    return reviews_df


def aggregate_by_pair(classified_df: pd.DataFrame, label_column: str, method_name: str) -> pd.DataFrame:
    """
    Aggregate reviews per (series_id, compareID1, compareID2) pair.

    Args:
        classified_df: DataFrame with classified reviews
        label_column: Name of the label column to aggregate
        method_name: Name for this method (for logging)

    Returns:
        DataFrame with one row per unique pair
    """
    # Step 0: Get compareFile1 and compareFile2 (same for all reviews of a pair)
    df_files = classified_df.groupby(
        ['series_id', 'compareID1', 'compareID2']
    ).agg({
        'compareFile1': 'first',
        'compareFile2': 'first'
    }).reset_index()

    # Step 1: Count votes LEFT vs RIGHT per pair
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
    df_votes_pivot['MaxVote'] = df_votes_pivot['compareID2']
    left_wins = df_votes_pivot['LEFT'] > df_votes_pivot['RIGHT']
    df_votes_pivot.loc[left_wins, 'MaxVote'] = df_votes_pivot.loc[left_wins, 'compareID1']

    # Step 2: Count labels per pair
    df_labels = classified_df.groupby(
        ['series_id', 'compareID1', 'compareID2', label_column]
    ).size().reset_index(name='count')

    df_labels_pivot = pd.pivot_table(
        df_labels,
        values='count',
        index=['series_id', 'compareID1', 'compareID2'],
        columns=label_column,
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # Rename columns to label_* format
    label_cols = [col for col in df_labels_pivot.columns
                  if col not in ['series_id', 'compareID1', 'compareID2']]

    rename_dict = {col: f'label_{col}' for col in label_cols}
    df_labels_pivot.rename(columns=rename_dict, inplace=True)

    # Step 3: Determine majority label
    label_count_cols = [col for col in df_labels_pivot.columns if col.startswith('label_')]

    if label_count_cols:
        df_labels_pivot['majority_label'] = df_labels_pivot[label_count_cols].idxmax(axis=1)
        df_labels_pivot['majority_label'] = df_labels_pivot['majority_label'].str.replace('label_', '')
    else:
        df_labels_pivot['majority_label'] = 'unknown'

    # Step 4: Merge files + votes + labels
    result = pd.merge(
        df_files,
        df_votes_pivot,
        on=['series_id', 'compareID1', 'compareID2'],
        how='left'
    )

    result = pd.merge(
        result,
        df_labels_pivot,
        on=['series_id', 'compareID1', 'compareID2'],
        how='left'
    )

    result['num_reviewers'] = result['Total']

    # Fill NaN label counts with 0
    for col in result.columns:
        if col.startswith('label_'):
            result[col] = result[col].fillna(0).astype(int)

    logger.info(f"{method_name}: Aggregated into {len(result)} unique pairs")

    return result


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create labeled pairs CSV with dual methods (keyword + embedding)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python create_labeled_pairs_dual_method.py
  python create_labeled_pairs_dual_method.py --input-dir /path/to/reviews
        """
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval"),
        help='Root directory containing review JSON files'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(r"D:\Similar Images\automatic_triage_photo_series"),
        help='Output directory for CSV files'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name'
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all reviews
    logger.info(f"Loading reviews from {args.input_dir}")
    reviews_df = load_all_reviews(args.input_dir)
    logger.info(f"Loaded {len(reviews_df)} individual reviews")

    # Count non-empty reasons
    non_empty = (reviews_df['reason_text'] != '').sum()
    logger.info(f"Non-empty reasons: {non_empty} ({100*non_empty/len(reviews_df):.1f}%)")

    # ========== METHOD 1: KEYWORD-BASED ==========
    logger.info("\n" + "="*70)
    logger.info("METHOD 1: KEYWORD-BASED CLASSIFICATION")
    logger.info("="*70)

    reviews_keyword = reviews_df.copy()
    reviews_keyword = classify_reasons_keyword(reviews_keyword)

    pairs_keyword = aggregate_by_pair(reviews_keyword, 'label_keyword', 'Keyword method')

    # Reorder columns
    id_cols = ['series_id', 'compareID1', 'compareID2', 'compareFile1', 'compareFile2']
    vote_cols = ['LEFT', 'RIGHT', 'Total', 'max_count', 'Agreement', 'MaxVote', 'num_reviewers']
    label_meta_cols = ['majority_label']
    label_count_cols = sorted([col for col in pairs_keyword.columns if col.startswith('label_')])

    column_order = id_cols + vote_cols + label_meta_cols + label_count_cols
    pairs_keyword = pairs_keyword[column_order]

    # Save keyword CSV
    output_keyword = args.output_dir / 'photo_triage_pairs_keyword_labels.csv'
    pairs_keyword.to_csv(output_keyword, index=False)
    logger.info(f"Saved keyword results to: {output_keyword}")

    # ========== METHOD 2: EMBEDDING-BASED ==========
    logger.info("\n" + "="*70)
    logger.info("METHOD 2: EMBEDDING-BASED CLASSIFICATION")
    logger.info("="*70)

    reviews_embedding = reviews_df.copy()
    reviews_embedding = classify_reasons_embedding(reviews_embedding, args.embedding_model)

    pairs_embedding = aggregate_by_pair(reviews_embedding, 'label_embedding', 'Embedding method')

    # Reorder columns
    label_count_cols_emb = sorted([col for col in pairs_embedding.columns if col.startswith('label_')])
    column_order_emb = id_cols + vote_cols + label_meta_cols + label_count_cols_emb
    pairs_embedding = pairs_embedding[column_order_emb]

    # Save embedding CSV
    output_embedding = args.output_dir / 'photo_triage_pairs_embedding_labels.csv'
    pairs_embedding.to_csv(output_embedding, index=False)
    logger.info(f"Saved embedding results to: {output_embedding}")

    # ========== COMPARISON STATISTICS ==========
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    print(f"\nTotal unique pairs: {len(pairs_keyword)}")
    print(f"Total individual reviews: {len(reviews_df)}")
    print(f"Average reviews per pair: {len(reviews_df) / len(pairs_keyword):.2f}")

    print("\n" + "-"*70)
    print("METHOD 1: KEYWORD-BASED")
    print("-"*70)
    print(f"Mean agreement: {pairs_keyword['Agreement'].mean():.3f}")
    print(f"Pairs with >=70% agreement: {(pairs_keyword['Agreement'] >= 0.70).sum()} "
          f"({100 * (pairs_keyword['Agreement'] >= 0.70).sum() / len(pairs_keyword):.1f}%)")

    print("\nLabel distribution:")
    keyword_counts = pairs_keyword['majority_label'].value_counts()
    for label, count in keyword_counts.head(10).items():
        pct = 100 * count / len(pairs_keyword)
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "-"*70)
    print("METHOD 2: EMBEDDING-BASED")
    print("-"*70)
    print(f"Mean agreement: {pairs_embedding['Agreement'].mean():.3f}")
    print(f"Pairs with >=70% agreement: {(pairs_embedding['Agreement'] >= 0.70).sum()} "
          f"({100 * (pairs_embedding['Agreement'] >= 0.70).sum() / len(pairs_embedding):.1f}%)")

    print("\nLabel distribution:")
    embedding_counts = pairs_embedding['majority_label'].value_counts()
    for label, count in embedding_counts.head(10).items():
        pct = 100 * count / len(pairs_embedding)
        print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "="*70)
    print(f"Output files:")
    print(f"  Keyword:   {output_keyword}")
    print(f"  Embedding: {output_embedding}")
    print("="*70)


if __name__ == "__main__":
    main()
