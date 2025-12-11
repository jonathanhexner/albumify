"""
Utility functions for working with PhotoTriage dataset.

Handles filename normalization, label loading, and data merging.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List


def normalize_phototriage_filename(filename: str) -> str:
    """
    Normalize PhotoTriage filename from short format to long format.

    Converts from format like "1-1.JPG" to "000001-01.JPG"

    Args:
        filename: Original filename (e.g., "1-1.JPG", "123-4.JPG")

    Returns:
        Normalized filename (e.g., "000001-01.JPG", "000123-04.JPG")

    Examples:
        >>> normalize_phototriage_filename("1-1.JPG")
        "000001-01.JPG"
        >>> normalize_phototriage_filename("123-4.JPG")
        "000123-04.JPG"
        >>> normalize_phototriage_filename("1234-56.JPG")
        "001234-56.JPG"
    """
    if not filename or pd.isna(filename):
        return filename

    # Split on the dash
    parts = filename.rsplit('.', 1)  # Split extension
    if len(parts) != 2:
        return filename

    name_part, ext = parts

    # Split the name part on dash
    if '-' not in name_part:
        return filename

    series_num, image_num = name_part.split('-', 1)

    # Pad to 6 digits and 2 digits respectively
    series_padded = series_num.zfill(6)
    image_padded = image_num.zfill(2)

    return f"{series_padded}-{image_padded}.{ext}"


def denormalize_phototriage_filename(filename: str) -> str:
    """
    Convert normalized filename back to short format.

    Converts from "000001-01.JPG" to "1-1.JPG"

    Args:
        filename: Normalized filename (e.g., "000001-01.JPG")

    Returns:
        Short format filename (e.g., "1-1.JPG")
    """
    if not filename or pd.isna(filename):
        return filename

    # Split on the dash
    parts = filename.rsplit('.', 1)  # Split extension
    if len(parts) != 2:
        return filename

    name_part, ext = parts

    # Split the name part on dash
    if '-' not in name_part:
        return filename

    series_num, image_num = name_part.split('-', 1)

    # Remove leading zeros
    series_short = str(int(series_num))
    image_short = str(int(image_num))

    return f"{series_short}-{image_short}.{ext}"


def load_phototriage_keyword_labels(
    csv_path: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval\photo_triage_pairs_keyword_labels.csv"
) -> pd.DataFrame:
    """
    Load PhotoTriage keyword labels CSV with normalized filenames.

    This adds normalized filename columns that match the format used in
    other parts of the codebase (000001-01.JPG instead of 1-1.JPG).

    Args:
        csv_path: Path to the keyword labels CSV file

    Returns:
        DataFrame with added columns:
        - compareFile1_normalized: normalized version of compareFile1
        - compareFile2_normalized: normalized version of compareFile2
    """
    df = pd.read_csv(csv_path)

    # Add normalized filename columns
    df['compareFile1_normalized'] = df['compareFile1'].apply(normalize_phototriage_filename)
    df['compareFile2_normalized'] = df['compareFile2'].apply(normalize_phototriage_filename)

    return df


def merge_with_keyword_labels(
    predictions_df: pd.DataFrame,
    keyword_labels_path: str = r"D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval\photo_triage_pairs_keyword_labels.csv",
    img1_col: str = 'img1',
    img2_col: str = 'img2',
    series_col: str = 'series_id',
    how: str = 'left',
    label_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Merge predictions with PhotoTriage keyword labels.

    Handles the filename format mismatch between the two datasets:
    - predictions_df uses: "000001-01.JPG"
    - keyword_labels uses: "1-1.JPG"

    Args:
        predictions_df: DataFrame with predictions, must have columns for img1, img2, series_id
        keyword_labels_path: Path to keyword labels CSV
        img1_col: Column name for first image in predictions_df
        img2_col: Column name for second image in predictions_df
        series_col: Column name for series ID in predictions_df
        how: Type of merge ('left', 'inner', 'outer', 'right')
        label_columns: Specific label columns to include. If None, includes all label_* columns

    Returns:
        Merged DataFrame with keyword label information

    Example:
        >>> df_predictions = pd.read_csv("outputs/train_labels_epoch1.csv")
        >>> df_merged = merge_with_keyword_labels(df_predictions)
        >>> print(df_merged.columns)
        # Will include: img1, img2, series_id, true_winner, pred_winner,
        #               majority_label, label_sharpness, label_composition, etc.
    """
    # Load keyword labels with normalized filenames
    df_labels = load_phototriage_keyword_labels(keyword_labels_path)

    # Create merge keys
    # For predictions: series_id + img1 + img2
    predictions_df['_merge_key'] = (
        predictions_df[series_col].astype(str) + '|' +
        predictions_df[img1_col] + '|' +
        predictions_df[img2_col]
    )

    # For labels: series_id + normalized filenames
    df_labels['_merge_key'] = (
        df_labels['series_id'].astype(str) + '|' +
        df_labels['compareFile1_normalized'] + '|' +
        df_labels['compareFile2_normalized']
    )

    # Select columns to merge
    if label_columns is None:
        # Get all label columns
        label_columns = [col for col in df_labels.columns if col.startswith('label_')]

    # Add metadata columns
    merge_columns = ['_merge_key', 'series_id', 'compareFile1', 'compareFile2'] + \
                   ['LEFT', 'RIGHT', 'Total', 'Agreement', 'MaxVote', 'num_reviewers', 'majority_label'] + \
                   label_columns

    # Only keep columns that exist in df_labels
    merge_columns = [col for col in merge_columns if col in df_labels.columns]

    df_labels_subset = df_labels[merge_columns].copy()

    # Merge
    df_merged = predictions_df.merge(
        df_labels_subset,
        on='_merge_key',
        how=how,
        suffixes=('', '_keyword')
    )

    # Clean up merge key
    df_merged = df_merged.drop(columns=['_merge_key'])

    # Drop duplicate series_id column if it exists
    if 'series_id_keyword' in df_merged.columns:
        df_merged = df_merged.drop(columns=['series_id_keyword'])

    return df_merged


def get_pairs_by_attribute(
    df: pd.DataFrame,
    attribute: str,
    attribute_value: Optional[str] = None,
    min_count: int = 1
) -> pd.DataFrame:
    """
    Filter pairs by a specific quality attribute.

    Args:
        df: DataFrame with keyword labels (output of merge_with_keyword_labels)
        attribute: Attribute column name (e.g., 'label_sharpness', 'majority_label')
        attribute_value: Optional specific value to filter for
        min_count: Minimum number of reviewers who selected this attribute

    Returns:
        Filtered DataFrame

    Example:
        >>> df_merged = merge_with_keyword_labels(df_predictions)
        >>>
        >>> # Get all pairs where sharpness was labeled
        >>> df_sharpness = get_pairs_by_attribute(df_merged, 'label_sharpness', min_count=1)
        >>>
        >>> # Get pairs where majority vote was "composition"
        >>> df_composition = get_pairs_by_attribute(df_merged, 'majority_label', 'composition')
    """
    if attribute not in df.columns:
        raise ValueError(f"Attribute '{attribute}' not found in dataframe")

    # Filter by attribute value if specified
    if attribute_value is not None:
        df_filtered = df[df[attribute] == attribute_value].copy()
    else:
        # For numeric columns (label counts), filter by min_count
        if df[attribute].dtype in ['int64', 'float64']:
            df_filtered = df[df[attribute] >= min_count].copy()
        else:
            # For categorical, just remove NaN/empty
            df_filtered = df[df[attribute].notna() & (df[attribute] != '')].copy()

    return df_filtered


def analyze_attribute_accuracy(
    df: pd.DataFrame,
    true_col: str = 'true_winner',
    pred_col: str = 'pred_winner',
    min_samples: int = 5
) -> pd.DataFrame:
    """
    Analyze prediction accuracy by quality attributes.

    Args:
        df: Merged DataFrame with keyword labels
        true_col: Column name for ground truth
        pred_col: Column name for predictions
        min_samples: Minimum samples to include an attribute category

    Returns:
        DataFrame with accuracy statistics per attribute

    Example:
        >>> df_merged = merge_with_keyword_labels(df_predictions)
        >>> accuracy_stats = analyze_attribute_accuracy(df_merged)
        >>> print(accuracy_stats.sort_values('accuracy', ascending=False))
    """
    results = []

    # Analyze majority_label if available
    if 'majority_label' in df.columns:
        for label in df['majority_label'].dropna().unique():
            df_subset = df[df['majority_label'] == label]
            if len(df_subset) >= min_samples:
                accuracy = (df_subset[true_col] == df_subset[pred_col]).mean()
                results.append({
                    'attribute': 'majority_label',
                    'value': label,
                    'count': len(df_subset),
                    'accuracy': accuracy,
                    'correct': (df_subset[true_col] == df_subset[pred_col]).sum()
                })

    # Analyze individual label columns
    label_cols = [col for col in df.columns if col.startswith('label_') and
                  col not in ['label_no_specific_attribute']]

    for col in label_cols:
        if df[col].dtype in ['int64', 'float64']:
            # Binary or count column
            df_with_label = df[df[col] > 0]
            if len(df_with_label) >= min_samples:
                accuracy = (df_with_label[true_col] == df_with_label[pred_col]).mean()
                results.append({
                    'attribute': col,
                    'value': 'present',
                    'count': len(df_with_label),
                    'accuracy': accuracy,
                    'correct': (df_with_label[true_col] == df_with_label[pred_col]).sum()
                })

    return pd.DataFrame(results).sort_values('accuracy', ascending=False)


# Example usage and tests
if __name__ == "__main__":
    # Test filename normalization
    print("Testing filename normalization:")
    test_files = ["1-1.JPG", "123-4.JPG", "1234-56.JPG"]
    for f in test_files:
        normalized = normalize_phototriage_filename(f)
        denormalized = denormalize_phototriage_filename(normalized)
        print(f"  {f} → {normalized} → {denormalized}")

    # Test loading keyword labels
    print("\nLoading keyword labels...")
    df_labels = load_phototriage_keyword_labels()
    print(f"Loaded {len(df_labels)} pairs")
    print(f"Columns: {df_labels.columns.tolist()}")
    print(f"\nFirst row:")
    print(df_labels.iloc[0][['series_id', 'compareFile1', 'compareFile1_normalized',
                               'compareFile2', 'compareFile2_normalized', 'majority_label']])
