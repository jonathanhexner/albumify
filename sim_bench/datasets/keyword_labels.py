"""
Utility for loading and merging keyword labels from photo_triage_pairs_keyword_labels.csv
"""
import re
import pandas as pd
from pathlib import Path
from typing import Optional


def convert_image_name(name: str) -> str:
    """
    Convert image name from CSV format to standard format.
    
    Examples:
        '1-1.JPG' -> '000001-01.JPG'
        '123-5.JPG' -> '000123-05.JPG'
    """
    match = re.match(r'(\d+)-(\d+)\.(\w+)', name)
    if match:
        series, img, ext = match.groups()
        return f'{int(series):06d}-{int(img):02d}.{ext}'
    return name


def load_keyword_labels(
    csv_path: str = r'D:\Similar Images\automatic_triage_photo_series\train_val\reviews_trainval\reviews_trainval\photo_triage_pairs_keyword_labels.csv'
) -> pd.DataFrame:
    """
    Load keyword labels CSV and convert image names to standard format.
    
    Returns:
        DataFrame with converted image1/image2 columns added.
    """
    df = pd.read_csv(csv_path)
    
    # Convert image names to standard format
    df['image1'] = df['compareFile1'].apply(convert_image_name)
    df['image2'] = df['compareFile2'].apply(convert_image_name)
    
    return df


def merge_keyword_labels(
    pairs_df: pd.DataFrame,
    keyword_labels_path: Optional[str] = None,
    image1_col: str = 'image1',
    image2_col: str = 'image2'
) -> pd.DataFrame:
    """
    Merge keyword labels into a pairs dataframe.
    
    Args:
        pairs_df: DataFrame with image pairs (must have image1, image2 columns)
        keyword_labels_path: Path to keyword labels CSV (uses default if None)
        image1_col: Column name for first image in pairs_df
        image2_col: Column name for second image in pairs_df
    
    Returns:
        Merged DataFrame with keyword label columns added.
    """
    # Load keyword labels
    if keyword_labels_path:
        labels_df = load_keyword_labels(keyword_labels_path)
    else:
        labels_df = load_keyword_labels()
    
    # Select only the label columns we want to merge
    label_columns = [col for col in labels_df.columns if col.startswith('label_')]
    merge_columns = ['image1', 'image2', 'majority_label'] + label_columns
    
    # Drop any existing label columns from pairs_df to avoid duplicates
    existing_label_cols = [c for c in pairs_df.columns if c.startswith('label_') or c == 'majority_label']
    if existing_label_cols:
        pairs_df = pairs_df.drop(columns=existing_label_cols)
    
    # Merge on image1, image2
    merged = pairs_df.merge(
        labels_df[merge_columns],
        left_on=[image1_col, image2_col],
        right_on=['image1', 'image2'],
        how='left'
    )
    
    # Clean up duplicate columns if merge created them
    for col in ['image1_y', 'image2_y']:
        if col in merged.columns:
            merged.drop(columns=[col], inplace=True)
    # Rename _x columns back
    merged.columns = [c.replace('_x', '') for c in merged.columns]
    
    return merged


# Label column names for reference
LABEL_COLUMNS = [
    'label_composition',
    'label_detail_visibility', 
    'label_distance_appropriateness',
    'label_dynamic_range',
    'label_exposure_quality',
    'label_field_of_view',
    'label_lighting_quality',
    'label_motion_blur',
    'label_no_specific_attribute',
    'label_sharpness',
    'label_subject_interest'
]


if __name__ == '__main__':
    # Demo usage
    from sim_bench.datasets.phototriage_data import PhotoTriageData
    
    # Load pairs data
    data_loader = PhotoTriageData(
        root_dir=r'D:\Similar Images\automatic_triage_photo_series',
        min_agreement=0.7,
        min_reviewers=2
    )
    train_df, val_df, test_df = data_loader.get_series_based_splits(0.8, 0.1, 0.1, 42)
    
    print(f'Train pairs: {len(train_df)}')
    print(f'Columns: {train_df.columns.tolist()}')
    
    # Merge keyword labels
    train_with_labels = merge_keyword_labels(train_df)
    
    print(f'\nAfter merge:')
    print(f'Columns: {train_with_labels.columns.tolist()}')
    print(f'\nLabel coverage: {train_with_labels["majority_label"].notna().sum()}/{len(train_with_labels)}')
    print(f'\nSample:')
    print(train_with_labels[['image1', 'image2', 'winner', 'majority_label']].head(10))

