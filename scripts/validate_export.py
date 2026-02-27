"""
Validate exported clustering data.

Checks CSV schema, data types, NaN values, and feature ranges.

Usage:
    python scripts/validate_export.py results/face_clustering_training/test_dataset
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def validate_faces_csv(df: pd.DataFrame) -> list[str]:
    """Validate faces.csv schema and data."""
    errors = []

    # Check required columns
    required_cols = [
        'face_id', 'image_path', 'cluster_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
        'blur_score', 'pose_yaw', 'pose_pitch', 'pose_roll', 'is_core', 'embedding_index'
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"faces.csv: Missing columns: {missing}")
        return errors

    # Check data types
    if not pd.api.types.is_integer_dtype(df['face_id']):
        errors.append("faces.csv: face_id should be integer")
    if not pd.api.types.is_integer_dtype(df['cluster_id']):
        errors.append("faces.csv: cluster_id should be integer")
    if not pd.api.types.is_bool_dtype(df['is_core']):
        errors.append("faces.csv: is_core should be boolean")

    # Check for invalid values
    if df['blur_score'].isna().any():
        errors.append(f"faces.csv: {df['blur_score'].isna().sum()} NaN blur_score values")
    if (df['blur_score'] < 0).any():
        errors.append(f"faces.csv: {(df['blur_score'] < 0).sum()} negative blur_score values")

    # Check bbox values
    for col in ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']:
        if df[col].isna().any():
            errors.append(f"faces.csv: NaN values in {col}")

    return errors


def validate_clusters_csv(df: pd.DataFrame) -> list[str]:
    """Validate clusters.csv schema and data."""
    errors = []

    # Check required columns
    required_cols = [
        'cluster_id', 'cluster_size', 'exemplar_ids', 'diameter', 'T_A', 'mean_blur', 'face_ids'
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"clusters.csv: Missing columns: {missing}")
        return errors

    # Check data types
    if not pd.api.types.is_integer_dtype(df['cluster_id']):
        errors.append("clusters.csv: cluster_id should be integer")
    if not pd.api.types.is_integer_dtype(df['cluster_size']):
        errors.append("clusters.csv: cluster_size should be integer")

    # Check for invalid values
    if (df['cluster_size'] < 2).any():
        errors.append(f"clusters.csv: {(df['cluster_size'] < 2).sum()} clusters with size < 2")

    if df['diameter'].isna().any():
        errors.append(f"clusters.csv: {df['diameter'].isna().sum()} NaN diameter values")

    if (df['diameter'] < 0).any():
        errors.append(f"clusters.csv: {(df['diameter'] < 0).sum()} negative diameter values")

    if (df['diameter'] > 2.0).any():
        errors.append(f"clusters.csv: {(df['diameter'] > 2.0).sum()} diameter > 2.0 (max cosine distance)")

    return errors


def validate_candidate_pairs_csv(df: pd.DataFrame) -> list[str]:
    """Validate candidate_pairs.csv schema and data."""
    errors = []

    # Check required columns
    required_cols = [
        'cluster_id_1', 'cluster_id_2', 'min_exemplar_dist', 'p10_cross_dist', 'p50_cross_dist',
        'support_fraction', 'diameter_ratio', 'cluster_size_min', 'cluster_size_ratio',
        'T_A', 'T_B', 'T_local', 'T_global'
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"candidate_pairs.csv: Missing columns: {missing}")
        return errors

    # Check for NaN values in any column
    nan_counts = df.isna().sum()
    if nan_counts.any():
        nan_cols = nan_counts[nan_counts > 0]
        errors.append(f"candidate_pairs.csv: NaN values found: {nan_cols.to_dict()}")

    # Check feature ranges
    for col, (min_val, max_val) in {
        'min_exemplar_dist': (0.0, 2.0),
        'p10_cross_dist': (0.0, 2.0),
        'p50_cross_dist': (0.0, 2.0),
        'support_fraction': (0.0, 1.0),
        'diameter_ratio': (1.0, None),
        'cluster_size_min': (1, None),
        'cluster_size_ratio': (1.0, None),
        'T_A': (0.0, 2.0),
        'T_B': (0.0, 2.0),
        'T_local': (0.0, 2.0),
        'T_global': (0.0, 2.0),
    }.items():
        if col not in df.columns:
            continue

        if (df[col] < min_val).any():
            errors.append(f"candidate_pairs.csv: {col} has values < {min_val}")

        if max_val is not None and (df[col] > max_val).any():
            errors.append(f"candidate_pairs.csv: {col} has values > {max_val}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description='Validate exported clustering data'
    )
    parser.add_argument(
        'export_dir',
        type=Path,
        help='Directory containing exported CSV files'
    )

    args = parser.parse_args()

    if not args.export_dir.exists():
        print(f"ERROR: Directory not found: {args.export_dir}")
        sys.exit(1)

    print(f"Validating exports in: {args.export_dir}")
    print("=" * 60)

    errors = []

    # Validate faces.csv
    faces_path = args.export_dir / 'faces.csv'
    if not faces_path.exists():
        errors.append(f"faces.csv not found")
    else:
        print(f"Validating faces.csv...")
        faces_df = pd.read_csv(faces_path)
        print(f"  Loaded {len(faces_df)} rows")
        errors.extend(validate_faces_csv(faces_df))

    # Validate clusters.csv
    clusters_path = args.export_dir / 'clusters.csv'
    if not clusters_path.exists():
        errors.append(f"clusters.csv not found")
    else:
        print(f"Validating clusters.csv...")
        clusters_df = pd.read_csv(clusters_path)
        print(f"  Loaded {len(clusters_df)} rows")
        errors.extend(validate_clusters_csv(clusters_df))

    # Validate candidate_pairs.csv
    pairs_path = args.export_dir / 'candidate_pairs.csv'
    if not pairs_path.exists():
        errors.append(f"candidate_pairs.csv not found")
    else:
        print(f"Validating candidate_pairs.csv...")
        pairs_df = pd.read_csv(pairs_path)
        print(f"  Loaded {len(pairs_df)} rows")
        errors.extend(validate_candidate_pairs_csv(pairs_df))

    print("=" * 60)

    if errors:
        print(f"\n[FAIL] Validation FAILED with {len(errors)} errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("\n[PASS] Validation PASSED - all checks successful!")

        # Print summary statistics
        if faces_path.exists() and clusters_path.exists() and pairs_path.exists():
            print("\nSummary:")
            print(f"  Faces: {len(faces_df)} total, {faces_df['is_core'].sum()} core")
            print(f"  Clusters: {len(clusters_df)} clusters")
            print(f"  Candidate pairs: {len(pairs_df)} pairs")
            print(f"  Feature ranges:")
            print(f"    min_exemplar_dist: [{pairs_df['min_exemplar_dist'].min():.3f}, {pairs_df['min_exemplar_dist'].max():.3f}]")
            print(f"    support_fraction: [{pairs_df['support_fraction'].min():.3f}, {pairs_df['support_fraction'].max():.3f}]")
            print(f"    cluster_size_min: [{pairs_df['cluster_size_min'].min()}, {pairs_df['cluster_size_min'].max()}]")


if __name__ == '__main__':
    main()
