#!/usr/bin/env python3
"""Verify that labeling app data is correctly formatted."""
import sys
from pathlib import Path
import pandas as pd

def main():
    data_dir = Path("results/test_face_clustering_e2e")

    print("=" * 70)
    print("LABELING APP DATA VERIFICATION")
    print("=" * 70)
    print(f"Directory: {data_dir}")
    print()

    # Check required files
    required_files = ['faces.csv', 'clusters.csv', 'export_summary.json', 'crops']
    missing = []
    for file in required_files:
        path = data_dir / file
        if not path.exists():
            missing.append(file)
            print(f"[FAIL] Missing: {file}")
        else:
            print(f"[OK] Found: {file}")

    if missing:
        print(f"\n[FAIL] Missing {len(missing)} required files")
        return 1

    print()

    # Load CSVs
    print("Loading CSV files...")
    faces_df = pd.read_csv(data_dir / 'faces.csv')
    clusters_df = pd.read_csv(data_dir / 'clusters.csv')

    print(f"  faces.csv: {len(faces_df)} rows")
    print(f"  clusters.csv: {len(clusters_df)} rows")
    print()

    # Verify faces.csv schema
    required_cols = ['face_id', 'image_path', 'cluster_id', 'is_core']
    missing_cols = [col for col in required_cols if col not in faces_df.columns]
    if missing_cols:
        print(f"[FAIL] faces.csv missing columns: {missing_cols}")
        return 1
    print(f"[OK] faces.csv has all required columns")

    # Verify no null image_paths
    null_paths = faces_df[faces_df['image_path'].isna()]
    if len(null_paths) > 0:
        print(f"[FAIL] {len(null_paths)} faces have null image_path!")
        return 1
    print(f"[OK] All faces have valid image_path")

    # Verify clusters.csv schema
    required_cols = ['cluster_id', 'size']
    missing_cols = [col for col in required_cols if col not in clusters_df.columns]
    if missing_cols:
        print(f"[FAIL] clusters.csv missing columns: {missing_cols}")
        return 1
    print(f"[OK] clusters.csv has all required columns")

    # Verify crops exist
    crops_dir = data_dir / 'crops'
    for face_id in faces_df['face_id']:
        crop_path = crops_dir / f"face_{face_id:04d}_aligned.jpg"
        if not crop_path.exists():
            print(f"[FAIL] Missing crop: {crop_path}")
            return 1

    print(f"[OK] All {len(faces_df)} face crops exist")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Faces: {len(faces_df)}")
    print(f"Clusters: {len(clusters_df)}")
    print(f"Core faces: {len(faces_df[faces_df['is_core']])}")
    print(f"Holdout faces: {len(faces_df[~faces_df['is_core']])}")
    print()

    # Cluster breakdown
    print("Clusters breakdown:")
    for _, row in clusters_df.iterrows():
        cluster_id = row['cluster_id']
        size = row['size']
        print(f"  Cluster {cluster_id}: {size} faces")

    print()
    print("=" * 70)
    print("[PASS] All checks passed - ready for labeling app")
    print("=" * 70)
    print()
    print("To open labeling app:")
    print(f"  streamlit run app/face_clustering_labeling.py")
    print(f"  Then select directory: {data_dir.absolute()}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
