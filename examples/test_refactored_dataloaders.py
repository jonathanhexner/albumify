"""
Test script to verify the refactored dataloader architecture.

This script demonstrates that:
1. Datasets provide metadata through get_dataframe() and get_image_dir()
2. get_dataset_from_loader() extracts dataset from loader
3. ExternalDatasetAdapter creates default metadata when needed
"""
import sys
from pathlib import Path

# Add sim_bench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from sim_bench.datasets.siamese_dataloaders import (
    EndToEndPairDataset,
    ExternalDatasetAdapter,
    get_dataset_from_loader
)


def test_endtoend_dataset():
    """Test EndToEndPairDataset provides metadata."""
    print("\n" + "="*60)
    print("Test 1: EndToEndPairDataset Metadata")
    print("="*60)

    # Create sample DataFrame
    pairs_df = pd.DataFrame({
        'image1': ['img1.jpg', 'img2.jpg'],
        'image2': ['img3.jpg', 'img4.jpg'],
        'winner': [0, 1],
        'series_id': ['series_1', 'series_1'],
        'agreement': [0.9, 0.8],
        'num_reviewers': [5, 4]
    })

    # Simple transform
    def dummy_transform(img):
        return torch.randn(3, 224, 224)

    # Create dataset
    dataset = EndToEndPairDataset(
        pairs_df=pairs_df,
        image_dir='/fake/path',
        transform=dummy_transform
    )

    # Test metadata methods
    df = dataset.get_dataframe()
    image_dir = dataset.get_image_dir()

    print(f"[OK] Dataset has {len(df)} pairs")
    print(f"[OK] DataFrame columns: {list(df.columns)}")
    print(f"[OK] Image directory: {image_dir}")
    print(f"[OK] Transform: {dataset.transform}")

    assert 'series_id' in df.columns, "Missing series_id"
    assert 'agreement' in df.columns, "Missing agreement"
    assert 'num_reviewers' in df.columns, "Missing num_reviewers"

    print("[OK] All metadata available!")


def test_external_adapter():
    """Test ExternalDatasetAdapter creates default metadata."""
    print("\n" + "="*60)
    print("Test 2: ExternalDatasetAdapter Default Metadata")
    print("="*60)

    # Mock external dataset (like MyDataset from Series-Photo-Selection)
    class MockExternalDataset:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            # Returns (imageA, imageB, winner) tuple
            img1 = torch.randn(3, 224, 224)
            img2 = torch.randn(3, 224, 224)
            winner = idx % 2
            return img1, img2, winner

    # Create adapter WITHOUT providing DataFrame
    external_dataset = MockExternalDataset()
    adapter = ExternalDatasetAdapter(external_dataset)

    # Test metadata methods
    df = adapter.get_dataframe()
    image_dir = adapter.get_image_dir()

    print(f"[OK] Adapter has {len(df)} pairs")
    print(f"[OK] DataFrame columns: {list(df.columns)}")
    print(f"[OK] Image directory: {image_dir}")

    # Check default values
    assert all(df['series_id'] == 'unknown'), "series_id should be 'unknown'"
    assert all(df['agreement'] == 1.0), "agreement should be 1.0"
    assert all(df['num_reviewers'] == 1), "num_reviewers should be 1"

    print(f"[OK] Default metadata created:")
    print(f"  - series_id: {df['series_id'].iloc[0]}")
    print(f"  - agreement: {df['agreement'].iloc[0]}")
    print(f"  - num_reviewers: {df['num_reviewers'].iloc[0]}")


def test_get_dataset_from_loader():
    """Test extracting dataset from loader."""
    print("\n" + "="*60)
    print("Test 3: get_dataset_from_loader()")
    print("="*60)

    # Create sample dataset
    pairs_df = pd.DataFrame({
        'image1': ['img1.jpg'],
        'image2': ['img2.jpg'],
        'winner': [0],
        'series_id': ['test_series'],
        'agreement': [0.95],
        'num_reviewers': [3]
    })

    def dummy_transform(img):
        return torch.randn(3, 224, 224)

    dataset = EndToEndPairDataset(
        pairs_df=pairs_df,
        image_dir='/test/path',
        transform=dummy_transform
    )

    # Create loader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Extract dataset
    extracted_dataset = get_dataset_from_loader(loader)

    # Verify we can access metadata
    df = extracted_dataset.get_dataframe()
    image_dir = extracted_dataset.get_image_dir()
    transform = extracted_dataset.transform

    print(f"[OK] Extracted dataset from loader")
    print(f"[OK] DataFrame: {len(df)} pairs")
    print(f"[OK] Image dir: {image_dir}")
    print(f"[OK] Transform: {transform}")
    print(f"[OK] Series ID: {df['series_id'].iloc[0]}")

    assert df['series_id'].iloc[0] == 'test_series'
    assert df['agreement'].iloc[0] == 0.95
    print("[OK] Metadata accessible from extracted dataset!")


def test_batch_format():
    """Test that adapter produces correct batch format."""
    print("\n" + "="*60)
    print("Test 4: Batch Format Compatibility")
    print("="*60)

    # Mock external dataset
    class MockExternalDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            img1 = torch.randn(3, 224, 224)
            img2 = torch.randn(3, 224, 224)
            winner = idx % 2
            return img1, img2, winner

    # Create adapter and loader
    external_dataset = MockExternalDataset()
    adapter = ExternalDatasetAdapter(external_dataset)
    loader = DataLoader(adapter, batch_size=2, shuffle=False)

    # Get a batch
    batch = next(iter(loader))

    # Verify batch format
    assert 'img1' in batch, "Missing img1"
    assert 'img2' in batch, "Missing img2"
    assert 'winner' in batch, "Missing winner"
    assert 'image1' in batch, "Missing image1"
    assert 'image2' in batch, "Missing image2"

    print(f"[OK] Batch keys: {list(batch.keys())}")
    print(f"[OK] img1 shape: {batch['img1'].shape}")
    print(f"[OK] img2 shape: {batch['img2'].shape}")
    print(f"[OK] winner shape: {batch['winner'].shape}")
    print(f"[OK] image1: {batch['image1']}")
    print(f"[OK] image2: {batch['image2']}")
    print("[OK] Batch format is correct!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("REFACTORED DATALOADER TESTS")
    print("="*60)

    try:
        test_endtoend_dataset()
        test_external_adapter()
        test_get_dataset_from_loader()
        test_batch_format()

        print("\n" + "="*60)
        print("ALL TESTS PASSED! [OK]")
        print("="*60)
        print("""
Summary:
1. [OK] Datasets provide metadata via get_dataframe() and get_image_dir()
2. [OK] ExternalDatasetAdapter creates default metadata when needed
3. [OK] get_dataset_from_loader() extracts dataset from loader
4. [OK] Batch format is compatible across all dataset types

The refactor is successful! You can now:
- Pass only loaders to training functions (not loaders + DataFrames)
- Use external datasets with the ExternalDatasetAdapter
- Extract metadata when needed using get_dataset_from_loader()
        """)

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
