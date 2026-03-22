"""Test crop filename validation."""

import pytest
from pathlib import Path
import tempfile
import shutil


def test_validate_crop_filenames_detects_saved_count_bug():
    """Test that validation detects when saved_count is used instead of metadata index."""
    # This test verifies the validation catches SIGHTING-006 bug

    # Import the validation function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.benchmark_face_clustering import validate_crop_filenames

    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        crops_dir = Path(tmpdir) / 'face_crops'
        crops_dir.mkdir()

        # Simulate the bug: metadata has indices [2, 3, 4, 5]
        # but crops were saved as face_0000, face_0001, face_0002, face_0003
        metadata = [
            {'bbox': {'w_px': 5, 'h_px': 5}},    # Index 0 - failed
            {'bbox': {'w_px': 5, 'h_px': 5}},    # Index 1 - failed
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 2 - saved
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 3 - saved
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 4 - saved
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 5 - saved
        ]

        saved_indices = [2, 3, 4, 5]  # Which metadata indices were saved

        # Create crop files with WRONG names (simulating the bug)
        for i in range(4):
            crop_file = crops_dir / f'face_{i:04d}_aligned.jpg'
            crop_file.write_text('fake image')

        # Validation should FAIL and detect the bug
        with pytest.raises(ValueError) as exc_info:
            validate_crop_filenames(metadata, saved_indices, crops_dir)

        # Check error message mentions the bug
        error_msg = str(exc_info.value)
        assert 'DETECTED BUG' in error_msg
        assert 'saved_count was used instead of metadata index' in error_msg


def test_validate_crop_filenames_passes_correct_alignment():
    """Test that validation passes when crops are correctly aligned."""

    # Import the validation function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.benchmark_face_clustering import validate_crop_filenames

    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        crops_dir = Path(tmpdir) / 'face_crops'
        crops_dir.mkdir()

        # Metadata with some failures
        metadata = [
            {'bbox': {'w_px': 5, 'h_px': 5}},     # Index 0 - failed
            {'bbox': {'w_px': 5, 'h_px': 5}},     # Index 1 - failed
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 2 - saved
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 3 - saved
        ]

        saved_indices = [2, 3]

        # Create crop files with CORRECT names (using metadata index)
        for meta_idx in saved_indices:
            crop_file = crops_dir / f'face_{meta_idx:04d}_aligned.jpg'
            crop_file.write_text('fake image')

        # Validation should PASS (no exception)
        validate_crop_filenames(metadata, saved_indices, crops_dir)
        # If we get here, validation passed


def test_validate_crop_filenames_detects_missing_files():
    """Test that validation detects when expected crop files are missing."""

    # Import the validation function
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.benchmark_face_clustering import validate_crop_filenames

    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        crops_dir = Path(tmpdir) / 'face_crops'
        crops_dir.mkdir()

        metadata = [
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 0
            {'bbox': {'w_px': 100, 'h_px': 100}}, # Index 1
        ]

        saved_indices = [0, 1]

        # Only create face_0000.jpg, not face_0001.jpg
        crop_file = crops_dir / 'face_0000_aligned.jpg'
        crop_file.write_text('fake image')

        # Validation should FAIL
        with pytest.raises(ValueError) as exc_info:
            validate_crop_filenames(metadata, saved_indices, crops_dir)

        error_msg = str(exc_info.value)
        assert 'face_0001_aligned.jpg' in error_msg or "doesn't exist" in error_msg


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
