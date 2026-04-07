"""Tests for face_cluster.crops.save_crops."""
import json
import numpy as np
import pytest
from pathlib import Path
from face_cluster.types import FaceRecord
from face_cluster.crops import save_crops


def make_face_with_crop(face_id: int = 0, has_aligned: bool = True) -> FaceRecord:
    """Create a synthetic FaceRecord with or without an aligned face."""
    aligned = np.zeros((112, 112, 3), dtype=np.uint8) if has_aligned else None
    return FaceRecord(
        face_id=face_id,
        image_id="test.jpg",
        bbox=(0, 0, 100, 100),
        area=10000.0,
        blur_score=100.0,
        aligned_face=aligned,
    )


def test_crops_saved_with_correct_filenames(tmp_path):
    """Saved crop should be named face_0000_aligned.jpg for face_id=0."""
    face = make_face_with_crop(face_id=0)
    result = save_crops([face], tmp_path)
    expected_file = tmp_path / "crops" / "face_0000_aligned.jpg"
    assert expected_file.exists(), f"Expected crop file {expected_file} not found"
    assert 0 in result
    assert result[0] == expected_file


def test_manifest_json_written(tmp_path):
    """crop_manifest.json should exist and contain the face_id entry."""
    face = make_face_with_crop(face_id=5)
    save_crops([face], tmp_path)
    manifest_path = tmp_path / "crop_manifest.json"
    assert manifest_path.exists(), "crop_manifest.json not written"
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert "5" in manifest, f"face_id 5 not found in manifest: {manifest}"
    assert manifest["5"] == "crops/face_0005_aligned.jpg"


def test_face_without_aligned_skipped(tmp_path):
    """Face with aligned_face=None should not produce a file or error."""
    face = make_face_with_crop(face_id=0, has_aligned=False)
    result = save_crops([face], tmp_path)
    # No file should be saved
    assert 0 not in result
    # crops dir may exist but should be empty of jpg files
    crop_files = list((tmp_path / "crops").glob("*.jpg"))
    assert len(crop_files) == 0, f"Unexpected crop files: {crop_files}"
    # manifest should be written but empty
    manifest_path = tmp_path / "crop_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert len(manifest) == 0
