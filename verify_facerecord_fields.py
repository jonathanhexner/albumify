"""Quick verification that FaceRecord has new fields."""

from face_cluster import FaceRecord
import numpy as np

# Test creating FaceRecord with new fields
try:
    face = FaceRecord(
        face_id=0,
        image_id="test.jpg",
        bbox=(0, 0, 112, 112),
        aligned_face=np.zeros((112, 112, 3), dtype=np.uint8),
        embedding=np.zeros(512),
        embedding_normalized=np.zeros(512),
        pose=(0.0, 0.0, 0.0),
        blur_score=100.0,
        area=12544.0,
        is_core=False,
        image_path="/path/to/test.jpg",
        face_index=0
    )

    print("✓ FaceRecord has new fields!")
    print(f"  image_path: {face.image_path}")
    print(f"  face_index: {face.face_index}")

except TypeError as e:
    print("✗ FaceRecord missing new fields!")
    print(f"  Error: {e}")
    print("\n  Solution: Restart kernel and re-run imports")
