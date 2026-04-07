"""Save aligned face crops and crop manifest."""
import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from face_cluster.types import FaceRecord

logger = logging.getLogger(__name__)


def save_crops(faces: List[FaceRecord], output_dir: Path) -> Dict[int, Path]:
    """Save aligned face crops and write crop_manifest.json.

    Args:
        faces: List of FaceRecord with aligned_face set
        output_dir: Directory to save crops into (created if needed)

    Returns:
        Dict mapping face_id -> absolute crop path
    """
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    manifest = {}  # face_id -> relative path string
    result = {}    # face_id -> absolute Path

    saved = 0
    skipped = 0
    for face in faces:
        if face.aligned_face is None:
            logger.debug(f"face_id {face.face_id}: no aligned_face, skipping crop save")
            skipped += 1
            continue
        filename = f"face_{face.face_id:04d}_aligned.jpg"
        crop_path = crops_dir / filename
        img = Image.fromarray(face.aligned_face)
        img.save(crop_path, "JPEG", quality=95)
        manifest[str(face.face_id)] = f"crops/{filename}"
        result[face.face_id] = crop_path
        saved += 1

    manifest_path = output_dir / "crop_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved {saved} crops to {crops_dir}, skipped {skipped} (no aligned_face)")
    return result
