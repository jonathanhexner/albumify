"""Artifact writer for per-face crop JPEGs + manifest (spec-057).

Extracted from RunExporter._write_crops. Two modes:
  * crop_source_dir provided: copy from a sibling dir (Phase 1 dual-write).
  * no source dir: render face.aligned_face → JPEG via PIL.

Returns a {face_id: relative_path} manifest so subsequent writers can
populate the faces.crop_path column.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from face_cluster.types import FaceRecord


def write_crops(
    output_dir: Path,
    faces: List[FaceRecord],
    crop_source_dir: Optional[Path],
) -> Dict[int, str]:
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[int, str] = {}

    if crop_source_dir is not None:
        src = Path(crop_source_dir)
        for face in faces:
            fname = f"face_{face.face_id:04d}_aligned.jpg"
            src_path = src / fname
            if not src_path.exists():
                continue
            dst_path = crops_dir / fname
            if dst_path.resolve() != src_path.resolve():
                shutil.copyfile(src_path, dst_path)
            manifest[face.face_id] = f"crops/{fname}"
        return manifest

    for face in faces:
        if face.aligned_face is None:
            continue
        fname = f"face_{face.face_id:04d}_aligned.jpg"
        dst_path = crops_dir / fname
        Image.fromarray(face.aligned_face).save(dst_path, "JPEG", quality=95)
        manifest[face.face_id] = f"crops/{fname}"
    return manifest
