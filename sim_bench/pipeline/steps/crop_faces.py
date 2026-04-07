"""Crop Faces step - simple bounding box cropping without alignment.

This step provides raw face crops using bounding box only,
without any rotation or affine alignment. Useful for:
- Debug comparison (raw vs aligned)
- Downstream tasks that don't need alignment
- Visualizations showing what detector found

Single responsibility: ONLY cropping, no alignment.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


def crop_face_bbox(
    image: np.ndarray,
    bbox: Dict[str, float],
    margin: float = 0.2,
    target_size: Optional[int] = None
) -> Optional[np.ndarray]:
    """Crop face from image using bounding box.

    Args:
        image: Full image (H, W, C)
        bbox: Bounding box dict with x_px, y_px, w_px, h_px
        margin: Margin to add around bbox (as fraction of size)
        target_size: If set, resize crop to this size

    Returns:
        Cropped face image or None if invalid
    """
    h, w = image.shape[:2]

    x = int(bbox.get('x_px', 0))
    y = int(bbox.get('y_px', 0))
    bw = int(bbox.get('w_px', 0))
    bh = int(bbox.get('h_px', 0))

    if bw <= 0 or bh <= 0:
        return None

    # Add margin
    margin_w = int(bw * margin)
    margin_h = int(bh * margin)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w, x + bw + margin_w)
    y2 = min(h, y + bh + margin_h)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if target_size is not None:
        crop = cv2.resize(crop, (target_size, target_size))

    return crop


@register_step
class CropFacesStep(BaseStep):
    """Crop faces using bounding box without alignment.

    Creates simple bbox crops for each face. These are stored
    separately from aligned_faces for comparison and debugging.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="crop_faces",
            display_name="Crop Faces (Raw)",
            description="Crop faces using bounding box only (no alignment).",
            category="people",
            requires={"insightface_faces"},
            produces={"raw_face_crops"},
            depends_on=["insightface_detect_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "margin": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Margin around bbox as fraction"
                    },
                    "target_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Output size (None for original)"
                    },
                    "skip_filtered": {
                        "type": "boolean",
                        "default": False,
                        "description": "Skip faces with filter_passed=False"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Crop all faces in context using bbox."""
        if not hasattr(context, 'insightface_faces') or not context.insightface_faces:
            logger.info("No faces found in context - skipping cropping")
            return

        margin = config.get("margin", 0.2)
        target_size = config.get("target_size", 256)
        skip_filtered = config.get("skip_filtered", False)
        cache = get_image_cache()

        raw_crops = {}
        stats = {
            "total": 0,
            "cropped": 0,
            "skipped_filtered": 0,
            "failed": 0
        }

        total_images = len(context.insightface_faces)
        for img_idx, (image_path, face_data) in enumerate(context.insightface_faces.items()):
            if not Path(image_path).exists():
                continue

            img_np = cache.get(image_path)
            if img_np is None:
                continue

            for face_info in face_data.get('faces', []):
                stats["total"] += 1
                face_idx = face_info.get('face_index', 0)

                # Check if filtered
                if skip_filtered:
                    if not face_info.get('filter_passed', True):
                        stats["skipped_filtered"] += 1
                        continue

                bbox = face_info.get('bbox', {})
                crop = crop_face_bbox(img_np, bbox, margin, target_size)

                if crop is None:
                    stats["failed"] += 1
                    continue

                key = f"{image_path}:face_{face_idx}"
                raw_crops[key] = crop
                stats["cropped"] += 1

                # Also store in face_info
                face_info['raw_crop'] = crop

            # Progress
            progress = (img_idx + 1) / total_images
            context.report_progress(
                "crop_faces", progress,
                f"Cropped {stats['cropped']} faces from {img_idx + 1}/{total_images} images"
            )

        # Store in context
        context.raw_face_crops = raw_crops

        # Log summary
        logger.info("=" * 60)
        logger.info("CROP_FACES: Summary")
        logger.info("=" * 60)
        logger.info(f"Total faces:    {stats['total']}")
        logger.info(f"Cropped:        {stats['cropped']}")
        logger.info(f"Skipped:        {stats['skipped_filtered']}")
        logger.info(f"Failed:         {stats['failed']}")
        logger.info("=" * 60)
