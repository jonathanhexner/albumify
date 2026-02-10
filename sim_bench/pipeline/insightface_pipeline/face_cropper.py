"""Face cropping utility for InsightFace pipeline."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


class InsightFaceCropper:
    """Crop faces from images using InsightFace bounding boxes."""

    def __init__(self, margin: float = 0.3, target_size: int = 256):
        """
        Initialize the face cropper.

        Args:
            margin: Margin to add around face bbox (0.3 = 30% on each side)
            target_size: Resize crop to this size (target_size x target_size)
        """
        self.margin = margin
        self.target_size = target_size

    def crop_face(
        self,
        image_path: Path,
        bbox: Dict[str, Any],
        image: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Crop face with margin and resize to target size.

        Args:
            image_path: Path to the image file
            bbox: Bounding box dict with keys: x_px, y_px, w_px, h_px
            image: Optional pre-loaded image (RGB numpy array)

        Returns:
            RGB numpy array of shape (target_size, target_size, 3), or None on failure
        """
        try:
            # Load image if not provided (EXIF-normalized via global cache)
            if image is None:
                cache = get_image_cache()
                img = cache.get_pil(image_path)
            else:
                img = Image.fromarray(image)

            img_width, img_height = img.size

            # Extract bbox coordinates
            x_px = bbox.get('x_px', 0)
            y_px = bbox.get('y_px', 0)
            w_px = bbox.get('w_px', 0)
            h_px = bbox.get('h_px', 0)

            if w_px <= 0 or h_px <= 0:
                logger.warning(f"Invalid bbox dimensions: {w_px}x{h_px}")
                return None

            # Calculate margin in pixels
            margin_w = int(w_px * self.margin)
            margin_h = int(h_px * self.margin)

            # Apply margin and clamp to image bounds
            x1 = max(0, x_px - margin_w)
            y1 = max(0, y_px - margin_h)
            x2 = min(img_width, x_px + w_px + margin_w)
            y2 = min(img_height, y_px + h_px + margin_h)

            # Validate crop coordinates
            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    f"Invalid crop coordinates for {image_path}: "
                    f"x1={x1}, y1={y1}, x2={x2}, y2={y2} (image: {img_width}x{img_height})"
                )
                return None

            # Crop the face region
            crop = img.crop((x1, y1, x2, y2))

            # Resize to target size
            crop_resized = crop.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS
            )

            # Convert to numpy array
            return np.array(crop_resized)

        except Exception as e:
            logger.error(f"Failed to crop face from {image_path}: {e}")
            return None

    def crop_face_from_detection(
        self,
        image_path: Path,
        face_data: Dict[str, Any],
        image: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Crop face from InsightFace detection data.

        Args:
            image_path: Path to the image file
            face_data: Face detection dict with 'bbox' key
            image: Optional pre-loaded image

        Returns:
            RGB numpy array of shape (target_size, target_size, 3), or None on failure
        """
        bbox = face_data.get('bbox', {})
        return self.crop_face(image_path, bbox, image)
