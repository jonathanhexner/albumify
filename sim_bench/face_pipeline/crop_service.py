"""Face cropping service using MediaPipe detection."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

from sim_bench.face_pipeline.types import CroppedFace, BoundingBox, MIN_FACE_RATIO
from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


class FaceCropService:
    """
    Service for detecting and cropping faces from images.

    Uses MediaPipe Face Detection for face localization.
    Filters faces by minimum size (face_ratio >= MIN_FACE_RATIO).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration.

        Args:
            config: Configuration dict with 'face_pipeline' section
        """
        self._config = config
        fp_config = config.get('face_pipeline', {})

        self._detection_confidence = fp_config.get('detection_confidence', 0.3)
        self._min_face_ratio = fp_config.get('min_face_ratio', MIN_FACE_RATIO)
        self._crop_padding = fp_config.get('crop_padding', 0.2)  # 20% padding around face

        self._face_detection = None
        logger.info(f"FaceCropService initialized: min_ratio={self._min_face_ratio}, padding={self._crop_padding}")

    def _load_model(self):
        """Lazy load MediaPipe face detection."""
        if self._face_detection is not None:
            return

        import mediapipe as mp
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # Full range model (better for varying distances)
            min_detection_confidence=self._detection_confidence
        )
        logger.info("MediaPipe face detection loaded")

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image with EXIF orientation correction (via global cache)."""
        cache = get_image_cache()
        return cache.get(image_path)

    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in image.

        Returns:
            List of face dicts with bbox, confidence, face_ratio
        """
        self._load_model()

        results = self._face_detection.process(image)

        faces = []
        if results.detections:
            img_height, img_width = image.shape[:2]

            for detection in results.detections:
                bbox_rel = detection.location_data.relative_bounding_box

                # Create bounding box
                bbox = BoundingBox(
                    x=max(0, bbox_rel.xmin),
                    y=max(0, bbox_rel.ymin),
                    w=min(bbox_rel.width, 1 - max(0, bbox_rel.xmin)),
                    h=min(bbox_rel.height, 1 - max(0, bbox_rel.ymin))
                ).to_absolute(img_width, img_height)

                face_ratio = bbox.area_ratio
                confidence = detection.score[0] if detection.score else 0.0

                faces.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'face_ratio': face_ratio
                })

        return faces

    def _crop_face(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        padding: float
    ) -> Image.Image:
        """
        Crop face from image with padding.

        Args:
            image: Source image as numpy array
            bbox: Face bounding box (with absolute coordinates)
            padding: Padding ratio (0.2 = 20% extra on each side)

        Returns:
            Cropped PIL Image
        """
        img_height, img_width = image.shape[:2]

        # Calculate padded crop region
        pad_w = int(bbox.w_px * padding)
        pad_h = int(bbox.h_px * padding)

        x1 = max(0, bbox.x_px - pad_w)
        y1 = max(0, bbox.y_px - pad_h)
        x2 = min(img_width, bbox.x_px + bbox.w_px + pad_w)
        y2 = min(img_height, bbox.y_px + bbox.h_px + pad_h)

        # Crop
        crop = image[y1:y2, x1:x2]

        # Convert to PIL
        return Image.fromarray(crop)

    def crop_faces(
        self,
        image_path: Path,
        min_face_ratio: Optional[float] = None
    ) -> List[CroppedFace]:
        """
        Detect and crop all faces from an image.

        Args:
            image_path: Path to image
            min_face_ratio: Minimum face area ratio (overrides config)

        Returns:
            List of CroppedFace objects
        """
        image_path = Path(image_path)
        min_ratio = min_face_ratio or self._min_face_ratio

        # Load image
        image = self._load_image(image_path)

        # Detect faces
        faces = self._detect_faces(image)

        # Filter by size and crop
        cropped_faces = []
        for idx, face in enumerate(faces):
            if face['face_ratio'] < min_ratio:
                continue

            crop = self._crop_face(image, face['bbox'], self._crop_padding)

            cropped_faces.append(CroppedFace(
                original_path=image_path,
                face_index=idx,
                image=crop,
                bbox=face['bbox'],
                detection_confidence=face['confidence'],
                face_ratio=face['face_ratio']
            ))

        if cropped_faces:
            logger.debug(f"Cropped {len(cropped_faces)} faces from {image_path.name}")

        return cropped_faces

    def crop_faces_batch(
        self,
        image_paths: List[Path],
        min_face_ratio: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> List[CroppedFace]:
        """
        Crop faces from multiple images.

        Args:
            image_paths: List of image paths
            min_face_ratio: Minimum face area ratio
            progress_callback: Optional callback(current, total)

        Returns:
            List of all cropped faces
        """
        all_faces = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths):
            faces = self.crop_faces(path, min_face_ratio)
            all_faces.extend(faces)

            if progress_callback:
                progress_callback(idx + 1, total)

        logger.info(f"Cropped {len(all_faces)} faces from {total} images")
        return all_faces

    def get_face_count(self, image_path: Path) -> int:
        """Get number of detectable faces in image (any size)."""
        image = self._load_image(image_path)
        faces = self._detect_faces(image)
        return len(faces)

    def get_dominant_face(self, image_path: Path) -> Optional[CroppedFace]:
        """Get the largest face in image, if any meet threshold."""
        faces = self.crop_faces(image_path)
        if not faces:
            return None
        return max(faces, key=lambda f: f.face_ratio)
