"""Align Faces step - applies rotation correction and 5-point affine alignment.

This step:
1. Pre-rotates the image by orientation_angle (from detect_face_orientation)
2. Transforms landmarks to the rotated coordinate system
3. Applies 5-point affine alignment to ArcFace reference template
4. Stores aligned face crops in context for embedding extraction

Single responsibility: ONLY alignment, no embedding extraction.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.utils.image_cache import get_image_cache
from sim_bench.pipeline.utils.face_alignment import ARCFACE_REF_POINTS_112

logger = logging.getLogger(__name__)


def rotate_image_and_landmarks(
    image: np.ndarray,
    landmarks: List[List[float]],
    angle: int
) -> Tuple[np.ndarray, List[List[float]]]:
    """Rotate image and transform landmarks by angle (0, 90, 180, 270).

    After rotation, landmarks are reordered to maintain correct left/right semantics:
    - landmarks[0] = left eye (left side of image)
    - landmarks[1] = right eye (right side of image)
    - landmarks[2] = nose
    - landmarks[3] = left mouth corner
    - landmarks[4] = right mouth corner

    Args:
        image: Input image as numpy array (H, W, C)
        landmarks: 5-point landmarks [[x, y], ...] in order [L_eye, R_eye, nose, L_mouth, R_mouth]
        angle: Rotation angle in degrees (0, 90, 180, 270)

    Returns:
        Tuple of (rotated_image, transformed_landmarks) with landmarks reordered
    """
    if angle == 0:
        return image, landmarks

    h, w = image.shape[:2]
    landmarks_np = np.array(landmarks, dtype=np.float32)

    if angle == 90:
        # Rotate 90° clockwise
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # Transform coords: new_x = h - old_y, new_y = old_x
        new_landmarks = np.zeros_like(landmarks_np)
        new_landmarks[:, 0] = h - landmarks_np[:, 1]
        new_landmarks[:, 1] = landmarks_np[:, 0]
        # NO swap needed - landmark labels refer to PERSON's left/right, not image position

    elif angle == 180:
        # Rotate 180°
        rotated = cv2.rotate(image, cv2.ROTATE_180)
        # Transform coords: new_x = w - old_x, new_y = h - old_y
        new_landmarks = np.zeros_like(landmarks_np)
        new_landmarks[:, 0] = w - landmarks_np[:, 0]
        new_landmarks[:, 1] = h - landmarks_np[:, 1]
        # NO swap needed - landmark labels refer to PERSON's left/right, not image position

    elif angle == 270:
        # Rotate 90° counter-clockwise (270° clockwise)
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Transform coords: new_x = old_y, new_y = w - old_x
        new_landmarks = np.zeros_like(landmarks_np)
        new_landmarks[:, 0] = landmarks_np[:, 1]
        new_landmarks[:, 1] = w - landmarks_np[:, 0]
        # NO swap needed - landmark labels refer to PERSON's left/right, not image position

    else:
        logger.warning(f"Unsupported rotation angle: {angle}, returning unchanged")
        return image, landmarks

    return rotated, new_landmarks.tolist()


def crop_face_generous(
    image: np.ndarray,
    landmarks: List[List[float]],
    margin: float = 0.5,
    bbox: Optional[Dict[str, int]] = None
) -> Tuple[Optional[np.ndarray], List[List[float]]]:
    """Crop face with generous margin for rotation.

    Uses the face bounding box (not landmarks) to compute margin, ensuring
    sufficient room for rotation without hitting image borders.

    Args:
        image: Full image (H, W, C)
        landmarks: 5-point landmarks in full image coords
        margin: Margin as fraction of face size (0.5 = 50%, 1.0 = 100%)
        bbox: Face bounding box dict with keys x, y, w, h. If provided,
              uses bbox dimensions for margin calculation (recommended).

    Returns:
        Tuple of (cropped_image, landmarks_in_crop_coords)
    """
    h, w = image.shape[:2]
    landmarks_np = np.array(landmarks[:5], dtype=np.float32)

    if bbox is not None:
        # Use bbox for margin calculation (CORRECT - full face size)
        # Handle both pipeline format (x_px, y_px, w_px, h_px) and simple format (x, y, w, h)
        bbox_x = bbox.get('x_px', bbox.get('x', 0))
        bbox_y = bbox.get('y_px', bbox.get('y', 0))
        bbox_w = bbox.get('w_px', bbox.get('w', 0))
        bbox_h = bbox.get('h_px', bbox.get('h', 0))
        face_size = max(bbox_w, bbox_h)
        # Use bbox as the crop center
        min_x = bbox_x
        max_x = bbox_x + bbox_w
        min_y = bbox_y
        max_y = bbox_y + bbox_h
    else:
        # Fallback: compute from landmarks (less accurate - landmarks are smaller than face)
        min_x = np.min(landmarks_np[:, 0])
        max_x = np.max(landmarks_np[:, 0])
        min_y = np.min(landmarks_np[:, 1])
        max_y = np.max(landmarks_np[:, 1])
        face_w = max_x - min_x
        face_h = max_y - min_y
        face_size = max(face_w, face_h)

    # Add margin based on face size (not landmark span)
    margin_px = int(face_size * margin)

    x1 = max(0, int(min_x - margin_px))
    y1 = max(0, int(min_y - margin_px))
    x2 = min(w, int(max_x + margin_px))
    y2 = min(h, int(max_y + margin_px))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, landmarks

    # Transform landmarks to crop coordinates
    crop_landmarks = [[pt[0] - x1, pt[1] - y1] for pt in landmarks[:5]]

    return crop, crop_landmarks


def align_face_with_orientation(
    image: np.ndarray,
    landmarks: List[List[float]],
    orientation_angle: int,
    target_size: int = 256,
    initial_margin: float = 0.5,
    bbox: Optional[Dict[str, int]] = None
) -> Optional[np.ndarray]:
    """Align face with orientation correction using two-stage crop.

    Stage 1: Crop face with generous margin (50-100%) for rotation room
    Stage 2: Rotate crop and transform landmarks
    Stage 3: Apply 5-point affine alignment to target size

    Args:
        image: Full image (H, W, C)
        landmarks: Original 5-point landmarks in full image coords
        orientation_angle: Rotation needed (0, 90, 180, 270)
        target_size: Output face crop size
        initial_margin: Margin for initial crop (0.5 = 50%, 1.0 = 100%)
        bbox: Face bounding box dict with x, y, w, h for proper margin calc

    Returns:
        Aligned face crop or None if alignment fails
    """
    # Stage 1: Generous crop around face using bbox for margin
    crop, crop_landmarks = crop_face_generous(
        image, landmarks, margin=initial_margin, bbox=bbox
    )
    if crop is None:
        logger.warning(f"Failed to create initial crop. bbox={bbox}, landmarks={landmarks[:2]}...")
        return None

    # Stage 2: Rotate crop if needed
    if orientation_angle != 0:
        crop, crop_landmarks = rotate_image_and_landmarks(crop, crop_landmarks, orientation_angle)

    # Stage 3: 5-point affine alignment
    scale = target_size / 112.0
    ref_points = ARCFACE_REF_POINTS_112 * scale

    src_pts = np.array(crop_landmarks[:5], dtype=np.float32)
    dst_pts = ref_points.astype(np.float32)

    # Validate landmarks are within reasonable bounds
    if np.any(src_pts < 0) or np.any(src_pts > max(crop.shape[:2])):
        logger.warning(f"Invalid crop landmarks: {src_pts.tolist()}, crop shape: {crop.shape}")
        return None

    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    if M is None:
        logger.warning(f"Failed to estimate affine transform. src_pts={src_pts.tolist()}")
        return None

    aligned = cv2.warpAffine(
        crop, M, (target_size, target_size),
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


@register_step
class AlignFacesStep(BaseStep):
    """Align faces using orientation correction and 5-point affine transform.

    Reads orientation_angle from each face (set by detect_face_orientation)
    and applies proper alignment. Stores aligned crops in context.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="align_faces",
            display_name="Align Faces",
            description="Apply orientation correction and 5-point alignment.",
            category="people",
            requires={"insightface_faces"},
            produces={"aligned_faces"},
            depends_on=["detect_face_orientation"],
            config_schema={
                "type": "object",
                "properties": {
                    "target_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Output face crop size"
                    },
                    "skip_filtered": {
                        "type": "boolean",
                        "default": True,
                        "description": "Skip faces with filter_passed=False"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Align all faces in context."""
        if not hasattr(context, 'insightface_faces') or not context.insightface_faces:
            logger.info("No faces found in context - skipping alignment")
            return

        target_size = config.get("target_size", 256)
        skip_filtered = config.get("skip_filtered", True)
        cache = get_image_cache()

        # spec-040 A1: index face_records so align can mutate the Pydantic
        # mirror in lockstep with the legacy aligned_faces dict.
        record_index = {
            (r.image_path, r.face_index): r
            for r in (context.face_records or [])
            if r.image_path is not None and r.face_index is not None
        }

        aligned_faces = {}
        stats = {
            "total": 0,
            "aligned": 0,
            "skipped_filtered": 0,
            "skipped_no_landmarks": 0,
            "failed": 0,
            "by_orientation": {0: 0, 90: 0, 180: 0, 270: 0}
        }

        total_images = len(context.insightface_faces)
        for img_idx, (image_path, face_data) in enumerate(context.insightface_faces.items()):
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            img_np = cache.get(image_path)
            if img_np is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue

            image_aligned = []

            for face_info in face_data.get('faces', []):
                stats["total"] += 1
                face_idx = face_info.get('face_index', 0)

                # Check if filtered
                if skip_filtered:
                    if not face_info.get('filter_passed', True):
                        stats["skipped_filtered"] += 1
                        continue
                    if not face_info.get('is_clusterable', True):
                        stats["skipped_filtered"] += 1
                        continue

                landmarks = face_info.get('landmarks')
                if not landmarks or len(landmarks) < 5:
                    stats["skipped_no_landmarks"] += 1
                    continue

                orientation = face_info.get('orientation_angle', 0)
                stats["by_orientation"][orientation] += 1

                # Get bbox for proper margin calculation
                bbox = face_info.get('bbox')

                # Debug: Log first few faces to understand data
                if stats["total"] <= 3:
                    logger.info(f"DEBUG face {face_idx}: bbox={bbox}, landmarks[:2]={landmarks[:2] if landmarks else None}, orientation={orientation}")

                # Align face with bbox for generous margin
                aligned = align_face_with_orientation(
                    img_np, landmarks, orientation, target_size, bbox=bbox
                )

                if aligned is None:
                    stats["failed"] += 1
                    if stats["failed"] <= 5:
                        logger.warning(f"Alignment failed for face {face_idx} in {image_path}")
                    continue

                # Store aligned crop
                key = f"{image_path}:face_{face_idx}"
                aligned_faces[key] = aligned
                stats["aligned"] += 1

                # Also store in face_info for easy access
                face_info['aligned_crop'] = aligned
                image_aligned.append(face_idx)

                record = record_index.get((image_path, face_idx))
                if record is not None:
                    record.aligned_face = aligned

            # Progress update
            progress = (img_idx + 1) / total_images
            context.report_progress(
                "align_faces", progress,
                f"Aligned {stats['aligned']} faces from {img_idx + 1}/{total_images} images"
            )

        # Store in context
        context.aligned_faces = aligned_faces

        # Log summary
        logger.info("=" * 60)
        logger.info("ALIGN_FACES: Summary")
        logger.info("=" * 60)
        logger.info(f"Total faces:        {stats['total']}")
        logger.info(f"Successfully aligned: {stats['aligned']}")
        logger.info(f"Skipped (filtered): {stats['skipped_filtered']}")
        logger.info(f"Skipped (no landmarks): {stats['skipped_no_landmarks']}")
        logger.info(f"Failed alignment:   {stats['failed']}")
        logger.info("By orientation:")
        for angle, count in stats["by_orientation"].items():
            logger.info(f"  {angle}°: {count}")
        logger.info("=" * 60)
