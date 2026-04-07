"""Save Face Debug Artifacts - saves all face crops to disk for debugging.

Outputs folder structure:
    debug_faces/
      {image_name}/
        face_{idx:03d}_raw.jpg              # bbox crop only
        face_{idx:03d}_raw_landmarks.jpg    # bbox crop + landmarks
        face_{idx:03d}_aligned.jpg          # orientation-corrected + 5-point aligned
        face_{idx:03d}_aligned_landmarks.jpg # aligned + reference landmarks
        face_{idx:03d}_info.txt             # metadata (orientation, confidence, etc.)

This allows easy debugging of the face alignment pipeline.
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
from sim_bench.pipeline.utils.face_alignment import ARCFACE_REF_POINTS_112
from sim_bench.pipeline.steps.align_faces import (
    rotate_image_and_landmarks,
    align_face_with_orientation,
)

logger = logging.getLogger(__name__)

# Colors for landmarks (BGR)
LANDMARK_COLORS = [
    (0, 255, 0),    # left_eye - green
    (0, 255, 0),    # right_eye - green
    (255, 0, 0),    # nose - blue
    (0, 0, 255),    # left_mouth - red
    (0, 0, 255),    # right_mouth - red
]
LANDMARK_LABELS = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']


def draw_landmarks(
    image: np.ndarray,
    landmarks: List[List[float]],
    radius: int = 3,
    draw_labels: bool = True
) -> np.ndarray:
    """Draw landmarks on image."""
    img = image.copy()
    for i, (pt, color, label) in enumerate(zip(landmarks[:5], LANDMARK_COLORS, LANDMARK_LABELS)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), radius, color, -1)
        if draw_labels:
            cv2.putText(img, label, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    return img


def crop_face_raw(
    image: np.ndarray,
    bbox: Dict[str, int],
    margin: float = 0.2
) -> Optional[np.ndarray]:
    """Crop face using bbox with margin."""
    h, w = image.shape[:2]
    x = int(bbox.get('x_px', 0))
    y = int(bbox.get('y_px', 0))
    bw = int(bbox.get('w_px', 0))
    bh = int(bbox.get('h_px', 0))

    if bw <= 0 or bh <= 0:
        return None

    margin_w = int(bw * margin)
    margin_h = int(bh * margin)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w, x + bw + margin_w)
    y2 = min(h, y + bh + margin_h)

    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def transform_landmarks_to_crop(
    landmarks: List[List[float]],
    bbox: Dict[str, int],
    margin: float = 0.2
) -> List[List[float]]:
    """Transform landmarks from full image coords to crop coords."""
    x = int(bbox.get('x_px', 0))
    y = int(bbox.get('y_px', 0))
    bw = int(bbox.get('w_px', 0))
    bh = int(bbox.get('h_px', 0))

    margin_w = int(bw * margin)
    margin_h = int(bh * margin)

    x1 = x - margin_w
    y1 = y - margin_h

    return [[pt[0] - x1, pt[1] - y1] for pt in landmarks]


@register_step
class SaveFaceDebugArtifactsStep(BaseStep):
    """Save face debug artifacts to disk for easy inspection."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="save_face_debug_artifacts",
            display_name="Save Face Debug Artifacts",
            description="Save raw and aligned face crops to disk for debugging.",
            category="debug",
            requires={"insightface_faces"},
            produces={"debug_artifacts_path"},
            depends_on=["detect_face_orientation"],
            config_schema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "default": "debug_faces",
                        "description": "Output directory for debug artifacts"
                    },
                    "target_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Size for aligned crops"
                    },
                    "save_raw": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save raw bbox crops"
                    },
                    "save_aligned": {
                        "type": "boolean",
                        "default": True,
                        "description": "Save aligned crops"
                    },
                    "margin": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Margin for raw crops"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Save debug artifacts for all faces."""
        if not hasattr(context, 'insightface_faces') or not context.insightface_faces:
            logger.info("No faces found - skipping debug artifacts")
            return

        output_dir = Path(config.get("output_dir", "debug_faces"))
        target_size = config.get("target_size", 256)
        save_raw = config.get("save_raw", True)
        save_aligned = config.get("save_aligned", True)
        margin = config.get("margin", 0.2)

        # Clear output dir
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cache = get_image_cache()

        # Reference landmarks for aligned crops (scaled to target_size)
        scale = target_size / 112.0
        ref_landmarks = (ARCFACE_REF_POINTS_112 * scale).tolist()

        stats = {"images": 0, "faces": 0, "raw": 0, "aligned": 0}

        for image_path, face_data in context.insightface_faces.items():
            img_np = cache.get(image_path)
            if img_np is None:
                logger.warning(f"Could not load: {image_path}")
                continue

            # Create folder for this image
            image_name = Path(image_path).stem
            image_dir = output_dir / image_name
            image_dir.mkdir(parents=True, exist_ok=True)
            stats["images"] += 1

            for face_info in face_data.get('faces', []):
                face_idx = face_info.get('face_index', 0)
                bbox = face_info.get('bbox', {})
                landmarks = face_info.get('landmarks', [])
                orientation = face_info.get('orientation_angle', 0)
                confidence = face_info.get('confidence', 0)
                frontal_score = face_info.get('frontal_score', 0)
                filter_passed = face_info.get('filter_passed', True)
                is_clusterable = face_info.get('is_clusterable', True)

                prefix = f"face_{face_idx:03d}"
                stats["faces"] += 1

                # Save info file
                info_path = image_dir / f"{prefix}_info.txt"
                with open(info_path, 'w') as f:
                    f.write(f"Image: {image_path}\n")
                    f.write(f"Face Index: {face_idx}\n")
                    f.write(f"Orientation: {orientation}°\n")
                    f.write(f"Confidence: {confidence:.3f}\n")
                    f.write(f"Frontal Score: {frontal_score:.3f}\n")
                    f.write(f"Filter Passed: {filter_passed}\n")
                    f.write(f"Is Clusterable: {is_clusterable}\n")
                    f.write(f"BBox: x={bbox.get('x_px')}, y={bbox.get('y_px')}, w={bbox.get('w_px')}, h={bbox.get('h_px')}\n")
                    f.write(f"Landmarks:\n")
                    for i, (pt, label) in enumerate(zip(landmarks[:5], LANDMARK_LABELS)):
                        f.write(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})\n")

                # RAW CROP (bbox only)
                if save_raw and bbox:
                    raw_crop = crop_face_raw(img_np, bbox, margin)
                    if raw_crop is not None:
                        # Save raw
                        cv2.imwrite(str(image_dir / f"{prefix}_raw.jpg"), raw_crop)
                        stats["raw"] += 1

                        # Save raw + landmarks
                        if landmarks and len(landmarks) >= 5:
                            crop_landmarks = transform_landmarks_to_crop(landmarks, bbox, margin)
                            raw_with_lm = draw_landmarks(raw_crop, crop_landmarks)
                            cv2.imwrite(str(image_dir / f"{prefix}_raw_landmarks.jpg"), raw_with_lm)

                # ALIGNED CROP (orientation + 5-point)
                if save_aligned and landmarks and len(landmarks) >= 5:
                    # Step 1: Rotate image if needed
                    if orientation != 0:
                        rotated_img, rotated_landmarks = rotate_image_and_landmarks(
                            img_np, landmarks, orientation
                        )

                        # Save intermediate rotated full image with landmarks
                        rotated_debug = rotated_img.copy()
                        for j, (pt, color) in enumerate(zip(rotated_landmarks[:5], LANDMARK_COLORS)):
                            x, y = int(pt[0]), int(pt[1])
                            cv2.circle(rotated_debug, (x, y), 8, color, -1)
                        # Crop to face region for saving (don't save full rotated image)
                        rotated_crop = crop_face_raw(rotated_debug, {
                            'x_px': int(min(p[0] for p in rotated_landmarks) - 50),
                            'y_px': int(min(p[1] for p in rotated_landmarks) - 50),
                            'w_px': int(max(p[0] for p in rotated_landmarks) - min(p[0] for p in rotated_landmarks) + 100),
                            'h_px': int(max(p[1] for p in rotated_landmarks) - min(p[1] for p in rotated_landmarks) + 100),
                        }, margin=0.3)
                        if rotated_crop is not None:
                            cv2.putText(
                                rotated_crop,
                                f"After {orientation}deg rotation",
                                (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1
                            )
                            cv2.imwrite(str(image_dir / f"{prefix}_rotated.jpg"), rotated_crop)

                        # Update info file with rotated landmarks
                        with open(info_path, 'a') as f:
                            f.write(f"\nAfter {orientation}° rotation:\n")
                            f.write(f"Rotated image shape: {rotated_img.shape}\n")
                            f.write(f"Rotated Landmarks:\n")
                            for i, (pt, label) in enumerate(zip(rotated_landmarks[:5], LANDMARK_LABELS)):
                                f.write(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f})\n")
                    else:
                        rotated_img = img_np
                        rotated_landmarks = landmarks

                    # Step 2: Compute affine transform
                    scale = target_size / 112.0
                    ref_pts = ARCFACE_REF_POINTS_112 * scale
                    src_pts = np.array(rotated_landmarks[:5], dtype=np.float32)
                    dst_pts = ref_pts.astype(np.float32)

                    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

                    if M is not None:
                        # Save transformation matrix
                        with open(info_path, 'a') as f:
                            f.write(f"\nAffine Transform Matrix:\n")
                            f.write(f"  [{M[0,0]:.4f}, {M[0,1]:.4f}, {M[0,2]:.4f}]\n")
                            f.write(f"  [{M[1,0]:.4f}, {M[1,1]:.4f}, {M[1,2]:.4f}]\n")
                            # Extract rotation angle from matrix
                            angle_rad = np.arctan2(M[1,0], M[0,0])
                            angle_deg = np.degrees(angle_rad)
                            scale_factor = np.sqrt(M[0,0]**2 + M[1,0]**2)
                            f.write(f"  Rotation: {angle_deg:.2f}°\n")
                            f.write(f"  Scale: {scale_factor:.4f}\n")

                        # Apply transform
                        aligned = cv2.warpAffine(
                            rotated_img, M, (target_size, target_size),
                            borderMode=cv2.BORDER_REPLICATE
                        )

                        # Save aligned
                        cv2.imwrite(str(image_dir / f"{prefix}_aligned.jpg"), aligned)
                        stats["aligned"] += 1

                        # Save aligned + reference landmarks
                        aligned_with_lm = draw_landmarks(aligned, ref_landmarks, radius=4)
                        cv2.putText(
                            aligned_with_lm,
                            f"orient={orientation} applied",
                            (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 0),
                            1
                        )
                        cv2.imwrite(str(image_dir / f"{prefix}_aligned_landmarks.jpg"), aligned_with_lm)

        # Log summary
        logger.info("=" * 60)
        logger.info("SAVE_FACE_DEBUG_ARTIFACTS: Summary")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir.absolute()}")
        logger.info(f"Images processed: {stats['images']}")
        logger.info(f"Faces processed: {stats['faces']}")
        logger.info(f"Raw crops saved: {stats['raw']}")
        logger.info(f"Aligned crops saved: {stats['aligned']}")
        logger.info("=" * 60)

        context.debug_artifacts_path = str(output_dir.absolute())
