"""Face alignment utilities.

Two alignment strategies:
1. Rotate-first, then crop (2-point): Uses roll angle from eyes only
2. 5-point affine alignment: Uses all 5 landmarks for proper similarity transform

The 5-point alignment is preferred for face recognition as it:
- Normalizes face position, scale, and rotation simultaneously
- Aligns to ArcFace reference template for optimal embedding quality
- Handles head tilt better than roll-only rotation
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ArcFace reference template for 112x112 face images
# Points: left_eye, right_eye, nose, left_mouth, right_mouth
ARCFACE_REF_POINTS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_face_5point(
    image: np.ndarray,
    landmarks: Union[np.ndarray, List[List[float]]],
    target_size: int = 256,
) -> Optional[np.ndarray]:
    """Align face using 5-point affine transform to ArcFace reference template.

    This is the standard alignment used by ArcFace and most modern face recognition
    models. It computes a similarity transform (rotation + scale + translation)
    from the detected landmarks to a normalized reference template.

    Args:
        image: Full image (H, W, C) in BGR or RGB format
        landmarks: 5 landmark points in pixel coordinates, shape (5, 2)
                   Order: [left_eye, right_eye, nose, left_mouth, right_mouth]
        target_size: Output square size in pixels (default 256 for display,
                     112 for direct embedding input)

    Returns:
        Aligned face image of shape (target_size, target_size, C), or None on failure
    """
    # Convert landmarks to numpy array if needed
    if isinstance(landmarks, list):
        src_pts = np.array(landmarks, dtype=np.float32)
    else:
        src_pts = np.asarray(landmarks, dtype=np.float32)

    # Validate landmarks shape
    if src_pts.shape != (5, 2):
        logger.warning(f"Invalid landmarks shape: {src_pts.shape}, expected (5, 2)")
        return None

    # Check for invalid coordinates (zeros or negative values indicate missing data)
    if np.any(src_pts <= 0):
        logger.warning("Landmarks contain invalid coordinates (<=0)")
        return None

    # Scale reference template to target size
    scale = target_size / 112.0
    dst_pts = ARCFACE_REF_POINTS_112 * scale

    # Estimate similarity transform (rotation + uniform scale + translation)
    # estimateAffinePartial2D returns a 2x3 affine matrix
    tform, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts,
        method=cv2.LMEDS  # Least median of squares - robust to outliers
    )

    if tform is None:
        logger.warning("Failed to estimate affine transform from landmarks")
        return None

    # Apply transform
    aligned = cv2.warpAffine(
        image, tform, (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned


def compute_alignment_quality(
    landmarks: Union[np.ndarray, List[List[float]]],
    target_size: int = 256,
) -> Dict[str, float]:
    """Compute alignment quality metrics from landmarks.

    Args:
        landmarks: 5 landmark points in pixel coordinates
        target_size: Target output size

    Returns:
        Dict with alignment metrics:
        - transform_error: Mean distance between transformed and reference points
        - scale: Estimated scale factor
        - rotation_deg: Estimated rotation in degrees
    """
    if isinstance(landmarks, list):
        src_pts = np.array(landmarks, dtype=np.float32)
    else:
        src_pts = np.asarray(landmarks, dtype=np.float32)

    if src_pts.shape != (5, 2):
        return {"transform_error": -1.0, "scale": -1.0, "rotation_deg": 0.0}

    scale = target_size / 112.0
    dst_pts = ARCFACE_REF_POINTS_112 * scale

    tform, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if tform is None:
        return {"transform_error": -1.0, "scale": -1.0, "rotation_deg": 0.0}

    # Transform source points and compute error
    src_pts_h = np.hstack([src_pts, np.ones((5, 1))])
    transformed = src_pts_h @ tform.T
    error = np.mean(np.linalg.norm(transformed - dst_pts, axis=1))

    # Extract scale and rotation from transform matrix
    # tform = [[s*cos(θ), -s*sin(θ), tx],
    #          [s*sin(θ),  s*cos(θ), ty]]
    cos_theta = tform[0, 0]
    sin_theta = tform[1, 0]
    est_scale = np.sqrt(cos_theta**2 + sin_theta**2)
    rotation_rad = np.arctan2(sin_theta, cos_theta)
    rotation_deg = np.degrees(rotation_rad)

    return {
        "transform_error": float(error),
        "scale": float(est_scale),
        "rotation_deg": float(rotation_deg),
    }


def rotate_image_and_transform_bbox(
    image: np.ndarray,
    bbox: Dict[str, Any],
    roll_angle: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotate full image and transform bounding box to match.

    Args:
        image: Full image (H, W, C)
        bbox: Dict with x_px, y_px, w_px, h_px
        roll_angle: Degrees to rotate (positive = counter-clockwise in OpenCV)

    Returns:
        (rotated_image, transformed_bbox)
    """
    if abs(roll_angle) < 0.1:
        return image, bbox

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    angle_rad = math.radians(abs(roll_angle))
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    rotation_matrix = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image, rotation_matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    bbox_center = np.array(
        [[bbox['x_px'] + bbox['w_px'] / 2, bbox['y_px'] + bbox['h_px'] / 2]],
        dtype=np.float32,
    )
    transformed = cv2.transform(bbox_center.reshape(-1, 1, 2), rotation_matrix)[0][0]

    transformed_bbox = {
        'x_px': int(transformed[0] - bbox['w_px'] / 2),
        'y_px': int(transformed[1] - bbox['h_px'] / 2),
        'w_px': bbox['w_px'],
        'h_px': bbox['h_px'],
    }
    return rotated, transformed_bbox


def crop_aligned_face(
    image: np.ndarray,
    bbox: Dict[str, Any],
    margin: float = 0.2,
    target_size: int = 256,
) -> np.ndarray:
    """Crop face from image with margin and resize to target_size.

    Args:
        image: Image (H, W, C) — should already be rotated upright
        bbox: Dict with x_px, y_px, w_px, h_px
        margin: Fractional margin around face (0.2 = 20 %)
        target_size: Output square size in pixels

    Returns:
        Cropped and resized face array, or None on invalid bbox
    """
    h, w = image.shape[:2]
    mw = int(bbox['w_px'] * margin)
    mh = int(bbox['h_px'] * margin)

    x1 = max(0, bbox['x_px'] - mw)
    y1 = max(0, bbox['y_px'] - mh)
    x2 = min(w, bbox['x_px'] + bbox['w_px'] + mw)
    y2 = min(h, bbox['y_px'] + bbox['h_px'] + mh)

    if x2 <= x1 or y2 <= y1:
        logger.warning("Invalid crop bbox: x1=%d y1=%d x2=%d y2=%d", x1, y1, x2, y2)
        return None

    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def align_and_crop_face(
    image: np.ndarray,
    bbox: Dict[str, Any],
    roll_angle: float,
    margin: float = 0.2,
    target_size: int = 256,
    angle_threshold: float = 5.0,
) -> np.ndarray:
    """Rotate-first alignment then crop — the canonical pipeline entry point.

    Args:
        image: Full RGB image (H, W, C)
        bbox: Face bounding box dict (x_px, y_px, w_px, h_px)
        roll_angle: Head roll in degrees (positive = clockwise tilt)
        margin: Fractional margin around face
        target_size: Output square size
        angle_threshold: Skip rotation when |angle| < threshold

    Returns:
        Aligned, cropped, resized face array; or None on failure
    """
    if abs(roll_angle) >= angle_threshold:
        image, bbox = rotate_image_and_transform_bbox(image, bbox, roll_angle)
    return crop_aligned_face(image, bbox, margin=margin, target_size=target_size)
