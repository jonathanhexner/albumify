#!/usr/bin/env python3
"""Test face orientation detection and alignment on an album or image.

Usage:
    python scripts/test_face_orientation.py /path/to/album
    python scripts/test_face_orientation.py /path/to/image.jpg
    python scripts/test_face_orientation.py /path/to/album --face-index 118
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.steps.detect_face_orientation import (
    detect_face_orientation,
    compute_orientation_confidence,
)
from sim_bench.pipeline.steps.align_faces import (
    rotate_image_and_landmarks,
    align_face_with_orientation,
)
from sim_bench.pipeline.utils.image_cache import get_image_cache

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def run_insightface_detection(image_path: str) -> dict:
    """Run InsightFace detection on a single image."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CPUExecutionProvider']
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    faces = app.get(img)

    result = {
        'image_path': image_path,
        'image_shape': img.shape,
        'faces': []
    }

    for i, face in enumerate(faces):
        face_info = {
            'face_index': i,
            'bbox': {
                'x_px': int(face.bbox[0]),
                'y_px': int(face.bbox[1]),
                'w_px': int(face.bbox[2] - face.bbox[0]),
                'h_px': int(face.bbox[3] - face.bbox[1]),
            },
            'confidence': float(face.det_score),
            'landmarks': face.kps.tolist() if face.kps is not None else None,
        }
        result['faces'].append(face_info)

    return result


def analyze_face(image_path: str, face_info: dict, output_dir: Path) -> dict:
    """Analyze a single face and save debug images."""
    img = cv2.imread(image_path)
    landmarks = face_info.get('landmarks')
    face_idx = face_info.get('face_index', 0)

    if not landmarks or len(landmarks) < 5:
        return {'error': 'No landmarks'}

    # Detect orientation
    orientation = detect_face_orientation(landmarks)
    confidence = compute_orientation_confidence(landmarks, orientation)

    # Align face
    aligned = align_face_with_orientation(img, landmarks, orientation, target_size=256)

    # Save debug images
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original with landmarks
    img_debug = img.copy()
    colors = [(0, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 255)]  # eyes=green, nose=blue, mouth=red
    labels = ['L_eye', 'R_eye', 'Nose', 'L_mouth', 'R_mouth']
    for j, (pt, color, label) in enumerate(zip(landmarks, colors, labels)):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img_debug, (x, y), 5, color, -1)
        cv2.putText(img_debug, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Add orientation text
    cv2.putText(
        img_debug,
        f"Orientation: {orientation}° (conf: {confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imwrite(str(output_dir / f"face_{face_idx}_original.jpg"), img_debug)

    # 2. Aligned face
    if aligned is not None:
        cv2.imwrite(str(output_dir / f"face_{face_idx}_aligned.jpg"), aligned)

    # 3. If rotated, show intermediate rotation
    if orientation != 0:
        rotated_img, rotated_lm = rotate_image_and_landmarks(img, landmarks, orientation)
        rotated_debug = rotated_img.copy()
        for j, (pt, color) in enumerate(zip(rotated_lm, colors)):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(rotated_debug, (x, y), 5, color, -1)
        cv2.putText(
            rotated_debug,
            f"After {orientation}° rotation",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        cv2.imwrite(str(output_dir / f"face_{face_idx}_rotated.jpg"), rotated_debug)

    return {
        'face_index': face_idx,
        'orientation': orientation,
        'confidence': confidence,
        'landmarks': landmarks,
        'aligned_shape': aligned.shape if aligned is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Test face orientation detection")
    parser.add_argument("path", type=Path, help="Path to image or album directory")
    parser.add_argument("--face-index", "-f", type=int, help="Only analyze specific face index")
    parser.add_argument("--output", "-o", type=Path, default=Path("orientation_test_output"),
                        help="Output directory for debug images")
    args = parser.parse_args()

    if not args.path.exists():
        logger.error(f"Path not found: {args.path}")
        sys.exit(1)

    # Collect images
    if args.path.is_file():
        images = [args.path]
    else:
        images = list(args.path.glob("**/*.jpg")) + list(args.path.glob("**/*.jpeg"))
        images += list(args.path.glob("**/*.JPG")) + list(args.path.glob("**/*.JPEG"))
        images += list(args.path.glob("**/*.png")) + list(args.path.glob("**/*.PNG"))

    logger.info(f"Found {len(images)} images")

    # Process each image
    all_results = []
    non_zero_orientations = []

    for img_path in images:
        logger.info(f"\nProcessing: {img_path.name}")

        try:
            detection = run_insightface_detection(str(img_path))
        except Exception as e:
            logger.error(f"  Detection failed: {e}")
            continue

        logger.info(f"  Found {len(detection['faces'])} faces")

        for face_info in detection['faces']:
            face_idx = face_info['face_index']

            if args.face_index is not None and face_idx != args.face_index:
                continue

            output_dir = args.output / img_path.stem
            result = analyze_face(str(img_path), face_info, output_dir)

            if 'error' in result:
                logger.warning(f"  Face {face_idx}: {result['error']}")
                continue

            orientation = result['orientation']
            confidence = result['confidence']

            logger.info(f"  Face {face_idx}: orientation={orientation}° (confidence={confidence:.2f})")

            if orientation != 0:
                non_zero_orientations.append({
                    'image': img_path.name,
                    'face_index': face_idx,
                    'orientation': orientation,
                    'confidence': confidence,
                })

            all_results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total faces analyzed: {len(all_results)}")
    logger.info(f"Non-zero orientations: {len(non_zero_orientations)}")

    if non_zero_orientations:
        logger.info("\nFaces with non-zero orientation:")
        for item in non_zero_orientations:
            logger.info(f"  {item['image']} face_{item['face_index']}: {item['orientation']}°")

    logger.info(f"\nDebug images saved to: {args.output}")


if __name__ == "__main__":
    main()
