"""Validate Alignment step - checks that face alignment worked correctly.

This step runs face detection on aligned crops and verifies that
detected landmarks are close to expected reference positions.

Single responsibility: ONLY validation, no modification of alignment.
"""

import logging
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.utils.face_alignment import ARCFACE_REF_POINTS_112

logger = logging.getLogger(__name__)


def compute_landmark_error(
    detected: np.ndarray,
    expected: np.ndarray
) -> float:
    """Compute mean Euclidean distance between detected and expected landmarks.

    Args:
        detected: Detected landmarks (5, 2)
        expected: Expected landmarks (5, 2)

    Returns:
        Mean error in pixels
    """
    if detected is None or len(detected) < 5:
        return float('inf')

    detected = np.array(detected[:5], dtype=np.float32)
    expected = np.array(expected[:5], dtype=np.float32)

    # Compute per-point errors
    errors = np.linalg.norm(detected - expected, axis=1)
    return float(np.mean(errors))


def detect_landmarks_in_crop(
    aligned_crop: np.ndarray,
    target_size: int = 256
) -> Optional[np.ndarray]:
    """Detect 5-point landmarks in an aligned face crop.

    Uses InsightFace for detection. If face is properly aligned,
    landmarks should be near reference positions.

    Args:
        aligned_crop: Aligned face image (H, W, 3)
        target_size: Expected size of the crop

    Returns:
        5-point landmarks or None if detection fails
    """
    try:
        from insightface.app import FaceAnalysis

        # Use minimal model for landmark detection only
        app = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider'],
            allowed_modules=['detection']
        )
        app.prepare(ctx_id=-1, det_size=(target_size, target_size))

        faces = app.get(aligned_crop)

        if not faces:
            return None

        # Get first detected face
        face = faces[0]
        if hasattr(face, 'kps') and face.kps is not None:
            return face.kps

        return None

    except Exception as e:
        logger.warning(f"Landmark detection failed: {e}")
        return None


@register_step
class ValidateAlignmentStep(BaseStep):
    """Validate face alignment quality by checking landmark positions.

    Runs face detection on aligned crops and measures distance
    from detected landmarks to expected reference positions.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="validate_alignment",
            display_name="Validate Alignment",
            description="Check that face alignment produced correct landmark positions.",
            category="people",
            requires={"aligned_faces"},
            produces={"alignment_validations"},
            depends_on=["align_faces"],
            config_schema={
                "type": "object",
                "properties": {
                    "error_threshold": {
                        "type": "number",
                        "default": 15.0,
                        "description": "Maximum allowed landmark error in pixels"
                    },
                    "validate_sample": {
                        "type": "boolean",
                        "default": True,
                        "description": "Only validate a sample of faces (faster)"
                    },
                    "sample_size": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of faces to validate if validate_sample=True"
                    },
                    "target_size": {
                        "type": "integer",
                        "default": 256,
                        "description": "Expected aligned crop size"
                    }
                }
            }
        )

    def process(self, context: PipelineContext, config: dict) -> None:
        """Validate alignment for faces in context."""
        if not hasattr(context, 'aligned_faces') or not context.aligned_faces:
            logger.info("No aligned faces found - skipping validation")
            return

        error_threshold = config.get("error_threshold", 15.0)
        validate_sample = config.get("validate_sample", True)
        sample_size = config.get("sample_size", 10)
        target_size = config.get("target_size", 256)

        # Scale reference points to target size
        scale = target_size / 112.0
        expected_landmarks = ARCFACE_REF_POINTS_112 * scale

        # Select faces to validate
        all_keys = list(context.aligned_faces.keys())
        if validate_sample and len(all_keys) > sample_size:
            # Sample evenly across faces
            indices = np.linspace(0, len(all_keys) - 1, sample_size, dtype=int)
            keys_to_validate = [all_keys[i] for i in indices]
        else:
            keys_to_validate = all_keys

        validations = {}
        stats = {
            "validated": 0,
            "passed": 0,
            "failed": 0,
            "no_face_detected": 0,
            "errors": []
        }

        for key in keys_to_validate:
            aligned_crop = context.aligned_faces[key]

            # Detect landmarks in aligned crop
            detected = detect_landmarks_in_crop(aligned_crop, target_size)

            if detected is None:
                stats["no_face_detected"] += 1
                validations[key] = {
                    "valid": False,
                    "error": float('inf'),
                    "reason": "no_face_detected"
                }
                continue

            # Compute error
            error = compute_landmark_error(detected, expected_landmarks)
            stats["errors"].append(error)

            is_valid = error < error_threshold
            if is_valid:
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            validations[key] = {
                "valid": is_valid,
                "error": error,
                "detected_landmarks": detected.tolist() if isinstance(detected, np.ndarray) else detected
            }
            stats["validated"] += 1

            # Update face_info in context if accessible
            self._update_face_info(context, key, is_valid, error)

        # Store validations in context
        context.alignment_validations = validations

        # Log summary
        logger.info("=" * 60)
        logger.info("VALIDATE_ALIGNMENT: Summary")
        logger.info("=" * 60)
        logger.info(f"Faces validated:    {stats['validated']}")
        logger.info(f"Passed (error < {error_threshold}px): {stats['passed']}")
        logger.info(f"Failed:             {stats['failed']}")
        logger.info(f"No face detected:   {stats['no_face_detected']}")
        if stats["errors"]:
            logger.info(f"Mean error:         {np.mean(stats['errors']):.2f}px")
            logger.info(f"Max error:          {np.max(stats['errors']):.2f}px")
        logger.info("=" * 60)

        # Warn if significant number of faces failed
        if stats["failed"] > stats["passed"]:
            logger.warning(
                f"More faces failed validation ({stats['failed']}) than passed ({stats['passed']}). "
                "This may indicate alignment issues."
            )

    def _update_face_info(
        self,
        context: PipelineContext,
        key: str,
        is_valid: bool,
        error: float
    ) -> None:
        """Update face_info in insightface_faces with validation result."""
        if not hasattr(context, 'insightface_faces'):
            return

        # Parse key: "image_path:face_N"
        parts = key.rsplit(':face_', 1)
        if len(parts) != 2:
            return

        image_path = parts[0]
        try:
            face_idx = int(parts[1])
        except ValueError:
            return

        if image_path not in context.insightface_faces:
            return

        face_data = context.insightface_faces[image_path]
        for face_info in face_data.get('faces', []):
            if face_info.get('face_index') == face_idx:
                face_info['alignment_valid'] = is_valid
                face_info['alignment_error'] = error
                break
