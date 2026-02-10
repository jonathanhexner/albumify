"""Face quality scoring using pose, eyes, smile, and sharpness.

⚠️ DEPRECATED: This module is legacy code for MediaPipe-based face quality scoring.

The active pipeline uses:
- pipeline/scoring/quality_strategy.py - Image quality scoring (IQA + AVA + Siamese)
- pipeline/scoring/person_penalty.py - Person/portrait penalties
- select_best.py - Composite scoring (quality + penalty)

This scorer is only used by legacy MediaPipe pipeline steps.
See: docs/architecture/PIPELINE_ARCHITECTURE_CURRENT_STATE.md
"""

import logging
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

from sim_bench.face_pipeline.pose_estimator import SixDRepNetEstimator
from sim_bench.face_pipeline.types import FaceQualityScore, CroppedFace
from sim_bench.portrait_analysis.eye_state import detect_eye_state
from sim_bench.portrait_analysis.smile_detection import detect_smile

logger = logging.getLogger(__name__)


class FaceQualityScorer:
    """
    Compute quality metrics for cropped faces.
    
    ⚠️ DEPRECATED: This is legacy MediaPipe-based face quality scoring.
    
    The active pipeline uses modular scoring in select_best.py:
    - ImageQualityStrategy (IQA + AVA + optional Siamese)
    - PersonPenaltyComputer (portrait-specific penalties)
    """

    def __init__(self, config: Dict[str, Any], pose_estimator: SixDRepNetEstimator = None):
        """
        Initialize with configuration.

        Args:
            config: Configuration dict with 'face_pipeline' and 'portrait_analysis' sections
            pose_estimator: Optional injected pose estimator
        """
        self._config = config
        fp_config = config.get('face_pipeline', {})
        pa_config = config.get('portrait_analysis', {})

        self._ear_threshold = fp_config.get(
            'eye_open_ear_threshold',
            pa_config.get('eye_open_ear_threshold', 0.2)
        )
        self._smile_width_threshold = fp_config.get(
            'smile_width_threshold',
            pa_config.get('smile_width_threshold', 0.15)
        )
        self._smile_elevation_threshold = fp_config.get(
            'smile_elevation_threshold',
            pa_config.get('smile_elevation_threshold', 0.005)
        )
        self._sharpness_norm = fp_config.get('sharpness_norm', 1000.0)

        self._pose_estimator = pose_estimator or SixDRepNetEstimator(config)
        self._face_mesh = None

        logger.info(
            "FaceQualityScorer initialized "
            f"(ear_threshold={self._ear_threshold}, sharpness_norm={self._sharpness_norm})"
        )

    def _load_face_mesh(self):
        """Lazy load MediaPipe face mesh."""
        if self._face_mesh:
            return
        import mediapipe as mp
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("MediaPipe face mesh loaded for quality scoring")

    def _compute_sharpness(self, image: Image.Image) -> float:
        """Compute normalized sharpness score (0-1)."""
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        sharpness_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, sharpness_raw / self._sharpness_norm)
        return float(sharpness_score)

    def _compute_landmark_scores(self, image: Image.Image) -> Tuple[float, bool, float, bool]:
        """Compute eyes-open and smile scores from landmarks."""
        self._load_face_mesh()
        image_np = np.array(image)
        results = self._face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            return 0.0, False, 0.0, False

        landmarks = results.multi_face_landmarks[0]
        eye_data = detect_eye_state(landmarks, image_np.shape, self._ear_threshold)
        smile_data = detect_smile(
            landmarks,
            image_np.shape,
            self._smile_width_threshold,
            self._smile_elevation_threshold
        )

        left_score = min(1.0, eye_data['left_ear'] / self._ear_threshold)
        right_score = min(1.0, eye_data['right_ear'] / self._ear_threshold)
        eyes_open_score = (left_score + right_score) / 2.0

        return (
            float(eyes_open_score),
            bool(eye_data['left_eye_open'] and eye_data['right_eye_open']),
            float(smile_data['smile_score']),
            bool(smile_data['is_smiling'])
        )

    def score_face(self, face: CroppedFace) -> FaceQualityScore:
        """
        Compute and attach quality metrics to a face.

        Args:
            face: CroppedFace to score

        Returns:
            FaceQualityScore
        """
        pose = self._pose_estimator.estimate_pose(face.image)
        eyes_open_score, both_eyes_open, smile_score, is_smiling = self._compute_landmark_scores(face.image)
        sharpness_score = self._compute_sharpness(face.image)

        face.pose = pose
        face.eyes_open_score = eyes_open_score
        face.both_eyes_open = both_eyes_open
        face.smile_score = smile_score
        face.is_smiling = is_smiling

        quality = FaceQualityScore(
            pose_score=pose.frontal_score,
            eyes_open_score=eyes_open_score,
            smile_score=smile_score,
            sharpness_score=sharpness_score,
            detection_confidence=face.detection_confidence
        )
        face.quality = quality

        return quality

    def score_faces(self, faces: list) -> list:
        """Score a list of faces in order."""
        scores = []
        for face in faces:
            scores.append(self.score_face(face))
        return scores
