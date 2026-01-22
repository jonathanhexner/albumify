"""
MediaPipe-based portrait analyzer combining face detection, eye state, and smile detection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from sim_bench.portrait_analysis.types import PortraitMetrics, EyeState, SmileState
from sim_bench.portrait_analysis.eye_state import detect_eye_state, create_eye_state
from sim_bench.portrait_analysis.smile_detection import detect_smile, create_smile_state

logger = logging.getLogger(__name__)


class MediaPipePortraitAnalyzer:
    """
    Portrait analyzer using MediaPipe for face detection, eye state, and smile detection.

    Config-only constructor - receives full config dict.
    Reads settings from config['portrait_analysis'].
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analyzer with configuration.

        Args:
            config: Full configuration dictionary
        """
        self._config = config
        pa = config.get('portrait_analysis', {})

        self._face_confidence = pa.get('face_detection_confidence', 0.5)
        self._portrait_face_ratio = pa.get('portrait_face_ratio_threshold', 0.0005)
        self._portrait_center_offset = pa.get('portrait_center_offset_threshold', 0.3)
        self._ear_threshold = pa.get('eye_open_ear_threshold', 0.2)
        self._smile_width_threshold = pa.get('smile_width_threshold', 0.15)
        self._smile_elevation_threshold = pa.get('smile_elevation_threshold', 0.005)

        self._face_detection = None
        self._face_mesh = None

        logger.info(
            f"PortraitAnalyzer initialized: "
            f"ear_threshold={self._ear_threshold}, "
            f"smile_thresholds=({self._smile_width_threshold}, {self._smile_elevation_threshold})"
        )

    def _load_models(self):
        """Lazy load MediaPipe models."""
        if self._face_detection is not None:
            return

        import mediapipe as mp
        self._face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=self._face_confidence
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("MediaPipe models loaded")

    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image and convert to RGB."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _is_portrait(
        self,
        face_bbox: Dict[str, float],
        image_shape: tuple
    ) -> tuple:
        """
        Determine if image is a portrait based on face size and position.

        Returns:
            (is_portrait, face_ratio, center_offset)
        """
        img_height, img_width = image_shape[:2]

        face_area = face_bbox['w'] * face_bbox['h'] * img_width * img_height
        image_area = img_width * img_height
        face_ratio = face_area / image_area

        face_center_x = (face_bbox['x'] + face_bbox['w'] / 2) * img_width
        image_center_x = img_width / 2
        center_offset = abs(face_center_x - image_center_x) / img_width

        is_portrait = (
            face_ratio > self._portrait_face_ratio and
            center_offset < self._portrait_center_offset
        )

        return is_portrait, face_ratio, center_offset

    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image."""
        results = self._face_detection.process(image)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_bbox = {
                    'x': bbox.xmin,
                    'y': bbox.ymin,
                    'w': bbox.width,
                    'h': bbox.height
                }

                is_portrait, face_ratio, center_offset = self._is_portrait(
                    face_bbox, image.shape
                )

                faces.append({
                    'bbox': face_bbox,
                    'confidence': detection.score[0] if detection.score else 0.0,
                    'is_portrait': is_portrait,
                    'face_ratio': face_ratio,
                    'center_offset': center_offset
                })

        return faces

    def analyze_image(self, image_path: Path) -> PortraitMetrics:
        """
        Perform complete portrait analysis on an image.

        Args:
            image_path: Path to image file

        Returns:
            PortraitMetrics with all analysis results
        """
        self._load_models()
        image_path = Path(image_path)
        image = self._load_image(image_path)

        faces = self._detect_faces(image)

        has_face = len(faces) > 0
        num_faces = len(faces)
        is_portrait = False
        face_ratio = None
        center_offset = None
        eye_state = None
        smile_state = None
        confidence = 0.0

        if num_faces == 1 and faces[0]['is_portrait']:
            is_portrait = True
            face_ratio = faces[0]['face_ratio']
            center_offset = faces[0]['center_offset']
            confidence = faces[0]['confidence']

            mesh_results = self._face_mesh.process(image)
            if mesh_results.multi_face_landmarks:
                landmarks = mesh_results.multi_face_landmarks[0]

                eye_data = detect_eye_state(
                    landmarks, image.shape, self._ear_threshold
                )
                eye_state = create_eye_state(eye_data)

                smile_data = detect_smile(
                    landmarks,
                    image.shape,
                    self._smile_width_threshold,
                    self._smile_elevation_threshold
                )
                smile_state = create_smile_state(smile_data)

        return PortraitMetrics(
            image_path=str(image_path),
            has_face=has_face,
            num_faces=num_faces,
            is_portrait=is_portrait,
            face_ratio=face_ratio,
            center_offset=center_offset,
            eye_state=eye_state,
            smile_state=smile_state,
            confidence=confidence
        )

    def analyze_batch(
        self,
        image_paths: List[Path],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, PortraitMetrics]:
        """
        Analyze batch of images.

        Args:
            image_paths: List of image paths
            progress_callback: Optional callback(current, total)

        Returns:
            Dict mapping image_path -> PortraitMetrics
        """
        self._load_models()
        results = {}
        total = len(image_paths)

        for idx, path in enumerate(image_paths):
            results[str(path)] = self.analyze_image(path)
            if progress_callback:
                progress_callback(idx + 1, total)

        logger.info(f"Analyzed {total} images: {sum(1 for r in results.values() if r.is_portrait)} portraits")
        return results
