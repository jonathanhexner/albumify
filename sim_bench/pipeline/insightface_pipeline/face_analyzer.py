"""InsightFace wrapper for face detection and analysis."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import insightface
from insightface.app import FaceAnalysis

from sim_bench.pipeline.insightface_pipeline.types import InsightFaceDetection
from sim_bench.pipeline.person_detection.types import BoundingBox
from sim_bench.pipeline.utils.image_cache import get_image_cache

logger = logging.getLogger(__name__)


class FacePersonAssociator:
    """Strategy for associating faces with persons."""
    
    def associate(self, face_bbox: BoundingBox, person_bbox: Optional[Dict[str, Any]]) -> bool:
        """Check if face is associated with person (center-inside-bbox check)."""
        person_bbox_obj = person_bbox if person_bbox is None else self._dict_to_bbox(person_bbox)
        
        has_person = person_bbox_obj is not None
        
        # Strategy: delegate to checker
        checker = PersonExistsChecker() if has_person else NoPersonChecker()
        return checker.check_association(face_bbox, person_bbox_obj)
    
    def _dict_to_bbox(self, bbox_dict: Dict[str, Any]) -> BoundingBox:
        """Convert dict to BoundingBox."""
        return BoundingBox(
            x=bbox_dict['x'],
            y=bbox_dict['y'],
            w=bbox_dict['w'],
            h=bbox_dict['h'],
            x_px=bbox_dict['x_px'],
            y_px=bbox_dict['y_px'],
            w_px=bbox_dict['w_px'],
            h_px=bbox_dict['h_px']
        )


class PersonExistsChecker:
    """Check association when person exists."""
    
    def check_association(self, face_bbox: BoundingBox, person_bbox: BoundingBox) -> bool:
        """Check if face center is inside person bbox."""
        face_center_x = face_bbox.x + face_bbox.w / 2
        face_center_y = face_bbox.y + face_bbox.h / 2
        
        inside_x = person_bbox.x <= face_center_x <= (person_bbox.x + person_bbox.w)
        inside_y = person_bbox.y <= face_center_y <= (person_bbox.y + person_bbox.h)
        
        return inside_x and inside_y


class NoPersonChecker:
    """No association check when person doesn't exist."""
    
    def check_association(self, face_bbox: BoundingBox, person_bbox: Optional[BoundingBox]) -> bool:
        """Always return False when no person."""
        return False


class InsightFaceFaceAnalyzer:
    """InsightFace wrapper for face detection and attribute analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name', 'buffalo_l')
        self.detection_threshold = config.get('detection_threshold', 0.5)
        self.device = config.get('device', 'cpu')
        
        self._app = None
        self._associator = FacePersonAssociator()
        
        logger.info(f"InsightFaceFaceAnalyzer configured: model={self.model_name}, device={self.device}")
    
    def _load_app(self):
        """Lazy load InsightFace app."""
        ctx_id = -1 if self.device == 'cpu' else 0
        self._app = FaceAnalysis(name=self.model_name, providers=['CPUExecutionProvider'])
        self._app.prepare(ctx_id=ctx_id, det_thresh=self.detection_threshold)
        logger.info(f"Loaded InsightFace model: {self.model_name}")
        return self._app
    
    def detect_faces(self, image_path: Path, person_data: Optional[Dict[str, Any]]) -> List[InsightFaceDetection]:
        """
        Detect faces in image.
        
        Args:
            image_path: Path to image
            person_data: Person detection data (optional)
            
        Returns:
            List of InsightFaceDetection objects
        """
        # Lazy load app
        self._app = self._app or self._load_app()

        # Load image (EXIF-normalized via global cache)
        cache = get_image_cache()
        img = cache.get(image_path)

        # Run detection
        faces = self._app.get(img)
        
        # Convert to InsightFaceDetection objects
        detections = []
        for i, face in enumerate(faces):
            detection = self._create_face_detection(face, image_path, i, img.shape, person_data)
            detections.append(detection)
        
        # Check for occlusion
        face_occluded = self._check_occlusion(person_data, detections)
        
        return detections if detections else self._create_occluded_detection(image_path, person_data, face_occluded)
    
    def _create_face_detection(self, face, image_path: Path, index: int, image_shape, person_data: Optional[Dict[str, Any]]) -> InsightFaceDetection:
        """Create InsightFaceDetection from face."""
        bbox_array = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox_array
        
        bbox = BoundingBox(
            x=float(x1) / image_shape[1],
            y=float(y1) / image_shape[0],
            w=float(x2 - x1) / image_shape[1],
            h=float(y2 - y1) / image_shape[0],
            x_px=int(x1),
            y_px=int(y1),
            w_px=int(x2 - x1),
            h_px=int(y2 - y1)
        )
        
        person_bbox_dict = person_data.get('bbox') if person_data else None
        person_bbox = self._associator._dict_to_bbox(person_bbox_dict) if person_bbox_dict else None
        
        return InsightFaceDetection(
            original_path=image_path,
            face_index=index,
            bbox=bbox,
            confidence=float(face.det_score),
            landmarks=face.kps,
            person_bbox=person_bbox,
            face_occluded=False
        )
    
    def _check_occlusion(self, person_data: Optional[Dict[str, Any]], detections: List[InsightFaceDetection]) -> bool:
        """Check if face is occluded (person exists but no face found)."""
        person_detected = person_data.get('person_detected', False) if person_data else False
        has_faces = len(detections) > 0
        return person_detected and not has_faces
    
    def _create_occluded_detection(self, image_path: Path, person_data: Optional[Dict[str, Any]], face_occluded: bool) -> List[InsightFaceDetection]:
        """Create empty list or occluded detection."""
        return []
    
    def get_face_attributes(self, face_img: np.ndarray) -> Dict[str, Any]:
        """
        Get face attributes (expression, age, gender).
        
        Args:
            face_img: Face image as numpy array
            
        Returns:
            Dictionary with attributes
        """
        # Lazy load app
        self._app = self._app or self._load_app()
        
        # Run analysis
        faces = self._app.get(face_img)
        
        face = faces[0] if faces else None
        
        attributes = self._extract_attributes(face) if face else self._get_default_attributes()
        
        return attributes
    
    def _extract_attributes(self, face) -> Dict[str, Any]:
        """Extract attributes from face."""
        return {
            'age': int(face.age) if hasattr(face, 'age') else 25,
            'gender': int(face.gender) if hasattr(face, 'gender') else 0,
            'emotion': self._get_emotion(face) if hasattr(face, 'emotion') else 'neutral'
        }
    
    def _get_default_attributes(self) -> Dict[str, Any]:
        """Get default attributes when no face detected."""
        return {
            'age': 25,
            'gender': 0,
            'emotion': 'neutral'
        }
    
    def _get_emotion(self, face) -> str:
        """Get emotion label from face."""
        return 'neutral'  # InsightFace doesn't provide emotion by default
