"""YOLOv8-Pose wrapper for person detection."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
from ultralytics import YOLO

from sim_bench.pipeline.person_detection.types import PersonDetection, BoundingBox
from sim_bench.pipeline.person_detection.body_orientation import BodyOrientationFactory

logger = logging.getLogger(__name__)


class PersonSelector:
    """Strategy for selecting primary person from multiple detections."""
    
    def select_primary(self, detections: List[PersonDetection]) -> Optional[PersonDetection]:
        """Select primary person (largest bbox)."""
        max_area = 0
        primary = None
        
        for detection in detections:
            area = detection.bbox.w_px * detection.bbox.h_px
            max_area = max(max_area, area)
            primary = detection if area == max_area else primary
        
        return primary


class YOLOPersonDetector:
    """YOLOv8-Pose wrapper for person detection and body orientation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_size = config.get('model_size', 'small')
        self.confidence_threshold = config.get('confidence_threshold', 0.25)
        self.device = config.get('device', 'cpu')
        
        self._model = None
        self._orientation_strategy = None
        self._person_selector = PersonSelector()
        
        logger.info(f"YOLOPersonDetector configured: model={self.model_size}, device={self.device}")
    
    def _load_model(self):
        """Lazy load YOLOv8-Pose model."""
        model_name = f"yolov8{self.model_size[0]}-pose.pt"
        self._model = YOLO(model_name)
        self._model.to(self.device)
        logger.info(f"Loaded YOLOv8-Pose model: {model_name}")
        return self._model
    
    def _load_orientation_strategy(self):
        """Lazy load body orientation strategy."""
        strategy_name = self.config.get('orientation_strategy', 'shoulder_hip')
        self._orientation_strategy = BodyOrientationFactory.create(strategy_name, self.config)
        return self._orientation_strategy
    
    def detect_person(self, image_path: Path) -> Optional[PersonDetection]:
        """
        Detect primary person in image.
        
        Returns PersonDetection with body orientation score, or None if no person detected.
        """
        # Lazy load model and strategy
        self._model = self._model or self._load_model()
        self._orientation_strategy = self._orientation_strategy or self._load_orientation_strategy()
        
        # Run detection
        results = self._model(str(image_path), conf=self.confidence_threshold, verbose=False)
        result = results[0]
        
        # Extract person detections (class 0 = person in COCO)
        person_detections = self._extract_person_detections(result)
        
        # Select primary person
        primary_person = self._person_selector.select_primary(person_detections)
        
        return primary_person
    
    def _extract_person_detections(self, result) -> List[PersonDetection]:
        """Extract person detections from YOLO result."""
        detections = []
        
        boxes = result.boxes
        keypoints = result.keypoints
        
        num_detections = len(boxes) if boxes is not None else 0
        
        for i in range(num_detections):
            detection = self._create_person_detection(boxes, keypoints, i, result.orig_shape)
            detections.append(detection)
        
        return detections
    
    def _create_person_detection(self, boxes, keypoints, index: int, image_shape) -> PersonDetection:
        """Create PersonDetection from YOLO output."""
        box = boxes[index]
        kpts = keypoints[index].data.cpu().numpy()[0] if keypoints is not None else np.zeros((17, 3))
        
        # Extract bbox
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
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
        
        # Compute keypoint confidence
        keypoint_confidence = float(np.mean(kpts[:, 2]))
        
        # Create detection
        person = PersonDetection(
            bbox=bbox,
            confidence=float(box.conf[0]),
            keypoints=kpts,
            body_facing_score=0.0,  # Computed next
            keypoint_confidence=keypoint_confidence
        )
        
        # Compute body orientation
        person.body_facing_score = self._orientation_strategy.compute_facing_score(person)
        
        return person
