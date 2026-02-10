"""Detect Persons step - YOLOv8-Pose person detection with body orientation."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.person_detection.yolo_detector import YOLOPersonDetector

logger = logging.getLogger(__name__)


@register_step
class DetectPersonsStep(BaseStep):
    """Detect persons and compute body orientation using YOLOv8-Pose."""
    
    def __init__(self):
        self._metadata = StepMetadata(
            name="detect_persons",
            display_name="Detect Persons",
            description="Detect persons and compute body facing score using YOLOv8-Pose.",
            category="people",
            requires={"image_paths"},
            produces={"persons"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "model_size": {
                        "type": "string",
                        "enum": ["nano", "small", "medium"],
                        "default": "small"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "default": 0.25
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "mps"],
                        "default": "cpu"
                    }
                }
            }
        )
        self._detector = None
    
    def _get_detector(self, config: dict) -> YOLOPersonDetector:
        """Lazy load person detector."""
        self._detector = self._detector or YOLOPersonDetector(config)
        return self._detector
    
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        """Get cache configuration for person detection."""
        image_paths = [str(p) for p in context.image_paths]
        
        return {
            "items": image_paths,
            "feature_type": "person_detection",
            "model_name": f"yolov8{config.get('model_size', 'small')[0]}-pose",
            "metadata": {"device": config.get("device", "cpu")}
        }
    
    def _process_uncached(self, items: List[str], context: PipelineContext, config: dict) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - detect persons."""
        detector = self._get_detector(config)
        results = {}
        
        for i, path_str in enumerate(items):
            person = detector.detect_person(Path(path_str))
            results[path_str] = self._serialize_person(person)
            
            progress = (i + 1) / len(items)
            context.report_progress("detect_persons", progress, f"Detecting {i + 1}/{len(items)}")
        
        return results
    
    def _serialize_person(self, person) -> Dict[str, Any]:
        """Serialize PersonDetection to JSON-serializable dict."""
        return {
            'person_detected': person is not None,
            'bbox': self._serialize_bbox(person.bbox) if person else None,
            'confidence': float(person.confidence) if person else 0.0,
            'body_facing_score': float(person.body_facing_score) if person else 0.0,
            'keypoint_confidence': float(person.keypoint_confidence) if person else 0.0
        }
    
    def _serialize_bbox(self, bbox) -> Dict[str, Any]:
        """Serialize BoundingBox to dict."""
        return {
            'x': float(bbox.x),
            'y': float(bbox.y),
            'w': float(bbox.w),
            'h': float(bbox.h),
            'x_px': int(bbox.x_px),
            'y_px': int(bbox.y_px),
            'w_px': int(bbox.w_px),
            'h_px': int(bbox.h_px)
        }
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize person detection to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to person detection."""
        return Serializers.json_deserialize(data)
    
    def _store_results(self, context: PipelineContext, results: Dict[str, Dict[str, Any]], config: dict) -> None:
        """Store persons in context."""
        context.persons = results
        
        detected_count = sum(1 for r in results.values() if r.get('person_detected', False))
        logger.info(f"Detected persons in {detected_count}/{len(results)} images")
