"""InsightFace Detect Faces step - face detection with person association."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.insightface_pipeline.face_analyzer import InsightFaceFaceAnalyzer

logger = logging.getLogger(__name__)


@register_step
class InsightFaceDetectFacesStep(BaseStep):
    """Detect faces using InsightFace SCRFD with person association."""
    
    def __init__(self):
        self._metadata = StepMetadata(
            name="insightface_detect_faces",
            display_name="InsightFace Detect Faces",
            description="Detect faces using InsightFace and associate with persons.",
            category="people",
            requires={"image_paths"},
            produces={"faces"},  # Common interface
            depends_on=["detect_persons"],
            config_schema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "default": "buffalo_l"
                    },
                    "detection_threshold": {
                        "type": "number",
                        "default": 0.5
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda"],
                        "default": "cpu"
                    },
                    "associate_to_person": {
                        "type": "boolean",
                        "default": True
                    }
                }
            }
        )
        self._analyzer = None
    
    def _get_analyzer(self, config: dict) -> InsightFaceFaceAnalyzer:
        """Lazy load face analyzer."""
        self._analyzer = self._analyzer or InsightFaceFaceAnalyzer(config)
        return self._analyzer
    
    def _get_cache_config(self, context: PipelineContext, config: dict) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face detection."""
        # Normalize paths to forward slashes for consistent keys
        image_paths = [str(p).replace('\\', '/') for p in context.image_paths]

        return {
            "items": image_paths,
            "feature_type": "insightface_detection",
            "model_name": config.get('model_name', 'buffalo_l'),
            "metadata": {"device": config.get("device", "cpu")}
        }
    
    def _process_uncached(self, items: List[str], context: PipelineContext, config: dict) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - detect faces."""
        analyzer = self._get_analyzer(config)
        results = {}
        
        for i, path_str in enumerate(items):
            person_data = context.persons.get(path_str) if hasattr(context, 'persons') else None
            faces = analyzer.detect_faces(Path(path_str), person_data)
            results[path_str] = self._serialize_faces(faces)
            
            progress = (i + 1) / len(items)
            context.report_progress("insightface_detect_faces", progress, f"Detecting {i + 1}/{len(items)}")
        
        return results
    
    def _serialize_faces(self, faces: List) -> Dict[str, Any]:
        """Serialize face detections to JSON-serializable dict."""
        return {
            'faces': [self._serialize_face(face) for face in faces]
        }
    
    def _serialize_face(self, face) -> Dict[str, Any]:
        """Serialize single face detection."""
        return {
            'face_index': face.face_index,
            'bbox': self._serialize_bbox(face.bbox),
            'confidence': float(face.confidence),
            'landmarks': face.landmarks.tolist(),
            'person_bbox': self._serialize_bbox(face.person_bbox) if face.person_bbox else None,
            'face_occluded': bool(face.face_occluded)
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
        """Serialize face detections to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to face detections."""
        return Serializers.json_deserialize(data)
    
    def _store_results(self, context: PipelineContext, results: Dict[str, Dict[str, Any]], config: dict) -> None:
        """Store faces in context."""
        context.insightface_faces = results
        
        total_faces = sum(len(r.get('faces', [])) for r in results.values())
        logger.info(f"Detected {total_faces} faces across {len(results)} images")
