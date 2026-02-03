"""Detect Faces step - MediaPipe face detection with caching."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


@register_step
class DetectFacesStep(BaseStep):
    """Detect and crop faces from images using MediaPipe."""

    def __init__(self):
        self._metadata = StepMetadata(
            name="detect_faces",
            display_name="Detect Faces",
            description="Detect faces using MediaPipe and store metadata for downstream processing.",
            category="people",
            requires={"image_paths"},
            produces={"faces"},
            depends_on=["discover_images"],
            config_schema={
                "type": "object",
                "properties": {
                    "min_face_ratio": {
                        "type": "number",
                        "default": 0.02,
                        "description": "Minimum face size as ratio of image area (0.02 = 2%)"
                    },
                    "detection_confidence": {
                        "type": "number",
                        "default": 0.5,
                        "description": "MediaPipe detection confidence threshold"
                    },
                    "crop_padding": {
                        "type": "number",
                        "default": 0.2,
                        "description": "Padding around face crop (0.2 = 20% extra)"
                    }
                }
            }
        )
        self._crop_service = None

    def _get_crop_service(self, config: dict):
        """Lazy load face crop service."""
        if self._crop_service is None:
            from sim_bench.face_pipeline.crop_service import FaceCropService
            logger.info("Loading MediaPipe face detection")
            self._crop_service = FaceCropService(config={'face_pipeline': config})
        return self._crop_service

    def _serialize_faces(self, cropped_faces) -> Dict[str, Any]:
        """Convert CroppedFace objects to JSON-serializable dict."""
        return {
            'faces': [
                {
                    'face_index': f.face_index,
                    'bbox': {
                        'x': f.bbox.x,
                        'y': f.bbox.y,
                        'w': f.bbox.w,
                        'h': f.bbox.h,
                        'x_px': f.bbox.x_px,
                        'y_px': f.bbox.y_px,
                        'w_px': f.bbox.w_px,
                        'h_px': f.bbox.h_px
                    },
                    'detection_confidence': f.detection_confidence,
                    'face_ratio': f.face_ratio
                }
                for f in cropped_faces
            ]
        }

    def _deserialize_faces(self, image_path: str, data: Dict[str, Any], service):
        """Reconstruct CroppedFace objects from cached JSON."""
        from sim_bench.face_pipeline.types import CroppedFace, BoundingBox
        from PIL import Image
        import numpy as np
        
        faces = []
        image = service._load_image(Path(image_path))
        
        for face_data in data['faces']:
            bbox_data = face_data['bbox']
            bbox = BoundingBox(
                x=bbox_data['x'],
                y=bbox_data['y'],
                w=bbox_data['w'],
                h=bbox_data['h'],
                x_px=bbox_data['x_px'],
                y_px=bbox_data['y_px'],
                w_px=bbox_data['w_px'],
                h_px=bbox_data['h_px']
            )
            
            crop = service._crop_face(image, bbox, service._crop_padding)
            
            faces.append(CroppedFace(
                original_path=Path(image_path),
                face_index=face_data['face_index'],
                image=crop,
                bbox=bbox,
                detection_confidence=face_data['detection_confidence'],
                face_ratio=face_data['face_ratio']
            ))
        
        return faces

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face detection."""
        image_paths = [str(p) for p in context.image_paths]
        if not image_paths:
            return None
        
        return {
            "items": image_paths,
            "feature_type": "face_detection",
            "model_name": "mediapipe",
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, Dict[str, Any]]:
        """Process uncached items - detect faces."""
        min_face_ratio = config.get("min_face_ratio", 0.02)
        service = self._get_crop_service(config)
        
        results = {}
        for i, path_str in enumerate(items):
            cropped_faces = service.crop_faces(Path(path_str), min_face_ratio)
            results[path_str] = self._serialize_faces(cropped_faces)
            
            progress = (i + 1) / len(items)
            context.report_progress(
                "detect_faces", progress,
                f"Detecting {i + 1}/{len(items)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: Dict[str, Any], item: str) -> bytes:
        """Serialize face detection dict to JSON bytes."""
        return Serializers.json_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> Dict[str, Any]:
        """Deserialize JSON bytes to face detection dict."""
        return Serializers.json_deserialize(data)
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, Dict[str, Any]],
        config: dict
    ) -> None:
        """Store faces in context, reconstructing CroppedFace objects."""
        service = self._get_crop_service(config)
        all_faces = []
        
        # Reconstruct CroppedFace objects from cached data
        for path_str, detection_data in results.items():
            faces = self._deserialize_faces(path_str, detection_data, service)
            all_faces.extend(faces)
        
        # Group faces by image path
        context.faces = {}
        for face in all_faces:
            path_key = str(face.original_path)
            if path_key not in context.faces:
                context.faces[path_key] = []
            context.faces[path_key].append(face)
