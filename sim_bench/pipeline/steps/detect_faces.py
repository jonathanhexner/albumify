"""Detect Faces step - MediaPipe face detection with caching."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers

logger = logging.getLogger(__name__)


def _get_faces_dir(image_path: Path) -> Path:
    """Get the .faces directory for storing cropped faces."""
    return image_path.parent / ".faces"


def _get_face_crop_path(image_path: Path, face_index: int) -> Path:
    """Get the path for a cropped face image."""
    faces_dir = _get_faces_dir(image_path)
    return faces_dir / f"{image_path.stem}_face_{face_index}.jpg"


def _save_face_crop(face_image: np.ndarray, save_path: Path) -> bool:
    """Save a cropped face image to disk."""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy array to PIL Image and save
        if face_image.dtype != np.uint8:
            face_image = (face_image * 255).astype(np.uint8)
        img = Image.fromarray(face_image)
        img.save(save_path, "JPEG", quality=90)
        return True
    except Exception as e:
        logger.warning(f"Failed to save face crop to {save_path}: {e}")
        return False


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
                        "default": 0.3,
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

    def _serialize_faces(self, cropped_faces, save_crops: bool = True) -> Dict[str, Any]:
        """Convert CroppedFace objects to JSON-serializable dict and optionally save crops."""
        faces_data = []
        for f in cropped_faces:
            # Determine crop path and save if requested
            crop_path = _get_face_crop_path(f.original_path, f.face_index)
            crop_path_str = None

            if save_crops and f.image is not None:
                if _save_face_crop(f.image, crop_path):
                    crop_path_str = str(crop_path)

            faces_data.append({
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
                'face_ratio': f.face_ratio,
                'crop_path': crop_path_str
            })

        return {'faces': faces_data}

    def _deserialize_faces(self, image_path: str, data: Dict[str, Any], service):
        """Reconstruct CroppedFace objects from cached JSON."""
        from sim_bench.face_pipeline.types import CroppedFace, BoundingBox

        faces = []
        image = None  # Lazy load only if needed

        for face_data in data['faces']:
            bbox = BoundingBox(**face_data['bbox'])
            crop_path = Path(face_data['crop_path']) if face_data.get('crop_path') else \
                        _get_face_crop_path(Path(image_path), face_data['face_index'])

            # Load from saved crop, or crop from original and save
            crop = self._load_or_create_crop(crop_path, image_path, bbox, service)
            if image is None and crop is None:
                image = service._load_image(Path(image_path))
                crop = service._crop_face(image, bbox, service._crop_padding)
                _save_face_crop(crop, crop_path)

            faces.append(CroppedFace(
                original_path=Path(image_path),
                face_index=face_data['face_index'],
                image=crop,
                bbox=bbox,
                detection_confidence=face_data['detection_confidence'],
                face_ratio=face_data['face_ratio'],
                crop_path=crop_path
            ))

        return faces

    def _load_or_create_crop(self, crop_path: Path, image_path: str, bbox, service):
        """Load crop from disk if exists, otherwise return None."""
        if not crop_path.exists():
            return None
        try:
            return np.array(Image.open(crop_path))
        except Exception as e:
            logger.warning(f"Failed to load crop {crop_path}: {e}")
            return None

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
