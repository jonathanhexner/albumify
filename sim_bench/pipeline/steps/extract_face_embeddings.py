"""Extract Face Embeddings step - ArcFace embeddings for face clustering.

Supports two backends via strategy pattern:
- custom: Your trained ArcFace model (checkpoint_path required)
- insightface: InsightFace's built-in w600k_r50 (rotation invariant, 600K identities)

Features:
- Skips faces marked as non-clusterable (filter_passed=False or is_clusterable=False)
- Applies roll alignment to rotate face crops for horizontal eye alignment
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np

from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.utils.image_cache import get_image_cache
from sim_bench.pipeline.face_embedding.base import BaseFaceEmbeddingExtractor
from sim_bench.pipeline.face_embedding.factory import FaceEmbeddingExtractorFactory

logger = logging.getLogger(__name__)


def align_face_by_roll(face_image: np.ndarray, roll_angle: float, threshold: float = 5.0) -> np.ndarray:
    """Rotate face crop to make eyes horizontal.

    Args:
        face_image: Face crop as numpy array (H, W, C)
        roll_angle: Roll angle in degrees (positive = clockwise tilt)
        threshold: Skip rotation if angle is below this threshold

    Returns:
        Aligned face image (same size as input)
    """
    if abs(roll_angle) < threshold:
        return face_image

    h, w = face_image.shape[:2]
    center = (w // 2, h // 2)

    # Rotate to counter the roll
    rotation_matrix = cv2.getRotationMatrix2D(center, roll_angle, 1.0)
    aligned = cv2.warpAffine(
        face_image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned


@register_step
class ExtractFaceEmbeddingsStep(BaseStep):
    """Extract face embeddings using configurable backend.

    Backends:
        - custom: Uses checkpoint_path to load your trained ArcFace model
        - insightface: Uses InsightFace's built-in w600k_r50 (better rotation invariance)
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="extract_face_embeddings",
            display_name="Extract Face Embeddings",
            description="Extract face embeddings for recognition and clustering.",
            category="people",
            requires=set(),  # faces is optional - step skips if none
            produces={"face_embeddings"},
            depends_on=["score_face_frontal"],  # Must run after filtering and frontal scoring
            config_schema={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["custom", "insightface"],
                        "default": "insightface",
                        "description": "Embedding extraction backend"
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to custom ArcFace checkpoint (for custom backend)"
                    },
                    "device": {
                        "type": "string",
                        "enum": ["cpu", "cuda", "mps"],
                        "default": "cpu",
                        "description": "Device to run model on"
                    },
                    "model_name": {
                        "type": "string",
                        "default": "buffalo_l",
                        "description": "InsightFace model name (for insightface backend)"
                    }
                }
            }
        )
        self._extractor: Optional[BaseFaceEmbeddingExtractor] = None
        self._extractor_config: Optional[Dict[str, Any]] = None

    def _get_extractor(self, config: dict) -> BaseFaceEmbeddingExtractor:
        """Lazy load extractor using factory.

        Caches the extractor instance for the same config.
        """
        # Check if we need to create a new extractor
        if self._extractor is None or self._extractor_config != config:
            self._extractor = FaceEmbeddingExtractorFactory.create(config)
            self._extractor_config = config.copy()
            logger.info(f"Using face embedding backend: {self._extractor.model_name}")
        return self._extractor

    def _generate_cache_key(self, face) -> str:
        """Generate unique cache key for a face."""
        # Normalize path to string for consistent keys
        path_str = str(face.original_path).replace('\\', '/')
        return f"{path_str}:face_{face.face_index}"
    
    def _get_all_faces(self, context: PipelineContext) -> List:
        """Get all faces from context (MediaPipe or InsightFace)."""
        # Check for pre-flattened list
        if context.all_faces:
            return context.all_faces

        # Check MediaPipe faces
        if context.faces:
            all_faces = []
            for faces_list in context.faces.values():
                all_faces.extend(faces_list)
            return all_faces

        # Check InsightFace faces - convert to compatible format
        if hasattr(context, 'insightface_faces') and context.insightface_faces:
            from sim_bench.face_pipeline.types import CroppedFace, BoundingBox
            from PIL import Image
            all_faces = []
            skipped = 0
            skipped_non_clusterable = 0
            cache = get_image_cache()

            for image_path, face_data in context.insightface_faces.items():
                if not Path(image_path).exists():
                    logger.warning(f"Image not found: {image_path}")
                    skipped += 1
                    continue

                for face_info in face_data.get('faces', []):
                    # Skip faces that didn't pass filtering or aren't clusterable
                    if not face_info.get('filter_passed', True):
                        skipped_non_clusterable += 1
                        continue
                    if not face_info.get('is_clusterable', True):
                        skipped_non_clusterable += 1
                        continue
                    bbox_data = face_info.get('bbox', {})
                    x_px = bbox_data.get('x_px', 0)
                    y_px = bbox_data.get('y_px', 0)
                    w_px = bbox_data.get('w_px', 0)
                    h_px = bbox_data.get('h_px', 0)

                    # Skip faces with invalid bbox
                    if w_px <= 0 or h_px <= 0:
                        logger.warning(f"Invalid bbox for face in {image_path}: w={w_px}, h={h_px}")
                        skipped += 1
                        continue

                    # Crop face from original image (EXIF-normalized via global cache)
                    face_image = None
                    try:
                        img = cache.get_pil(image_path)
                        pad = int(min(w_px, h_px) * 0.2)
                        left = max(0, x_px - pad)
                        top = max(0, y_px - pad)
                        right = min(img.width, x_px + w_px + pad)
                        bottom = min(img.height, y_px + h_px + pad)

                        # Validate crop coordinates
                        if right <= left or bottom <= top:
                            logger.warning(
                                f"Invalid crop coordinates for {image_path}: "
                                f"left={left}, top={top}, right={right}, bottom={bottom}"
                            )
                            skipped += 1
                            continue

                        face_crop = img.crop((left, top, right, bottom))
                        face_image = np.array(face_crop.convert('RGB'))

                        # Apply roll alignment if angle is available
                        roll_angle = face_info.get('roll_angle', 0.0)
                        if abs(roll_angle) >= 5.0:
                            face_image = align_face_by_roll(face_image, roll_angle)
                    except Exception as e:
                        logger.warning(f"Failed to crop face from {image_path}: {e}")
                        skipped += 1
                        continue

                    if face_image is None or face_image.size == 0:
                        skipped += 1
                        continue

                    bbox = BoundingBox(
                        x=bbox_data.get('x', 0),
                        y=bbox_data.get('y', 0),
                        w=bbox_data.get('w', 0),
                        h=bbox_data.get('h', 0),
                        x_px=x_px, y_px=y_px, w_px=w_px, h_px=h_px,
                    )
                    face = CroppedFace(
                        original_path=Path(image_path),
                        face_index=face_info.get('face_index', 0),
                        image=face_image,
                        bbox=bbox,
                        detection_confidence=face_info.get('confidence', 0),
                        face_ratio=0,
                    )
                    all_faces.append(face)

            logger.info("=" * 60)
            logger.info("EXTRACT_FACE_EMBEDDINGS: Face selection")
            logger.info("=" * 60)
            logger.info(f"Total faces in context: {sum(len(fd.get('faces', [])) for fd in context.insightface_faces.values())}")
            logger.info(f"Selected for embedding: {len(all_faces)}")
            logger.info(f"Skipped (invalid bbox): {skipped}")
            logger.info(f"Skipped (filter_passed=False or is_clusterable=False): {skipped_non_clusterable}")
            if skipped_non_clusterable > 0:
                logger.info("  -> Faces were filtered by filter_faces or score_face_frontal steps")
            logger.info("=" * 60)
            return all_faces

        return []

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face embedding extraction."""
        # Check for faces first - if no faces, skip gracefully
        all_faces = self._get_all_faces(context)
        if not all_faces:
            return None

        # Get extractor to determine model name for cache key
        extractor = self._get_extractor(config)

        # Use cache key strings as items
        cache_keys = [self._generate_cache_key(f) for f in all_faces]

        return {
            "items": cache_keys,
            "feature_type": "face_embedding",
            "model_name": extractor.model_name,
            "metadata": {}
        }
    
    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, np.ndarray]:
        """Process uncached items - extract face embeddings."""
        all_faces = self._get_all_faces(context)

        # Map cache keys back to faces
        key_to_face = {
            self._generate_cache_key(f): f
            for f in all_faces
        }
        
        uncached_faces = [key_to_face[key] for key in items if key in key_to_face]

        if not uncached_faces:
            return {}

        # Filter out faces with no image data
        valid_faces = [f for f in uncached_faces if f.image is not None]
        if not valid_faces:
            logger.warning(f"No valid face images found among {len(uncached_faces)} faces")
            return {}

        logger.info(f"Processing {len(valid_faces)} faces with valid images (skipped {len(uncached_faces) - len(valid_faces)})")

        # Get extractor using factory
        extractor = self._get_extractor(config)

        # Extract in batch - only valid faces
        face_images = [face.image for face in valid_faces]
        face_metadata = [{"path": str(f.original_path), "face_index": f.face_index} for f in valid_faces]
        
        embeddings_list = extractor.extract_batch(face_images, face_metadata)

        # Update uncached_faces reference to only valid ones
        uncached_faces = valid_faces
        
        results = {}
        for i, (face, embedding) in enumerate(zip(uncached_faces, embeddings_list)):
            face.embedding = embedding
            key = self._generate_cache_key(face)
            results[key] = embedding
            
            progress = (i + 1) / len(uncached_faces)
            context.report_progress(
                "extract_face_embeddings", progress,
                f"Embedding {i + 1}/{len(uncached_faces)}"
            )
        
        return results
    
    def _serialize_for_cache(self, result: np.ndarray, item: str) -> bytes:
        """Serialize numpy array to bytes."""
        return Serializers.numpy_serialize(result)
    
    def _deserialize_from_cache(self, data: bytes, item: str) -> np.ndarray:
        """Deserialize bytes to numpy array."""
        return Serializers.numpy_deserialize(data)
    
    def _store_results(
        self,
        context: PipelineContext,
        results: Dict[str, np.ndarray],
        config: dict
    ) -> None:
        """Store face embeddings in context and update face objects."""
        all_faces = self._get_all_faces(context)
        logger.info(f"Storing {len(results)} face embeddings for {len(all_faces)} faces")

        # Update face objects with embeddings
        key_to_face = {
            self._generate_cache_key(f): f
            for f in all_faces
        }

        for key, embedding in results.items():
            if key in key_to_face:
                key_to_face[key].embedding = embedding

        # Store in context dict
        context.face_embeddings = {
            key: embedding
            for key, embedding in results.items()
        }

        if results:
            sample_keys = list(results.keys())[:3]
            logger.info(f"Sample embedding keys stored: {sample_keys}")
