"""Extract Face Embeddings step - ArcFace embeddings for face clustering.

Single responsibility: Extract embeddings from pre-aligned face crops.

Expects:
- context.aligned_faces: Dict[str, np.ndarray] from align_faces step
- context.insightface_faces: Face metadata for filtering

Does NOT:
- Perform any alignment
- Load images
- Handle multiple backends for alignment
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

from sim_bench.face_pipeline.types import CroppedFace, BoundingBox
from sim_bench.pipeline.base import BaseStep, StepMetadata
from sim_bench.pipeline.context import PipelineContext
from sim_bench.pipeline.registry import register_step
from sim_bench.pipeline.serializers import Serializers
from sim_bench.pipeline.face_embedding.base import BaseFaceEmbeddingExtractor
from sim_bench.pipeline.face_embedding.factory import FaceEmbeddingExtractorFactory

logger = logging.getLogger(__name__)

# spec-079 / SIGHTING-099: output-schema version for the embedding cache. Bump
# when the embedding representation changes (model, normalization, dtype). "v1"
# tags the current normalized-arcface output; legacy rows (model_version=None)
# are recomputed by base.py. Stale embeddings were the actual cause of the
# Albumify 8-vs-12 over-split (cached rows differed from live computation).
EMBEDDING_OUTPUT_VERSION = "emb-v1-arcface-norm"


@register_step
class ExtractFaceEmbeddingsStep(BaseStep):
    """Extract face embeddings from pre-aligned face crops.

    Requires align_faces step to run first.
    """

    def __init__(self):
        self._metadata = StepMetadata(
            name="extract_face_embeddings",
            display_name="Extract Face Embeddings",
            description="Extract face embeddings from aligned face crops.",
            category="people",
            # spec-041 audit fix: also reads context.face_records (the A1
            # dual-write loop) and mutates each FaceRecord's embedding /
            # embedding_normalized fields. Declaring both sides so the
            # telemetry and dependency resolution are honest.
            requires={"aligned_faces", "face_records"},
            produces={"face_embeddings", "face_records"},
            depends_on=["align_faces"],
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
        """Lazy load extractor using factory."""
        if self._extractor is None or self._extractor_config != config:
            self._extractor = FaceEmbeddingExtractorFactory.create(config)
            self._extractor_config = config.copy()
            logger.info(f"Using face embedding backend: {self._extractor.model_name}")
        return self._extractor

    def release(self) -> None:
        """SIGHTING-117: free the ArcFace/InsightFace embedding model after
        extraction. This step was MISSED in the original release-per-step pass
        (it had no override), so its model stayed resident for the whole run —
        measured ~360 MB of leaked RSS. Downstream reads context.face_embeddings /
        face_records, not this extractor. Unlike the torch steps, this backend is
        ONNX, which does drop on gc (verified: insightface_detect_faces frees
        cleanly), so nulling the handle genuinely reclaims the memory.
        _extractor_config is reset so a later run re-lazy-loads via _get_extractor."""
        self._release_models("_extractor")
        self._extractor_config = None

    def _get_all_faces(self, context: PipelineContext) -> List[CroppedFace]:
        """Get all aligned faces from context."""
        if not hasattr(context, 'aligned_faces') or not context.aligned_faces:
            logger.warning("No aligned_faces in context. align_faces step must run first.")
            return []

        if not hasattr(context, 'insightface_faces') or not context.insightface_faces:
            logger.warning("No insightface_faces in context for metadata.")
            return []

        all_faces = []
        skipped_no_alignment = 0

        for image_path, face_data in context.insightface_faces.items():
            for face_info in face_data.get('faces', []):
                face_idx = face_info.get('face_index', 0)
                face_key = f"{image_path}:face_{face_idx}"

                # Skip faces that aren't clusterable
                if not face_info.get('filter_passed', True):
                    continue
                if not face_info.get('is_clusterable', True):
                    continue

                # Get aligned crop from context
                aligned_crop = context.aligned_faces.get(face_key)
                if aligned_crop is None:
                    skipped_no_alignment += 1
                    continue

                bbox_data = face_info.get('bbox', {})
                bbox = BoundingBox(
                    x=bbox_data.get('x', 0),
                    y=bbox_data.get('y', 0),
                    w=bbox_data.get('w', 0),
                    h=bbox_data.get('h', 0),
                    x_px=bbox_data.get('x_px', 0),
                    y_px=bbox_data.get('y_px', 0),
                    w_px=bbox_data.get('w_px', 0),
                    h_px=bbox_data.get('h_px', 0),
                )

                face = CroppedFace(
                    original_path=Path(image_path),
                    face_index=face_idx,
                    image=aligned_crop,
                    bbox=bbox,
                    detection_confidence=face_info.get('confidence', 0),
                    face_ratio=0,
                )
                all_faces.append(face)

        logger.info("=" * 60)
        logger.info("EXTRACT_FACE_EMBEDDINGS: Face selection")
        logger.info("=" * 60)
        logger.info(f"Aligned faces available: {len(context.aligned_faces)}")
        logger.info(f"Selected for embedding: {len(all_faces)}")
        logger.info(f"Skipped (no alignment): {skipped_no_alignment}")
        logger.info("=" * 60)

        return all_faces

    def _generate_cache_key(self, face: CroppedFace) -> str:
        """Generate unique cache key for a face."""
        path_str = str(face.original_path).replace('\\', '/')
        return f"{path_str}:face_{face.face_index}"

    def _get_cache_config(
        self,
        context: PipelineContext,
        config: dict
    ) -> Optional[Dict[str, Any]]:
        """Get cache configuration for face embedding extraction."""
        all_faces = self._get_all_faces(context)
        if not all_faces:
            return None

        extractor = self._get_extractor(config)
        cache_keys = [self._generate_cache_key(f) for f in all_faces]

        return {
            "items": cache_keys,
            "feature_type": "face_embedding",
            "model_name": extractor.model_name,
            # spec-079 / SIGHTING-099: schema version → stale rows recompute.
            "metadata": {"model_version": EMBEDDING_OUTPUT_VERSION},
        }

    def _process_uncached(
        self,
        items: List[str],
        context: PipelineContext,
        config: dict
    ) -> Dict[str, np.ndarray]:
        """Extract embeddings from uncached faces."""
        all_faces = self._get_all_faces(context)
        key_to_face = {self._generate_cache_key(f): f for f in all_faces}

        uncached_faces = [key_to_face[key] for key in items if key in key_to_face]
        if not uncached_faces:
            return {}

        valid_faces = [f for f in uncached_faces if f.image is not None and f.image.size > 0]
        if not valid_faces:
            logger.warning(f"No valid face images among {len(uncached_faces)} faces")
            return {}

        logger.info(f"Extracting embeddings for {len(valid_faces)} faces")

        extractor = self._get_extractor(config)
        face_images = [face.image for face in valid_faces]
        face_metadata = [{"path": str(f.original_path), "face_index": f.face_index} for f in valid_faces]

        embeddings_list = extractor.extract_batch(face_images, face_metadata)

        results = {}
        for i, (face, embedding) in enumerate(zip(valid_faces, embeddings_list)):
            face.embedding = embedding
            key = self._generate_cache_key(face)
            results[key] = embedding

            progress = (i + 1) / len(valid_faces)
            context.report_progress(
                "extract_face_embeddings", progress,
                f"Embedding {i + 1}/{len(valid_faces)}"
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
        """Store face embeddings in context."""
        all_faces = self._get_all_faces(context)
        key_to_face = {self._generate_cache_key(f): f for f in all_faces}

        for key, embedding in results.items():
            if key in key_to_face:
                key_to_face[key].embedding = embedding

        context.face_embeddings = dict(results)

        # spec-040 A1: mirror embeddings onto context.face_records so the v2
        # clustering chain has the data it needs. Cache key format is
        # ``{image_path}:face_{face_index}`` — see _generate_cache_key.
        record_index = {
            (r.image_path, r.face_index): r
            for r in (context.face_records or [])
            if r.image_path is not None and r.face_index is not None
        }
        for key, embedding in results.items():
            path_part, _, idx_part = key.rpartition(":face_")
            if not idx_part:
                continue
            try:
                face_idx = int(idx_part)
            except ValueError:
                continue
            record = record_index.get((path_part, face_idx))
            if record is None:
                continue
            record.embedding = embedding
            norm = float(np.linalg.norm(embedding)) if embedding is not None else 0.0
            if norm > 0:
                record.embedding_normalized = embedding / norm

        logger.info(f"Stored {len(results)} face embeddings")
