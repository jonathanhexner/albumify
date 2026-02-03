"""End-to-end face processing pipeline."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from sim_bench.face_pipeline.crop_service import FaceCropService
from sim_bench.face_pipeline.quality_scorer import FaceQualityScorer
from sim_bench.face_pipeline.types import (
    AlbumFaceResult,
    FaceCluster,
    CroppedFace
)
from sim_bench.album.services.face_embedding_service import FaceEmbeddingService

logger = logging.getLogger(__name__)


class FacePipelineService:
    """Complete face processing pipeline."""

    def __init__(
        self,
        config: Dict[str, Any],
        crop_service: Optional[FaceCropService] = None,
        quality_scorer: Optional[FaceQualityScorer] = None,
        face_embedder: Optional[FaceEmbeddingService] = None
    ):
        """
        Initialize pipeline with configuration and optional injected services.

        Args:
            config: Configuration dict
            crop_service: Optional FaceCropService
            quality_scorer: Optional FaceQualityScorer
            face_embedder: Optional FaceEmbeddingService
        """
        self._config = config
        fp_config = config.get('face_pipeline', {})

        self._crop_service = crop_service or FaceCropService(config)
        self._quality_scorer = quality_scorer or FaceQualityScorer(config)
        self._face_embedder = face_embedder or FaceEmbeddingService(config)

        self._embedding_batch_size = fp_config.get('embedding_batch_size', 32)
        self._cluster_distance_threshold = fp_config.get('cluster_distance_threshold', 0.5)
        self._enable_embedding_cache = fp_config.get('enable_embedding_cache', True)
        self._embedding_cache = {}

        logger.info("FacePipelineService initialized")

    def _collect_faces(self, image_paths: List[Path]) -> tuple:
        """Crop faces from images and track images without faces."""
        all_faces = []
        images_without_faces = []

        for path in image_paths:
            faces = self._crop_service.crop_faces(path)
            if not faces:
                images_without_faces.append(path)
                continue
            all_faces.extend(faces)

        return all_faces, images_without_faces

    def _embed_faces(self, faces: List[CroppedFace]):
        """Attach embeddings to faces (with optional in-memory cache)."""
        faces_to_embed = []
        for face in faces:
            if self._enable_embedding_cache and face.crop_key in self._embedding_cache:
                face.embedding = self._embedding_cache[face.crop_key]
                continue
            faces_to_embed.append(face)

        if not faces_to_embed:
            return

        images = [f.image for f in faces_to_embed]
        embeddings = self._face_embedder.extract_embeddings_batch(
            images,
            batch_size=self._embedding_batch_size,
            show_progress=False
        )

        for face, emb in zip(faces_to_embed, embeddings):
            face.embedding = emb
            if self._enable_embedding_cache:
                self._embedding_cache[face.crop_key] = emb

    def _build_clusters(self, faces: List[CroppedFace]) -> List[FaceCluster]:
        """Cluster faces by identity and build cluster objects."""
        embeddings = [f.embedding for f in faces if f.embedding is not None]
        if not embeddings:
            return []

        emb_array = np.stack(embeddings, axis=0)
        labels = self._face_embedder.cluster_faces(
            emb_array,
            distance_threshold=self._cluster_distance_threshold
        )

        clusters = {}
        for face, label in zip(faces, labels):
            clusters.setdefault(int(label), []).append(face)
            face.cluster_id = int(label)

        result = []
        for cluster_id, cluster_faces in clusters.items():
            cluster = FaceCluster(cluster_id=cluster_id, faces=cluster_faces)
            cluster.compute_centroid()
            cluster.select_representative()
            result.append(cluster)

        return result

    def process_album(self, image_paths: List[Path]) -> AlbumFaceResult:
        """
        Process a list of images through the full face pipeline.

        Returns:
            AlbumFaceResult with faces, clusters, and summary stats
        """
        image_paths = [Path(p) for p in image_paths]
        total_images = len(image_paths)

        all_faces, images_without_faces = self._collect_faces(image_paths)

        if all_faces:
            self._quality_scorer.score_faces(all_faces)
            self._embed_faces(all_faces)
            clusters = self._build_clusters(all_faces)
        else:
            clusters = []

        images_with_faces = len(set(f.original_path for f in all_faces))
        faces_meeting_threshold = len(all_faces)

        result = AlbumFaceResult(
            all_faces=all_faces,
            clusters=clusters,
            images_without_faces=images_without_faces,
            total_images=total_images,
            images_with_faces=images_with_faces,
            total_faces_detected=len(all_faces),
            faces_meeting_threshold=faces_meeting_threshold
        )

        logger.info(
            f"Face pipeline complete: images={total_images}, "
            f"faces={len(all_faces)}, clusters={len(clusters)}"
        )

        return result
