"""InsightFace native extractor using built-in w600k_r50 ArcFace."""

import logging
from typing import List, Dict, Any

import cv2
import numpy as np

from sim_bench.pipeline.face_embedding.base import BaseFaceEmbeddingExtractor

logger = logging.getLogger(__name__)


class InsightFaceNativeExtractor(BaseFaceEmbeddingExtractor):
    """Extractor using InsightFace's built-in w600k_r50 ArcFace.

    This model is trained on WebFace600K (600K+ identities) with extensive
    augmentation including rotation, making it more robust to head pose variations
    than models trained on smaller datasets without augmentation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name_config = config.get("model_name", "buffalo_l")
        self._app = None
        self._rec_model = None

    def _get_app(self):
        """Lazy load InsightFace FaceAnalysis app."""
        if self._app is None:
            from insightface.app import FaceAnalysis

            providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            logger.info(f"Loading InsightFace model '{self.model_name_config}' with providers: {providers}")

            self._app = FaceAnalysis(name=self.model_name_config, providers=providers)
            ctx_id = 0 if self.device == 'cuda' else -1
            self._app.prepare(ctx_id=ctx_id)
            
            # Get the recognition model directly
            self._rec_model = None
            for model in self._app.models.values():
                if hasattr(model, 'get_feat'):
                    self._rec_model = model
                    logger.info(f"Found recognition model: {type(model).__name__}")
                    break
            
            if self._rec_model is None:
                logger.warning("No recognition model found, will fall back to full detection")

        return self._app

    def extract_batch(
        self,
        face_images: List[np.ndarray],
        face_metadata: List[Dict[str, Any]]
    ) -> List[np.ndarray]:
        """Extract embeddings from already-cropped face images.

        These are pre-cropped faces, so we use the recognition model directly
        rather than re-running face detection.
        
        InsightFace expects BGR images in [H, W, C] format.
        """
        app = self._get_app()
        embeddings = []

        # Try to use recognition model directly (faster, more reliable for crops)
        if self._rec_model is not None:
            logger.info(f"Using direct recognition model for {len(face_images)} faces")
            for idx, face_img in enumerate(face_images):
                # Validate input
                if face_img is None or face_img.size == 0:
                    logger.warning(f"Face {idx}: Empty or None image, using zero vector")
                    embeddings.append(np.zeros(512, dtype=np.float32))
                    continue

                # InsightFace expects BGR format [H, W, C]
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                    # Assume RGB input, convert to BGR
                    face_bgr = face_img[:, :, ::-1].copy()
                elif len(face_img.shape) == 2:
                    # Grayscale image - convert to BGR
                    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                else:
                    logger.warning(f"Face {idx}: Unexpected shape {face_img.shape}, using zero vector")
                    embeddings.append(np.zeros(512, dtype=np.float32))
                    continue

                # Resize to expected input size (112x112 for most InsightFace models)
                face_resized = cv2.resize(face_bgr, (112, 112))
                
                # Get embedding directly from recognition model
                embedding = self._rec_model.get_feat([face_resized])[0]
                # Normalize
                embedding_norm = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding_norm.astype(np.float32))

        else:
            # Fallback: use full face analysis (slower, may fail on tight crops)
            logger.info(f"Using fallback face analysis for {len(face_images)} faces")
            for idx, face_img in enumerate(face_images):
                # Validate input
                if face_img is None or face_img.size == 0:
                    logger.warning(f"Face {idx}: Empty or None image, using zero vector")
                    embeddings.append(np.zeros(512, dtype=np.float32))
                    continue

                # InsightFace expects BGR format [H, W, C]
                if len(face_img.shape) == 3 and face_img.shape[2] == 3:
                    face_bgr = face_img[:, :, ::-1].copy()
                elif len(face_img.shape) == 2:
                    face_bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                else:
                    logger.warning(f"Face {idx}: Unexpected shape {face_img.shape}, using zero vector")
                    embeddings.append(np.zeros(512, dtype=np.float32))
                    continue

                # Run face analysis on the crop
                faces = app.get(face_bgr)

                if faces and hasattr(faces[0], 'normed_embedding') and faces[0].normed_embedding is not None:
                    embeddings.append(faces[0].normed_embedding.astype(np.float32))
                else:
                    logger.warning(f"Face {idx}: Couldn't extract embedding, using zero vector")
                    embeddings.append(np.zeros(512, dtype=np.float32))

        return embeddings

    def extract_single(
        self,
        face_image: np.ndarray,
        face_metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Extract embedding for a single face."""
        return self.extract_batch([face_image], [face_metadata])[0]

    @property
    def embedding_dim(self) -> int:
        """InsightFace w600k_r50 produces 512-dim embeddings."""
        return 512

    @property
    def model_name(self) -> str:
        """Cache key identifier."""
        return "arcface_insightface"
