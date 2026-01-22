"""
Face detection and embedding extraction for person clustering.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import logging

from sim_bench.specialized_models.base import SpecializedModel

logger = logging.getLogger(__name__)


class FaceModel(SpecializedModel):
    """
    Face detection and embedding extraction model.
    
    Extracts face embeddings for person clustering. Supports multiple backends.
    """
    
    def __init__(
        self,
        backend: str = 'deepface',
        model_name: Optional[str] = None,
        device: str = 'cpu',
        enable_cache: bool = True
    ):
        """
        Initialize face model.
        
        Args:
            backend: Backend to use ('deepface', 'insightface', or 'mediapipe')
            model_name: Specific model name (optional, uses backend default)
            device: Device for computation
            enable_cache: Whether to cache results
        """
        super().__init__(device=device, enable_cache=enable_cache)
        self.backend = backend
        self.model_name = model_name or self._get_default_model()
        self._model = None
        
    def _get_default_model(self) -> str:
        """Get default model name for backend."""
        defaults = {
            'deepface': 'VGG-Face',
            'insightface': 'arcface_r100_v1',
            'mediapipe': 'face_detection'
        }
        return defaults.get(self.backend, 'VGG-Face')
    
    def _load_model(self):
        """Lazy load the face model."""
        if self._model is not None:
            return

        backend_loaders = {
            'deepface': self._load_deepface,
            'insightface': self._load_insightface,
            'mediapipe': self._load_mediapipe
        }

        loader = backend_loaders.get(self.backend)
        if loader is None:
            raise ValueError(f"Unknown backend: {self.backend}")
        loader()

    def _load_deepface(self):
        """Load DeepFace backend."""
        from deepface import DeepFace
        self._model = DeepFace
        logger.info(f"Loaded DeepFace backend (model: {self.model_name})")

    def _load_insightface(self):
        """Load InsightFace backend."""
        import insightface
        self._model = insightface.app.FaceAnalysis()
        ctx_id = -1 if self.device == 'cpu' else 0
        self._model.prepare(ctx_id=ctx_id)
        logger.info("Loaded InsightFace backend")

    def _load_mediapipe(self):
        """Load MediaPipe backend."""
        import mediapipe as mp
        self._face_detection = mp.solutions.face_detection.FaceDetection()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh()
        logger.info("Loaded MediaPipe backend")
    
    def extract_embeddings(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract face embeddings from images.

        Args:
            image_paths: List of image file paths

        Returns:
            Dict mapping image_path -> embedding array
            Shape: [n_faces, embedding_dim] if multiple faces, or [embedding_dim] if single face
        """
        self._load_model()
        embeddings = {}

        for image_path in image_paths:
            result = self.process_image(image_path)
            face_embs = self._extract_embeddings_from_result(result)
            embeddings[image_path] = face_embs

        return embeddings

    def _extract_embeddings_from_result(self, result: Dict[str, Any]) -> np.ndarray:
        """Extract embeddings array from process_image result."""
        if not result.get('embeddings'):
            return np.array([]).reshape(0, 512)

        face_embs = np.array(result['embeddings'])
        if face_embs.ndim == 1:
            face_embs = face_embs.reshape(1, -1)
        return face_embs
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process single image and extract face information.

        Args:
            image_path: Path to image file

        Returns:
            Dict with:
                - embeddings: List of face embedding arrays
                - detections: List of face detection info (bbox, confidence)
                - face_count: Number of faces detected
        """
        self._load_model()
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        processors = {
            'deepface': self._process_deepface,
            'insightface': self._process_insightface,
            'mediapipe': self._process_mediapipe
        }

        processor = processors.get(self.backend)
        if processor is None:
            raise ValueError(f"Unsupported backend: {self.backend}")
        return processor(image_path)
    
    def _process_deepface(self, image_path: Path) -> Dict[str, Any]:
        """Process image using DeepFace backend."""
        from deepface import DeepFace

        objs = DeepFace.represent(
            str(image_path),
            model_name=self.model_name,
            enforce_detection=False,
            detector_backend='opencv'
        )

        embeddings = [np.array(obj['embedding']) for obj in objs]
        detections = [
            {
                'bbox': obj.get('facial_area', {}),
                'confidence': 1.0,
                'region': obj.get('region', {})
            }
            for obj in objs
        ]

        return {
            'embeddings': embeddings,
            'detections': detections,
            'face_count': len(embeddings)
        }
    
    def _process_insightface(self, image_path: Path) -> Dict[str, Any]:
        """Process image using InsightFace backend."""
        import cv2
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        faces = self._model.get(img)
        
        embeddings = []
        detections = []
        
        for face in faces:
            embeddings.append(face.embedding)
            detections.append({
                'bbox': face.bbox.tolist(),
                'confidence': float(face.det_score),
                'landmarks': face.landmark.tolist() if hasattr(face, 'landmark') else None
            })
        
        return {
            'embeddings': embeddings,
            'detections': detections,
            'face_count': len(embeddings)
        }
    
    def _process_mediapipe(self, image_path: Path) -> Dict[str, Any]:
        """Process image using MediaPipe backend."""
        import cv2
        from mediapipe.python.solutions import drawing_utils
        
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detection_results = self._face_detection.process(img_rgb)
        
        embeddings = []
        detections = []
        
        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                detections.append({
                    'bbox': {
                        'x': bbox.xmin,
                        'y': bbox.ymin,
                        'w': bbox.width,
                        'h': bbox.height
                    },
                    'confidence': detection.score[0] if detection.score else 0.0
                })
                
                # MediaPipe doesn't provide embeddings directly
                # Would need additional model for embeddings
                embeddings.append(np.zeros(512))  # Placeholder
        
        return {
            'embeddings': embeddings,
            'detections': detections,
            'face_count': len(embeddings),
            'note': 'MediaPipe provides detection only, embeddings are placeholders'
        }
    
    def _get_routing_key(self) -> str:
        """Get routing key for face detection."""
        return 'face_detection'
    
    def __repr__(self) -> str:
        return f"FaceModel(backend={self.backend}, model={self.model_name}, device={self.device})"






