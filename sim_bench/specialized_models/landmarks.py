"""
Landmark/place recognition and embedding extraction.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import logging

from sim_bench.specialized_models.base import SpecializedModel
from sim_bench.vision_language import CLIPModel

logger = logging.getLogger(__name__)


class LandmarkModel(SpecializedModel):
    """
    Landmark/place recognition model.
    
    Extracts place-specific embeddings for location-based clustering.
    Uses enhanced CLIP embeddings with landmark-specific prompts.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        pretrained: Optional[str] = None,
        device: str = 'cpu',
        enable_cache: bool = True
    ):
        """
        Initialize landmark model.
        
        Args:
            model_name: CLIP model name (default: ViT-B-32)
            pretrained: Pretrained checkpoint (default: laion2b_s34b_b79k)
            device: Device for computation
            enable_cache: Whether to cache results
        """
        super().__init__(device=device, enable_cache=enable_cache)
        self.model_name = model_name or 'ViT-B-32'
        self.pretrained = pretrained or 'laion2b_s34b_b79k'
        self._clip_model = None
        
        # Landmark-specific prompts for enhanced recognition
        self.landmark_prompts = [
            "a famous landmark",
            "a historical monument",
            "a tourist attraction",
            "a recognizable building",
            "a well-known place",
            "an architectural landmark",
            "a city landmark",
            "a cultural site"
        ]
    
    def _load_model(self):
        """Lazy load the CLIP model."""
        if self._clip_model is not None:
            return
        
        self._clip_model = CLIPModel(
            model_name=self.model_name,
            pretrained=self.pretrained,
            device=self.device,
            enable_cache=self.enable_cache
        )
        logger.info(f"Loaded CLIP model for landmarks: {self.model_name}")
    
    def extract_embeddings(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract landmark embeddings from images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dict mapping image_path -> embedding array [embedding_dim]
        """
        self._load_model()
        embeddings = {}
        
        # Batch encode images
        image_embeddings = self._clip_model.encode_images(
            [str(p) for p in image_paths],
            batch_size=32
        )
        
        for image_path, embedding in zip(image_paths, image_embeddings):
            embeddings[image_path] = embedding
        
        return embeddings
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process single image and extract landmark information.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dict with:
                - embedding: Landmark embedding array
                - landmark_score: Confidence that image contains a landmark
                - metadata: Additional information
        """
        self._load_model()
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Encode image
        image_embedding = self._clip_model.encode_images([str(image_path)])[0]
        
        # Encode landmark prompts
        prompt_embeddings = self._clip_model.encode_texts(self.landmark_prompts)
        
        # Compute similarity to landmark prompts
        similarities = self._clip_model.compute_similarity(
            image_embedding.reshape(1, -1),
            prompt_embeddings
        )[0]
        
        # Average similarity as landmark confidence
        landmark_score = float(np.mean(similarities))
        
        return {
            'embedding': image_embedding,
            'landmark_score': landmark_score,
            'prompt_scores': {
                prompt: float(score)
                for prompt, score in zip(self.landmark_prompts, similarities)
            },
            'is_landmark': landmark_score > 0.3  # Threshold for landmark detection
        }
    
    def _get_routing_key(self) -> str:
        """Get routing key for landmark detection."""
        return 'landmark_detection'
    
    def __repr__(self) -> str:
        return f"LandmarkModel(model={self.model_name}, device={self.device})"




