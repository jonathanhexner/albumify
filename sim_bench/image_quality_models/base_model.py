"""Base interface for all image quality assessment models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class BaseQualityModel(ABC):
    """Base interface for all quality assessment models.
    
    All models (Siamese, AVA, IQA, etc.) implement this interface for unified benchmarking.
    """
    
    def __init__(self, name: str, device: str = 'cpu'):
        """
        Initialize base quality model.
        
        Args:
            name: Human-readable model name
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.name = name
        self.device = device
        self.model_type = self.__class__.__name__
    
    @abstractmethod
    def score_image(self, image_path: Path) -> float:
        """
        Score a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality score (higher = better). Range depends on model type.
            
        Raises:
            NotImplementedError: If model only supports pairwise comparison
        """
        pass
    
    def compare_images(self, image1_path: Path, image2_path: Path) -> Dict:
        """
        Compare two images to determine which has higher quality.
        
        Default implementation uses score_image() to compute difference.
        Models with native pairwise comparison (e.g., Siamese) can override.
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            
        Returns:
            Dict with:
                - prediction: int (1 if img1 better, 0 if img2 better)
                - confidence: float (0-1, confidence in prediction)
                - score_img1: float (score for img1, or None if not applicable)
                - score_img2: float (score for img2, or None if not applicable)
        """
        score1 = self.score_image(image1_path)
        score2 = self.score_image(image2_path)
        
        prediction = 1 if score1 > score2 else 0
        
        # Compute relative confidence
        max_score = max(abs(score1), abs(score2), 1e-6)
        confidence = abs(score1 - score2) / max_score
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'score_img1': score1,
            'score_img2': score2
        }
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict) -> 'BaseQualityModel':
        """
        Create model instance from config dict.
        
        Args:
            config: Configuration dictionary from YAML
            
        Returns:
            Initialized model instance
        """
        pass
    
    def __repr__(self):
        return f"{self.model_type}(name='{self.name}', device='{self.device}')"
