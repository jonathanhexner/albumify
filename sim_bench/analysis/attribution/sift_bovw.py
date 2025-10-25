"""
SIFT BoVW feature attribution using keypoint visualization.

Provides spatial attribution for SIFT Bag-of-Visual-Words features.
Shows which image regions (keypoints) contribute to specific visual words.
"""

import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging

from .base import BaseAttributionExtractor

logger = logging.getLogger(__name__)


class SIFTBoVWAttributionExtractor(BaseAttributionExtractor):
    """
    Extract features and visualize keypoint locations for SIFT BoVW.
    
    Shows which keypoints (with orientation arrows) are assigned to
    specific visual words in the vocabulary.
    """
    
    def __init__(self, codebook_path: str, n_features: int = 800):
        """
        Initialize SIFT BoVW attribution extractor.
        
        Args:
            codebook_path: Path to codebook.pkl file
            n_features: Max number of SIFT features per image
        """
        self.codebook_path = Path(codebook_path)
        self.n_features = n_features
        
        # Load codebook
        if not self.codebook_path.exists():
            raise FileNotFoundError(f"Codebook not found: {codebook_path}")
        
        with open(self.codebook_path, 'rb') as f:
            codebook = pickle.load(f)
        
        self.visual_words = codebook['cluster_centers']  # Shape: (512, 128)
        self.vocab_size = self.visual_words.shape[0]
        
        logger.info(f"Loaded codebook: {self.vocab_size} visual words")
    
    def _extract_sift_keypoints_and_descriptors(
        self, 
        image_path: str
    ) -> Tuple[List, np.ndarray]:
        """
        Extract SIFT keypoints and descriptors from image.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (keypoints, descriptors)
            keypoints: List of cv2.KeyPoint objects
            descriptors: Array of shape (N, 128) or None
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        sift = cv2.SIFT_create(nfeatures=self.n_features)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        
        return keypoints, descriptors
    
    def _assign_to_visual_words(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Assign descriptors to visual words.
        
        Args:
            descriptors: SIFT descriptors, shape (N, 128)
            
        Returns:
            assignments: Array of visual word IDs, shape (N,)
        """
        if descriptors is None or len(descriptors) == 0:
            return np.array([])
        
        # Compute distances to all visual words
        d2 = (np.sum(descriptors**2, axis=1, keepdims=True)
              + np.sum(self.visual_words**2, axis=1, keepdims=True).T
              - 2.0 * descriptors.dot(self.visual_words.T))
        
        # Find nearest visual word for each descriptor
        assignments = np.argmin(d2, axis=1)
        
        return assignments
    
    def _create_bovw_histogram(self, assignments: np.ndarray) -> np.ndarray:
        """
        Create BoVW histogram from assignments.
        
        Args:
            assignments: Visual word assignments, shape (N,)
            
        Returns:
            histogram: Normalized histogram, shape (vocab_size,)
        """
        if len(assignments) == 0:
            return np.zeros(self.vocab_size, dtype='float32')
        
        hist = np.bincount(assignments, minlength=self.vocab_size).astype('float32')
        hist /= (np.linalg.norm(hist) + 1e-12)
        
        return hist
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """
        Extract BoVW feature vector (same as used in sim-bench).
        
        Args:
            image_path: Path to image
            
        Returns:
            Feature vector of shape (vocab_size,) e.g., (512,)
        """
        _, descriptors = self._extract_sift_keypoints_and_descriptors(image_path)
        
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size, dtype='float32')
        
        assignments = self._assign_to_visual_words(descriptors)
        histogram = self._create_bovw_histogram(assignments)
        
        return histogram
    
    def compute_attribution(
        self,
        image_path: str,
        feature_indices: Optional[List[int]] = None
    ) -> Tuple[Dict[int, List], np.ndarray]:
        """
        Compute keypoint locations for specific visual words.
        
        Args:
            image_path: Path to image
            feature_indices: Visual word IDs to visualize (None = all)
            
        Returns:
            Tuple of (keypoints_by_word, original_image)
            keypoints_by_word: Dict mapping word_id -> list of cv2.KeyPoint
            original_image: Original RGB image array [H, W, 3]
        """
        # Load original image for visualization
        img_bgr = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract keypoints and descriptors
        keypoints, descriptors = self._extract_sift_keypoints_and_descriptors(image_path)
        
        if descriptors is None or len(descriptors) == 0:
            return {}, img_rgb
        
        # Assign to visual words
        assignments = self._assign_to_visual_words(descriptors)
        
        # Group keypoints by visual word
        keypoints_by_word = {}
        
        if feature_indices is None:
            feature_indices = range(self.vocab_size)
        
        for word_id in feature_indices:
            # Find keypoints assigned to this visual word
            mask = assignments == word_id
            word_keypoints = [keypoints[i] for i in range(len(keypoints)) if mask[i]]
            keypoints_by_word[word_id] = word_keypoints
        
        return keypoints_by_word, img_rgb
    
    def analyze_feature_importance(
        self,
        image_path: str,
        top_k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze which visual words are most frequent in an image.
        
        Args:
            image_path: Path to image
            top_k: Number of top visual words to return
            
        Returns:
            Dictionary with:
                - 'features': Full BoVW histogram
                - 'top_indices': Indices of top-k visual words
                - 'top_values': Frequencies of top-k visual words
        """
        histogram = self.extract_features(image_path)
        
        # Find top-k most frequent visual words
        top_indices = np.argsort(histogram)[-top_k:][::-1]
        top_values = histogram[top_indices]
        
        return {
            'features': histogram,
            'top_indices': top_indices,
            'top_values': top_values
        }


def draw_keypoints_for_word(
    image: np.ndarray,
    keypoints: List,
    color: Tuple[int, int, int] = (255, 0, 0),
    draw_rich: bool = True,
    line_thickness: int = 3,
    circle_radius_multiplier: float = 2.0
) -> np.ndarray:
    """
    Draw keypoints with orientation arrows on image (ENHANCED VISIBILITY).
    
    Args:
        image: RGB image array
        keypoints: List of cv2.KeyPoint objects
        color: RGB color tuple for keypoints
        draw_rich: If True, draw with orientation arrows and scale
        line_thickness: Thickness of arrow lines (default 3 for visibility)
        circle_radius_multiplier: Scale factor for keypoint circles (default 2.0)
        
    Returns:
        Image with keypoints drawn
    """
    img_rgb = image.copy()
    
    if len(keypoints) == 0:
        return image
    
    # Manual drawing for better visibility and control
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size * circle_radius_multiplier)
        angle = kp.angle
        
        # Draw circle (keypoint location and scale)
        cv2.circle(img_rgb, (x, y), size, color, line_thickness, cv2.LINE_AA)
        
        if draw_rich and angle >= 0:
            # Draw orientation arrow
            angle_rad = np.deg2rad(angle)
            end_x = int(x + size * np.cos(angle_rad))
            end_y = int(y + size * np.sin(angle_rad))
            cv2.arrowedLine(img_rgb, (x, y), (end_x, end_y), 
                          color, line_thickness, cv2.LINE_AA, tipLength=0.3)
    
    return img_rgb

