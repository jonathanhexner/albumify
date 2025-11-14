"""
Rule-based image quality assessment using hand-crafted features.
Implements sharpness, exposure, colorfulness, contrast, and noise metrics.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

from sim_bench.quality_assessment.base import QualityAssessor


class RuleBasedQuality(QualityAssessor):
    """Rule-based quality assessment using multiple hand-crafted metrics."""
    
    def __init__(
        self,
        weights: Dict[str, float] = None,
        device: str = 'cpu'
    ):
        """
        Initialize rule-based quality assessor.
        
        Args:
            weights: Dictionary of feature weights
                    {'sharpness': 0.4, 'exposure': 0.3, 'colorfulness': 0.2, 'contrast': 0.1}
            device: Device (not used for rule-based, kept for API consistency)
        """
        super().__init__(device)
        
        # Default weights based on literature
        self.weights = weights or {
            'sharpness': 0.40,
            'exposure': 0.30,
            'colorfulness': 0.20,
            'contrast': 0.10
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
    def assess_image(self, image_path: str) -> float:
        """
        Assess image quality using combined metrics.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Quality score [0-1], higher is better
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute individual metrics
        sharpness = self._compute_sharpness(gray)
        exposure = self._compute_exposure_quality(gray)
        colorfulness = self._compute_colorfulness(img)
        contrast = self._compute_contrast(gray)
        
        # Normalize metrics to [0, 1] range
        sharpness_norm = self._normalize_sharpness(sharpness)
        exposure_norm = exposure  # Already in [0, 1]
        color_norm = self._normalize_colorfulness(colorfulness)
        contrast_norm = contrast  # Already normalized
        
        # Weighted combination
        quality = (
            self.weights.get('sharpness', 0) * sharpness_norm +
            self.weights.get('exposure', 0) * exposure_norm +
            self.weights.get('colorfulness', 0) * color_norm +
            self.weights.get('contrast', 0) * contrast_norm
        )
        
        return float(quality)
    
    def _compute_sharpness(self, gray: np.ndarray) -> float:
        """
        Compute sharpness using Laplacian variance.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Laplacian variance (raw score)
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _compute_sharpness_tenengrad(self, gray: np.ndarray) -> float:
        """
        Alternative: Tenengrad sharpness measure using Sobel operators.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tenengrad score
        """
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Only consider gradients above threshold
        threshold = gradient.mean()
        gradient_filtered = gradient[gradient > threshold]
        
        if len(gradient_filtered) == 0:
            return 0.0
            
        return float(np.sum(gradient_filtered**2))
    
    def _compute_exposure_quality(self, gray: np.ndarray) -> float:
        """
        Compute exposure quality based on histogram clipping and dynamic range.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Exposure score [0, 1], higher is better
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()  # Normalize
        
        # Penalize clipping (pure black/white)
        clip_black = hist[0]
        clip_white = hist[255]
        clip_penalty = clip_black + clip_white
        
        # Compute histogram entropy (dynamic range usage)
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log(hist_nonzero))
        entropy_normalized = entropy / np.log(256)  # [0, 1]
        
        # Combine: good exposure = low clipping + high dynamic range
        exposure_score = (1.0 - clip_penalty) * 0.5 + entropy_normalized * 0.5
        
        return float(np.clip(exposure_score, 0, 1))
    
    def _compute_colorfulness(self, img: np.ndarray) -> float:
        """
        Compute colorfulness using Hasler & Süsstrunk metric.
        
        Args:
            img: BGR image
            
        Returns:
            Colorfulness score (raw)
        """
        # Split into channels
        (B, G, R) = cv2.split(img.astype(float))
        
        # Compute opponent color space
        rg = R - G
        yb = 0.5 * (R + G) - B
        
        # Compute statistics
        rg_mean = np.mean(rg)
        rg_std = np.std(rg)
        yb_mean = np.mean(yb)
        yb_std = np.std(yb)
        
        # Colorfulness metric
        std_root = np.sqrt(rg_std**2 + yb_std**2)
        mean_root = np.sqrt(rg_mean**2 + yb_mean**2)
        
        colorfulness = std_root + 0.3 * mean_root
        
        return float(colorfulness)
    
    def _compute_contrast(self, gray: np.ndarray) -> float:
        """
        Compute RMS contrast.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Normalized contrast [0, 1]
        """
        mean_val = gray.mean()
        if mean_val == 0:
            return 0.0
            
        std_val = gray.std()
        rms_contrast = std_val / mean_val
        
        # Normalize: typical good images have RMS contrast 0.2-0.8
        contrast_normalized = np.clip(rms_contrast / 0.8, 0, 1)
        
        return float(contrast_normalized)
    
    def _normalize_sharpness(self, sharpness: float) -> float:
        """
        Normalize sharpness score to [0, 1].
        
        Args:
            sharpness: Raw Laplacian variance
            
        Returns:
            Normalized score
        """
        # Typical range: 0-1000 for natural images
        # Good sharp images: >100, excellent: >500
        normalized = sharpness / 1000.0
        return float(np.clip(normalized, 0, 1))
    
    def _normalize_colorfulness(self, colorfulness: float) -> float:
        """
        Normalize colorfulness score to [0, 1].
        
        Args:
            colorfulness: Raw colorfulness score
            
        Returns:
            Normalized score
        """
        # Typical range: 0-150
        # Good colorful images: 40-80, very vibrant: >100
        normalized = colorfulness / 100.0
        return float(np.clip(normalized, 0, 1))
    
    def get_config(self) -> Dict[str, Any]:
        """Get method configuration."""
        config = super().get_config()
        config.update({
            'weights': self.weights
        })
        return config
    
    def get_detailed_scores(self, image_path: str) -> Dict[str, float]:
        """
        Get detailed breakdown of all quality metrics.
        
        Args:
            image_path: Path to image
            
        Returns:
            Dictionary with individual metric scores
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sharpness = self._compute_sharpness(gray)
        exposure = self._compute_exposure_quality(gray)
        colorfulness = self._compute_colorfulness(img)
        contrast = self._compute_contrast(gray)
        
        return {
            'sharpness_raw': sharpness,
            'sharpness_normalized': self._normalize_sharpness(sharpness),
            'exposure': exposure,
            'colorfulness_raw': colorfulness,
            'colorfulness_normalized': self._normalize_colorfulness(colorfulness),
            'contrast': contrast,
            'overall': self.assess_image(image_path)
        }


