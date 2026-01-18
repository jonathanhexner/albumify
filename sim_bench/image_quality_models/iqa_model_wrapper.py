"""IQA model wrappers for unified quality assessment interface."""

from pathlib import Path
from typing import Dict

from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.quality_assessment.rule_based import RuleBasedQuality


class RuleBasedIQAModel(BaseQualityModel):
    """Rule-based IQA using hand-crafted features."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize rule-based IQA."""
        super().__init__(name='Rule-Based-IQA', device=device)
        self.iqa = RuleBasedQuality(device=device)
    
    def score_image(self, image_path: Path) -> float:
        """Return overall quality score (0-1 range)."""
        scores = self.iqa.get_detailed_scores(str(image_path))
        return scores['overall']
    
    @classmethod
    def from_config(cls, config: Dict) -> 'RuleBasedIQAModel':
        """Create from config dict."""
        device = config.get('device', 'cpu')
        return cls(device)


class SharpnessOnlyIQAModel(BaseQualityModel):
    """Sharpness-only IQA metric."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize sharpness-only IQA."""
        super().__init__(name='Sharpness-Only', device=device)
        self.iqa = RuleBasedQuality(device=device)
    
    def score_image(self, image_path: Path) -> float:
        """Return normalized sharpness score."""
        scores = self.iqa.get_detailed_scores(str(image_path))
        return scores['sharpness_normalized']
    
    @classmethod
    def from_config(cls, config: Dict) -> 'SharpnessOnlyIQAModel':
        """Create from config dict."""
        device = config.get('device', 'cpu')
        return cls(device)


class ExposureOnlyIQAModel(BaseQualityModel):
    """Exposure-only IQA metric."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize exposure-only IQA."""
        super().__init__(name='Exposure-Only', device=device)
        self.iqa = RuleBasedQuality(device=device)
    
    def score_image(self, image_path: Path) -> float:
        """Return exposure quality score."""
        scores = self.iqa.get_detailed_scores(str(image_path))
        return scores['exposure']
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ExposureOnlyIQAModel':
        """Create from config dict."""
        device = config.get('device', 'cpu')
        return cls(device)


class ColorfulnessOnlyIQAModel(BaseQualityModel):
    """Colorfulness-only IQA metric."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize colorfulness-only IQA."""
        super().__init__(name='Colorfulness-Only', device=device)
        self.iqa = RuleBasedQuality(device=device)
    
    def score_image(self, image_path: Path) -> float:
        """Return normalized colorfulness score."""
        scores = self.iqa.get_detailed_scores(str(image_path))
        return scores['colorfulness_normalized']
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ColorfulnessOnlyIQAModel':
        """Create from config dict."""
        device = config.get('device', 'cpu')
        return cls(device)


class ContrastOnlyIQAModel(BaseQualityModel):
    """Contrast-only IQA metric."""
    
    def __init__(self, device: str = 'cpu'):
        """Initialize contrast-only IQA."""
        super().__init__(name='Contrast-Only', device=device)
        self.iqa = RuleBasedQuality(device=device)
    
    def score_image(self, image_path: Path) -> float:
        """Return contrast quality score."""
        scores = self.iqa.get_detailed_scores(str(image_path))
        return scores['contrast']
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ContrastOnlyIQAModel':
        """Create from config dict."""
        device = config.get('device', 'cpu')
        return cls(device)
