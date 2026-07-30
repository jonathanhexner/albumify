"""Model factory for creating quality assessment models from config."""

import logging
from typing import Dict

from sim_bench.image_quality_models.base_model import BaseQualityModel
from sim_bench.image_quality_models.siamese_model_wrapper import SiameseQualityModel
from sim_bench.image_quality_models.ava_model_wrapper import AVAQualityModel
from sim_bench.image_quality_models.iqa_model_wrapper import (
    RuleBasedIQAModel,
    SharpnessOnlyIQAModel,
    ExposureOnlyIQAModel,
    ColorfulnessOnlyIQAModel,
    ContrastOnlyIQAModel,
    NoiseOnlyIQAModel
)
from sim_bench.image_quality_models.pyiqa_model_wrapper import PyIQAModel, PYIQA_METRICS
from sim_bench.image_quality_models.clip_prompt_model import ClipPromptModel

logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    'siamese': SiameseQualityModel,
    'ava': AVAQualityModel,
    'rule_based_iqa': RuleBasedIQAModel,
    'sharpness_iqa': SharpnessOnlyIQAModel,
    'exposure_iqa': ExposureOnlyIQAModel,
    'colorfulness_iqa': ColorfulnessOnlyIQAModel,
    'contrast_iqa': ContrastOnlyIQAModel,
    'noise_iqa': NoiseOnlyIQAModel,  # spec-098: wavelet noise score
}

# spec-093: pyiqa no-reference metrics (MANIQA/MUSIQ/HyperIQA/BRISQUE/NIQE/CLIP-IQA)
# all front the single PyIQAModel class, keyed by metric name.
for _metric in PYIQA_METRICS:
    MODEL_REGISTRY[_metric] = PyIQAModel

# spec-094 follow-up: CLIP prompt-based occlusion detector (finger over lens etc.)
MODEL_REGISTRY['clip_occlusion'] = ClipPromptModel


def create_model(model_config: Dict) -> BaseQualityModel:
    """
    Create model from configuration dictionary.
    
    Args:
        model_config: Dict with 'type' key and model-specific parameters
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model type is not registered
        
    Example:
        >>> config = {'type': 'ava', 'checkpoint': 'path/to/model.pt', 'device': 'cpu'}
        >>> model = create_model(config)
    """
    model_type = model_config.get('type')
    
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: '{model_type}'. Available: {available}")
    
    model_class = MODEL_REGISTRY[model_type]
    model = model_class.from_config(model_config)
    
    logger.info(f"Created model: {model}")
    
    return model


def register_model(model_type: str, model_class):
    """
    Register a new model type.
    
    Args:
        model_type: String identifier for model
        model_class: Class implementing BaseQualityModel
    """
    if not issubclass(model_class, BaseQualityModel):
        raise ValueError(f"Model class must inherit from BaseQualityModel")
    
    MODEL_REGISTRY[model_type] = model_class
    logger.info(f"Registered model type: {model_type}")
