"""
Specialized models for domain-specific photo analysis.

Provides face detection/recognition and landmark recognition models
that are triggered by routing decisions from photo analysis.
"""

from sim_bench.specialized_models.base import SpecializedModel
from sim_bench.specialized_models.faces import FaceModel
from sim_bench.specialized_models.landmarks import LandmarkModel


def create_specialized_model(model_type: str, **config) -> SpecializedModel:
    """
    Factory function for creating specialized models.

    Args:
        model_type: Type of model ('face', 'landmark')
        **config: Model-specific configuration

    Returns:
        SpecializedModel instance

    Raises:
        ValueError: If unknown model type

    Example:
        >>> from sim_bench.specialized_models import create_specialized_model
        >>>
        >>> face_model = create_specialized_model('face', backend='deepface')
        >>> results = face_model.process_batch(image_paths, routing_hints)
    """
    if model_type == 'face':
        return FaceModel(**config)
    elif model_type == 'landmark':
        return LandmarkModel(**config)
    else:
        raise ValueError(
            f"Unknown specialized model type: {model_type}. "
            f"Supported types: 'face', 'landmark'"
        )


__all__ = [
    'SpecializedModel',
    'FaceModel',
    'LandmarkModel',
    'create_specialized_model'
]




