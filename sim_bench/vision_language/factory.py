"""
Factory function for loading vision-language models.
"""

# Conditional imports
try:
    from sim_bench.vision_language.clip import CLIPModel
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False
    CLIPModel = None


def load_vision_language_model(model_type: str, **config):
    """
    Factory function to load vision-language model.

    Args:
        model_type: Type of model ('clip', 'blip', 'llava')
        **config: Model-specific configuration

    Returns:
        BaseVisionLanguageModel instance

    Raises:
        ValueError: If unknown model type
        ImportError: If required dependencies not available

    Example:
        >>> from sim_bench.vision_language.factory import load_vision_language_model
        >>> model = load_vision_language_model('clip', model_name='ViT-B-32')
        >>> embeddings = model.encode_images(['photo1.jpg', 'photo2.jpg'])
    """
    if model_type == 'clip':
        if not _HAS_CLIP:
            raise ImportError(
                "CLIP requires PyTorch and OpenCLIP. "
                "Install with: pip install torch open-clip-torch"
            )
        return CLIPModel(**config)

    else:
        raise ValueError(
            f"Unknown vision-language model type: {model_type}. "
            f"Supported types: 'clip'"
        )
