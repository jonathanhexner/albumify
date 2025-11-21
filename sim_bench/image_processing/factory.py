"""
Factory function for creating image processors.
"""

from sim_bench.image_processing.thumbnail import ThumbnailGenerator


def create_image_processor(processor_type: str, **config):
    """
    Factory function for creating image processors.

    Args:
        processor_type: Type of processor ('thumbnail', 'enhancer', 'cropper')
        **config: Processor-specific configuration

    Returns:
        ImageProcessor instance

    Raises:
        ValueError: If unknown processor type

    Example:
        >>> from sim_bench.image_processing.factory import create_image_processor
        >>>
        >>> generator = create_image_processor('thumbnail', cache_dir='.cache/thumbs')
        >>> thumbnails = generator.generate('photo.jpg', sizes=['tiny', 'small'])
    """
    if processor_type == 'thumbnail':
        return ThumbnailGenerator(**config)
    else:
        raise ValueError(
            f"Unknown image processor type: {processor_type}. "
            f"Supported types: 'thumbnail'"
        )
