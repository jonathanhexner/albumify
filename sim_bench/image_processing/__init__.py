"""
Image processing module for sim-bench.
"""

from sim_bench.image_processing.thumbnail import ThumbnailGenerator
from sim_bench.image_processing.degradation import (
    ImageDegradationProcessor,
    create_degradation_processor
)

__all__ = [
    'ThumbnailGenerator',
    'ImageDegradationProcessor',
    'create_degradation_processor'
]
