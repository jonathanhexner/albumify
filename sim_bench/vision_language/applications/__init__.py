"""
Application-specific wrappers for vision-language models.

Provides ready-to-use implementations for common use cases:
- aesthetic: Aesthetic quality assessment via text prompts
- retrieval: Semantic image retrieval
- classification: Zero-shot classification
"""

from sim_bench.vision_language.applications.aesthetic import AestheticAssessor
from sim_bench.vision_language.applications.retrieval import SemanticRetrieval
from sim_bench.vision_language.applications.classification import ZeroShotClassifier

__all__ = [
    'AestheticAssessor',
    'SemanticRetrieval',
    'ZeroShotClassifier',
]
