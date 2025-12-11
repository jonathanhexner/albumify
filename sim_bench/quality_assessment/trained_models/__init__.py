"""
Trained models for quality assessment.

This module contains trainable models that learn from PhotoTriage and other datasets.
"""

from sim_bench.quality_assessment.trained_models.phototriage_binary import (
    BinaryClassifierConfig,
    PhotoTriageBinaryClassifier,
    BinaryClassifierTrainer
)

__all__ = [
    'BinaryClassifierConfig',
    'PhotoTriageBinaryClassifier',
    'BinaryClassifierTrainer'
]
