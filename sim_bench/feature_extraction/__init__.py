"""Similarity methods for sim-bench."""

from .base import BaseMethod, load_method
from .resnet_features import ResNetFeatureExtractor
from .vgg_features import VGGFeatureExtractor

__all__ = ['BaseMethod', 'load_method', 'ResNetFeatureExtractor', 'VGGFeatureExtractor']
