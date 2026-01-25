"""
Phase 2: Multitask Face Pretraining

Joint pretraining on AffectNet with:
- Expression classification (8 classes)
- Landmark regression (5-10 key points)

This pretrained backbone can then be transferred to face recognition tasks.
"""

from sim_bench.training.phase2_pretraining.multitask_model import MultitaskFaceModel
from sim_bench.training.phase2_pretraining.affectnet_dataset import AffectNetDataset
from sim_bench.training.phase2_pretraining.landmark_extractor import LandmarkExtractor

__all__ = ['MultitaskFaceModel', 'AffectNetDataset', 'LandmarkExtractor']
