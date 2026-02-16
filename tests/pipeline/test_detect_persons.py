"""Tests for person detection pipeline step."""

import logging
from pathlib import Path

import pytest
import numpy as np

from sim_bench.pipeline.person_detection.types import PersonDetection, BoundingBox
from sim_bench.pipeline.person_detection.body_orientation import ShoulderHipOrientationStrategy

logger = logging.getLogger(__name__)


def test_person_detection_types():
    """Test PersonDetection dataclass creation."""
    bbox = BoundingBox(
        x=0.1, y=0.2, w=0.3, h=0.4,
        x_px=10, y_px=20, w_px=30, h_px=40
    )
    
    keypoints = np.random.rand(17, 3)
    keypoints[:, 2] = 0.8  # High confidence
    
    person = PersonDetection(
        bbox=bbox,
        confidence=0.95,
        keypoints=keypoints,
        body_facing_score=0.8,
        keypoint_confidence=0.8
    )
    
    assert person.bbox == bbox
    assert person.confidence == 0.95
    assert person.is_reliable(min_confidence=0.5)


def test_body_orientation_strategy():
    """Test body orientation scoring strategy."""
    config = {'keypoint_confidence_threshold': 0.5}
    strategy = ShoulderHipOrientationStrategy(config)
    
    # Create mock person with symmetric keypoints (front-facing)
    keypoints = np.zeros((17, 3))
    keypoints[5] = [100, 100, 0.9]  # Left shoulder
    keypoints[6] = [200, 100, 0.9]  # Right shoulder
    keypoints[11] = [110, 200, 0.9]  # Left hip
    keypoints[12] = [190, 200, 0.9]  # Right hip
    
    bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4, x_px=50, y_px=50, w_px=200, h_px=300)
    person = PersonDetection(bbox=bbox, confidence=0.9, keypoints=keypoints, body_facing_score=0.0, keypoint_confidence=0.9)
    
    score = strategy.compute_facing_score(person)
    
    # Front-facing should have high symmetry score
    assert 0.0 <= score <= 1.0
    logger.info(f"Body facing score: {score}")


def test_unreliable_keypoints():
    """Test handling of unreliable keypoints."""
    config = {'keypoint_confidence_threshold': 0.5}
    strategy = ShoulderHipOrientationStrategy(config)
    
    # Create mock person with low-confidence keypoints
    keypoints = np.zeros((17, 3))
    keypoints[:, 2] = 0.1  # Low confidence
    
    bbox = BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4, x_px=50, y_px=50, w_px=200, h_px=300)
    person = PersonDetection(bbox=bbox, confidence=0.9, keypoints=keypoints, body_facing_score=0.0, keypoint_confidence=0.1)
    
    score = strategy.compute_facing_score(person)
    
    # Unreliable keypoints should return neutral score (0.5)
    assert score == 0.5
    logger.info(f"Neutral score for unreliable keypoints: {score}")
