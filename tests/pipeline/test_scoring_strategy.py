"""Tests for scoring strategy."""

import logging

import pytest

from sim_bench.pipeline.scoring.strategy import (
    PenaltyComputer,
    InsightFacePenaltyScoring,
    PersonPenaltyStrategy,
    NoPersonPenaltyStrategy,
    ScoringStrategyFactory
)
from sim_bench.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)


def test_penalty_computer():
    """Test penalty computation."""
    config = {
        'penalties': {
            'body_orientation': 0.1,
            'face_occlusion': 0.2,
            'eyes_closed': 0.15,
            'no_smile': 0.1,
            'face_turned': 0.15
        }
    }
    
    computer = PenaltyComputer(config)
    
    # Test body penalty
    person_data = {'body_facing_score': 0.5}
    body_penalty = computer.compute_body_penalty(person_data)
    assert body_penalty == 0.1 * 0.5  # weight * (1 - score)
    
    # Test occlusion penalty
    occlusion_penalty = computer.compute_occlusion_penalty()
    assert occlusion_penalty == 0.2
    
    logger.info(f"Penalties computed: body={body_penalty}, occlusion={occlusion_penalty}")


def test_scoring_strategy_factory():
    """Test scoring strategy factory."""
    strategy = ScoringStrategyFactory.create('insightface_penalty')
    
    assert isinstance(strategy, InsightFacePenaltyScoring)
    logger.info("Strategy factory created InsightFacePenaltyScoring")


def test_person_penalty_strategy():
    """Test person penalty strategy."""
    strategy = PersonPenaltyStrategy()
    computer = PenaltyComputer({'penalties': {'body_orientation': 0.1}})
    
    # Create mock context
    context = PipelineContext()
    context.persons = {
        'test.jpg': {
            'person_detected': True,
            'body_facing_score': 0.8
        }
    }
    context.insightface_faces = {}
    
    penalties = strategy.compute('test.jpg', context, computer)
    
    # Should have body orientation penalty
    expected = 0.1 * (1.0 - 0.8)  # weight * (1 - score)
    assert penalties == pytest.approx(expected, rel=0.01)
    logger.info(f"Person penalties: {penalties}")


def test_no_person_penalty_strategy():
    """Test no-person penalty strategy."""
    strategy = NoPersonPenaltyStrategy()
    computer = PenaltyComputer({'penalties': {}})
    
    context = PipelineContext()
    context.persons = {}
    
    penalties = strategy.compute('test.jpg', context, computer)
    
    # Should have no penalties
    assert penalties == 0.0
    logger.info("No penalties for no person")
