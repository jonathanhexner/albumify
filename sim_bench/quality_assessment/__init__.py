"""
Minimal quality assessment module.

Legacy benchmark code archived in sim_bench/legacy/quality_assessment/
New unified interface in sim_bench/image_quality_models/

This module now only provides the base rule-based IQA used by the unified
image quality models system.

For new development, use:
    from sim_bench.image_quality_models import create_model
    model = create_model({'type': 'rule_based_iqa', ...})
"""

from sim_bench.quality_assessment.base import QualityAssessor
from sim_bench.quality_assessment.rule_based import RuleBasedQuality

__all__ = ['QualityAssessor', 'RuleBasedQuality']
