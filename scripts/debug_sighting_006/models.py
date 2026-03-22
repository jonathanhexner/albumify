"""Data models for hypothesis testing."""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class Verdict(Enum):
    """Test verdict enum."""
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    SUSPICIOUS = "⚠️  SUSPICIOUS"
    INFO = "ℹ️  INFO"


@dataclass
class TestResult:
    """Result of a hypothesis test."""
    hypothesis: str
    verdict: Verdict
    evidence: Dict[str, any]
    conclusion: str
    recommendation: Optional[str] = None
