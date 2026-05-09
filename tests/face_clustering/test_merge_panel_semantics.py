"""Tests for the three-state outcome label and margin-disabled rendering
introduced by spec-030 Phase 3.

The user-reported regression in SIGHTING-058 was: an iter-1 row for C0+C1 with
action="passed" (passed all gates but lost the iteration tie-break) rendered
as "REJECTED" with all four gates green — semantically nonsense.  These tests
pin down the post-fix labels.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# The panel module lives under app/face_clustering, not on the default import path.
APP_DIR = Path(__file__).resolve().parents[2] / "app" / "face_clustering"
sys.path.insert(0, str(APP_DIR))

from _merge_decisions_panel import _outcome_label_color, _format_margin_value  # noqa: E402


@dataclass
class _Pair:
    """Stand-in with the attribute shape the helpers read."""
    action: str = "merged"
    actually_merged: bool = False
    margin_gap: float = 0.05


def test_label_merged_when_actually_merged():
    label, color = _outcome_label_color(_Pair(action="merged", actually_merged=True))
    assert (label, color) == ("MERGED", "green")


def test_label_passed_when_action_passed():
    """The exact case that misled the user on face_clustering_20260508_000446
    iter 1 C0+C1: passed all four gates, but C0+C11 had a lower distance and
    won the iteration.  Used to render as REJECTED; must now render as PASSED.
    """
    label, color = _outcome_label_color(_Pair(action="passed", actually_merged=False))
    assert (label, color) == ("PASSED", "orange")


def test_label_rejected_when_action_rejected():
    label, color = _outcome_label_color(_Pair(action="rejected", actually_merged=False))
    assert (label, color) == ("REJECTED", "red")


def test_label_merged_overrides_passed_action():
    """Defensive: the algorithm sets action='merged' AND actually_merged=True
    on the winner.  If anything ever inverts these, prefer the boolean."""
    label, _ = _outcome_label_color(_Pair(action="passed", actually_merged=True))
    assert label == "MERGED"


def test_margin_disabled_when_inf():
    """merge_margin=0 in config disables the gate; merger returns
    worst_gap=float('inf').  Used to print 'inf'; must now print 'disabled'."""
    assert _format_margin_value(_Pair(margin_gap=float("inf"))) == "disabled"


def test_margin_disabled_when_neg_inf():
    assert _format_margin_value(_Pair(margin_gap=float("-inf"))) == "disabled"


def test_margin_value_when_set():
    assert _format_margin_value(_Pair(margin_gap=0.123)) == "0.123"


def test_margin_na_when_none():
    assert _format_margin_value(_Pair(margin_gap=None)) == "n/a"
