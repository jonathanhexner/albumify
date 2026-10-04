"""Baseline-eval harness (spec-111 P0) — the AC1 gate every checkpoint must pass.

AI double-checking *sounds* obviously good but can add cost + noise without adding quality. So a
checkpoint ships only if it is *measured* to beat the deterministic baseline on a human-labeled set.
This compares {baseline} vs {baseline+verifier} against human truth, and — the honest part — breaks
the changed items into helped (wrong→right) vs hurt (right→wrong). Pure; reuses the spec-102 notion of
agreement-with-human.
"""
from __future__ import annotations

from typing import Dict, Hashable

from sim_bench.verification.types import EvalResult

__all__ = ["evaluate"]


def evaluate(baseline: Dict[Hashable, object],
             verified: Dict[Hashable, object],
             truth: Dict[Hashable, object],
             checkpoint: str = "") -> EvalResult:
    """Score a checkpoint against human-labeled decisions.

    Each dict maps item id → decision. ``truth`` defines the labeled set evaluated; ``baseline`` is the
    pipeline's decision, ``verified`` is the decision after the verifier's corrections. Returns an
    ``EvalResult`` whose ``improved`` is the ship/no-ship gate.
    """
    ids = list(truth)
    if not ids:
        return EvalResult(checkpoint, 0, 0.0, 0.0, 0, 0, 0)

    def agree(src: Dict) -> float:
        return sum(1 for i in ids if src.get(i) == truth[i]) / len(ids)

    base_agree = agree(baseline)
    ver_agree = agree(verified)

    changed = [i for i in ids if baseline.get(i) != verified.get(i)]
    helped = sum(1 for i in changed if baseline.get(i) != truth[i] and verified.get(i) == truth[i])
    hurt = sum(1 for i in changed if baseline.get(i) == truth[i] and verified.get(i) != truth[i])

    return EvalResult(
        checkpoint=checkpoint, n=len(ids),
        baseline_agreement=round(base_agree, 4), verified_agreement=round(ver_agree, 4),
        n_changed=len(changed), helped=helped, hurt=hurt,
    )
