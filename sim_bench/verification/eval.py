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

__all__ = ["evaluate", "evaluate_clustering"]


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


def evaluate_clustering(baseline_labels: Dict[Hashable, object],
                        verified_labels: Dict[Hashable, object],
                        truth_labels: Dict[Hashable, object],
                        checkpoint: str = "") -> EvalResult:
    """AC1 gate for *clustering* checkpoints (faces/scenes) via pairwise reduction.

    Cluster ids are arbitrary (label permutation), so we can't compare labels directly. Instead reduce
    to pairwise decisions over the labeled faces: for each pair, ``decision = same cluster?`` vs
    ``truth = same person?``. Then the generic ``evaluate`` applies. O(n²) pairs — fine for eval sets of
    a few hundred faces.
    """
    ids = sorted(truth_labels)
    pairs_base: Dict = {}
    pairs_ver: Dict = {}
    pairs_truth: Dict = {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            key = (i, j)
            pairs_truth[key] = truth_labels[i] == truth_labels[j]
            pairs_base[key] = baseline_labels.get(i) == baseline_labels.get(j)
            pairs_ver[key] = verified_labels.get(i) == verified_labels.get(j)
    return evaluate(pairs_base, pairs_ver, pairs_truth, checkpoint)
