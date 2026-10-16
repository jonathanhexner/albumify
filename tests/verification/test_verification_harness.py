"""spec-111 P0 — verification spine: escalation trigger + baseline-eval gate + types."""
from __future__ import annotations

from sim_bench.verification.escalation import borderline
from sim_bench.verification.eval import evaluate
from sim_bench.verification.types import Correction, VerifyResult, MERGE


class ut_Borderline:
    def test_selects_only_items_near_the_boundary(self):
        # margin = distance from the fence; band=0.1 → only the near-0 ones escalate
        items = [("a", 0.02), ("b", -0.05), ("c", 0.4), ("d", -0.3)]
        out = borderline(items, margin=lambda it: it[1], band=0.1)
        assert [i[0] for i in out] == ["a", "b"]           # confident c,d left alone

    def test_band_zero_only_exact_fence(self):
        items = [("a", 0.0), ("b", 0.001)]
        assert [i[0] for i in borderline(items, lambda it: it[1], 0.0)] == ["a"]

    def test_negative_band_raises(self):
        import pytest
        with pytest.raises(ValueError):
            borderline([], lambda x: 0.0, -1)

    def test_empty(self):
        assert borderline([], lambda x: 0.0, 0.5) == []


class ut_Evaluate:
    def test_verifier_that_helps_improves(self):
        # baseline wrong on b,c; verifier fixes both → improved
        truth    = {"a": 1, "b": 1, "c": 1, "d": 0}
        baseline = {"a": 1, "b": 0, "c": 0, "d": 0}
        verified = {"a": 1, "b": 1, "c": 1, "d": 0}
        r = evaluate(baseline, verified, truth, "demo")
        assert r.baseline_agreement == 0.5 and r.verified_agreement == 1.0
        assert r.delta == 0.5 and r.n_changed == 2 and r.helped == 2 and r.hurt == 0
        assert r.improved is True

    def test_verifier_that_hurts_does_not_ship(self):
        # baseline right on all; verifier breaks one → hurt, not improved
        truth    = {"a": 1, "b": 1}
        baseline = {"a": 1, "b": 1}
        verified = {"a": 1, "b": 0}
        r = evaluate(baseline, verified, truth, "demo")
        assert r.hurt == 1 and r.helped == 0
        assert r.improved is False

    def test_net_zero_change_does_not_ship(self):
        # fixes one, breaks one → agreement unchanged → not improved
        truth    = {"a": 1, "b": 1}
        baseline = {"a": 0, "b": 1}
        verified = {"a": 1, "b": 0}
        r = evaluate(baseline, verified, truth, "demo")
        assert r.helped == 1 and r.hurt == 1
        assert r.improved is False                         # no net gain → stays off

    def test_empty_truth_is_safe(self):
        r = evaluate({}, {}, {}, "demo")
        assert r.n == 0 and r.improved is False


class ut_Types:
    def test_verify_result_counts(self):
        vr = VerifyResult("verify_face_clusters", n_reviewed=10, escalated=["p1", "p2"],
                          corrections=[Correction(MERGE, ["c1", "c2"], "same person")])
        assert vr.n_escalated == 2 and vr.n_corrections == 1
