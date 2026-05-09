"""Unit tests for spec-012 observability types."""
import pytest
from face_cluster.types import GateResult, QualityVerdict, ClusterOrigin, ClusterMetadata


class ut_QualityVerdict:
    def test_all_passed_true(self):
        verdict = QualityVerdict(gates={
            "blur": GateResult(value=120.0, threshold=50.0, passed=True),
            "pose_yaw": GateResult(value=10.0, threshold=30.0, passed=True),
        })
        assert verdict.all_passed() is True
        assert verdict.rejection_reason is None

    def test_all_passed_false_on_any_fail(self):
        verdict = QualityVerdict(
            gates={
                "blur": GateResult(value=20.0, threshold=50.0, passed=False),
                "pose_yaw": GateResult(value=10.0, threshold=30.0, passed=True),
            },
            rejection_reason="blur",
        )
        assert verdict.all_passed() is False
        assert verdict.rejection_reason == "blur"

    def test_gate_result_fields(self):
        g = GateResult(value=25.3, threshold=50.0, passed=False)
        assert g.value == pytest.approx(25.3)
        assert g.threshold == pytest.approx(50.0)
        assert g.passed is False

    def test_serialise_to_dict(self):
        verdict = QualityVerdict(
            gates={"blur": GateResult(value=80.0, threshold=50.0, passed=True)},
            rejection_reason=None,
        )
        d = {
            "gates": {k: {"value": v.value, "threshold": v.threshold, "passed": v.passed}
                      for k, v in verdict.gates.items()},
            "rejection_reason": verdict.rejection_reason,
        }
        assert d["gates"]["blur"]["passed"] is True
        assert d["rejection_reason"] is None

    def test_empty_gates_all_passed(self):
        verdict = QualityVerdict(gates={})
        assert verdict.all_passed() is True


class ut_ClusterOrigin:
    def test_values(self):
        assert ClusterOrigin.BASE == "base"
        assert ClusterOrigin.AUTO_MERGE == "auto_merge"
        assert ClusterOrigin.MANUAL_MERGE == "manual_merge"
        assert ClusterOrigin.REMERGE == "remerge"
        assert ClusterOrigin.UNKNOWN == "unknown"

    def test_from_string(self):
        assert ClusterOrigin("base") is ClusterOrigin.BASE

    def test_cluster_metadata_defaults(self):
        meta = ClusterMetadata(origin=ClusterOrigin.BASE)
        assert meta.parent_cluster_ids == []

    def test_cluster_metadata_with_parents(self):
        meta = ClusterMetadata(origin=ClusterOrigin.AUTO_MERGE, parent_cluster_ids=[3, 7])
        assert meta.parent_cluster_ids == [3, 7]
