"""spec-088 / SIGHTING-107: the "Export for analysis" toggle reaches a real step
in the unified chain and fires the FC-app export.
"""

import sim_bench.pipeline.steps.face_cluster_analysis_export as mod
from sim_bench.pipeline.steps.face_cluster_analysis_export import FaceClusterAnalysisExportStep
from sim_bench.api.services.pipeline_service import PipelineService

_UNIFIED = ["quality_gate", "build_face_knn_graph", "assign_people_clusters",
            "face_cluster_analysis_export"]


class _Ctx:
    def __init__(self):
        self.face_records = [object()]
        self.cluster_result = object()
        self.merged_cluster_result = None
        self.core_indices = [0]
        self.merge_log = None
        self.fc_export_dir = None


class ut_ExportFlagRouting:
    """_broadcast_clustering_config routes the toggle to the export step (AC)."""

    def test_flag_on_routes_to_step_with_params(self):
        svc = PipelineService(None)
        out = svc._broadcast_clustering_config(
            _UNIFIED, {"cluster_people": {"K": 7, "export_for_analysis": True}}
        )
        cfg = out["face_cluster_analysis_export"]
        assert cfg["export_for_analysis"] is True
        assert cfg["K"] == 7  # clustering params travel with it (for serialization)

    def test_flag_off_does_not_route(self):
        svc = PipelineService(None)
        out = svc._broadcast_clustering_config(
            _UNIFIED, {"cluster_people": {"K": 7, "export_for_analysis": False}}
        )
        assert "face_cluster_analysis_export" not in out


class ut_ExportStep:
    """The step honours the flag and reuses export_for_analysis()."""

    def test_off_is_noop(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mod, "export_for_analysis", lambda **k: calls.append(k))
        FaceClusterAnalysisExportStep().process(_Ctx(), {"export_for_analysis": False})
        assert calls == []

    def test_on_calls_export_with_context_artifacts(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mod, "export_for_analysis", lambda **k: calls.append(k))
        ctx = _Ctx()
        FaceClusterAnalysisExportStep().process(ctx, {"export_for_analysis": True, "K": 5})
        assert len(calls) == 1
        # merged falls back to base when no merge ran; base is the cluster_result.
        assert calls[0]["base_cluster_result"] is ctx.cluster_result
        assert calls[0]["merged_cluster_result"] is ctx.cluster_result

    def test_on_but_no_clusters_skips(self, monkeypatch):
        calls = []
        monkeypatch.setattr(mod, "export_for_analysis", lambda **k: calls.append(k))
        ctx = _Ctx()
        ctx.cluster_result = None
        FaceClusterAnalysisExportStep().process(ctx, {"export_for_analysis": True})
        assert calls == []
