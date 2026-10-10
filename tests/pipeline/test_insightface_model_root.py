"""spec-110 T1.3 - both InsightFace loaders honor INSIGHTFACE_HOME.

Why this is a test and not a config line: on the Modal worker the default root (~/.insightface) is
container-local scratch, so buffalo_l was re-downloaded on every cold start while the models Volume
sat unused (AC5). There are *two* `FaceAnalysis` call sites -- detection and embeddings -- and
missing either one keeps the download. The fakes below stand in for FaceAnalysis so no weights are
touched.
"""
from __future__ import annotations

import pytest

import sim_bench.pipeline.face_embedding.insightface_native as native_mod
import sim_bench.pipeline.insightface_pipeline.face_analyzer as fa_mod
from sim_bench.pipeline.face_embedding.insightface_native import InsightFaceNativeExtractor
from sim_bench.pipeline.insightface_pipeline.face_analyzer import InsightFaceFaceAnalyzer
from sim_bench.pipeline.utils.insightface_home import ENV_VAR, model_root_kwargs


class _FakeSession:
    def get_providers(self):
        return ["CPUExecutionProvider"]


class _FakeModel:
    session = _FakeSession()

    def get_feat(self, *a, **k):   # the embedding extractor looks for this
        return None


class _FakeFaceAnalysis:
    """Records the kwargs InsightFace would have been constructed with."""

    calls: list = []

    def __init__(self, name=None, providers=None, **kwargs):
        self.models = {"recognition": _FakeModel()}
        type(self).calls.append({"name": name, "providers": providers, **kwargs})

    def prepare(self, **kwargs):
        pass


@pytest.fixture
def fake_detection(monkeypatch):
    _FakeFaceAnalysis.calls = []
    monkeypatch.setattr(fa_mod, "FaceAnalysis", _FakeFaceAnalysis)
    return _FakeFaceAnalysis


@pytest.fixture
def fake_embedding(monkeypatch):
    """The embedding extractor imports FaceAnalysis inside the method, so patch the source module."""
    _FakeFaceAnalysis.calls = []
    import insightface.app

    monkeypatch.setattr(insightface.app, "FaceAnalysis", _FakeFaceAnalysis)
    return _FakeFaceAnalysis


class ut_InsightFaceHome:
    def test_env_var_becomes_the_root_kwarg(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert model_root_kwargs() == {"root": str(tmp_path)}

    def test_unset_means_no_kwarg_so_insightface_applies_its_default(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert model_root_kwargs() == {}

    def test_empty_value_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "")
        assert model_root_kwargs() == {}


class ut_DetectionLoader:
    def test_env_var_redirects_the_model_root(self, fake_detection, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "volume_models"))
        InsightFaceFaceAnalyzer({"device": "cpu"})._load_app()
        assert fake_detection.calls[0]["root"] == str(tmp_path / "volume_models")

    def test_unset_env_keeps_the_upstream_default(self, fake_detection, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        InsightFaceFaceAnalyzer({"device": "cpu"})._load_app()
        assert "root" not in fake_detection.calls[0]

    def test_the_cuda_provider_is_still_requested_on_gpu(self, fake_detection, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "/root/models/insightface")
        InsightFaceFaceAnalyzer({"device": "cuda"})._load_app()
        providers = fake_detection.calls[0]["providers"]
        assert providers[0][0] == "CUDAExecutionProvider"       # spec-109 P1 regression guard
        assert providers[-1] == "CPUExecutionProvider"


class ut_EmbeddingLoader:
    """`extract_face_embeddings` loads its own FaceAnalysis - missing it keeps the re-download."""

    def test_env_var_redirects_the_model_root(self, fake_embedding, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "volume_models"))
        InsightFaceNativeExtractor({"device": "cpu"})._get_app()
        assert fake_embedding.calls[0]["root"] == str(tmp_path / "volume_models")

    def test_unset_env_keeps_the_upstream_default(self, fake_embedding, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        InsightFaceNativeExtractor({"device": "cpu"})._get_app()
        assert "root" not in fake_embedding.calls[0]

    def test_the_cuda_provider_is_still_requested_on_gpu(self, fake_embedding, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "/root/models/insightface")
        InsightFaceNativeExtractor({"device": "cuda"})._get_app()
        providers = fake_embedding.calls[0]["providers"]
        assert providers[0][0] == "CUDAExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"
