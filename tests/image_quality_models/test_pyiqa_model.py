"""ut_PyIQAModel (spec-093) - direction normalization, availability, registry.

Direction logic is tested offline with a fake pyiqa metric (deterministic, no
weight download). One real BRISQUE check confirms the live integration; BRISQUE
is classical (no network), so it is safe in CI.
"""

import numpy as np
import pytest
from PIL import Image

from sim_bench.image_quality_models import create_model, MODEL_REGISTRY
from sim_bench.image_quality_models.pyiqa_model_wrapper import PyIQAModel, PYIQA_METRICS


@pytest.fixture
def real_image(tmp_path):
    p = tmp_path / "img.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(p, "JPEG")
    return p


class _FakeMetric:
    def __init__(self, value, lower_better):
        self._value = value
        self.lower_better = lower_better

    def __call__(self, _path):
        import torch
        return torch.tensor(self._value)


def _patch_metric(monkeypatch, value, lower_better):
    import pyiqa
    monkeypatch.setattr(
        pyiqa, "create_metric",
        lambda name, device="cpu": _FakeMetric(value, lower_better),
    )


def test_all_pyiqa_metrics_registered():
    for m in PYIQA_METRICS:
        assert MODEL_REGISTRY[m] is PyIQAModel


def test_is_available_true_when_pyiqa_installed():
    # pyiqa is a hard dependency of spec-093; the spike installed it.
    assert PyIQAModel.is_available() is True


def test_lower_better_metric_is_negated(monkeypatch, real_image):
    _patch_metric(monkeypatch, value=19.7, lower_better=True)
    model = create_model({"type": "brisque", "device": "cpu"})
    assert model.lower_better is True
    assert model.raw_score(real_image) == pytest.approx(19.7)
    # higher=better contract -> distortion score is flipped
    assert model.score_image(real_image) == pytest.approx(-19.7)


def test_higher_better_metric_is_passthrough(monkeypatch, real_image):
    _patch_metric(monkeypatch, value=0.34, lower_better=False)
    model = create_model({"type": "maniqa", "device": "cpu"})
    assert model.lower_better is False
    assert model.score_image(real_image) == pytest.approx(0.34)
    assert model.raw_score(real_image) == pytest.approx(0.34)


def test_from_config_uses_type_as_metric_name(monkeypatch, real_image):
    _patch_metric(monkeypatch, value=1.0, lower_better=False)
    model = create_model({"type": "clipiqa", "device": "cpu"})
    assert model.metric_name == "clipiqa"


def test_real_brisque_offline(real_image):
    """Live pyiqa path with the real (classical, no-download) BRISQUE metric."""
    model = create_model({"type": "brisque", "device": "cpu"})
    assert model.lower_better is True
    score = model.score_image(real_image)
    raw = model.raw_score(real_image)
    assert isinstance(score, float)
    assert score == pytest.approx(-raw)
