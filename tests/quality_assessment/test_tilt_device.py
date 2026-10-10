"""Offline regression tests for GeoCalib device routing."""

from unittest.mock import Mock

import pytest

import sim_bench.quality_assessment.tilt_geocalib as tilt_module


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scorer_forwards_device_to_raw_estimator(monkeypatch, device):
    raw_estimator = Mock(return_value=tilt_module.RawTilt(
        angle_deg=8.0, roll_uncertainty_deg=0.0))
    monkeypatch.setattr(tilt_module, "estimate_tilt_raw", raw_estimator)

    result = tilt_module.TiltScorer(device=device).calc(
        tilt_module.TiltInputs(image_paths=["photo.jpg"]))

    raw_estimator.assert_called_once_with("photo.jpg", device=device)
    assert result.angles == {"photo.jpg": 8.0}
    assert result.confidences == {"photo.jpg": 1.0}
    assert result.skipped == []
