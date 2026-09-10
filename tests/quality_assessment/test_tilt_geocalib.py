"""
Unit tests for the GeoCalib tilt backend -- spec-100 T1.2.

Marked `slow`: GeoCalib loads a torch model (~seconds) and runs ~2 s/inference on
CPU. Recovery accuracy at scale is the benchmark's job (spec-100 T2, 122 real
photos); these tests pin the *contract* and the *abstention* behaviour that the
spec-099 penalty depends on, plus a sign/recovery smoke check when the Budapest
album is present on this machine.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from sim_bench.quality_assessment.tilt import TiltResult
from sim_bench.quality_assessment.tilt_geocalib import estimate_tilt, estimate_tilt_raw

pytestmark = [pytest.mark.slow, pytest.mark.needs_data]

_BUDAPEST = Path(r"D:\Budapest2025_Google")


def _rotated_crop(rgb: np.ndarray, deg: float) -> np.ndarray:
    """Rotate content clockwise by deg (spec-099 sign), central 72% crop."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    rot = cv2.warpAffine(rgb, m, (w, h))
    mh, mw = int(0.14 * h), int(0.14 * w)
    return rot[mh:h - mh, mw:w - mw]


class ut_TiltGeoCalibContract:
    def test_returns_tiltresult_with_valid_ranges(self):
        img = np.full((400, 600, 3), 128, np.uint8)
        r = estimate_tilt(img)
        assert isinstance(r, TiltResult)
        assert 0.0 <= r.confidence <= 1.0
        assert r.n_lines == 0  # learned backend keeps the field for contract parity

    def test_accepts_grayscale_input(self):
        gray = np.full((300, 400), 100, np.uint8)
        r = estimate_tilt(gray)  # must not raise on 2-D input
        assert isinstance(r, TiltResult)


class ut_TiltGeoCalibAbstains:
    def test_structureless_image_low_confidence(self):
        """A guessed tilt must not move a score: flat + noise => high uncertainty."""
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        raw = estimate_tilt_raw(noise)
        assert raw.roll_uncertainty_deg > 3.0  # model is unsure
        assert estimate_tilt(noise).confidence < 0.5


@pytest.mark.skipif(not _BUDAPEST.exists(), reason="Budapest album not on this machine")
class ut_TiltGeoCalibRecovery:
    def _load(self) -> np.ndarray:
        from PIL import Image, ImageOps
        p = sorted(_BUDAPEST.glob("*.jpg"))[0]
        with Image.open(p) as pil:
            rgb = np.array(ImageOps.exif_transpose(pil).convert("RGB"))
        s = 1400 / max(rgb.shape[:2])
        return cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)))

    def test_sign_matches_spec099_convention(self):
        """Injecting +clockwise must increase the reported (clockwise-positive) roll."""
        rgb = self._load()
        base = estimate_tilt_raw(rgb).angle_deg
        plus = estimate_tilt_raw(_rotated_crop(rgb, 6.0)).angle_deg
        assert (plus - base) > 2.0  # positive injection -> positive delta

    def test_relative_recovery_within_tolerance(self):
        rgb = self._load()
        base = estimate_tilt_raw(rgb).angle_deg
        for d in (6.0, -6.0):
            got = estimate_tilt_raw(_rotated_crop(rgb, d)).angle_deg
            assert abs((got - base) - d) < 2.5
