"""Unit tests for spec-099 tilt estimation."""

import cv2
import numpy as np
import pytest

from sim_bench.quality_assessment.tilt import TiltResult, estimate_tilt


def _structured_scene(size: int = 640) -> np.ndarray:
    """Synthetic upright scene: strong horizontal + vertical line structure."""
    img = np.full((size, size), 200, dtype=np.uint8)
    for y in range(80, size, 120):        # horizontals (horizon, shelves)
        cv2.line(img, (0, y), (size - 1, y), 60, 3)
    for x in range(100, size, 160):       # verticals (walls, frames)
        cv2.line(img, (x, 0), (x, size - 1), 90, 3)
    cv2.rectangle(img, (150, 200), (420, 470), 30, 2)
    return img


def _rotated(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate content by angle_deg (positive = clockwise), keep central crop
    so black corners never enter the estimate."""
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_deg, 1.0)
    rot = cv2.warpAffine(img, m, (w, h), borderValue=200)
    margin = int(0.2 * min(h, w))
    return rot[margin:h - margin, margin:w - margin]


class TestEstimateTilt:
    @pytest.mark.parametrize("angle", [-8.0, -4.0, -2.0, 2.0, 4.0, 8.0])
    def test_known_rotation_recovered(self, angle):
        result = estimate_tilt(_rotated(_structured_scene(), angle))
        assert result.confidence > 0.3
        assert result.angle_deg == pytest.approx(angle, abs=1.0)

    def test_upright_scene_near_zero(self):
        result = estimate_tilt(_structured_scene())
        assert abs(result.angle_deg) < 0.5
        assert result.confidence > 0.3

    def test_sign_convention_clockwise_positive(self):
        assert estimate_tilt(_rotated(_structured_scene(), 6.0)).angle_deg > 0
        assert estimate_tilt(_rotated(_structured_scene(), -6.0)).angle_deg < 0

    def test_structureless_image_low_confidence(self):
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, (640, 640), dtype=np.uint8)
        result = estimate_tilt(noise)
        assert result.confidence < 0.2  # must abstain -> zero penalty downstream

    def test_blank_image_zero_confidence(self):
        result = estimate_tilt(np.full((480, 640), 128, dtype=np.uint8))
        assert result == TiltResult(0.0, 0.0, 0)

    def test_diagonal_only_scene_abstains_or_agrees_low(self):
        """45-degree diagonals belong to neither axis family: no false tilt."""
        img = np.full((640, 640), 200, dtype=np.uint8)
        for off in range(-640, 640, 90):
            cv2.line(img, (0, off), (640, off + 640), 60, 3)
        result = estimate_tilt(img)
        assert result.confidence < 0.2 or abs(result.angle_deg) < 1.0

    def test_rejects_color_input(self):
        with pytest.raises(ValueError):
            estimate_tilt(np.zeros((10, 10, 3), dtype=np.uint8))

    def test_large_image_downscaled_not_crashing(self):
        big = cv2.resize(_structured_scene(), (2600, 1950))
        result = estimate_tilt(big)
        assert abs(result.angle_deg) < 1.0 and result.confidence > 0.2
