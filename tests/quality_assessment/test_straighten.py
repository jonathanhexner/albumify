"""ut for auto-straighten geometry + transform (spec-101)."""

import math

import cv2
import numpy as np
import pytest

from sim_bench.quality_assessment.straighten import (
    aspect_preserving_rect,
    largest_inscribed_rect,
    retained_area_fraction,
    straighten,
)


def _brute_max_area(w, h, angle_deg, n=240):
    """Feasible centred upright box maximising area: corners rotated by -a must
    stay within +-W/2, +-H/2 (necessary & sufficient for a centred box)."""
    a = math.radians(abs(angle_deg))
    s, c = abs(math.sin(a)), abs(math.cos(a))
    best = 0.0
    for i in range(1, n + 1):
        p = 0.5 * w * i / n              # half-width
        if p * c > 0.5 * w:             # even q=0 fails
            continue
        # p*c + q*s <= w/2  and  p*s + q*c <= h/2
        q_max = min((0.5 * w - p * c) / (s or 1e9), (0.5 * h - p * s) / (c or 1e9))
        if q_max <= 0:
            continue
        best = max(best, 4.0 * p * q_max)
    return best


class ut_LargestInscribedRect:
    def test_zero_angle_is_full_frame(self):
        assert largest_inscribed_rect(200, 100, 0.0) == pytest.approx((200, 100))

    def test_square_45deg_is_half_area(self):
        wr, hr = largest_inscribed_rect(100, 100, 45.0)
        assert wr * hr == pytest.approx(5000.0, rel=1e-3)  # 100^2 / 2

    @pytest.mark.parametrize("w,h,ang", [(200, 100, 10), (200, 100, 30), (160, 120, 22),
                                         (100, 100, 15), (300, 100, 8)])
    def test_matches_brute_force_max_area(self, w, h, ang):
        wr, hr = largest_inscribed_rect(w, h, ang)
        assert wr * hr == pytest.approx(_brute_max_area(w, h, ang), rel=0.01)

    def test_sign_symmetry(self):
        assert largest_inscribed_rect(200, 100, 17) == pytest.approx(
            largest_inscribed_rect(200, 100, -17))


class ut_AspectPreservingRect:
    def test_ratio_is_preserved(self):
        w, h = 200, 100
        wr, hr = aspect_preserving_rect(w, h, 20.0)
        assert wr / hr == pytest.approx(w / h, rel=1e-6)

    def test_not_larger_than_max_area(self):
        # a same-aspect box is one feasible box, so its area <= the max-area box
        w, h = 200, 100
        ap = aspect_preserving_rect(w, h, 25.0)
        mx = largest_inscribed_rect(w, h, 25.0)
        assert ap[0] * ap[1] <= mx[0] * mx[1] * 1.001

    def test_zero_angle_full_frame(self):
        assert aspect_preserving_rect(200, 100, 0.0) == pytest.approx((200, 100))


class ut_Straighten:
    def test_zero_roll_is_identity(self):
        img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), np.uint8)
        assert np.array_equal(straighten(img, 0.0), img)

    def test_no_invented_pixels_in_crop(self):
        """A1: rotate with a sentinel border; the inscribed crop must exclude it."""
        img = np.full((120, 200, 3), 120, np.uint8)          # uniform interior
        roll = 18.0
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
        SENT = (255, 0, 255)
        rot = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=SENT)
        wr, hr = aspect_preserving_rect(w, h, roll)
        cw, ch = int(round(wr)), int(round(hr))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        crop = rot[y0:y0 + ch, x0:x0 + cw]
        magenta = np.all(crop == np.array(SENT), axis=-1)
        assert not magenta.any()                             # zero border pixels

    def test_preserve_aspect_keeps_ratio(self):
        img = np.zeros((100, 200, 3), np.uint8)
        out = straighten(img, 12.0, preserve_aspect=True)
        assert out.shape[1] / out.shape[0] == pytest.approx(200 / 100, rel=0.02)

    def test_max_area_retains_more_than_aspect(self):
        assert (retained_area_fraction(200, 100, 20, preserve_aspect=False)
                >= retained_area_fraction(200, 100, 20, preserve_aspect=True) - 1e-9)

    def test_output_is_smaller_than_input_for_real_roll(self):
        img = np.zeros((100, 100, 3), np.uint8)
        out = straighten(img, 10.0)
        assert out.shape[0] < 100 and out.shape[1] < 100
