"""
Auto-straighten: roll correction + inscribed-rectangle crop -- spec-101.

Rotating a photo by -roll to level it exposes triangular corners. Filling them
(black or edge-replicated) invents pixels; the honest fix is to CROP to the
largest axis-aligned rectangle that lies entirely inside the rotated frame --
zero invented pixels, at the cost of a small zoom-in (lost field of view).

Two crop modes:
  * preserve_aspect=True  -> largest centred box with the ORIGINAL W:H ratio
                            (no shape change; the photo default).
  * preserve_aspect=False -> largest-AREA box (aspect floats; classic
                            'rotatedRectWithMaxArea').

Consumes the roll from spec-099/100 (score_tilt). Framework-agnostic
(numpy in / numpy out), spec-053 helper style. Never mutates the caller's array.
"""

import math

import cv2
import numpy as np

_EPS = 1e-9


def largest_inscribed_rect(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Largest-AREA upright rectangle fully inside a w x h image rotated by angle.

    Closed form (rotatedRectWithMaxArea). angle is taken modulo the rectangle's
    symmetry; sign does not matter.
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    a = math.radians(abs(angle_deg) % 180.0)
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    width_is_longer = w >= h
    long_side, short_side = (w, h) if width_is_longer else (h, w)

    if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < _EPS:
        # half-constrained by the short side
        x = 0.5 * short_side
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a
    return max(wr, 0.0), max(hr, 0.0)


def aspect_preserving_rect(w: float, h: float, angle_deg: float) -> tuple[float, float]:
    """Largest centred box with the SAME W:H ratio, inside the rotated w x h frame.

    For a centred box rotated by -a inside the upright frame, its axis-aligned
    bounding half-extents are (p|cos|+q|sin|, p|sin|+q|cos|) for half-sizes p,q.
    Fitting both <= W/2, H/2 with (p,q) = s*(W,H)/2 gives a closed-form scale s.
    """
    if w <= 0 or h <= 0:
        return 0.0, 0.0
    a = math.radians(abs(angle_deg) % 180.0)
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))
    s_w = w / (w * cos_a + h * sin_a)
    s_h = h / (w * sin_a + h * cos_a)
    s = min(s_w, s_h)
    return s * w, s * h


def straighten(rgb: np.ndarray, roll_deg: float, *, preserve_aspect: bool = True,
               interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Level a photo: rotate by -roll, then crop to the inscribed rectangle.

    roll_deg follows the spec-099 convention (+ = content tilted clockwise); the
    image is rotated counter-clockwise by roll_deg to level it. No border fill is
    ever visible in the output -- every output pixel is real source content.
    A ~0 roll returns the input unchanged (identity).
    """
    if abs(roll_deg) < _EPS:
        return rgb
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), roll_deg, 1.0)  # +deg = CCW in cv2
    rot = cv2.warpAffine(rgb, m, (w, h), flags=interp)

    wr, hr = (aspect_preserving_rect(w, h, roll_deg) if preserve_aspect
              else largest_inscribed_rect(w, h, roll_deg))
    cw, ch = int(round(wr)), int(round(hr))
    if cw <= 0 or ch <= 0:
        return rot
    x0 = max(0, (w - cw) // 2)
    y0 = max(0, (h - ch) // 2)
    return rot[y0:y0 + ch, x0:x0 + cw]


def retained_area_fraction(w: float, h: float, angle_deg: float,
                           preserve_aspect: bool = True) -> float:
    """Fraction of the original area kept after the inscribed crop (the fix cost)."""
    if w <= 0 or h <= 0:
        return 0.0
    wr, hr = (aspect_preserving_rect(w, h, angle_deg) if preserve_aspect
              else largest_inscribed_rect(w, h, angle_deg))
    return (wr * hr) / (w * h)
