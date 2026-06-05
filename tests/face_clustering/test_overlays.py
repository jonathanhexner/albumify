"""spec-070 — unit tests for the pose-axis projection + overlay drawing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from face_cluster.overlays import draw_overlay, pose_axes_2d


def _vec(line):
    """(dx, dy) of an axis line from its anchor."""
    x0, y0, x1, y1 = line
    return (x1 - x0, y1 - y0)


def test_frontal_pose_x_horizontal_y_vertical_z_degenerate():
    """AC1: at yaw=pitch=roll=0, X is horizontal, Y vertical, Z ~zero-length."""
    axes = pose_axes_2d((100, 100), scale=50, yaw=0, pitch=0, roll=0)
    dx, dy = _vec(axes["x"])
    assert dx == pytest.approx(50, abs=1e-6) and dy == pytest.approx(0, abs=1e-6)
    dx, dy = _vec(axes["y"])
    assert dx == pytest.approx(0, abs=1e-6) and dy == pytest.approx(50, abs=1e-6)
    dx, dy = _vec(axes["z"])
    assert math.hypot(dx, dy) == pytest.approx(0, abs=1e-6)


def test_yaw_turns_z_axis_sideways():
    """AC6: positive yaw swings the Z (forward) axis horizontally — the
    signature that yaw is wired to the right angle (not pitch)."""
    axes = pose_axes_2d((0, 0), scale=100, yaw=90, pitch=0, roll=0)
    dx, dy = _vec(axes["z"])
    assert dx == pytest.approx(100, abs=1e-4)   # sin(90)=1
    assert dy == pytest.approx(0, abs=1e-4)


def test_pitch_moves_z_vertically():
    """Pitch (not yaw) drives the vertical component of the Z axis. At
    pitch=90 the Z axis points fully up: dx=0, dy=-scale (since +y is down)."""
    axes = pose_axes_2d((0, 0), scale=100, yaw=0, pitch=90, roll=0)
    dx, dy = _vec(axes["z"])
    assert dx == pytest.approx(0, abs=1e-4)
    assert dy == pytest.approx(-100, abs=1e-4)   # +y down → pitch-up is negative
    # And a partial pitch still tilts Z upward.
    _, dy45 = _vec(pose_axes_2d((0, 0), scale=100, yaw=0, pitch=45, roll=0)["z"])
    assert dy45 < 0


def test_anchor_is_respected():
    axes = pose_axes_2d((30, 40), scale=10, yaw=0, pitch=0, roll=0)
    for line in axes.values():
        assert line[0] == 30 and line[1] == 40


def test_draw_overlay_changes_pixels_and_keeps_shape():
    """AC2: drawing bbox + landmarks + pose produces a different image of the
    same shape (didn't crash, drew something)."""
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    out = draw_overlay(
        img,
        bbox=(20, 20, 100, 100),
        landmarks=[(40, 50), (70, 50)],
        pose=(15.0, -10.0, 2.0),
    )
    assert out.shape == img.shape
    assert int(np.abs(out.astype(int) - img.astype(int)).sum()) > 0


def test_draw_overlay_none_pose_is_safe():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    out = draw_overlay(img, bbox=(5, 5, 45, 45), landmarks=None, pose=None)
    assert out.shape == img.shape
