"""spec-070 — face debug overlay geometry + drawing.

Pure helpers (no Streamlit). ``pose_axes_2d`` projects the 3 head-pose axes
from (yaw, pitch, roll); ``draw_overlay`` renders bbox + landmarks + axes onto
an image with cv2. Shared by the pipeline ``render_face_overlays`` step and the
live UI overlay so the projection math lives in exactly one place (unit-tested).

Coordinate convention: image pixels, +y DOWN (standard image/cv2). The Plotly
UI flips y where needed. Axis colours follow the usual head-pose convention:
X = red (subject left/right), Y = green (up/down), Z = blue (out of the face).
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

Line = Tuple[float, float, float, float]  # (x0, y0, x1, y1), image px, +y down


def pose_axes_2d(
    center: Sequence[float],
    scale: float,
    yaw: float,
    pitch: float,
    roll: float,
) -> Dict[str, Line]:
    """Project the 3 head-pose axes to 2D line segments from ``center``.

    Args:
        center: (cx, cy) image-pixel anchor (usually the bbox centre).
        scale: axis length in pixels.
        yaw, pitch, roll: head pose in DEGREES (the repo (yaw,pitch,roll)
            convention — already remapped from InsightFace at the detector).

    Returns:
        ``{"x": line, "y": line, "z": line}`` where each line is
        ``(cx, cy, x_end, y_end)`` in image px (+y down). Standard head-pose
        projection: at yaw=pitch=roll=0 the X axis is horizontal, Y vertical,
        Z degenerate (points straight out of the screen).
    """
    cx, cy = float(center[0]), float(center[1])
    y = math.radians(yaw)
    p = math.radians(pitch)
    r = math.radians(roll)
    s = float(scale)

    # X axis (red).
    x_end = s * (math.cos(y) * math.cos(r))
    y_end = s * (math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y))
    line_x = (cx, cy, cx + x_end, cy + y_end)

    # Y axis (green).
    x_end = s * (-math.cos(y) * math.sin(r))
    y_end = s * (math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r))
    line_y = (cx, cy, cx + x_end, cy + y_end)

    # Z axis (blue) — out of the face.
    x_end = s * (math.sin(y))
    y_end = s * (-math.cos(y) * math.sin(p))
    line_z = (cx, cy, cx + x_end, cy + y_end)

    return {"x": line_x, "y": line_y, "z": line_z}


def draw_overlay(
    img,
    bbox: Optional[Sequence[float]] = None,
    landmarks=None,
    pose: Optional[Sequence[float]] = None,
    scale: Optional[float] = None,
):
    """Return a copy of ``img`` (BGR ndarray) with bbox + landmarks + pose axes.

    ``bbox`` = (x1, y1, x2, y2) image px. ``pose`` = (yaw, pitch, roll) deg or
    None. Anything None is skipped. Used by the pipeline render step.
    """
    import cv2  # local import — keep the module importable without cv2

    out = img.copy()
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        if scale is None:
            scale = 0.5 * min(abs(x2 - x1), abs(y2 - y1))
    else:
        cy, cx = out.shape[0] / 2.0, out.shape[1] / 2.0
        if scale is None:
            scale = 0.3 * min(out.shape[0], out.shape[1])

    if landmarks is not None:
        for pt in landmarks:
            cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 200, 0), -1)

    if pose is not None and len(pose) == 3 and all(v is not None for v in pose):
        axes = pose_axes_2d((cx, cy), scale, pose[0], pose[1], pose[2])
        colours = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}  # BGR
        for k, (ax0, ay0, ax1, ay1) in axes.items():
            cv2.line(out, (int(ax0), int(ay0)), (int(ax1), int(ay1)), colours[k], 2)

    return out


__all__ = ["pose_axes_2d", "draw_overlay", "Line"]
