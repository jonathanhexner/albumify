"""Group G: per-cluster quality and pose distribution features."""

from typing import List, Dict, Any, Optional
import numpy as np


def _pose_stats(nodes: List[int], faces: List[Any], frontal_threshold: float):
    """Return (frontal_frac, mean_yaw, mean_pitch, yaw_std) for a cluster."""
    yaws, pitches = [], []
    for idx in nodes:
        pose = faces[idx].pose
        if pose is not None and pose != (0.0, 0.0, 0.0):
            yaw, pitch, _ = pose
            yaws.append(yaw)
            pitches.append(pitch)

    if not yaws:
        return 1.0, 0.0, 0.0, 0.0

    frontal = sum(
        1 for y, p in zip(yaws, pitches)
        if abs(y) <= frontal_threshold and abs(p) <= frontal_threshold
    )
    return (
        frontal / len(yaws),
        float(np.mean(yaws)),
        float(np.mean(pitches)),
        float(np.std(yaws)),
    )


def compute_quality_features(
    nodes_a: List[int],
    nodes_b: List[int],
    faces: List[Any],
    frontal_threshold: float = 15.0,
) -> Dict[str, float]:
    """Compute quality and pose distribution features for a cluster pair.

    Returns a flat dict of feature name -> value.
    """
    blurs_a = [faces[i].blur_score for i in nodes_a]
    blurs_b = [faces[i].blur_score for i in nodes_b]
    areas_a = [faces[i].area for i in nodes_a]
    areas_b = [faces[i].area for i in nodes_b]

    mean_blur_a = float(np.mean(blurs_a))
    mean_blur_b = float(np.mean(blurs_b))
    mean_area_a = float(np.mean(areas_a))
    mean_area_b = float(np.mean(areas_b))
    area_max = max(mean_area_a, mean_area_b)
    area_min = min(mean_area_a, mean_area_b)

    ff_a, yaw_a, pitch_a, yaw_std_a = _pose_stats(nodes_a, faces, frontal_threshold)
    ff_b, yaw_b, pitch_b, yaw_std_b = _pose_stats(nodes_b, faces, frontal_threshold)

    pose_diff = float(np.sqrt((yaw_a - yaw_b) ** 2 + (pitch_a - pitch_b) ** 2))

    return {
        "mean_blur_a": mean_blur_a,
        "mean_blur_b": mean_blur_b,
        "blur_min_a": float(min(blurs_a)),
        "blur_min_b": float(min(blurs_b)),
        "frontal_frac_a": ff_a,
        "frontal_frac_b": ff_b,
        "frontal_frac_min": min(ff_a, ff_b),
        "pose_diff": pose_diff,
        "yaw_std_a": yaw_std_a,
        "yaw_std_b": yaw_std_b,
        "mean_area_a": mean_area_a,
        "mean_area_b": mean_area_b,
        "area_ratio": area_max / area_min if area_min > 0 else 1.0,
    }
