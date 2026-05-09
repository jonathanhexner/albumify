"""Group D: source image diversity and shared-image features."""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def _image_stems(nodes: List[int], faces: List[Any]) -> set:
    """Unique image stems for the given face indices."""
    stems = set()
    for idx in nodes:
        path = faces[idx].image_path
        if path:
            stems.add(Path(path).stem)
    return stems


def compute_source_image_features(
    nodes_a: List[int],
    nodes_b: List[int],
    faces: List[Any],
    distance_matrix: np.ndarray,
) -> Dict[str, float]:
    """Compute source image diversity and shared-image anti-merge features.

    Returns a flat dict of feature name -> value.
    """
    images_a = _image_stems(nodes_a, faces)
    images_b = _image_stems(nodes_b, faces)
    shared = images_a & images_b

    n_images_a = len(images_a)
    n_images_b = len(images_b)
    n_shared = len(shared)
    n_min = min(n_images_a, n_images_b)

    # Minimum cross-cluster distance for face pairs sharing a source image.
    # A small distance + same photo = likely a detection duplicate, not two people.
    same_image_min_dist = _same_image_min_dist(nodes_a, nodes_b, faces, distance_matrix, shared)

    return {
        "n_images_a": n_images_a,
        "n_images_b": n_images_b,
        "shared_source_images": n_shared,
        "shared_source_ratio": n_shared / n_min if n_min > 0 else 0.0,
        "same_image_min_dist": same_image_min_dist,
    }


def _same_image_min_dist(
    nodes_a: List[int],
    nodes_b: List[int],
    faces: List[Any],
    distance_matrix: np.ndarray,
    shared_stems: set,
) -> float:
    """Min distance between cross-cluster pairs whose faces share a source image."""
    if not shared_stems:
        return float("nan")

    min_dist = float("inf")
    for i in nodes_a:
        stem_i = Path(faces[i].image_path).stem if faces[i].image_path else None
        if stem_i not in shared_stems:
            continue
        for j in nodes_b:
            stem_j = Path(faces[j].image_path).stem if faces[j].image_path else None
            if stem_i == stem_j:
                min_dist = min(min_dist, distance_matrix[i, j])

    return float(min_dist) if min_dist < float("inf") else float("nan")
