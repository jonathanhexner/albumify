"""Helper functions for debugging."""

import re
import glob
import numpy as np
from pathlib import Path


def extract_face_id_from_filename(filename: str) -> int:
    """Extract numeric face_id from 'face_0123_aligned.jpg'."""
    match = re.search(r'face_(\d+)_aligned', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot extract face_id from filename: {filename}")


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine distance between embeddings."""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    distance = 1.0 - similarity
    return distance


def resolve_glob_path(pattern: str) -> Path:
    """Resolve glob pattern to single file path."""
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No files match pattern: {pattern}")
    if len(matches) > 1:
        print(f"Warning: Multiple files match pattern {pattern}, using first: {matches[0]}")
    return Path(matches[0])
