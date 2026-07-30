"""Artifact writer for embeddings.npy + embedding_face_ids.npy (spec-057).

Extracted from RunExporter._write_embeddings_npy.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from face_cluster.types import FaceRecord


_EMB_DIM = 512


def write_embeddings(output_dir: Path, faces: List[FaceRecord]) -> None:
    """Save the (n_faces, 512) float32 matrix + the matching face_id array.

    Row i of embeddings.npy corresponds to face_ids[i] in
    embedding_face_ids.npy — they must be loaded together.
    """
    matrix = np.zeros((len(faces), _EMB_DIM), dtype=np.float32)
    face_ids = np.array([f.face_id for f in faces], dtype=np.int32)
    for i, face in enumerate(faces):
        emb = (
            face.embedding_normalized
            if face.embedding_normalized is not None
            else face.embedding
        )
        if emb is not None:
            matrix[i] = np.asarray(emb, dtype=np.float32)
    np.save(output_dir / "embeddings.npy", matrix)
    np.save(output_dir / "embedding_face_ids.npy", face_ids)
