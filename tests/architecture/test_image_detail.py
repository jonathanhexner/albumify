"""spec-033 P-D / P-E: RunStore.image_detail completeness contract.

Rules:
  1. ImageDetail is a Pydantic BaseModel with extra="forbid" — typo'd
     attribute names at construction raise ValidationError.
  2. RunStore exposes ``image_detail(image_path)`` as the canonical single
     join. The method must reach faces + filter_decisions + cluster_assignments
     in one call (verified by inspecting the SQL used).
  3. End-to-end: a synthetic run with one image, two faces, and one
     image-level + two face-level filter decisions returns a fully populated
     ImageDetail with non-NULL fields where the schema marks them required.
"""
from __future__ import annotations

import inspect
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from face_cluster.image_detail import FaceDetail, FaceFilterDecision, ImageDetail
from face_cluster.run_store import RunStore


def test_image_detail_forbids_extra_fields():
    with pytest.raises(ValidationError):
        ImageDetail(image_path="x.jpg", bogus_field=1)  # type: ignore[call-arg]


def test_face_detail_forbids_extra_fields():
    with pytest.raises(ValidationError):
        FaceDetail(
            face_id=0, bbox=(0, 0, 1, 1), blur_score=0.0, area=0.0, is_core=False,
            unknown=1,  # type: ignore[call-arg]
        )


def test_run_store_exposes_image_detail():
    """spec-033 P-D acceptance — the method must exist with the right signature."""
    sig = inspect.signature(RunStore.image_detail)
    assert "image_path" in sig.parameters, (
        "RunStore.image_detail(image_path) is the spec-033 P-D canonical API."
    )


def test_image_detail_queries_load_bearing_tables():
    """The method must touch faces, cluster_assignments, and filter_decisions."""
    src = inspect.getsource(RunStore.image_detail)
    for table in ("faces", "cluster_assignments", "filter_decisions"):
        assert re.search(rf"\bFROM {table}\b", src), (
            f"RunStore.image_detail must query {table!r} — that's the spec-033 P-D "
            "single-join contract. Found:\n" + src[:500]
        )


def _build_synthetic_run(run_dir: Path) -> None:
    """Construct a minimal v4 run dir with one image, two faces, decisions."""
    from face_cluster.db import SCHEMA_DDL, SCHEMA_VERSION

    run_dir.mkdir(parents=True, exist_ok=True)
    db_path = run_dir / "face_clustering.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

        image_path = "D:/photos/img.jpg"
        # 21 columns matching the schema order — see run_exporter._write_faces_and_scores.
        for fid in (0, 1):
            conn.execute(
                "INSERT INTO faces VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fid, image_path, "img.jpg", fid,
                    0.0, 0.0, 100.0, 100.0,
                    f"crops/{fid}.jpg",
                    0.95, 100.0, 10000.0, 5.0, 2.0, 1.0,
                    1, None,
                    0.8, 0.7, 0.6, 3,
                ),
            )
            conn.execute(
                "INSERT INTO face_scores VALUES (?,?,?,?,?,?)",
                (fid, 0.9, 0.8, 0.7, 0.6, 1),
            )
            conn.execute(
                "INSERT INTO cluster_assignments VALUES (?,?,?,?,?)",
                (fid, 7, 0, 1 if fid == 0 else 0, 0.1),
            )

        # filter_decisions: image-level + per-face
        conn.execute(
            "INSERT INTO filter_decisions VALUES (?,?,?,?,?,?,?)",
            (image_path, "image", None, "image_quality", 0, "passed", json.dumps({"iqa": 0.8})),
        )
        conn.execute(
            "INSERT INTO filter_decisions VALUES (?,?,?,?,?,?,?)",
            ("0", "face", image_path, "face_blur", 0, "passed", json.dumps({"blur_score": 100.0})),
        )

        # run_metadata — single row so RunStore.metadata() doesn't blow up.
        conn.execute(
            "INSERT INTO run_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "run-1", "test_album", "fc_app", None, "{}", None, None,
                1, 2, 2, 1, 1, 0, 0,
                "2026-05-15T00:00:00", "2026-05-15T00:00:01",
                SCHEMA_VERSION,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Embeddings file the RunStore validates on construction.
    embeddings = np.zeros((2, 512), dtype=np.float32)
    np.save(run_dir / "embeddings.npy", embeddings)
    np.save(run_dir / "embedding_face_ids.npy", np.array([0, 1], dtype=np.int32))

    (run_dir / "pipeline_run.json").write_text(json.dumps({
        "run_id": "run-1",
        "source_album": "test_album",
        "producer": "fc_app",
        "schema_version": SCHEMA_VERSION,
        "db_path": "face_clustering.db",
        "started_at": "2026-05-15T00:00:00",
        "finished_at": "2026-05-15T00:00:01",
    }))

    crops = run_dir / "crops"
    crops.mkdir(exist_ok=True)
    for fid in (0, 1):
        (crops / f"face_{fid:04d}_aligned.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # tiny JPEG


def test_image_detail_returns_populated_for_synthetic_run(tmp_path):
    run_dir = tmp_path / "run"
    _build_synthetic_run(run_dir)

    store = RunStore(run_dir)
    detail = store.image_detail("D:/photos/img.jpg")

    assert detail.image_path == "D:/photos/img.jpg"
    assert detail.iqa_score == pytest.approx(0.8)
    assert detail.ava_score == pytest.approx(0.7)
    assert detail.sharpness_score == pytest.approx(0.6)
    assert detail.scene_cluster_id == 3
    assert len(detail.faces) == 2
    # First face should be the exemplar of cluster 7 (per the synthetic fixture).
    assert detail.faces[0].cluster_id == 7
    assert detail.faces[0].is_exemplar is True
    assert detail.faces[1].is_exemplar is False
    # Image-level filter decision is present.
    assert len(detail.image_filter_decisions) == 1
    assert detail.image_filter_decisions[0].filter_name == "image_quality"
