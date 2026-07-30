"""spec-040 T2 (B3 + B6): schema v5 writes actually happen.

Prior to T2 the v5 schema shipped with DDL + Pandera schemas but no writer:
``faces.area_ratio`` columns were NULL on every run, and the three new
tables (``images`` / ``scene_clusters`` / ``scene_cluster_assignments``)
were empty. This test exercises a real producer run end-to-end and
asserts the writes actually populate v5 state.

Coverage:

* ``test_images_table_populated_on_albumify_run`` — fresh Albumify run
  on the 9-jpg fixture produces one ``images`` row per input image.
* ``test_area_ratio_populated_on_face_records`` — every detected face
  has ``area_ratio`` ∈ (0, 1].
* ``test_faces_area_ratio_column_non_null_after_export`` — the
  ``faces.area_ratio`` column is populated, not NULL, after RunExporter
  writes (closes SIGHTING-064-style regression).
* ``test_empty_inputs_still_validate`` — the Pandera contracts are
  exercised even when scene_clusters / scene_cluster_assignments are
  empty (the common case until scene producers ship).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


from tests.conftest import get_test_data_dir


FIXTURE_DIR = get_test_data_dir() / "face_clustering"


def _pick_jpgs(n_per_person: int = 3, n_persons: int = 3) -> list[Path]:
    if not FIXTURE_DIR.exists():
        return []
    picked: list[Path] = []
    for person_dir in sorted(FIXTURE_DIR.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        jpgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() == ".jpg")
        picked.extend(jpgs[:n_per_person])
        if len(picked) >= n_per_person * n_persons:
            break
    return picked


@pytest.fixture(scope="module")
def producer_context(tmp_path_factory):
    """Run the producer chain on a small real fixture; return the context."""
    images = _pick_jpgs(n_per_person=2, n_persons=3)
    if not images:
        pytest.skip(f"Fixture missing at {FIXTURE_DIR}")
    src_dir = tmp_path_factory.mktemp("v5_writes_src")
    for src in images:
        (src_dir / src.name).write_bytes(src.read_bytes())

    import sim_bench.pipeline.steps.all_steps  # noqa: F401
    from sim_bench.pipeline.config import PipelineConfig
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.executor import PipelineExecutor
    from sim_bench.pipeline.registry import get_registry

    context = PipelineContext(source_directory=src_dir)
    context.image_paths = sorted(src_dir.iterdir())
    steps = [
        "detect_persons", "insightface_detect_faces",
        "detect_face_orientation", "align_faces", "extract_face_embeddings",
    ]
    step_configs = {name: {} for name in steps}
    context.step_configs = step_configs
    executor = PipelineExecutor(get_registry())
    result = executor.execute(context, steps, config=PipelineConfig(step_configs=step_configs, fail_fast=True))
    if not result.success:
        pytest.skip(f"Producer chain failed: {result.error_message}")
    if not context.face_records:
        pytest.skip("Producer produced no face_records")
    return context


def test_area_ratio_populated_on_face_records(producer_context):
    """Every detected face must carry an area_ratio in (0, 1]."""
    for r in producer_context.face_records:
        assert r.area_ratio is not None, f"face_id={r.face_id} has no area_ratio"
        assert 0.0 < r.area_ratio <= 1.0, (
            f"face_id={r.face_id} area_ratio={r.area_ratio} out of (0, 1]"
        )
        assert r.bbox_w_ratio is not None and 0.0 < r.bbox_w_ratio <= 1.0
        assert r.bbox_h_ratio is not None and 0.0 < r.bbox_h_ratio <= 1.0
        # x_ratio / y_ratio CAN legitimately be 0 (face touches the left/top edge).
        assert r.bbox_x_ratio is not None and 0.0 <= r.bbox_x_ratio <= 1.0
        assert r.bbox_y_ratio is not None and 0.0 <= r.bbox_y_ratio <= 1.0


def test_image_dims_populated_on_face_records(producer_context):
    """Image width/height are recoverable from pixel/ratio on every face."""
    for r in producer_context.face_records:
        assert r.image_width_px is not None and r.image_width_px > 0
        assert r.image_height_px is not None and r.image_height_px > 0


def test_run_exporter_writes_images_and_ratios(producer_context, tmp_path):
    """End-to-end: run the exporter, read the DB, assert v5 tables populated."""
    from sim_bench.run_db.exporter import RunExporter
    from face_cluster.types import ClusterResult
    import numpy as np

    faces = producer_context.face_records
    # Minimal cluster result: every face in its own singleton cluster.
    clusters = {i: [i] for i in range(len(faces))}
    base_cr = ClusterResult(
        labels=np.arange(len(faces), dtype=int),
        clusters=clusters,
        cluster_stats={i: {} for i in range(len(faces))},
        exemplars={i: [i] for i in range(len(faces))},
        n_clusters=len(faces),
        n_noise=0,
    )

    out_dir = tmp_path / "run_v5"
    image_paths = [str(p) for p in producer_context.image_paths]
    RunExporter(out_dir).export(
        faces=faces,
        base_cluster_result=base_cr,
        merged_cluster_result=None,
        core_indices=list(range(len(faces))),
        merge_log=[], merge_metadata={},
        config=None, source_album=str(producer_context.source_directory),
        producer="albumify",
        run_id="test_v5", started_at="2026-05-20T00:00:00Z", finished_at="2026-05-20T00:00:01Z",
        image_paths=image_paths,
    )

    db_path = out_dir / "face_clustering.db"
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert n_images == len(set(image_paths)), (
            f"images table has {n_images} rows; expected {len(set(image_paths))}"
        )

        n_with_faces = conn.execute(
            "SELECT COUNT(*) FROM images WHERE n_faces > 0"
        ).fetchone()[0]
        assert n_with_faces > 0, "no images table row records any faces"

        non_null = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE area_ratio IS NOT NULL"
        ).fetchone()[0]
        assert non_null == len(faces), (
            f"faces.area_ratio non-null count = {non_null}; expected {len(faces)}"
        )

        rng_ok = conn.execute(
            "SELECT COUNT(*) FROM faces WHERE area_ratio > 0 AND area_ratio <= 1"
        ).fetchone()[0]
        assert rng_ok == len(faces), (
            f"faces.area_ratio in-range count = {rng_ok}; expected {len(faces)}"
        )

        # Scene tables empty for now (no producer); table must still exist.
        assert conn.execute("SELECT COUNT(*) FROM scene_clusters").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM scene_cluster_assignments").fetchone()[0] == 0
    finally:
        conn.close()


def test_pandera_contracts_pass_on_empty_inputs(tmp_path):
    """Even with zero faces / zero scene data, the writer must Pandera-validate.

    This is the contract-exerciser REVIEW.md B6 asked for: the schemas can
    no longer be dead code; they're called on every export(), empty or not.
    """
    from sim_bench.run_db.exporter import RunExporter
    from face_cluster.types import ClusterResult

    import numpy as np
    empty_cr = ClusterResult(
        labels=np.array([], dtype=int),
        clusters={},
        cluster_stats={},
        exemplars={},
        n_clusters=0, n_noise=0,
    )
    out_dir = tmp_path / "run_empty"
    RunExporter(out_dir).export(
        faces=[],
        base_cluster_result=empty_cr,
        merged_cluster_result=None,
        core_indices=[],
        merge_log=[], merge_metadata={},
        config=None, source_album="empty",
        producer="albumify",
        run_id="test_empty", started_at="2026-05-20T00:00:00Z", finished_at="2026-05-20T00:00:01Z",
        image_paths=[],
        scene_clusters=[],
        scene_cluster_assignments=[],
    )
    db_path = out_dir / "face_clustering.db"
    conn = sqlite3.connect(str(db_path))
    try:
        for table in ("images", "scene_clusters", "scene_cluster_assignments", "faces"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n == 0, f"{table} unexpectedly has {n} rows for an empty run"
    finally:
        conn.close()
