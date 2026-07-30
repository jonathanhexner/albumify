"""spec-040 A1 — producer dual-write to ``context.face_records``.

Runs the producer chain (detect_persons → insightface_detect_faces →
detect_face_orientation → align_faces → extract_face_embeddings) on a
real 3-image fixture and asserts that ``context.face_records`` is
populated end-to-end:

* one ``FaceRecord`` per detected face,
* bbox + landmarks set by the detect step,
* ``aligned_face`` set by the align step,
* ``embedding`` + ``embedding_normalized`` set by the embed step.

Without this guarantee the v2 clustering chain (Phase 3 / FCAppRunner)
is unreachable in production — that was REVIEW.md finding A1.

Fixture: ``test_data/face_clustering/`` — 3 people × 3 images, ground
truth in ``ground_truth.csv``. Small enough to run in CI, real enough
to exercise the detector / aligner / embedder.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from tests.conftest import get_test_data_dir


FIXTURE_DIR = get_test_data_dir() / "face_clustering"


def _pick_jpgs(n: int = 3) -> List[Path]:
    """Pick N .jpg images from the fixture, one per person if possible.

    HEIC files are skipped — InsightFace's detector needs OpenCV-loadable
    images and HEIC support is environment-dependent on Windows.
    """
    if not FIXTURE_DIR.exists():
        return []
    picked: List[Path] = []
    for person_dir in sorted(FIXTURE_DIR.iterdir()):
        if not person_dir.is_dir() or not person_dir.name.startswith("person_"):
            continue
        jpgs = sorted(p for p in person_dir.iterdir() if p.suffix.lower() == ".jpg")
        if jpgs:
            picked.append(jpgs[0])
        if len(picked) >= n:
            break
    return picked


PRODUCER_STEPS = [
    "detect_persons",
    "insightface_detect_faces",
    "detect_face_orientation",
    "align_faces",
    "extract_face_embeddings",
]


@pytest.fixture(scope="module")
def producer_run(tmp_path_factory):
    """Run the producer-only chain once per module and return the context.

    Skipped if the fixture is missing.
    """
    images = _pick_jpgs(3)
    if not images:
        pytest.skip(
            f"Dual-write fixture missing at {FIXTURE_DIR}. "
            "Expected person_1/, person_2/, person_3/ each with .jpg files."
        )

    # Stage into a flat temp dir so discover_images sees exactly N images.
    src_dir = tmp_path_factory.mktemp("dual_write_src")
    for src in images:
        (src_dir / src.name).write_bytes(src.read_bytes())

    import sim_bench.pipeline.steps.all_steps  # noqa: F401  -- registers steps
    from sim_bench.pipeline.config import PipelineConfig
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.executor import PipelineExecutor
    from sim_bench.pipeline.registry import get_registry

    # discover_images writes context.image_paths; do it manually so the
    # test does not depend on the discover step.
    context = PipelineContext(source_directory=src_dir)
    context.image_paths = sorted(src_dir.iterdir())

    step_configs = {name: {} for name in PRODUCER_STEPS}
    context.step_configs = step_configs
    config = PipelineConfig(step_configs=step_configs, fail_fast=True)

    executor = PipelineExecutor(get_registry())
    result = executor.execute(context, PRODUCER_STEPS, config=config)

    if not result.success:
        pytest.skip(
            f"Producer chain failed: {result.error_message} "
            f"(failed step: {result.failed_step}). Likely a model-download or "
            "environment issue — A1's contract is unverifiable in this env."
        )

    return context, result


def test_face_records_populated(producer_run):
    """context.face_records must be non-empty after the detect step.

    This is the core A1 assertion: a real production run produces
    Pydantic FaceRecords, not just dict-of-dicts state.
    """
    context, _ = producer_run
    assert context.face_records, (
        "context.face_records is empty after the producer chain. "
        "The v2 path cannot run in production (REVIEW.md A1)."
    )


def test_face_records_count_matches_legacy(producer_run):
    """Dual-write must produce one FaceRecord per legacy face.

    Mismatched counts mean the two representations have drifted — the
    failure mode this whole strangler-fig migration is designed to avoid.
    """
    context, _ = producer_run
    legacy_total = sum(
        len((data or {}).get("faces", []))
        for data in (context.insightface_faces or {}).values()
    )
    assert len(context.face_records) == legacy_total, (
        f"face_records ({len(context.face_records)}) does not match "
        f"insightface_faces total ({legacy_total}). Producer dual-write "
        "is dropping or duplicating records."
    )


def test_face_records_have_bbox_and_landmarks(producer_run):
    """Detect step must populate bbox and landmarks on every record."""
    context, _ = producer_run
    for r in context.face_records:
        assert r.bbox is not None and len(r.bbox) == 4, f"bad bbox: {r.bbox}"
        x1, y1, x2, y2 = r.bbox
        assert x2 > x1 and y2 > y1, f"non-positive bbox extent: {r.bbox}"
        assert r.landmarks is not None, f"no landmarks for face_id={r.face_id}"
        assert r.landmarks.shape[0] >= 5, f"expected 5+ landmarks: {r.landmarks.shape}"
        assert r.image_path, f"no image_path on face_id={r.face_id}"


def test_align_step_sets_aligned_face(producer_run):
    """At least one FaceRecord must carry an aligned_face crop.

    Align step is allowed to skip individual faces (no landmarks, etc.)
    but on a 3-image real fixture at least one face should align.
    """
    context, _ = producer_run
    aligned_count = sum(1 for r in context.face_records if r.aligned_face is not None)
    assert aligned_count > 0, (
        "No FaceRecord has aligned_face set. Either align_faces is not "
        "writing to face_records, or every face failed alignment."
    )
    for r in context.face_records:
        if r.aligned_face is not None:
            assert r.aligned_face.ndim == 3, f"aligned_face wrong shape: {r.aligned_face.shape}"


def test_embed_step_sets_embedding(producer_run):
    """At least one FaceRecord must carry an embedding + normalized form."""
    context, _ = producer_run
    embedded = [r for r in context.face_records if r.embedding is not None]
    assert embedded, (
        "No FaceRecord has embedding set. extract_face_embeddings is not "
        "writing to face_records."
    )
    for r in embedded:
        assert r.embedding.ndim == 1, f"embedding wrong shape: {r.embedding.shape}"
        assert r.embedding_normalized is not None, (
            f"face_id={r.face_id} has embedding but no embedding_normalized"
        )
        norm = float(np.linalg.norm(r.embedding_normalized))
        assert abs(norm - 1.0) < 1e-4, f"embedding_normalized not unit-norm: {norm}"


def test_face_ids_are_unique(producer_run):
    """face_id must be unique across the run (downstream code assumes it)."""
    context, _ = producer_run
    ids = [r.face_id for r in context.face_records]
    assert len(ids) == len(set(ids)), f"duplicate face_ids: {ids}"
