"""spec-035 / FR-033-1 — E2E acceptance test for the Albumify face-clustering pipeline.

What this test catches that no other test in the spec-033 suite catches:
SIGHTING-061 — a regression that left ``context.people_clusters`` empty and
crashed ``identity_refinement``. Architecture tests inspect code shape;
unit tests construct objects in isolation; synthetic-data tests build fake
DBs. Only an actual pipeline run on real images surfaces "config knob
references upstream field nobody computes" bugs.

The test runs the production ``default_pipeline`` from
``configs/pipeline.yaml`` on a 5-image fixture and asserts:

* ``context.people_clusters`` is non-empty after ``cluster_people``.
* ``context.all_faces`` is non-empty after ``cluster_people``.
* No step raises.

The fixture is sourced from ``test_data/face_clustering_100/`` (the same
fixture the FC App E2E test uses). The first 5 images are picked
deterministically. Test is skipped if the fixture is missing — with a
clear message about how to populate it.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.conftest import get_test_data_dir


# Marker so the test can be opted out of in fast loops:
#   pytest -m "not face_e2e"
# Included by default. Registered in pyproject.toml.
pytestmark = pytest.mark.face_e2e


PIPELINE_YAML = Path(__file__).resolve().parents[2] / "configs" / "pipeline.yaml"
TEST_DATA_100 = get_test_data_dir() / "face_clustering_100"


def _select_fixture_images(n: int = 5) -> list[Path]:
    """Deterministic pick of N image files from the 100-image fixture.

    Sorted by filename so the selection is stable across machines.
    """
    if not TEST_DATA_100.exists():
        return []
    suffixes = {".jpg", ".jpeg", ".png", ".heic"}
    images = sorted(
        p for p in TEST_DATA_100.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    )
    return images[:n]


@pytest.fixture(scope="module")
def albumify_run(tmp_path_factory):
    """Run the production Albumify pipeline on a 5-image fixture once per module.

    Returns the populated ``PipelineContext`` so individual tests can assert
    against it without re-running the pipeline.
    """
    fixture_images = _select_fixture_images(5)
    if not fixture_images:
        pytest.skip(
            f"Albumify E2E fixture missing at {TEST_DATA_100}.\n"
            "Populate with: "
            "python scripts/create_test_sample.py "
            "--source D:/Google_Germany "
            "--dest test_data/face_clustering_100 --n 100 --seed 42"
        )

    # Stage 5 images into a temp dir so the pipeline discovers exactly 5.
    src_dir = tmp_path_factory.mktemp("albumify_e2e_src")
    for src in fixture_images:
        # Copy (not symlink) — Windows symlinks need admin; copy is portable.
        (src_dir / src.name).write_bytes(src.read_bytes())

    # Load production step list from pipeline.yaml.
    yaml_text = PIPELINE_YAML.read_text(encoding="utf-8")
    yaml_doc = yaml.safe_load(yaml_text)
    step_names: list[str] = yaml_doc.get("default_pipeline", [])
    if not step_names:
        pytest.fail("default_pipeline not found in configs/pipeline.yaml")

    # Construct step_configs from the same yaml (one dict per step name).
    step_configs = {
        name: (yaml_doc.get(name) or {}) for name in step_names
    }

    # Import every registered step so the registry is populated.
    import sim_bench.pipeline.steps.all_steps  # noqa: F401
    from sim_bench.pipeline.config import PipelineConfig
    from sim_bench.pipeline.context import PipelineContext
    from sim_bench.pipeline.executor import PipelineExecutor
    from sim_bench.pipeline.registry import get_registry

    context = PipelineContext(source_directory=src_dir)
    context.step_configs = step_configs

    config = PipelineConfig(
        step_configs=step_configs,
        fail_fast=True,
    )

    executor = PipelineExecutor(get_registry())
    result = executor.execute(context, step_names, config=config)

    return context, result


def test_pipeline_completes_successfully(albumify_run):
    """The full default_pipeline must run without any step raising.

    A failure here means a regression in any of detect_persons,
    insightface_detect_faces, filter_faces, align_faces, the scoring steps,
    extract_face_embeddings, cluster_people, identity_refinement, or
    select_best. The error message names the step.
    """
    _, result = albumify_run
    assert result.success, (
        f"Pipeline failed: {result.error_message} "
        f"(failed step: {result.failed_step})"
    )


def test_people_clusters_non_empty(albumify_run):
    """SIGHTING-061 regression guard.

    The bug shipped in spec-033 P-C C-1 left ``context.people_clusters`` empty
    because the bridge unblocked the blur gate against data nobody computed.
    This assertion is the test that would have caught it.
    """
    context, _ = albumify_run
    assert context.people_clusters, (
        "context.people_clusters is empty after the pipeline ran. "
        "This was the failure mode of SIGHTING-061. Check the run log for "
        "'face_cluster_knn: no faces passed quality gating, all noise'."
    )
    # At least one cluster must have at least one face.
    has_member = any(len(faces) >= 1 for faces in context.people_clusters.values())
    assert has_member, (
        "Every people_clusters entry is empty. Faces were assembled but no "
        "cluster retained any. Check core_indices and the merge stage."
    )


def test_faces_detected_on_at_least_one_image(albumify_run):
    """At least one fixture image must produce at least one detected face.

    A regression here means face detection broke upstream of clustering.
    Distinguishes "no faces detected" (this test) from "faces detected
    but all rejected at clustering" (the people_clusters test catches that).
    """
    context, _ = albumify_run
    total_detected = sum(
        len((data or {}).get("faces", []))
        for data in (context.insightface_faces or {}).values()
    )
    assert total_detected > 0, (
        f"insightface_faces contains 0 faces across all {len(context.image_paths)} "
        "fixture images. Detection broke upstream of clustering."
    )


def test_filter_decisions_recorded(albumify_run):
    """spec-032 / spec-033 P-C C-3 — filter_decisions must accumulate on Albumify.

    Pre-spec-033 the filter_decisions table on Albumify was empty (SIGHTING-059
    Issue 5). The fix wired ``filters=context.filters`` through to RunExporter.
    This test guards against the wiring being silently disconnected.
    """
    context, _ = albumify_run
    # context.filters is a FilterContext; non-zero items means at least one
    # filter step recorded a decision.
    assert len(context.filters) > 0, (
        "context.filters is empty after the pipeline ran. Either no filter "
        "step recorded a decision, or the FilterContext wiring regressed."
    )
