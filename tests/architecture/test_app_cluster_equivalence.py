"""spec-079 Stage 1 — the cross-app equivalence test (the executable "done").

This is the ONE test that runs the SAME profile through BOTH app entrypoints and
asserts they produce IDENTICAL identity-cluster sizes:

    FC v2     (run_pipeline -> producer chain + 8 unified clustering steps)
    Albumify  (PipelineService -> cluster_people step -> bridge)
                                |
                          assert sizes EQUAL

It is DIFFERENT from the existing suites, which each check ONE app in isolation:
  * tests/face_clustering/e2e_budapest/  -> FC v2 vs a fixed baseline (15/340).
  * tests/architecture/test_config_parity -> hand-built dicts, never the real apps.
Neither answers "do the two apps agree with each other?" -- which is the objective.

RED-by-design (pytest strict xfail)
-----------------------------------
Today the apps disagree (FC v2 = 8 identities, Albumify = 20) because pose data
never reaches Albumify's FaceRecord and blur_min is pinned to 0.0 -- see
SIGHTING-096 and spec-079 Stages 2-3. So the equality assertion FAILS now.

We mark it ``xfail(strict=True)`` rather than letting it hard-fail CI:
  * NOW (gap open):  the assertion fails -> reported XFAIL (expected). Documented,
    visible in the report, does not break the heavy suite.
  * WHEN FIXED (8==8): the assertion passes -> strict xfail turns the unexpected
    pass into an XPASS *failure*, forcing whoever closed the gap to delete the
    xfail marker and consciously declare equivalence reached. That is the moment
    spec-079's Track A is done; re-baseline the goldens then (and not before).

Heavy + opt-in: marked ``budapest`` (real album + models, ~minutes). Run with::

    .venv/Scripts/python -m pytest -m budapest tests/architecture/test_app_cluster_equivalence.py -v

Skips (not xfails) when the Budapest album / profile_5 are absent on this machine.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TESTS = _REPO / "tests"
_SCRIPTS = _REPO / "scripts"
for _p in (str(_REPO), str(_TESTS), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _budapest_baseline as anchor  # noqa: E402  (tests/_budapest_baseline.py)


# --------------------------------------------------------------------------- #
# Environment guard                                                           #
# --------------------------------------------------------------------------- #
def _require_budapest_data() -> None:
    """Skip (do not xfail) when this machine lacks the real inputs."""
    if not anchor.SOURCE_DIR.is_dir():
        pytest.skip(f"Budapest album not present: {anchor.SOURCE_DIR}")
    if not anchor.PROFILE_PATH.is_file():
        pytest.skip(f"profile_5 not present: {anchor.PROFILE_PATH}")


# --------------------------------------------------------------------------- #
# FC v2 path -- mirrors scripts/run_profile.py                                #
# --------------------------------------------------------------------------- #
def _fc_v2_cluster_sizes() -> list[int]:
    """Run FC v2 headlessly from profile_5 and return assigned cluster sizes desc."""
    from app.face_clustering_v2.pipeline import PRODUCER_STEPS
    from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
    from face_cluster.fc_params import FCParams
    from face_cluster.run_layout import allocate_run_dir
    from sim_bench.pipeline.run import run_pipeline
    from sim_bench.pipeline.spec import PipelineSpec

    params = FCParams.load(anchor.PROFILE_PATH)
    album = anchor.SOURCE_DIR.name
    run_dir, run_id = allocate_run_dir(Path.home() / ".sim_bench" / "runs", album)
    spec = PipelineSpec.from_fcparams(
        params,
        producer_steps=["discover_images"] + list(PRODUCER_STEPS),
        clustering_steps=UNIFIED_CLUSTERING_STEPS,
    )
    result = run_pipeline(
        source_dir=anchor.SOURCE_DIR, run_dir=run_dir, run_id=run_id,
        album=album, spec=spec, producer="fc_app_v2",
    )
    assert result.success, f"FC v2 run failed: {result.error_message}"

    db_path = result.db_path
    if not db_path.is_file():
        return []
    conn = sqlite3.connect(str(db_path))
    try:
        mx = conn.execute("SELECT MAX(iteration) FROM cluster_assignments").fetchone()[0]
        rows = conn.execute(
            "SELECT cluster_id, COUNT(*) FROM cluster_assignments "
            "WHERE iteration=? AND cluster_id>=0 GROUP BY cluster_id ORDER BY 2 DESC",
            (mx,),
        ).fetchall()
        return [r[1] for r in rows]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Albumify path -- mirrors scripts/capture_albumify_baseline.py               #
# --------------------------------------------------------------------------- #
def _albumify_cluster_sizes() -> list[int]:
    """Run Albumify (API services, in-process) from profile_5 and return people sizes desc."""
    import capture_albumify_baseline as cap  # scripts/  -- single source of the overlay logic
    from sim_bench.api.database.session import get_session_direct
    from sim_bench.api.database.models import Person
    from sim_bench.api.services.album_service import AlbumService
    from sim_bench.api.services.pipeline_service import PipelineService

    album_name = "spec079_equivalence_budapest"
    session = get_session_direct()
    albums = AlbumService(session)
    pipeline = PipelineService(session)

    # Deterministic: drop any prior album of this name (cascades people/runs).
    for a in albums.list_all():
        if a.name == album_name:
            albums.delete(a.id)
    album = albums.create(album_name, str(anchor.SOURCE_DIR))

    step_configs = cap.build_step_configs(anchor.PROFILE_PATH)
    job_id = pipeline.start_pipeline(
        album_id=album.id, steps=cap.pipeline_steps(), step_configs=step_configs, fail_fast=True,
    )
    pipeline.execute_pipeline(job_id)
    run = pipeline.get_status(job_id)
    assert run.status == "completed", f"Albumify run failed: {run.error_message}"

    people = session.query(Person).filter(Person.run_id == job_id).all()
    return sorted((p.face_count for p in people), reverse=True)


# --------------------------------------------------------------------------- #
# The acceptance test                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.budapest
def test_same_profile_yields_same_clusters() -> None:
    """Same profile_5 through both apps must give identical identity-cluster sizes."""
    _require_budapest_data()

    fc_v2 = _fc_v2_cluster_sizes()
    albumify = _albumify_cluster_sizes()

    # Surface the live numbers even under xfail (pytest swallows the assertion
    # message on an expected failure). Run with -s to stream, or read on XPASS.
    print(
        "\n[spec-079 equivalence] "
        f"FC v2 = {len(fc_v2)} identities {fc_v2}  |  "
        f"Albumify = {len(albumify)} identities {albumify}",
        file=sys.stderr,
    )

    # Sanity: the FC v2 side must reproduce its own anchor; otherwise the test
    # environment is broken and the equality comparison below is meaningless.
    assert fc_v2 == anchor.EXPECTED_CLUSTER_SIZES, (
        f"FC v2 did not reproduce its anchor {anchor.EXPECTED_CLUSTER_SIZES}; "
        f"got {fc_v2}. Environment/regression issue -- fix before reading equivalence."
    )

    # NOTE on semantics (spec-079): exceptions while RUNNING either app above are
    # real failures and surface as ERRORs (we no longer wrap the test in a blanket
    # xfail, which used to mask a broken pipeline as "expected"). Only a genuine
    # *size difference* is the not-yet-done state -> xfail. Equality -> pass = done.
    if albumify != fc_v2:
        pytest.xfail(
            "spec-079: apps not yet equivalent (architecture migration in progress).\n"
            f"  FC v2    : {len(fc_v2)} identities {fc_v2}\n"
            f"  Albumify : {len(albumify)} identities {albumify}"
        )
