"""spec-063 — synthetic-data tests for ReclusterService.

Uses the spec-045 synthetic fixture as the prior-run input. Covers the
6 cases listed in spec §"Tests":

1. list_recent_runs returns typed entries
2. recluster(valid prior) returns ReclusterResult
3. recluster(missing prior) raises ValueError
4. snapshot dir exists post-recluster
5. parent_run_id matches input dir name
6. reclustering with tightened params produces a different n_clusters
"""
from __future__ import annotations

from pathlib import Path

import pytest

from face_cluster.fc_params import FCParams
from face_cluster.repositories import (
    RunHistoryRepoConfig,
    RunHistoryRepository,
)
from face_cluster.views.recluster import (
    ReclusterResult,
    ReclusterService,
    RunPickerEntry,
)
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


# Permissive params: the synthetic fixture has no pose/blur metadata, so
# everything must pass the quality gate. Mirrors the constants used by
# test_legacy_vs_v2_equivalence and test_fc_app_runner.
_DEFAULT_PARAMS = dict(
    K=3,
    distance_threshold=0.5,
    min_cluster_size=2,
    blur_min=0.0,
    yaw_max=999.0,
    pitch_max=999.0,
    roll_max=999.0,
    max_faces_per_image_core=50,
    merge_enabled=False,
    attach_enabled=False,
)


@pytest.fixture
def prior_run_dir(tmp_path: Path) -> Path:
    """Schema-valid v5 run dir built from the spec-045 synthetic seed."""
    return _build_synthetic_run_dir(tmp_path)


@pytest.fixture
def isolated_history_repo(tmp_path: Path) -> RunHistoryRepository:
    """RunHistoryRepository pointed at a tmp-path DB so the test does not
    read/write the user's real ~/.sim_bench DB."""
    db_path = tmp_path / "history.db"
    return RunHistoryRepository(RunHistoryRepoConfig(db_path=db_path))


@pytest.fixture
def service(isolated_history_repo: RunHistoryRepository) -> ReclusterService:
    return ReclusterService(isolated_history_repo)


@pytest.fixture
def runs_base(tmp_path: Path) -> Path:
    """Where the recluster snapshot dirs land. Tmp so no user files touched."""
    base = tmp_path / "snapshots"
    base.mkdir()
    return base


def test_list_recent_runs_returns_typed_entries(
    service: ReclusterService, isolated_history_repo: RunHistoryRepository,
) -> None:
    """list_recent_runs returns RunPickerEntry typed objects (possibly empty)."""
    # Seed one fc_app_v2 row so list_recent_runs has something to return.
    action_id = isolated_history_repo.start_action("fc_app_v2_run", payload={
        "run_id": "abc123",
        "output_dir": "C:/nonexistent/run_abc",  # orphan — directory doesn't exist
        "source_album": "test_album",
        "producer": "fc_app_v2",
    })
    isolated_history_repo.complete_action(action_id, result_fields={
        "n_faces": 10, "n_clusters": 2, "run_id": "abc123",
    })

    entries = service.list_recent_runs(limit=5)
    assert isinstance(entries, list)
    assert all(isinstance(e, RunPickerEntry) for e in entries)
    assert len(entries) == 1
    assert entries[0].run_id == "abc123"
    assert entries[0].album == "test_album"
    # Directory doesn't exist on disk → flagged orphan.
    assert entries[0].is_orphan is True


def test_recluster_valid_prior_returns_typed_result(
    service: ReclusterService, prior_run_dir: Path, runs_base: Path,
) -> None:
    """recluster() on a valid prior returns a ReclusterResult."""
    params = FCParams(**_DEFAULT_PARAMS)
    result = service.recluster(prior_run_dir, params, runs_base_dir=runs_base)
    assert isinstance(result, ReclusterResult)
    assert result.n_faces > 0
    assert result.n_clusters >= 0


def test_recluster_missing_prior_raises_clear_value_error(
    service: ReclusterService, tmp_path: Path, runs_base: Path,
) -> None:
    """A nonexistent prior_run_dir raises ValueError with a useful message."""
    params = FCParams(**_DEFAULT_PARAMS)
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="prior_run_dir does not exist"):
        service.recluster(missing, params, runs_base_dir=runs_base)


def test_recluster_creates_snapshot_dir(
    service: ReclusterService, prior_run_dir: Path, runs_base: Path,
) -> None:
    """The returned snapshot_dir exists and contains the v5 artifacts."""
    params = FCParams(**_DEFAULT_PARAMS)
    result = service.recluster(prior_run_dir, params, runs_base_dir=runs_base)
    assert result.snapshot_dir.is_dir()
    assert (result.snapshot_dir / "face_clustering.db").is_file()
    assert (result.snapshot_dir / "pipeline_run.json").is_file()
    assert (result.snapshot_dir / "embeddings.npy").is_file()


def test_recluster_parent_run_id_matches_input(
    service: ReclusterService, prior_run_dir: Path, runs_base: Path,
) -> None:
    """The snapshot records the parent's dir name as parent_run_id, both on
    the ReclusterResult and inside pipeline_run.json."""
    import json
    params = FCParams(**_DEFAULT_PARAMS)
    result = service.recluster(prior_run_dir, params, runs_base_dir=runs_base)
    assert result.parent_run_id == prior_run_dir.name

    payload = json.loads(
        (result.snapshot_dir / "pipeline_run.json").read_text(encoding="utf-8")
    )
    assert payload.get("parent_run_id") == prior_run_dir.name


def test_recluster_tighter_threshold_changes_cluster_count(
    service: ReclusterService, prior_run_dir: Path, runs_base: Path,
) -> None:
    """Tightening distance_threshold from 0.5 to 0.15 should not produce
    the same cluster count on the synthetic fixture (3 clusters by design
    at the loose threshold). At very tight thresholds the components
    fragment into more clusters or collapse to noise — either way the
    cluster count must differ from the loose-threshold run."""
    loose = FCParams(**_DEFAULT_PARAMS)
    tight = FCParams(**{**_DEFAULT_PARAMS, "distance_threshold": 0.15})

    r_loose = service.recluster(prior_run_dir, loose, runs_base_dir=runs_base)
    r_tight = service.recluster(prior_run_dir, tight, runs_base_dir=runs_base)
    assert r_loose.n_clusters != r_tight.n_clusters, (
        f"Reclustering with a much tighter threshold ({tight.distance_threshold}) "
        f"produced the same cluster count ({r_loose.n_clusters}) as the loose "
        f"threshold ({loose.distance_threshold}). Either the synthetic fixture "
        f"is degenerate or the params didn't propagate to the clustering step."
    )
