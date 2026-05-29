"""Unit tests for the Cluster Analysis tab's run-dir resolver helpers.

Targets the bug surfaced 2026-05-29 where spec-050's run_tab wrote
``v2_last_run_dir`` BEFORE the pipeline ran (deliberate for failure
recovery), and the Cluster Analysis tab then handed that allocated-but-
empty dir to the Repository, which raised ValueError on the missing
``face_clustering.db``.

The predicate ``_run_dir_is_loadable`` is the contract that prevents the
regression: if a dir has no DB yet, resolver must skip it.
"""
from __future__ import annotations

from pathlib import Path

from app.face_clustering_v2.tabs.cluster_analysis_tab import _run_dir_is_loadable


def test_loadable_false_for_nonexistent_dir(tmp_path):
    assert _run_dir_is_loadable(tmp_path / "does_not_exist") is False


def test_loadable_false_for_empty_dir(tmp_path):
    """The exact bug shape: spec-050 allocates an empty UUID dir BEFORE
    running the pipeline. Resolver must reject it."""
    allocated = tmp_path / "deadbeef1234"
    allocated.mkdir()
    assert _run_dir_is_loadable(allocated) is False


def test_loadable_false_for_dir_with_other_files_but_no_db(tmp_path):
    """A run that crashed after writing pipeline_run.json but before the
    DB export still must be rejected."""
    partial = tmp_path / "partial_run"
    partial.mkdir()
    (partial / "pipeline_run.json").write_text('{"schema_version": 5}', encoding="utf-8")
    assert _run_dir_is_loadable(partial) is False


def test_loadable_true_when_db_present(tmp_path):
    """Sufficient signal: dir exists + DB file exists. The Repository
    does the deeper schema validation; this predicate is just a gate
    against the empty-dir shape."""
    complete = tmp_path / "complete_run"
    complete.mkdir()
    (complete / "face_clustering.db").touch()
    assert _run_dir_is_loadable(complete) is True


def test_loadable_false_when_db_path_is_a_dir(tmp_path):
    """Defensive: if something pathological put a directory named
    face_clustering.db in the run dir, treat as not loadable."""
    weird = tmp_path / "weird_run"
    weird.mkdir()
    (weird / "face_clustering.db").mkdir()
    assert _run_dir_is_loadable(weird) is False
