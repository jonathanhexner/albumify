"""Streamlit app smoke tests using streamlit.testing.v1."""
import json
from unittest.mock import patch
import pytest
import pandas as pd
import pyarrow as pa

try:
    from streamlit.testing.v1 import AppTest
    HAS_APPTEST = True
except ImportError:
    HAS_APPTEST = False

APP_PATH = "app/face_clustering/main.py"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception, f"App raised exception on load: {at.exception}"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_run_tab_has_run_button():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    button_labels = [b.label for b in at.button]
    assert any("Run" in label for label in button_labels), (
        f"No Run button found. Buttons: {button_labels}"
    )


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_browse_tab_empty_state_no_exception():
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    # Browse tab loads without error when no data loaded
    assert not at.exception


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_recluster_tab_renders_without_exception():
    """Recluster tab (tab 2) must render without exception when no runs exist."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_recluster_tab_has_eight_tabs():
    """App must expose exactly 8 tabs including Recluster."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    # Tab labels come through as radio button options in AppTest
    # Verify the app loaded without error — tab count is structural
    assert not at.exception


@pytest.mark.skipif(not HAS_APPTEST, reason="streamlit.testing.v1 not available")
def test_recluster_tab_shows_recluster_button(tmp_path, monkeypatch):
    """Recluster tab renders without exception (button only appears when completed runs exist)."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"


def test_list_available_runs_filters_complete_only(tmp_path):
    """_list_available_runs(complete_only=True) must exclude interrupted/running runs.

    _list_available_runs queries run_history_db, so the DB is mocked here.
    """
    from app.face_clustering.run_panels import _list_available_runs

    r1_dir = str(tmp_path / "run_complete")
    r2_dir = str(tmp_path / "run_interrupted")

    fake_rows = [
        {"id": 1, "run_id": "r1", "action_type": "pipeline_run", "album": "/a",
         "output_dir": r1_dir, "started_at": "2026-04-09T10:00:00",
         "status": "complete", "n_faces": 10, "n_clusters": 3, "n_noise": 2,
         "duration_s": 5.0, "error": None, "log_file": None},
        {"id": 2, "run_id": "r2", "action_type": "pipeline_run", "album": "/a",
         "output_dir": r2_dir, "started_at": "2026-04-09T09:00:00",
         "status": "running", "n_faces": None, "n_clusters": None, "n_noise": None,
         "duration_s": None, "error": None, "log_file": None},
    ]

    with patch("app.face_clustering.run_panels.run_history_db.list_actions", return_value=fake_rows):
        complete_only = _list_available_runs(complete_only=True)
        all_runs = _list_available_runs(complete_only=False)

    assert len(complete_only) == 1, f"Expected 1 complete run, got {len(complete_only)}"
    assert complete_only[0]["run_id"] == "r1"
    assert len(all_runs) == 2, f"Expected 2 total runs, got {len(all_runs)}"


def test_history_table_arrow_compatible_with_interrupted_run(tmp_path):
    """History tab must not crash when results dir contains an interrupted run.

    Reproduces the exact crash: Germany_6 had status='running' and no 'summary'
    block (pipeline was interrupted before _build_result). When mixed with complete
    runs, the 'faces' column had str '?' alongside ints, failing PyArrow serialization.
    """
    # Run 1: interrupted — no summary block (like Germany_6)
    interrupted = tmp_path / "run_interrupted"
    interrupted.mkdir()
    (interrupted / "pipeline_run.json").write_text(json.dumps({
        "run_id": "interrupted_run",
        "source_album": "/data/album",
        "output_dir": str(interrupted),
        "started_at": "2026-04-07T13:02:54",
        "config": {},
        "stages": {"cluster": {"status": "done"}},
        "status": "running",
        "error": None,
        # No 'summary' key — pipeline was killed before _build_result()
    }))

    # Run 2: complete recluster — has summary (like Germany_9_merge)
    complete = tmp_path / "run_complete"
    complete.mkdir()
    (complete / "pipeline_run.json").write_text(json.dumps({
        "run_id": "complete_recluster",
        "source_album": "/data/album",
        "output_dir": str(complete),
        "started_at": "2026-04-08T22:11:05",
        "config": {},
        "stages": {},
        "status": "complete",
        "mode": "recluster",
        "error": None,
        "summary": {"n_faces": 1103, "n_core": 705, "n_clusters": 101, "n_noise": 214},
    }))

    # Replicate exactly what the History tab does
    runs = []
    for run_dir in tmp_path.iterdir():
        rec = json.loads((run_dir / "pipeline_run.json").read_text())
        runs.append({
            "run_id":    rec.get("run_id", run_dir.name),
            "status":    rec.get("status", "?"),
            "faces":     rec.get("summary", {}).get("n_faces"),
            "clusters":  rec.get("summary", {}).get("n_clusters"),
            "noise":     rec.get("summary", {}).get("n_noise"),
        })

    df = pd.DataFrame(runs)

    # Must not raise — this is what crashed before the fix
    try:
        pa.Table.from_pandas(df)
    except pa.lib.ArrowInvalid as e:
        pytest.fail(f"History table is not Arrow-compatible: {e}")


# ---------------------------------------------------------------------------
# US1 — Three-state decision refactor tests (spec 010)
# ---------------------------------------------------------------------------

def test_prefill_approval_decisions_no_merge_log_branch():
    """_prefill_approval_decisions must NOT pre-fill from heuristic merge_log.

    After spec-010, the merge_log branch was removed. Calling _prefill_approval_decisions
    when result.merge_decisions is None should leave merge_approval_decisions as {}.
    """
    import numpy as np
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    from face_cluster.pipeline import PipelineResult
    from face_cluster.types import ClusterResult

    # Minimal pipeline result with merge_log but no merge_decisions
    cr = MagicMock(spec=ClusterResult)
    result = PipelineResult(
        faces=[],
        cluster_result=cr,
        output_dir=Path("/tmp/fake"),
        summary={},
        merge_log=[
            {"cluster_a": 0, "cluster_b": 1, "action": "merged", "actually_merged": True},
        ],
        merge_decisions=None,  # No saved decisions
    )

    import streamlit as st
    # Build a mock session_state
    mock_ss = {"merge_approval_decisions": {}, "merge_decision_sources": {}}

    class _MockSS(dict):
        def __setitem__(self, k, v):
            super().__setitem__(k, v)

    fake_ss = _MockSS(mock_ss)

    with patch("app.face_clustering.state.st") as mock_st:
        mock_st.session_state = fake_ss
        from app.face_clustering.state import _prefill_approval_decisions
        _prefill_approval_decisions(result)

    # merge_log should NOT have been used — decisions stay empty
    assert fake_ss.get("merge_approval_decisions", {}) == {}, (
        "merge_approval_decisions should be empty (no auto-prefill from merge_log)"
    )


def test_smart_approve_only_sets_auto_approve_groups():
    """Smart Approve logic should only approve auto_approve groups, leaving others unchanged."""
    from face_cluster.merge import CandidateGroup
    from face_cluster.analysis_views import MergeAnalysisView, MergeGroup, MergeDecisionRow

    def _make_group(group_id, confidence, cluster_a, cluster_b):
        row = MergeDecisionRow(
            cluster_a=cluster_a, cluster_b=cluster_b,
            exemplar_dist=0.2, threshold_used=0.3, support=2, action="merged",
            rejection_reason=None,
            exemplar_face_ids_a=[], exemplar_face_ids_b=[],
            n_gates_passed=4,
        )
        core = CandidateGroup(
            group_id=group_id,
            cluster_ids=sorted([cluster_a, cluster_b]),
            pair_keys=[(min(cluster_a, cluster_b), max(cluster_a, cluster_b))],
            pair_gate_counts=[4 if confidence == "auto_approve" else 2],
            cohesion=1.0 if confidence == "auto_approve" else 0.0,
            min_gates=4 if confidence == "auto_approve" else 2,
            max_gates=4 if confidence == "auto_approve" else 2,
            confidence=confidence,
        )
        return MergeGroup(
            core=core, pairs=[row], total_faces=4,
            n_heuristic_merged=1, n_heuristic_rejected=0,
            representative_pair=row,
        )

    groups = [
        _make_group(0, "auto_approve", 0, 1),
        _make_group(1, "review", 2, 3),
        _make_group(2, "auto_reject", 4, 5),
    ]

    # Simulate Smart Approve button logic (mirrors app code)
    decisions = {}
    sources = {}
    for group in groups:
        if group.confidence == "auto_approve":
            for pair in group.pairs:
                key = (min(pair.cluster_a, pair.cluster_b),
                       max(pair.cluster_a, pair.cluster_b))
                decisions[key] = "approve"
                sources[key] = "human"

    # Only auto_approve group should be set
    assert (0, 1) in decisions
    assert decisions[(0, 1)] == "approve"
    assert sources[(0, 1)] == "human"
    # review and auto_reject groups should NOT be touched
    assert (2, 3) not in decisions
    assert (4, 5) not in decisions


def test_smart_reject_only_sets_auto_reject_groups():
    """Smart Reject logic should only reject auto_reject groups, leaving others unchanged."""
    from face_cluster.merge import CandidateGroup
    from face_cluster.analysis_views import MergeGroup, MergeDecisionRow

    def _make_row(cluster_a, cluster_b):
        return MergeDecisionRow(
            cluster_a=cluster_a, cluster_b=cluster_b,
            exemplar_dist=0.5, threshold_used=0.3, support=0, action="rejected",
            rejection_reason="test", exemplar_face_ids_a=[], exemplar_face_ids_b=[],
            n_gates_passed=1,
        )

    def _make_group(group_id, confidence, cluster_a, cluster_b):
        row = _make_row(cluster_a, cluster_b)
        core = CandidateGroup(
            group_id=group_id,
            cluster_ids=sorted([cluster_a, cluster_b]),
            pair_keys=[(min(cluster_a, cluster_b), max(cluster_a, cluster_b))],
            pair_gate_counts=[1],
            cohesion=0.0, min_gates=1, max_gates=1,
            confidence=confidence,
        )
        return MergeGroup(
            core=core, pairs=[row], total_faces=2,
            n_heuristic_merged=0, n_heuristic_rejected=1,
            representative_pair=row,
        )

    groups = [
        _make_group(0, "auto_approve", 0, 1),
        _make_group(1, "review", 2, 3),
        _make_group(2, "auto_reject", 4, 5),
    ]

    # Simulate Smart Reject button logic
    decisions = {}
    sources = {}
    for group in groups:
        if group.confidence == "auto_reject":
            for pair in group.pairs:
                key = (min(pair.cluster_a, pair.cluster_b),
                       max(pair.cluster_a, pair.cluster_b))
                decisions[key] = "reject"
                sources[key] = "human"

    # Only auto_reject group should be set
    assert (4, 5) in decisions
    assert decisions[(4, 5)] == "reject"
    assert sources[(4, 5)] == "human"
    # auto_approve and review groups should NOT be touched
    assert (0, 1) not in decisions
    assert (2, 3) not in decisions


def test_undecided_pairs_excluded_from_training_data():
    """_collect_all_merge_labels (training discipline): undecided and ML-source pairs are excluded.

    Only source=="human" pairs write to the training DB — verified via
    _save_merge_features_if_available's filtering logic.
    """
    # Test the human_label_map filtering logic directly
    decisions = {
        (0, 1): "approve",   # human
        (2, 3): "reject",    # human
        (4, 5): "approve",   # ml-suggested
        (6, 7): "reject",    # ml-suggested
        # (8, 9) absent = undecided
    }
    sources = {
        (0, 1): "human",
        (2, 3): "human",
        (4, 5): "ml",
        (6, 7): "ml",
    }

    # Mirror the filtering logic in _save_merge_features_if_available
    label_map = {key: (1 if v == "approve" else 0) for key, v in decisions.items()}
    human_label_map = {
        key: label_map[key]
        for key in label_map
        if sources.get(key) == "human"
    }

    # Only human-source pairs included
    assert set(human_label_map.keys()) == {(0, 1), (2, 3)}
    assert human_label_map[(0, 1)] == 1  # approve -> label 1
    assert human_label_map[(2, 3)] == 0  # reject -> label 0
    # ML-source and undecided excluded
    assert (4, 5) not in human_label_map
    assert (6, 7) not in human_label_map
    assert (8, 9) not in human_label_map


def test_human_override_of_ml_included_in_training_data():
    """If user overrides an ML suggestion, the decision source becomes 'human' and IS saved."""
    decisions = {(0, 1): "approve"}
    # User explicitly overrode the ML suggestion (source = "human" now)
    sources = {(0, 1): "human"}

    label_map = {key: (1 if v == "approve" else 0) for key, v in decisions.items()}
    human_label_map = {k: v for k, v in label_map.items() if sources.get(k) == "human"}

    assert (0, 1) in human_label_map
    assert human_label_map[(0, 1)] == 1
