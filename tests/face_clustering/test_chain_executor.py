"""Unit tests for face_cluster.chain_executor.

Test class: ut_ChainExecutor

These tests use minimal mocking — the executor is tested with a fake pipeline
runner to avoid requiring InsightFace on the test machine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from face_cluster.chain_executor import ChainExecutor, write_step_labels, _load_labels, ExecutionContext
from face_cluster.session_manager import SessionManager, Step


@pytest.fixture
def mgr() -> SessionManager:
    return SessionManager()


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    return tmp_path / "album"


@pytest.fixture
def base_session(mgr, session_root):
    session = mgr.create(session_root, "/photos/album", {"n_faces": 50, "n_images": 20})
    # Create base dir with minimal required files so the executor can find it
    (session_root / "base").mkdir(exist_ok=True)
    return session


class ut_ChainExecutor:

    def test_execute_chain_from_cluster_step(self, base_session, session_root):
        """Cluster step calls FaceClusteringPipeline.run with recluster config."""
        mgr = SessionManager()
        session = base_session
        chain = mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {"K": 5, "distance_threshold": 0.35})

        executor = ChainExecutor()
        with patch("face_cluster.chain_executor.FaceClusteringPipeline") as MockPipeline:
            mock_instance = MockPipeline.return_value
            mock_instance.run.return_value = MagicMock()

            executor.execute_chain_from(session, "chain_01", from_step=0)

            mock_instance.run.assert_called_once()
            call_config = mock_instance.run.call_args[0][0]
            assert call_config.source_dir == str(session_root / "base")
            assert call_config.output_dir == str(session_root / "chain_01" / "step_00_cluster")
            assert call_config.K == 5
            assert call_config.distance_threshold == 0.35

    def test_execute_chain_from_creates_step_folder(self, base_session, session_root):
        """Step folder is created before pipeline runs."""
        mgr = SessionManager()
        session = base_session
        chain = mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})

        executor = ChainExecutor()
        with patch("face_cluster.chain_executor.FaceClusteringPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = MagicMock()
            executor.execute_chain_from(session, "chain_01", from_step=0)

        assert (session_root / "chain_01" / "step_00_cluster").exists()

    def test_execute_chain_uses_previous_step_as_input(self, base_session, session_root):
        """Second step uses first step's output dir as source."""
        mgr = SessionManager()
        session = base_session
        chain = mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})
        mgr.append_step(session, "chain_01", "recluster", {"K": 8})

        executor = ChainExecutor()
        calls = []

        def fake_run(config):
            calls.append(config)
            return MagicMock()

        with patch("face_cluster.chain_executor.FaceClusteringPipeline") as MockPipeline:
            MockPipeline.return_value.run.side_effect = fake_run
            executor.execute_chain_from(session, "chain_01", from_step=0)

        assert len(calls) == 2
        assert calls[0].source_dir == str(session_root / "base")
        assert calls[1].source_dir == str(session_root / "chain_01" / "step_00_cluster")

    def test_execute_from_middle_uses_correct_input(self, base_session, session_root):
        """execute_chain_from(from_step=1) uses step_00's output as input."""
        mgr = SessionManager()
        session = base_session
        chain = mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})
        mgr.append_step(session, "chain_01", "recluster", {"K": 8})
        # Create step_00_cluster folder so the executor can resolve it
        (session_root / "chain_01" / "step_00_cluster").mkdir(parents=True, exist_ok=True)

        executor = ChainExecutor()
        with patch("face_cluster.chain_executor.FaceClusteringPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = MagicMock()
            executor.execute_chain_from(session, "chain_01", from_step=1)

            call_config = MockPipeline.return_value.run.call_args[0][0]
            assert call_config.source_dir == str(session_root / "chain_01" / "step_00_cluster")

    def test_execute_returns_last_step_dir(self, base_session, session_root):
        """execute_chain_from returns the final step's output directory."""
        mgr = SessionManager()
        session = base_session
        chain = mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})

        executor = ChainExecutor()
        with patch("face_cluster.chain_executor.FaceClusteringPipeline") as MockPipeline:
            MockPipeline.return_value.run.return_value = MagicMock()
            result_dir = executor.execute_chain_from(session, "chain_01", from_step=0)

        assert result_dir == session_root / "chain_01" / "step_00_cluster"

    def test_execute_empty_chain_returns_base(self, base_session, session_root):
        """Empty chain (no steps) starting from step 0 returns base dir."""
        mgr = SessionManager()
        session = base_session
        mgr.create_chain(session)

        executor = ChainExecutor()
        result_dir = executor.execute_chain_from(session, "chain_01", from_step=0)
        assert result_dir == session_root / "base"


class ut_ChainExecutorLabels:

    def test_write_step_labels(self, tmp_path):
        step_dir = tmp_path / "step_01_merge"
        step_dir.mkdir()
        write_step_labels(step_dir, [(1, 2), (3, 4)], [(5, 6)])

        with open(step_dir / "labels.json", encoding="utf-8") as f:
            data = json.load(f)

        assert data["approved"] == [[1, 2], [3, 4]]
        assert data["rejected"] == [[5, 6]]

    def test_load_labels_from_own_folder(self, tmp_path):
        step_dir = tmp_path / "step_01_merge"
        step_dir.mkdir()
        write_step_labels(step_dir, [(10, 20)], [])

        session = MagicMock()
        session.chains = []
        step = Step(step_index=1, folder="step_01_merge", type="merge",
                    timestamp="2026-01-01T00:00:00Z", params={})
        ctx = ExecutionContext(session=session, chain_id="chain_01", step=step,
                               input_dir=tmp_path, output_dir=step_dir)

        labels = _load_labels(ctx)
        assert labels["approved"] == [[10, 20]]
        assert labels["rejected"] == []

    def test_load_labels_from_source_chain(self, tmp_path):
        """If labels.json absent in own folder, fall back to source chain folder."""
        src_step_dir = tmp_path / "chain_01" / "step_01_merge"
        src_step_dir.mkdir(parents=True)
        write_step_labels(src_step_dir, [(100, 200)], [])

        own_step_dir = tmp_path / "chain_02" / "step_01_merge"
        own_step_dir.mkdir(parents=True)

        from face_cluster.session_manager import Chain, Step
        src_chain = Chain(chain_id="chain_01", folder="chain_01",
                          created_at="2026-01-01T00:00:00Z",
                          branched_from=None, has_pending_labels=False, steps=[])
        new_chain = Chain(chain_id="chain_02", folder="chain_02",
                          created_at="2026-01-01T00:00:00Z",
                          branched_from={"chain_id": "chain_01", "step_index": 1},
                          has_pending_labels=False, steps=[])
        session = MagicMock()
        session.chains = [src_chain, new_chain]
        session.session_root = tmp_path

        step = Step(step_index=1, folder="step_01_merge", type="merge",
                    timestamp="2026-01-01T00:00:00Z", params={})
        ctx = ExecutionContext(session=session, chain_id="chain_02", step=step,
                               input_dir=tmp_path / "chain_01" / "step_00_cluster",
                               output_dir=own_step_dir)

        labels = _load_labels(ctx)
        assert labels["approved"] == [[100, 200]]

    def test_load_labels_returns_empty_when_not_found(self, tmp_path):
        step_dir = tmp_path / "step_01_merge"
        step_dir.mkdir()

        session = MagicMock()
        session.chains = []
        step = Step(step_index=1, folder="step_01_merge", type="merge",
                    timestamp="2026-01-01T00:00:00Z", params={})
        ctx = ExecutionContext(session=session, chain_id="chain_01", step=step,
                               input_dir=tmp_path, output_dir=step_dir)

        labels = _load_labels(ctx)
        assert labels["approved"] == []
        assert labels["rejected"] == []
