"""Unit tests for face_cluster.session_manager.

Test class: ut_SessionManager
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from face_cluster.session_manager import (
    SessionManager,
    Session,
    Chain,
    Step,
    load_session_json,
)


@pytest.fixture
def mgr() -> SessionManager:
    return SessionManager()


@pytest.fixture
def session_root(tmp_path: Path) -> Path:
    return tmp_path / "my_album"


class ut_SessionManager:
    # ------------------------------------------------------------------ create

    def test_create_session(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {"n_faces": 100, "n_images": 50})

        assert session_root.exists()
        assert (session_root / "base").exists()

        path = session_root / "session.json"
        assert path.exists()

        data = load_session_json(path)
        assert data["schema_version"] == 1
        assert data["source_album"] == "/photos/album"
        assert data["base"]["folder"] == "base"
        assert data["base"]["n_faces"] == 100
        assert data["base"]["n_images"] == 50
        assert data["chains"] == []
        assert data["active_chain"] is None

    def test_create_session_returns_base_dir(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        base_dir = mgr.get_base_dir(session)
        assert base_dir == session_root / "base"
        assert base_dir.exists()

    def test_update_base_summary(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {"n_faces": 0})
        mgr.update_base_summary(session, n_faces=342, n_images=120)

        data = load_session_json(session_root / "session.json")
        assert data["base"]["n_faces"] == 342
        assert data["base"]["n_images"] == 120

    # ------------------------------------------------------------------ chains

    def test_create_chain(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        chain = mgr.create_chain(session)

        assert chain.chain_id == "chain_01"
        assert chain.folder == "chain_01"
        assert chain.branched_from is None
        assert chain.steps == []
        assert (session_root / "chain_01").exists()
        assert session.active_chain == "chain_01"

        data = load_session_json(session_root / "session.json")
        assert len(data["chains"]) == 1
        assert data["chains"][0]["chain_id"] == "chain_01"
        assert data["active_chain"] == "chain_01"

    def test_create_multiple_chains(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.create_chain(session)

        assert len(session.chains) == 2
        assert session.chains[0].chain_id == "chain_01"
        assert session.chains[1].chain_id == "chain_02"
        assert session.active_chain == "chain_02"

    # ------------------------------------------------------------------ steps

    def test_append_steps(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        chain = mgr.create_chain(session)

        step0, dir0 = mgr.append_step(session, "chain_01", "cluster", {"K": 5, "distance_threshold": 0.35})
        assert step0.step_index == 0
        assert step0.folder == "step_00_cluster"
        assert step0.type == "cluster"
        assert step0.params == {"K": 5, "distance_threshold": 0.35}
        assert dir0.exists()
        assert dir0 == session_root / "chain_01" / "step_00_cluster"

        step1, dir1 = mgr.append_step(session, "chain_01", "merge", {"n_pairs": 8})
        assert step1.step_index == 1
        assert step1.folder == "step_01_merge"
        assert dir1.exists()

        step2, dir2 = mgr.append_step(session, "chain_01", "merge", {"n_pairs": 5})
        assert step2.step_index == 2
        assert step2.folder == "step_02_merge"

    def test_update_step_result(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        step, _ = mgr.append_step(session, "chain_01", "cluster", {})

        mgr.update_step_result(session, "chain_01", 0, "28 clusters")

        data = load_session_json(session_root / "session.json")
        assert data["chains"][0]["steps"][0]["result_summary"] == "28 clusters"

    def test_get_step_output_dir(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})
        mgr.append_step(session, "chain_01", "merge", {})

        dir0 = mgr.get_step_output_dir(session, "chain_01", 0)
        dir1 = mgr.get_step_output_dir(session, "chain_01", 1)
        assert dir0 == session_root / "chain_01" / "step_00_cluster"
        assert dir1 == session_root / "chain_01" / "step_01_merge"

    def test_get_latest_step_dir_with_steps(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {})
        mgr.append_step(session, "chain_01", "merge", {})

        latest = mgr.get_latest_step_dir(session, "chain_01")
        assert latest == session_root / "chain_01" / "step_01_merge"

    def test_get_latest_step_dir_empty_chain_returns_base(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)

        latest = mgr.get_latest_step_dir(session, "chain_01")
        assert latest == session_root / "base"

    # ------------------------------------------------------------------ branching

    def test_branch_chain_copies_steps(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {"K": 5})
        mgr.append_step(session, "chain_01", "merge", {"n_pairs": 8})
        mgr.append_step(session, "chain_01", "merge", {"n_pairs": 5})

        chain2 = mgr.create_chain(session, branched_from=("chain_01", 0))

        assert chain2.chain_id == "chain_02"
        assert chain2.branched_from == {"chain_id": "chain_01", "step_index": 0}
        # Only step_00 (index 0) copied
        assert len(chain2.steps) == 1
        assert chain2.steps[0].type == "cluster"
        assert chain2.steps[0].params == {"K": 5}
        # chain_01 unchanged
        c1 = mgr.get_chain(session, "chain_01")
        assert len(c1.steps) == 3

    def test_branch_chain_copies_multiple_steps(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {"K": 5})
        mgr.append_step(session, "chain_01", "merge", {"n_pairs": 8})
        mgr.append_step(session, "chain_01", "merge", {"n_pairs": 5})

        chain2 = mgr.create_chain(session, branched_from=("chain_01", 1))

        assert len(chain2.steps) == 2  # steps 0 and 1 copied
        assert chain2.steps[0].type == "cluster"
        assert chain2.steps[1].type == "merge"

    # ------------------------------------------------------------------ persistence

    def test_load_session_survives_restart(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {"n_faces": 200, "n_images": 80})
        chain = mgr.create_chain(session)
        step, _ = mgr.append_step(session, "chain_01", "cluster", {"K": 5})
        mgr.update_step_result(session, "chain_01", 0, "28 clusters")

        mgr2 = SessionManager()
        loaded = mgr2.load(session_root)

        assert loaded is not None
        assert loaded.session_id == session.session_id
        assert loaded.source_album == "/photos/album"
        assert loaded.base["n_faces"] == 200
        assert len(loaded.chains) == 1
        c = loaded.chains[0]
        assert c.chain_id == "chain_01"
        assert len(c.steps) == 1
        assert c.steps[0].result_summary == "28 clusters"
        assert loaded.active_chain == "chain_01"

    def test_load_returns_none_if_no_session_json(self, mgr: SessionManager, tmp_path: Path):
        non_session = tmp_path / "not_a_session"
        non_session.mkdir()
        result = mgr.load(non_session)
        assert result is None

    # ------------------------------------------------------------------ chain independence

    def test_multiple_chains_independent(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.append_step(session, "chain_01", "cluster", {"K": 5})

        mgr.create_chain(session)
        mgr.append_step(session, "chain_02", "cluster", {"K": 8})
        mgr.append_step(session, "chain_02", "merge", {"n_pairs": 3})

        c1 = mgr.get_chain(session, "chain_01")
        c2 = mgr.get_chain(session, "chain_02")
        assert len(c1.steps) == 1
        assert len(c2.steps) == 2

    # ------------------------------------------------------------------ pending labels

    def test_pending_labels_accumulate(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)

        batch1 = [{"pair": ["c1", "c2"], "decision": "approve"}]
        batch2 = [{"pair": ["c3", "c4"], "decision": "reject"}]

        mgr.save_pending_labels(session, "chain_01", batch1)
        mgr.save_pending_labels(session, "chain_01", batch2)

        pending = mgr.get_pending_labels(session, "chain_01")
        assert len(pending) == 2
        assert pending[0]["pair"] == ["c1", "c2"]
        assert pending[1]["pair"] == ["c3", "c4"]

    def test_consume_pending_labels_clears_file(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)

        mgr.save_pending_labels(session, "chain_01", [{"pair": ["c1", "c2"], "decision": "approve"}])
        assert mgr.has_pending_labels(session, "chain_01")

        consumed = mgr.consume_pending_labels(session, "chain_01")
        assert len(consumed) == 1
        assert not mgr.has_pending_labels(session, "chain_01")
        assert not (session_root / "chain_01" / "pending_labels.json").exists()

    def test_has_pending_labels_false_when_empty(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        assert not mgr.has_pending_labels(session, "chain_01")

    def test_has_pending_labels_persisted_in_json(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)

        mgr.save_pending_labels(session, "chain_01", [{"pair": ["c1", "c2"], "decision": "approve"}])

        # Reload and verify flag
        reloaded = mgr.load(session_root)
        assert reloaded is not None
        assert reloaded.chains[0].has_pending_labels is True

    def test_consume_clears_pending_flag_in_json(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.save_pending_labels(session, "chain_01", [{"pair": ["c1", "c2"], "decision": "approve"}])
        mgr.consume_pending_labels(session, "chain_01")

        reloaded = mgr.load(session_root)
        assert reloaded is not None
        assert reloaded.chains[0].has_pending_labels is False

    # ------------------------------------------------------------------ active chain

    def test_set_active_chain(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        mgr.create_chain(session)
        mgr.create_chain(session)
        assert session.active_chain == "chain_02"

        mgr.set_active_chain(session, "chain_01")
        assert session.active_chain == "chain_01"

        data = load_session_json(session_root / "session.json")
        assert data["active_chain"] == "chain_01"

    def test_get_active_chain_none_when_not_set(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        assert mgr.get_active_chain(session) is None

    def test_get_chain_raises_on_unknown(self, mgr: SessionManager, session_root: Path):
        session = mgr.create(session_root, "/photos/album", {})
        with pytest.raises(KeyError):
            mgr.get_chain(session, "chain_99")
