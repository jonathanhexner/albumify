"""Tests for spec-032 P2: RunExporter writes filter_decisions, RunStore reads them.

Verifies the round-trip and that the table behaves correctly on the dual-write
transition (a run with no FilterContext should produce a queryable but empty
filter_decisions table — not a missing table).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.filter_context import FilterContext
from face_cluster.run_exporter import RunExporter
from face_cluster.run_store import RunStore, FilterDecisionRow
from face_cluster.types import ClusterResult, FaceRecord


def _minimal_face(face_id: int, image_path: str) -> FaceRecord:
    rng = np.random.default_rng(face_id)
    emb = rng.standard_normal(512).astype(np.float32)
    return FaceRecord(
        face_id=face_id,
        image_id=Path(image_path).stem,
        bbox=(0.0, 0.0, 100.0, 100.0),
        landmarks=None,
        aligned_face=rng.integers(0, 255, size=(112, 112, 3), dtype=np.uint8),
        embedding=emb,
        embedding_normalized=emb / np.linalg.norm(emb),
        pose=(0.0, 0.0, 0.0),
        blur_score=200.0,
        area=10000.0,
        is_core=True,
        image_path=image_path,
        face_index=0,
        det_score=0.95,
    )


def _minimal_cluster_result(n_faces: int) -> ClusterResult:
    labels = np.zeros(n_faces, dtype=np.int32)
    return ClusterResult(
        labels=labels,
        clusters={0: list(range(n_faces))},
        cluster_stats={0: {"size": n_faces, "diameter": 0.1, "origin": "base", "parent_ids": []}},
        exemplars={0: list(range(min(3, n_faces)))},
        n_clusters=1,
        n_noise=0,
    )


def _run_export(tmp_path: Path, filters):
    faces = [_minimal_face(i, str(tmp_path / "img.jpg")) for i in range(3)]
    cr = _minimal_cluster_result(len(faces))
    cfg = PipelineConfig()
    exporter = RunExporter(tmp_path / "out")
    exporter.export(
        faces=faces,
        base_cluster_result=cr,
        merged_cluster_result=None,
        core_indices=list(range(len(faces))),
        merge_log=[],
        merge_metadata=None,
        config=cfg,
        source_album="test_album",
        producer="fc_app",
        run_id="20260512_000000",
        started_at=datetime.now().isoformat(),
        finished_at=datetime.now().isoformat(),
        filters=filters,
    )
    return tmp_path / "out"


class ut_FilterDecisionsRoundTrip:
    def test_export_with_no_filters_writes_empty_table(self, tmp_path):
        out = _run_export(tmp_path, filters=None)
        store = RunStore(out)
        # No exception, empty list.
        assert store.filter_decisions() == []

    def test_export_with_empty_filters_writes_empty_table(self, tmp_path):
        out = _run_export(tmp_path, filters=FilterContext())
        store = RunStore(out)
        assert store.filter_decisions() == []

    def test_round_trip_preserves_decision_fields(self, tmp_path):
        fc = FilterContext()
        fc.record(
            "face_0042",
            filter_name="face_blur",
            rejected=True,
            reason="blur 12.3 < 50.0",
            measured={"value": 12.3, "threshold": 50.0},
            parent_id="img.jpg",
        )
        out = _run_export(tmp_path, filters=fc)
        rows = RunStore(out).filter_decisions()
        assert len(rows) == 1
        row = rows[0]
        assert row.item_id == "face_0042"
        assert row.item_type == "face"
        assert row.parent_id == "img.jpg"
        assert row.filter_name == "face_blur"
        assert row.rejected is True
        assert row.reason == "blur 12.3 < 50.0"
        assert row.measured == {"value": 12.3, "threshold": 50.0}

    def test_round_trip_preserves_multiple_decisions(self, tmp_path):
        fc = FilterContext()
        fc.record("face_0", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        fc.record("face_0", filter_name="face_pose_yaw",
                  rejected=True, reason="too far", measured={"yaw": 60.0})
        fc.record("img.jpg", filter_name="image_quality",
                  rejected=False, reason="ok", measured={"iqa": 0.5})

        out = _run_export(tmp_path, filters=fc)
        rows = RunStore(out).filter_decisions()
        assert len(rows) == 3

        by_key = {(r.item_id, r.filter_name): r for r in rows}
        assert by_key[("face_0", "face_blur")].rejected is False
        assert by_key[("face_0", "face_pose_yaw")].rejected is True
        assert by_key[("face_0", "face_pose_yaw")].measured == {"yaw": 60.0}
        assert by_key[("img.jpg", "image_quality")].item_type == "image"

    def test_filter_decisions_table_ordered_by_item_id_then_filter(self, tmp_path):
        fc = FilterContext()
        fc.record("z_face", filter_name="face_blur",
                  rejected=False, reason="ok", measured={})
        fc.record("a_image", filter_name="image_quality",
                  rejected=True, reason="bad", measured={})
        out = _run_export(tmp_path, filters=fc)
        rows = RunStore(out).filter_decisions()
        assert [r.item_id for r in rows] == ["a_image", "z_face"]


class ut_FilterDecisionsSchemaContract:
    def test_v4_layout_listdir_unchanged_by_filters_addition(self, tmp_path):
        """Adding filters MUST NOT change the v4 5-artifact layout — the
        decisions live in the DB, not as a sidecar file."""
        from face_cluster.run_exporter import EXPECTED_ARTIFACTS
        out = _run_export(tmp_path, filters=FilterContext())
        actual = sorted(p.name for p in out.iterdir())
        expected = sorted(EXPECTED_ARTIFACTS)
        assert actual == expected

    def test_table_schema_has_primary_key_on_item_filter(self, tmp_path):
        """Re-recording (item, filter) doesn't create dupes in the DB."""
        import sqlite3
        out = _run_export(tmp_path, filters=FilterContext())
        with sqlite3.connect(out / "face_clustering.db") as conn:
            cur = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='filter_decisions'"
            )
            sql = cur.fetchone()[0]
        assert "PRIMARY KEY (item_id, filter_name)" in sql
