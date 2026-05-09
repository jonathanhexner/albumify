"""Tests for face_cluster.run_exporter (spec-030, Phase 1).

Pin down the writer's contract:
  - Produces exactly the 5-artifact layout (FR-003).
  - merge_decisions round-trips all 28 MergeDecisionRow fields (FR-004).
  - Strict-write rejects rows with unknown / missing keys (FR-005).
  - Same input from "albumify" or "fc_app" producers yields identical
    on-disk content modulo the producer column itself (US2).
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import List

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.run_exporter import (
    EXPECTED_ARTIFACTS,
    SCHEMA_VERSION,
    RunExporter,
    RunExporterError,
)
from face_cluster.types import (
    ClusterResult,
    FaceRecord,
    MergeDecisionRow,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_face(face_id: int, *, is_core: bool = True) -> FaceRecord:
    rng = np.random.default_rng(seed=face_id)
    emb = rng.standard_normal(512).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return FaceRecord(
        face_id=face_id,
        image_id=f"img_{face_id // 4}.jpg",
        image_path=f"/tmp/img_{face_id // 4}.jpg",
        bbox=(10.0, 20.0, 30.0, 40.0),
        area=1200.0,
        blur_score=80.0,
        is_core=is_core,
        face_index=face_id % 4,
        embedding=emb,
        embedding_normalized=emb,
        pose=(5.0, -2.0, 1.0),
        det_score=0.9,
    )


def _make_cluster_result(clusters: dict, n_total: int) -> ClusterResult:
    labels = np.full(n_total, -1, dtype=np.int32)
    for cid, members in clusters.items():
        for idx in members:
            labels[idx] = cid
    return ClusterResult(
        labels=labels,
        clusters=clusters,
        cluster_stats={cid: {"diameter": 0.3, "mean_dist": 0.15} for cid in clusters},
        exemplars={cid: members[:1] for cid, members in clusters.items()},
        n_clusters=len(clusters),
        n_noise=int((labels == -1).sum()),
    )


def _full_merge_row(
    iteration: int,
    cluster_a: int,
    cluster_b: int,
    *,
    action: str = "merged",
    actually_merged: bool = True,
) -> dict:
    """Produce a dict whose keys exactly equal MergeDecisionRow.field_names()."""
    return {
        "iteration": iteration,
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "cluster_a_size": 22,
        "cluster_b_size": 33,
        "exemplar_dist": 0.580,
        "threshold_used": 0.6,
        "T_a": None,
        "T_b": None,
        "T_global": None,
        "p25_cross_dist": 0.597,
        "passes_cross": True,
        "support": 191,
        "unique_support": 16,
        "required_support": 2,
        "post_diameter": 0.972,
        "max_allowed_diameter": 2.294,
        "margin_gap": float("inf"),
        "margin_dist_to_b": 0.0,
        "margin_competitor_dist": 0.0,
        "margin_competitor_id": -1,
        "passes_exemplar": True,
        "passes_support": True,
        "passes_margin": True,
        "passes_diameter": True,
        "action": action,
        "actually_merged": actually_merged,
        "rejection_reason": None,
    }


@pytest.fixture
def small_run(tmp_path: Path):
    """A minimal but realistic run: 6 faces, 2 clusters, 1 actual merge."""
    faces = [_make_face(i) for i in range(6)]
    base = _make_cluster_result({0: [0, 1, 2], 1: [3, 4, 5]}, n_total=6)
    merged = _make_cluster_result({0: [0, 1, 2, 3, 4, 5]}, n_total=6)
    merge_log = [
        _full_merge_row(1, 0, 1, action="merged", actually_merged=True),
    ]
    return {
        "tmp": tmp_path,
        "faces": faces,
        "base": base,
        "merged": merged,
        "merge_log": merge_log,
        "core_indices": list(range(6)),
        "config": PipelineConfig(),
    }


def _export_small_run(small_run, *, output_dir: Path, producer: str = "fc_app") -> None:
    RunExporter(output_dir).export(
        faces=small_run["faces"],
        base_cluster_result=small_run["base"],
        merged_cluster_result=small_run["merged"],
        core_indices=small_run["core_indices"],
        merge_log=small_run["merge_log"],
        merge_metadata={
            "n_iterations": 1,
            "merge_exemplar_threshold": 0.6,
            "merge_candidate_threshold": 0.65,
        },
        config=small_run["config"],
        source_album="ut_album",
        producer=producer,
        run_id="20260509_120000",
        started_at="2026-05-09T12:00:00",
        finished_at="2026-05-09T12:00:30",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_writes_exact_artifact_set(small_run):
    """FR-003: a run dir contains exactly the EXPECTED_ARTIFACTS set, nothing else."""
    out = small_run["tmp"] / "run_a"
    _export_small_run(small_run, output_dir=out)

    actual = sorted(p.name for p in out.iterdir())
    expected = sorted(EXPECTED_ARTIFACTS)
    assert actual == expected, f"unexpected listdir: {actual}, expected: {expected}"


def test_pipeline_run_json_is_pointer_only(small_run):
    """FR-007: pipeline_run.json carries pointer fields, no config blob."""
    out = small_run["tmp"] / "run_b"
    _export_small_run(small_run, output_dir=out)

    payload = json.loads((out / "pipeline_run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["db_path"] == "face_clustering.db"
    assert payload["producer"] == "fc_app"
    assert payload["run_id"] == "20260509_120000"
    assert payload["status"] == "complete"
    # Anti-assertion: nothing in this file is a config dump.
    assert "config" not in payload
    assert "merge_thresholds" not in payload


def test_db_user_version_pragma(small_run):
    """SCHEMA_VERSION is reflected as PRAGMA user_version on the DB itself."""
    out = small_run["tmp"] / "run_c"
    _export_small_run(small_run, output_dir=out)

    conn = sqlite3.connect(out / "face_clustering.db")
    user_ver = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert user_ver == SCHEMA_VERSION


def test_merge_decisions_full_fidelity(small_run):
    """FR-004: every field of every merge_log row survives the DB round-trip."""
    out = small_run["tmp"] / "run_d"
    _export_small_run(small_run, output_dir=out)

    conn = sqlite3.connect(out / "face_clustering.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM merge_decisions").fetchall()
    conn.close()

    expected_fields = MergeDecisionRow.field_names()
    assert len(rows) == 1
    row = dict(rows[0])

    # SQLite has no boolean type; pass_* and actually_merged come back as int.
    bool_fields = {
        "passes_cross", "passes_exemplar", "passes_support",
        "passes_margin", "passes_diameter", "actually_merged",
    }

    src = small_run["merge_log"][0]
    for fname in expected_fields:
        got = row[fname]
        want = src[fname]
        if fname in bool_fields and want is not None:
            assert got == (1 if want else 0), f"{fname}: got {got!r}, want {want!r}"
        elif isinstance(want, float) and math.isinf(want):
            assert math.isinf(got), f"{fname}: lost inf, got {got!r}"
        else:
            assert got == want, f"{fname}: got {got!r}, want {want!r}"


def test_writer_raises_on_unknown_field(small_run):
    """FR-005: strict-write — a merge_log row with an extra key is rejected."""
    out = small_run["tmp"] / "run_e"
    bad_log = [_full_merge_row(1, 0, 1)]
    bad_log[0]["future_extra_field"] = 42

    with pytest.raises(RunExporterError, match="extra=.*future_extra_field"):
        RunExporter(out).export(
            faces=small_run["faces"],
            base_cluster_result=small_run["base"],
            merged_cluster_result=small_run["merged"],
            core_indices=small_run["core_indices"],
            merge_log=bad_log,
            merge_metadata=None,
            config=small_run["config"],
            source_album="ut_album",
            producer="fc_app",
            run_id="x", started_at="x", finished_at="x",
        )


def test_writer_raises_on_missing_field(small_run):
    """FR-005: strict-write — a merge_log row missing a key is rejected."""
    out = small_run["tmp"] / "run_f"
    bad_log = [_full_merge_row(1, 0, 1)]
    del bad_log[0]["actually_merged"]

    with pytest.raises(RunExporterError, match="missing=.*actually_merged"):
        RunExporter(out).export(
            faces=small_run["faces"],
            base_cluster_result=small_run["base"],
            merged_cluster_result=small_run["merged"],
            core_indices=small_run["core_indices"],
            merge_log=bad_log,
            merge_metadata=None,
            config=small_run["config"],
            source_album="ut_album",
            producer="fc_app",
            run_id="x", started_at="x", finished_at="x",
        )


def test_invalid_producer_rejected(small_run):
    """Only the four allow-listed producer values are accepted."""
    out = small_run["tmp"] / "run_g"
    with pytest.raises(RunExporterError, match="producer must be one of"):
        RunExporter(out).export(
            faces=small_run["faces"],
            base_cluster_result=small_run["base"],
            merged_cluster_result=small_run["merged"],
            core_indices=small_run["core_indices"],
            merge_log=[],
            merge_metadata=None,
            config=small_run["config"],
            source_album="ut_album",
            producer="rogue_writer",
            run_id="x", started_at="x", finished_at="x",
        )


def test_run_metadata_n_merges_matches_log(small_run):
    """run_metadata.n_merges == count of actually_merged=True rows."""
    out = small_run["tmp"] / "run_h"
    _export_small_run(small_run, output_dir=out)

    conn = sqlite3.connect(out / "face_clustering.db")
    conn.row_factory = sqlite3.Row
    meta = dict(conn.execute("SELECT * FROM run_metadata").fetchone())
    conn.close()
    assert meta["n_merges"] == 1
    assert meta["n_iterations"] == 1
    assert meta["producer"] == "fc_app"
    assert meta["schema_version"] == SCHEMA_VERSION


def test_albumify_and_fcapp_produce_identical_layout(small_run, tmp_path):
    """US2 sanity — same input, two producer values, byte-equal artifact set
    and byte-equal merge_decisions content (only run_metadata.producer differs)."""
    out_a = tmp_path / "out_albumify"
    out_b = tmp_path / "out_fcapp"
    _export_small_run(small_run, output_dir=out_a, producer="albumify")
    _export_small_run(small_run, output_dir=out_b, producer="fc_app")

    assert sorted(p.name for p in out_a.iterdir()) == sorted(p.name for p in out_b.iterdir())

    # The npy files should be byte-identical.
    assert (out_a / "embeddings.npy").read_bytes() == (out_b / "embeddings.npy").read_bytes()
    assert (out_a / "embedding_face_ids.npy").read_bytes() == (out_b / "embedding_face_ids.npy").read_bytes()

    def _merge_rows(p: Path):
        conn = sqlite3.connect(p / "face_clustering.db")
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute("SELECT * FROM merge_decisions ORDER BY iteration, cluster_a, cluster_b")]
        conn.close()
        return rows

    assert _merge_rows(out_a) == _merge_rows(out_b)


def test_embeddings_npy_alignment(small_run):
    """Row i of embeddings.npy ↔ face_id at index i in embedding_face_ids.npy."""
    out = small_run["tmp"] / "run_i"
    _export_small_run(small_run, output_dir=out)

    matrix = np.load(out / "embeddings.npy")
    ids = np.load(out / "embedding_face_ids.npy")
    assert matrix.shape == (len(small_run["faces"]), 512)
    assert matrix.dtype == np.float32
    assert ids.tolist() == [f.face_id for f in small_run["faces"]]


def test_no_embeddings_table_in_db(small_run):
    """FR-006: embeddings table is not part of the v4 schema."""
    out = small_run["tmp"] / "run_j"
    _export_small_run(small_run, output_dir=out)

    conn = sqlite3.connect(out / "face_clustering.db")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    conn.close()
    assert "embeddings" not in tables, f"embeddings table should not exist in v4 schema, got tables: {tables}"


def test_merge_decisions_column_order_matches_dataclass(small_run):
    """The DB column order must match MergeDecisionRow.field_names() so the
    writer's positional INSERT stays correct over time."""
    out = small_run["tmp"] / "run_k"
    _export_small_run(small_run, output_dir=out)

    conn = sqlite3.connect(out / "face_clustering.db")
    cols = [r[1] for r in conn.execute("PRAGMA table_info(merge_decisions)").fetchall()]
    conn.close()
    assert tuple(cols) == MergeDecisionRow.field_names()
