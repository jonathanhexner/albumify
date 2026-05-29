"""Tests for face_cluster.run_store (spec-030, Phase 2).

Pin down the reader's contract:
  - Construction validates the layout up-front; any missing artifact raises.
  - schema_version mismatches raise immediately.
  - Round-trip — RunStore reads back exactly what RunExporter wrote.
  - merge_log() returns full-fidelity MergeDecisionRow objects (no defaults).
  - Static check: zero `.exists()` chains in run_store.py source (FR-008).
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.run_exporter import RunExporter
from sim_bench.run_db.store import (
    EmbeddingMatrix,
    RunMetadata,
    RunStore,
    RunStoreError,
)
from face_cluster.types import (
    ClusterResult,
    FaceRecord,
    MergeDecisionRow,
)


# ---------------------------------------------------------------------------
# Fixture builders (mirroring test_run_exporter.py)
# ---------------------------------------------------------------------------

def _make_face(face_id: int) -> FaceRecord:
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
        is_core=True,
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


def _full_merge_row(iteration, cluster_a, cluster_b, *, action="merged", actually_merged=True):
    return {
        "iteration": iteration, "cluster_a": cluster_a, "cluster_b": cluster_b,
        "cluster_a_size": 22, "cluster_b_size": 33,
        "exemplar_dist": 0.580, "threshold_used": 0.6,
        "T_a": None, "T_b": None, "T_global": None,
        "p25_cross_dist": 0.597, "passes_cross": True,
        "support": 191, "unique_support": 16, "required_support": 2,
        "post_diameter": 0.972, "max_allowed_diameter": 2.294,
        "margin_gap": float("inf"),
        "margin_dist_to_b": 0.0,
        "margin_competitor_dist": 0.0,
        "margin_competitor_id": -1,
        "passes_exemplar": True, "passes_support": True,
        "passes_margin": True, "passes_diameter": True,
        "action": action, "actually_merged": actually_merged,
        "rejection_reason": None,
    }


@pytest.fixture
def written_run(tmp_path: Path) -> Path:
    """A complete run on disk produced by RunExporter — used as input to RunStore."""
    out = tmp_path / "run"
    faces = [_make_face(i) for i in range(6)]
    base = _make_cluster_result({0: [0, 1, 2], 1: [3, 4, 5]}, n_total=6)
    merged = _make_cluster_result({0: [0, 1, 2, 3, 4, 5]}, n_total=6)
    log = [_full_merge_row(1, 0, 1, action="merged", actually_merged=True)]
    RunExporter(out).export(
        faces=faces,
        base_cluster_result=base,
        merged_cluster_result=merged,
        core_indices=list(range(6)),
        merge_log=log,
        merge_metadata={"n_iterations": 1, "merge_exemplar_threshold": 0.6},
        config=PipelineConfig(),
        source_album="ut_album",
        producer="fc_app",
        run_id="20260509_120000",
        started_at="2026-05-09T12:00:00",
        finished_at="2026-05-09T12:00:30",
    )
    # Crops would normally be present from face.aligned_face; the writer
    # creates an empty crops/ dir when none are provided.  Drop in two stub
    # files so crop_path() round-trip tests have something to find.
    crops = out / "crops"
    crops.mkdir(exist_ok=True)
    for f in faces:
        (crops / f"face_{f.face_id:04d}_aligned.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    # Update DB to match (writer didn't record paths because aligned_face was None).
    conn = sqlite3.connect(out / "face_clustering.db")
    for f in faces:
        conn.execute(
            "UPDATE faces SET crop_path = ? WHERE face_id = ?",
            (f"crops/face_{f.face_id:04d}_aligned.jpg", f.face_id),
        )
    conn.commit()
    conn.close()
    return out


# ---------------------------------------------------------------------------
# Construction / validation tests
# ---------------------------------------------------------------------------

def test_constructs_on_valid_run(written_run):
    store = RunStore(written_run)
    assert store.run_dir == written_run


def test_raises_on_missing_directory(tmp_path):
    with pytest.raises(RunStoreError, match="run directory not found"):
        RunStore(tmp_path / "does-not-exist")


def test_raises_on_missing_pipeline_run_json(written_run):
    (written_run / "pipeline_run.json").unlink()
    with pytest.raises(RunStoreError, match="pipeline_run.json"):
        RunStore(written_run)


def test_raises_on_missing_db(written_run):
    (written_run / "face_clustering.db").unlink()
    with pytest.raises(RunStoreError, match="face_clustering.db"):
        RunStore(written_run)


def test_raises_on_missing_embeddings(written_run):
    (written_run / "embeddings.npy").unlink()
    with pytest.raises(RunStoreError, match="embeddings.npy"):
        RunStore(written_run)


def test_raises_on_missing_crops_dir(written_run):
    shutil.rmtree(written_run / "crops")
    with pytest.raises(RunStoreError, match="crops"):
        RunStore(written_run)


def test_raises_on_schema_version_mismatch(written_run):
    pr = written_run / "pipeline_run.json"
    payload = json.loads(pr.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    pr.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunStoreError, match="schema_version mismatch"):
        RunStore(written_run)


def test_raises_on_db_user_version_mismatch(written_run):
    """JSON says v4 but DB says something else → reject."""
    conn = sqlite3.connect(written_run / "face_clustering.db")
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(RunStoreError, match="user_version"):
        RunStore(written_run)


def test_raises_on_corrupt_pipeline_run_json(written_run):
    (written_run / "pipeline_run.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(RunStoreError, match="not valid JSON"):
        RunStore(written_run)


# ---------------------------------------------------------------------------
# Read-method round-trip tests
# ---------------------------------------------------------------------------

def test_metadata_round_trip(written_run):
    meta = RunStore(written_run).metadata()
    assert isinstance(meta, RunMetadata)
    assert meta.run_id == "20260509_120000"
    assert meta.source_album == "ut_album"
    assert meta.producer == "fc_app"
    assert meta.parent_run_id is None
    assert meta.n_faces == 6
    assert meta.n_merges == 1
    assert meta.n_iterations == 1
    assert meta.n_clusters_base == 2
    assert meta.n_clusters_final == 1
    assert meta.schema_version == 5


def test_merge_log_full_fidelity(written_run):
    """Every field of every row reads back exactly as written.  No defaults."""
    rows = RunStore(written_run).merge_log()
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, MergeDecisionRow)
    # Identity
    assert (r.iteration, r.cluster_a, r.cluster_b) == (1, 0, 1)
    # Booleans came back as bool, not int
    assert r.passes_exemplar is True and r.passes_support is True
    assert r.passes_margin is True and r.passes_diameter is True
    assert r.actually_merged is True
    # inf round-trip
    assert r.margin_gap == float("inf")
    # Action and rejection
    assert r.action == "merged"
    assert r.rejection_reason is None


def test_embeddings_round_trip(written_run):
    emb = RunStore(written_run).embeddings()
    assert isinstance(emb, EmbeddingMatrix)
    assert emb.matrix.shape == (6, 512)
    assert emb.matrix.dtype == np.float32
    assert emb.face_ids.tolist() == list(range(6))


def test_faces_round_trip(written_run):
    faces = RunStore(written_run).faces()
    assert len(faces) == 6
    assert all(isinstance(f, FaceRecord) for f in faces)
    assert [f.face_id for f in faces] == list(range(6))
    # Embeddings attached from npy
    assert all(f.embedding_normalized is not None for f in faces)


def test_clusters_base_and_final(written_run):
    store = RunStore(written_run)
    base = store.clusters("base")
    final = store.clusters("final")
    assert isinstance(base, ClusterResult)
    assert base.n_clusters == 2
    assert final.n_clusters == 1


def test_clusters_iter_label_invalid(written_run):
    with pytest.raises(RunStoreError, match="iteration label"):
        RunStore(written_run).clusters("bogus")


def test_iteration_count(written_run):
    assert RunStore(written_run).iteration_count() == 1


def test_crop_path_resolves(written_run):
    p = RunStore(written_run).crop_path(0)
    assert p.is_file()
    assert p.name == "face_0000_aligned.jpg"


def test_crop_path_unknown_face_raises(written_run):
    with pytest.raises(RunStoreError, match="unknown face_id"):
        RunStore(written_run).crop_path(999)


def test_crop_path_missing_file_raises(written_run):
    (written_run / "crops" / "face_0000_aligned.jpg").unlink()
    with pytest.raises(RunStoreError, match="crop file missing"):
        RunStore(written_run).crop_path(0)


# ---------------------------------------------------------------------------
# Architecture invariant — no .exists() chain in the reader (FR-008, FR-012)
# ---------------------------------------------------------------------------

def test_no_existence_chain_in_run_store():
    """run_store.py must use exists() only for raise-on-miss validation,
    never inside `if x.exists(): use_x else use_y` style fallback chains.
    Specifically: the file may not contain `elif .*\\.exists\\(\\):` patterns.
    """
    src = Path(__file__).resolve().parents[2] / "sim_bench" / "run_db" / "store.py"
    text = src.read_text(encoding="utf-8")

    # Disallow elif <anything>.exists(): — that is the fallback-chain pattern.
    forbidden = re.findall(r"^\s*elif\s+.*\.exists\s*\(\s*\)\s*:", text, flags=re.M)
    assert not forbidden, (
        f"run_store.py contains forbidden fallback chain(s): {forbidden}"
    )

    # Disallow the pattern: if x.exists(): (without raise/return on the next line).
    # Approximate: each `if .exists():` must be paired with a `raise` or `return`
    # within ~5 lines.  Cheap heuristic that catches the common shape.
    if_exists = list(re.finditer(r"^\s*if\s+.*\.exists\s*\(\s*\)\s*:", text, flags=re.M))
    lines = text.splitlines()
    for m in if_exists:
        line_no = text[: m.start()].count("\n") + 1
        block = "\n".join(lines[line_no - 1: line_no + 4])
        assert "raise" in block or "return" in block, (
            f"run_store.py:{line_no}: `if .exists():` without raise/return nearby:\n"
            f"{block}"
        )
