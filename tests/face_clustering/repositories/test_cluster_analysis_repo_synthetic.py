"""spec-045 Phase 1 — synthetic-data tests for ClusterAnalysisRepository.

Builds a small in-temp-dir run dir (3 real clusters + a noise cluster,
~30 faces total) and exercises the Repository's read surface. No
filesystem state outside ``tmp_path``; no dependency on the
spec-046 ``transactional_session`` fixture (per-run DB isn't
Alembic-managed — D3).

Covers spec.§8.1 cases #1–#10 (mutation cases #11/#12 land in Phase 2).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

from sim_bench.run_db._schema import SCHEMA_DDL, SCHEMA_VERSION
from face_cluster.repositories._errors import ValidationError
from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisCriteria,
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from sim_bench.run_db.store import RunMetadata
from face_cluster.views._base import Assignment, ClusterRow
from face_cluster.views.cluster_analysis import ForceMergeResult
from sim_bench.pipeline.clustering_labels import NOISE_LABEL


# ---------------------------------------------------------------------------
# Synthetic fixture — small but schema-valid run dir
# ---------------------------------------------------------------------------

# Three real clusters (sizes 10 / 10 / 10) + a noise bucket (size 2).
# Exemplars: first 2 faces of each real cluster are flagged.
_CLUSTERS: List[Tuple[int, int]] = [(0, 10), (1, 10), (2, 10), (NOISE_LABEL, 2)]
_N_FACES = sum(size for _, size in _CLUSTERS)  # 32
# 512 matches the InsightFace production embedding dim. The legacy
# manual_merge_snapshot writer hardcodes 512 — the fixture must agree.
_EMBED_DIM = 512


def _build_synthetic_run_dir(tmp_path: Path) -> Path:
    """Write a minimal, schema-valid run dir under ``tmp_path``.

    Returns the run dir path. Files written: face_clustering.db (full v5
    schema), embeddings.npy, embedding_face_ids.npy, pipeline_run.json,
    crops/ (empty dir). The DB is seeded with 3 real clusters + 1 noise
    cluster.
    """
    run_dir = tmp_path / "synthetic_run"
    run_dir.mkdir()
    (run_dir / "crops").mkdir()

    # Embeddings: deterministic L2-normalized vectors (faces in the same
    # cluster get nearby vectors so distance-based tests stay stable).
    rng = np.random.default_rng(seed=0)
    face_ids: List[int] = []
    embeddings: List[np.ndarray] = []
    fid = 1000
    for cid, size in _CLUSTERS:
        center = rng.standard_normal(_EMBED_DIM).astype(np.float32)
        center /= np.linalg.norm(center) + 1e-9
        for _ in range(size):
            v = center + 0.05 * rng.standard_normal(_EMBED_DIM).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            embeddings.append(v)
            face_ids.append(fid)
            fid += 1
    np.save(run_dir / "embeddings.npy", np.stack(embeddings))
    np.save(run_dir / "embedding_face_ids.npy", np.array(face_ids, dtype=np.int64))

    # pipeline_run.json (RunStore validates schema_version matches).
    (run_dir / "pipeline_run.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "run_id": "synthetic"}),
        encoding="utf-8",
    )

    # face_clustering.db — full v5 schema + seed rows.
    db_path = run_dir / "face_clustering.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_DDL)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

        # faces
        # SIGHTING-080 follow-up: crop_path populated with the v5 writer's
        # filename pattern (face_{id:04d}_aligned.jpg). Real crop files
        # written below as 0-byte placeholders so Path.is_file() passes.
        faces_rows = []
        for i, fid in enumerate(face_ids):
            crop_rel = f"crops/face_{fid:04d}_aligned.jpg"
            (run_dir / crop_rel).touch()  # placeholder is enough for is_file()
            faces_rows.append(
                (fid, f"img_{i:03d}.jpg", f"img_{i:03d}", 0, 0, 0, 0, 0, crop_rel,
                 0.99, 0.5, 100.0, None, None, None, 1, None)
            )
        conn.executemany(
            "INSERT INTO faces (face_id, image_path, image_id, face_index, "
            "bbox_x, bbox_y, bbox_w, bbox_h, crop_path, det_score, blur_score, "
            "area, yaw, pitch, roll, is_core, rejection_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            faces_rows,
        )

        # clusters + cluster_assignments (iteration=0 only — "final" resolves to max).
        offset = 0
        for cid, size in _CLUSTERS:
            conn.execute(
                "INSERT INTO clusters (cluster_id, iteration, size, diameter, "
                "avg_intra_dist, origin, parent_ids) VALUES (?,0,?,?,?,'base','[]')",
                (cid, size, 0.1 if cid != NOISE_LABEL else 0.0,
                 0.05 if cid != NOISE_LABEL else 0.0),
            )
            for j in range(size):
                fid = face_ids[offset + j]
                is_exemplar = 1 if (cid != NOISE_LABEL and j < 2) else 0
                conn.execute(
                    "INSERT INTO cluster_assignments (face_id, cluster_id, "
                    "iteration, is_exemplar, d10_score) VALUES (?,?,0,?,0.1)",
                    (fid, cid, is_exemplar),
                )
            offset += size

        # run_metadata — minimal but valid (RunStore.metadata() parses this).
        conn.execute(
            "INSERT INTO run_metadata (run_id, source_album, producer, "
            "parent_run_id, config_json, merge_thresholds_json, "
            "merge_iter_summary_json, n_images, n_faces, n_core, "
            "n_clusters_base, n_clusters_final, n_merges, n_iterations, "
            "started_at, finished_at, schema_version) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("synthetic", "synthetic_album", "test", None,
             json.dumps({"merge_candidate_threshold": 0.45}),
             None, None,
             _N_FACES, _N_FACES, _N_FACES - 2, 3, 3, 0, 1,
             "2026-01-01T00:00:00", "2026-01-01T00:00:01", SCHEMA_VERSION),
        )
        conn.commit()
    finally:
        conn.close()
    return run_dir


@pytest.fixture
def synthetic_run_dir(tmp_path) -> Path:
    return _build_synthetic_run_dir(tmp_path)


@pytest.fixture
def repo(synthetic_run_dir) -> ClusterAnalysisRepository:
    return ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=synthetic_run_dir))


# ---------------------------------------------------------------------------
# Tests — spec.§8.1 #1–#10
# ---------------------------------------------------------------------------

def test_init_validates_run_dir(tmp_path):  # #1
    missing = tmp_path / "does_not_exist"
    with pytest.raises(ValueError, match="run_dir does not exist"):
        ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=missing))


def test_init_validates_db_exists(tmp_path):  # #2
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(ValueError, match="face_clustering.db not found"):
        ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=empty))


def test_get_cluster_rows_returns_typed_list(repo):  # #3
    rows = repo.get_cluster_rows()
    assert isinstance(rows, list)
    assert all(isinstance(r, ClusterRow) for r in rows)
    assert len(rows) == 3  # 3 real clusters; noise excluded
    assert [r.cluster_id for r in rows] == sorted([r.cluster_id for r in rows])


def test_get_cluster_rows_excludes_noise_by_default(repo):  # #4
    rows = repo.get_cluster_rows()
    assert NOISE_LABEL not in {r.cluster_id for r in rows}


def test_get_cluster_ids_matches_rows(repo):  # #5
    assert repo.get_cluster_ids() == [r.cluster_id for r in repo.get_cluster_rows()]


def test_find_assignments_by_cluster(repo):  # #6
    rows = repo.find_assignments(ClusterAnalysisCriteria(cluster_id=2))
    assert rows  # cluster 2 exists
    assert {a.cluster_id for a in rows} == {2}
    assert all(isinstance(a, Assignment) for a in rows)


def test_find_assignments_exemplars_only(repo):  # #7
    rows = repo.find_assignments(ClusterAnalysisCriteria(exemplars_only=True))
    assert rows
    assert all(a.is_exemplar for a in rows)
    # 2 exemplars per real cluster, 3 clusters → 6 exemplars
    assert len(rows) == 6


def test_find_assignments_include_noise(repo):  # #8
    without = repo.find_assignments(ClusterAnalysisCriteria())
    with_noise = repo.find_assignments(ClusterAnalysisCriteria(include_noise=True))
    assert NOISE_LABEL not in {a.cluster_id for a in without}
    assert NOISE_LABEL in {a.cluster_id for a in with_noise}
    assert len(with_noise) == len(without) + 2  # 2 noise faces in the fixture


def test_get_face_records_filters_by_id(repo):  # #9
    all_assignments = repo.find_assignments(ClusterAnalysisCriteria(include_noise=True))
    sample = [a.face_id for a in all_assignments[:5]]
    records = repo.get_face_records(sample + [999_999])  # one unknown id
    assert len(records) == 5  # unknown silently dropped
    assert {r.face_id for r in records} == set(sample)


def test_get_run_metadata_shape(repo):  # #10
    meta = repo.get_run_metadata()
    assert isinstance(meta, RunMetadata)
    assert meta.n_clusters_final == 3
    assert meta.n_faces == _N_FACES
    assert isinstance(meta.n_merges, int)
    assert meta.schema_version == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Mutation tests — spec.§8.1 #11–#12 (Phase 2)
# ---------------------------------------------------------------------------

def _legacy_config():
    """Minimal legacy PipelineConfig the snapshot writer wants. Imported lazily
    so the read-only tests above don't pay for the legacy config import path."""
    from face_cluster.config import PipelineConfig
    return PipelineConfig()


def _hash_dir(d: Path) -> bytes:
    """Concatenated bytes-len signature of a dir — cheap "unchanged" proof."""
    import hashlib
    h = hashlib.sha256()
    for p in sorted(d.iterdir()):
        if p.is_file():
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
            h.update(p.read_bytes())
    return h.digest()


def test_save_manual_merge_snapshot_writes_snapshot(repo, synthetic_run_dir):  # #11
    before = _hash_dir(synthetic_run_dir)
    result = repo.save_manual_merge_snapshot(
        cluster_a=0, cluster_b=1, merge_round=1, config=_legacy_config(),
    )
    assert isinstance(result, ForceMergeResult)
    assert result.snapshot_dir.exists()
    assert result.snapshot_dir != synthetic_run_dir
    assert result.parent_run_dir == synthetic_run_dir
    assert result.merge_round == 1
    assert result.new_cluster_id == 0  # min(0, 1)
    assert result.n_merged == 1
    # Parent run dir contents unchanged (force-merge is non-destructive).
    assert _hash_dir(synthetic_run_dir) == before


def test_save_manual_merge_snapshot_read_only_raises(synthetic_run_dir):  # #12
    ro_repo = ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=synthetic_run_dir, read_only=True)
    )
    before = _hash_dir(synthetic_run_dir)
    with pytest.raises(ValidationError, match="read-only"):
        ro_repo.save_manual_merge_snapshot(
            cluster_a=0, cluster_b=1, merge_round=1, config=_legacy_config(),
        )
    # Nothing written: parent untouched, no sibling snapshot dir.
    assert _hash_dir(synthetic_run_dir) == before
    assert not (synthetic_run_dir.parent / f"{synthetic_run_dir.name}_merge_snap_1").exists()


# ---------------------------------------------------------------------------
# Regression: real-world "merger ran, merged nothing" scenario (2026-05-29)
# ---------------------------------------------------------------------------

def _add_no_op_merge_round(run_dir: Path) -> None:
    """Add a merge_decisions row at iteration=1 with actually_merged=0.

    Mirrors the user's failing Budapest run (52a70e6f...): clusters table
    has rows only at iteration=0; merger considered 3 pairs at iter=1
    and merged none, so merge_decisions has iter=1 rows but clusters /
    cluster_assignments don't.

    Before the 2026-05-29 fix, RunStore.clusters("final") resolved
    "final" to MAX(iteration) FROM merge_decisions = 1, queried clusters
    WHERE iteration=1, found nothing, and raised RunStoreError. The
    Repository now resolves "final" against the clusters table itself
    (so "final" → 0 in this case) and passes the int to RunStore.
    """
    conn = sqlite3.connect(str(run_dir / "face_clustering.db"))
    try:
        # Minimum required columns per MERGE_DECISIONS_DDL.
        conn.execute(
            "INSERT INTO merge_decisions ("
            "  iteration, cluster_a, cluster_b, cluster_a_size, cluster_b_size,"
            "  exemplar_dist, threshold_used, support, required_support,"
            "  post_diameter, max_allowed_diameter, margin_gap, margin_dist_to_b,"
            "  margin_competitor_dist, margin_competitor_id,"
            "  passes_exemplar, passes_support, passes_margin, passes_diameter,"
            "  action, actually_merged"
            ") VALUES (1, 0, 1, 10, 10, 0.5, 0.45, 0, 1, 0.1, 0.3, 0.1, 0.5, 0.6, 2, 0, 0, 0, 1, 'reject', 0)"
        )
        conn.commit()
    finally:
        conn.close()


def test_get_cluster_result_final_when_merger_ran_but_merged_nothing(synthetic_run_dir):
    """Regression for the 2026-05-29 Budapest crash: RunStore's "final"
    resolver picks the merge_decisions max iteration, but clusters live
    at the previous iteration. The Repository's get_cluster_result must
    resolve "final" against the clusters table so it returns iteration 0
    (which has rows) instead of iteration 1 (which doesn't)."""
    _add_no_op_merge_round(synthetic_run_dir)
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=synthetic_run_dir))
    # Must NOT raise. Must return the iteration-0 ClusterResult (noise + 3 real).
    cr = repo.get_cluster_result("final")
    assert cr is not None
    real_clusters = {cid for cid in cr.clusters if cid != NOISE_LABEL}
    assert real_clusters == {0, 1, 2}
    # Belt-and-braces: list_clusters() (which uses the same resolver) agrees.
    rows = repo.get_cluster_rows("final")
    assert {r.cluster_id for r in rows} == {0, 1, 2}
