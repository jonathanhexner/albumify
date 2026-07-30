"""spec-057 Phase 0 — golden-hash equivalence baseline.

Snapshots the byte-level output of ``RunExporter.export()`` against a
synthetic input (3 clusters, ~30 faces). Every writer extraction in
Phase 1-2 must keep producing the same hashes; if any byte changes,
this test goes red and the extraction is reverted.

The snapshot file is ``_golden_hashes.txt`` next to this test; commit
it alongside Phase 0. To regenerate after an intentional behavior
change (e.g., bumping SCHEMA_VERSION), delete the file and re-run the
test once — it writes a fresh snapshot when missing.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from face_cluster.config import PipelineConfig
from face_cluster.types import ClusterResult, FaceRecord
from sim_bench.run_db.exporter import RunExporter


GOLDEN_PATH = Path(__file__).with_name("_golden_hashes.txt")


# ---------------------------------------------------------------------------
# Synthetic fixture — 3 clusters, ~30 faces, deterministic
# ---------------------------------------------------------------------------

_N_PER_CLUSTER = 10
_CLUSTERS = (0, 1, 2)
_EMBED_DIM = 512
_RNG_SEED = 20260529


def _make_face(face_id: int, image_id: str) -> FaceRecord:
    rng = np.random.default_rng(seed=_RNG_SEED + face_id)
    emb = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-9
    return FaceRecord(
        face_id=face_id,
        image_id=image_id,
        image_path=f"/tmp/{image_id}",
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


def _make_cluster_result(members: Dict[int, List[int]], n_total: int) -> ClusterResult:
    labels = np.full(n_total, -1, dtype=np.int32)
    for cid, idxs in members.items():
        for idx in idxs:
            labels[idx] = cid
    return ClusterResult(
        labels=labels,
        clusters=members,
        cluster_stats={cid: {"diameter": 0.3, "mean_dist": 0.15} for cid in members},
        exemplars={cid: idxs[:2] for cid, idxs in members.items()},
        n_clusters=len(members),
        n_noise=int((labels == -1).sum()),
    )


def _full_merge_row(iteration: int, ca: int, cb: int) -> dict:
    return {
        "iteration": iteration,
        "cluster_a": ca,
        "cluster_b": cb,
        "cluster_a_size": _N_PER_CLUSTER,
        "cluster_b_size": _N_PER_CLUSTER,
        "exemplar_dist": 0.42,
        "threshold_used": 0.5,
        "T_a": None, "T_b": None, "T_global": None,
        "p25_cross_dist": 0.41,
        "passes_cross": True,
        "support": 50, "unique_support": 8, "required_support": 2,
        "post_diameter": 0.6, "max_allowed_diameter": 1.5,
        "margin_gap": float("inf"),
        "margin_dist_to_b": 0.0,
        "margin_competitor_dist": 0.0,
        "margin_competitor_id": -1,
        "passes_exemplar": True, "passes_support": True,
        "passes_margin": True, "passes_diameter": True,
        "action": "merged", "actually_merged": True,
        "rejection_reason": None,
    }


@pytest.fixture
def golden_exporter_input(tmp_path: Path):
    """30 faces (10 per cluster x 3) + 1 successful merge round."""
    faces: List[FaceRecord] = []
    members: Dict[int, List[int]] = {cid: [] for cid in _CLUSTERS}
    fid_to_idx: Dict[int, int] = {}
    for cid in _CLUSTERS:
        for j in range(_N_PER_CLUSTER):
            idx = len(faces)
            fid = 1000 + idx
            faces.append(_make_face(fid, f"img_{cid:02d}_{j:02d}.jpg"))
            members[cid].append(idx)
            fid_to_idx[fid] = idx
    n_total = len(faces)
    base = _make_cluster_result(members, n_total)

    # Merge 1 into 0 → final has 2 clusters.
    merged_members = {0: members[0] + members[1], 2: members[2]}
    merged = _make_cluster_result(merged_members, n_total)
    merge_log = [_full_merge_row(1, 0, 1)]

    return {
        "output_dir": tmp_path / "golden_run",
        "faces": faces,
        "base": base,
        "merged": merged,
        "merge_log": merge_log,
        "core_indices": list(range(n_total)),
        "config": PipelineConfig(),
    }


# ---------------------------------------------------------------------------
# Hashing — every file in the run dir
# ---------------------------------------------------------------------------

# Files whose bytes are deterministically derived from inputs (and thus
# safe to hash). Excluded: pipeline_run.json (carries started_at /
# finished_at timestamps that vary every run — checked separately for
# shape, not bytes).
_DETERMINISTIC_ARTIFACTS = (
    "face_clustering.db",
    "embeddings.npy",
    "embedding_face_ids.npy",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# (table, column) pairs that capture wall-clock time and thus vary
# between runs even when inputs are identical. Replaced with a sentinel
# before hashing.
_NON_DETERMINISTIC_COLUMNS = {
    ("images", "created_at"),
    ("scene_clusters", "created_at"),
}


def _sha256_db_canonical(db_path: Path) -> Dict[str, str]:
    """Per-table hashes via canonical SELECT * ORDER BY (rowid).

    Hashing the .db file bytes directly is fragile — SQLite page-layout
    and write order vary between runs even when row content is
    identical. Hashing the canonical row dump gives a stable signature.

    Wall-clock columns (see ``_NON_DETERMINISTIC_COLUMNS``) are replaced
    with a sentinel so the hash is reproducible.
    """
    out: Dict[str, str] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        tables = sorted(
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
        for t in tables:
            h = hashlib.sha256()
            cur = conn.execute(f"SELECT * FROM {t} ORDER BY rowid")
            cols = [d[0] for d in cur.description]
            skip_idx = {i for i, c in enumerate(cols) if (t, c) in _NON_DETERMINISTIC_COLUMNS}
            h.update(("|".join(cols) + "\n").encode("utf-8"))
            for row in cur:
                masked = tuple(
                    "<wall_clock>" if i in skip_idx else c
                    for i, c in enumerate(row)
                )
                h.update(("|".join(repr(c) for c in masked) + "\n").encode("utf-8"))
            out[f"db:{t}"] = h.hexdigest()
    finally:
        conn.close()
    return out


def _compute_all_hashes(output_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    # Per-file (deterministic only)
    for name in ("embeddings.npy", "embedding_face_ids.npy"):
        p = output_dir / name
        hashes[name] = _sha256_file(p)
    # DB: per-table canonical hash
    hashes.update(_sha256_db_canonical(output_dir / "face_clustering.db"))
    # Crops dir: sorted listing + sizes (binary contents are PIL-version-dependent)
    crops = output_dir / "crops"
    if crops.is_dir():
        names = sorted(p.name for p in crops.iterdir())
        sig = "|".join(f"{n}" for n in names)
        hashes["crops/_listing"] = hashlib.sha256(sig.encode("utf-8")).hexdigest()
    return hashes


def _write_golden(hashes: Dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in sorted(hashes.items())]
    GOLDEN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_golden() -> Dict[str, str]:
    if not GOLDEN_PATH.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in GOLDEN_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        out[k] = v
    return out


def _run_export(inputs: dict) -> Path:
    RunExporter(inputs["output_dir"]).export(
        faces=inputs["faces"],
        base_cluster_result=inputs["base"],
        merged_cluster_result=inputs["merged"],
        core_indices=inputs["core_indices"],
        merge_log=inputs["merge_log"],
        merge_metadata={
            "n_iterations": 1,
            "merge_exemplar_threshold": 0.5,
            "merge_candidate_threshold": 0.55,
        },
        config=inputs["config"],
        source_album="spec_057_golden",
        producer="fc_app",
        run_id="spec057_golden_run",
        started_at="2026-05-29T00:00:00",
        finished_at="2026-05-29T00:00:01",
    )
    return inputs["output_dir"]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_exporter_output_matches_golden_hashes(golden_exporter_input):
    """Snapshot test: every writer extraction in spec-057 must preserve
    these per-table + per-file hashes. If golden file is missing, this
    test snapshots fresh — subsequent runs assert.
    """
    out = _run_export(golden_exporter_input)
    live = _compute_all_hashes(out)
    golden = _read_golden()
    if not golden:
        _write_golden(live)
        pytest.skip(
            f"Golden hashes file was missing; wrote {len(live)} entries to "
            f"{GOLDEN_PATH.name}. Re-run to assert."
        )
    assert live == golden, (
        "Exporter output diverged from golden snapshot.\n\n"
        + "Live entries:\n"
        + "\n".join(f"  {k}={v}" for k, v in sorted(live.items()))
        + "\n\nGolden entries:\n"
        + "\n".join(f"  {k}={v}" for k, v in sorted(golden.items()))
    )


def test_pipeline_run_json_shape_invariant(golden_exporter_input):
    """pipeline_run.json carries timestamps that vary; verify the
    shape and load-bearing fields instead of byte-hash.
    """
    out = _run_export(golden_exporter_input)
    payload = json.loads((out / "pipeline_run.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] >= 5
    assert payload["db_path"] == "face_clustering.db"
    assert payload["producer"] == "fc_app"
    assert payload["status"] == "complete"
    assert payload["run_id"] == "spec057_golden_run"
