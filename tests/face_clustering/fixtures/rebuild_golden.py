"""Rebuild tests/face_clustering/fixtures/golden_run_history.db.

Seeds a deterministic 24-row action_log DB via the new SQLAlchemy
``RunHistoryRepository``. This is the read+mutation oracle for
every repo-level equivalence test.

spec-048: ported off the legacy free functions in
``face_cluster.run_history_db`` (scheduled for deletion 2026-06-08).
Using the new Repository here means the script keeps working after
the burn-in window.

The fixture covers every code path the equivalence test exercises:
- statuses: running, complete, failed
- NULL hot fields (start with empty payload)
- unicode comment, multiline comment, SQL-special characters
- parent/child run_id chain (parent_run_id linkage)
- multiple action_types (face_cluster_run, recluster, merge_apply,
  profile_save, ml_training)
- populated and missing source_album
- multiple producers (albumify, fc_app_v2, fc_app)

Usage:
    .venv/Scripts/python -m tests.face_clustering.fixtures.rebuild_golden
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# The new Repository emits no deprecation warnings, but importing
# face_cluster.* still triggers the legacy module-level warnings.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from face_cluster.repositories.run_history_repo import (
    RunHistoryRepoConfig,
    RunHistoryRepository,
)


FIXTURE_DIR = Path(__file__).resolve().parent
FIXTURE_DB = FIXTURE_DIR / "golden_run_history.db"


def _seed(repo: RunHistoryRepository) -> None:
    """Seed 24 rows hitting every code path."""

    # 1-3: complete runs on the same source_album with different producers
    for i, (producer, n_faces) in enumerate(
        [("albumify", 120), ("fc_app_v2", 95), ("fc_app", 110)], 1
    ):
        aid = repo.start_action(
            "face_cluster_run",
            payload={
                "source_album": "album_alpha",
                "run_id": f"run_{i:03d}",
                "source_dir": "D:/test_data/album_alpha",
                "output_dir": f"D:/clustering_output/run_{i:03d}",
                "producer": producer,
                "run_name": f"alpha run {i}",
                "run_kind": "initial",
                "config_json": "{\"min_cluster_size\":5}",
            },
        )
        repo.complete_action(
            aid,
            result_fields={
                "n_faces": n_faces,
                "n_clusters": 6 + i,
                "n_noise": 2,
                "n_core": 30 + i * 5,
                "log_file": f"run_{i:03d}.log",
            },
        )

    # 4-5: completed runs on album_beta, one with unicode comment
    for i, comment in enumerate(
        [
            "good run, kept exemplars",
            "test échec — ünicode pør checking",
        ],
        4,
    ):
        aid = repo.start_action(
            "face_cluster_run",
            payload={
                "source_album": "album_beta",
                "run_id": f"run_{i:03d}",
                "source_dir": "D:/test_data/album_beta",
                "output_dir": f"D:/clustering_output/run_{i:03d}",
                "producer": "fc_app_v2",
                "run_name": f"beta run {i}",
            },
        )
        repo.complete_action(
            aid,
            result_fields={"n_faces": 80, "n_clusters": 4, "n_noise": 1, "n_core": 28},
        )
        repo.update_comment(aid, comment)

    # 6: failed run
    aid = repo.start_action(
        "face_cluster_run",
        payload={
            "source_album": "album_beta",
            "run_id": "run_006",
            "producer": "fc_app_v2",
        },
    )
    repo.fail_action(aid, "ValueError: insufficient samples for clustering")

    # 7: running (left open)
    repo.start_action(
        "face_cluster_run",
        payload={
            "source_album": "album_gamma",
            "run_id": "run_007",
            "producer": "fc_app_v2",
        },
    )

    # 8: action with NULL hot fields (empty payload)
    aid = repo.start_action("profile_save", payload={})
    repo.complete_action(aid)

    # 9-11: recluster chain — parent_run_id linking to run #1
    for i, (n_clusters, comment) in enumerate(
        [(7, "first recluster"), (8, None), (9, "third pass")], 9
    ):
        aid = repo.start_action(
            "recluster",
            payload={
                "source_album": "album_alpha",
                "run_id": f"recl_{i:03d}",
                "parent_run_id": 1,
                "producer": "fc_app_v2",
                "run_kind": "recluster",
            },
        )
        repo.complete_action(
            aid,
            result_fields={"n_clusters": n_clusters, "n_faces": 120, "n_core": 50},
        )
        if comment is not None:
            repo.update_comment(aid, comment)

    # 12-14: merge_apply actions
    for _ in range(12, 15):
        aid = repo.start_action(
            "merge_apply",
            payload={
                "source_album": "album_alpha",
                "parent_run_id": 1,
                "producer": "fc_app_v2",
            },
        )
        repo.complete_action(aid)

    # 15-16: rows with missing source_album (older legacy data shape)
    for i in range(15, 17):
        aid = repo.start_action(
            "ml_training",
            payload={"run_id": f"ml_{i:03d}"},
        )
        repo.complete_action(aid, result_fields={"n_faces": 50})

    # 17-20: more face_cluster_runs across diverse albums for filter coverage
    for i in range(17, 21):
        aid = repo.start_action(
            "face_cluster_run",
            payload={
                "source_album": f"album_delta_{i}",
                "run_id": f"run_{i:03d}",
                "producer": "albumify",
            },
        )
        repo.complete_action(
            aid,
            result_fields={"n_faces": 40 + i, "n_clusters": 3, "n_noise": 0, "n_core": 15},
        )

    # 21: row with multiline comment + special chars (SQL-injection-ish)
    aid = repo.start_action(
        "face_cluster_run",
        payload={"source_album": "album_epsilon", "run_id": "run_021", "producer": "fc_app_v2"},
    )
    repo.complete_action(aid, result_fields={"n_faces": 100, "n_clusters": 5})
    repo.update_comment(aid, "line1\nline2\n'quote' \"double\" \\backslash")

    # 22-24: more failures with different error texts
    for i, err in enumerate(
        [
            "TimeoutError: insightface load",
            "FileNotFoundError: missing config.yaml",
            "RuntimeError: cuda OOM",
        ],
        22,
    ):
        aid = repo.start_action(
            "face_cluster_run",
            payload={"source_album": f"album_zeta_{i}", "producer": "fc_app_v2"},
        )
        repo.fail_action(aid, err)


def rebuild() -> None:
    if FIXTURE_DB.exists():
        FIXTURE_DB.unlink()
    FIXTURE_DB.parent.mkdir(parents=True, exist_ok=True)

    repo = RunHistoryRepository(RunHistoryRepoConfig(db_path=FIXTURE_DB))
    _seed(repo)

    with sqlite3.connect(str(FIXTURE_DB)) as conn:
        n = conn.execute("SELECT COUNT(*) FROM action_log").fetchone()[0]
        print(f"Rebuilt {FIXTURE_DB} with {n} rows.")


if __name__ == "__main__":
    rebuild()
    sys.exit(0)
