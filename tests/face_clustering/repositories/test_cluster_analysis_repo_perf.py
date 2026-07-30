"""spec-059 T001 — performance baseline for ClusterAnalysisRepository.get_cluster_rows.

Records the pre-refactor mean (committed as ``BASELINE_MS``) and gates the
spec-059 SQLAlchemy migration to within 1.2× of it (AC7).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from tests.face_clustering.repositories.test_cluster_analysis_repo_synthetic import (
    _build_synthetic_run_dir,
)


# Recorded 2026-05-30 on the pre-spec-059 implementation (raw sqlite3).
# Mean of 1000 get_cluster_rows() calls against the synthetic fixture (32
# faces, 3 real clusters + noise). Spec-059 AC7: post-migration mean must
# stay within 1.2× this number.
BASELINE_MS: float = 0.954  # measured 2026-05-30 on raw-sqlite3 implementation
SLOW_FACTOR: float = 1.2

_ITERATIONS = 1000


def _mean_ms_for_get_cluster_rows(run_dir: Path) -> float:
    repo = ClusterAnalysisRepository(ClusterAnalysisRepoConfig(run_dir=run_dir))
    # Warm-up — first call pays connection-open cost on every backend.
    repo.get_cluster_rows()
    start = time.perf_counter()
    for _ in range(_ITERATIONS):
        repo.get_cluster_rows()
    elapsed_s = time.perf_counter() - start
    return (elapsed_s / _ITERATIONS) * 1000.0


@pytest.mark.skipif(
    os.environ.get("RUN_PERF_TESTS") != "1",
    reason="perf bench — set RUN_PERF_TESTS=1 to enable",
)
def test_get_cluster_rows_within_baseline(tmp_path: Path) -> None:
    run_dir = _build_synthetic_run_dir(tmp_path)
    mean_ms = _mean_ms_for_get_cluster_rows(run_dir)
    print(f"\nget_cluster_rows() mean over {_ITERATIONS} iterations: {mean_ms:.3f} ms")
    assert mean_ms <= BASELINE_MS * SLOW_FACTOR, (
        f"Performance regression: {mean_ms:.3f} ms > {BASELINE_MS:.3f} * "
        f"{SLOW_FACTOR} = {BASELINE_MS * SLOW_FACTOR:.3f} ms"
    )


if __name__ == "__main__":
    # Hand-run path: `python tests/.../test_cluster_analysis_repo_perf.py`
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        run_dir = _build_synthetic_run_dir(Path(td))
        ms = _mean_ms_for_get_cluster_rows(run_dir)
        print(f"BASELINE_MS = {ms:.3f}")
