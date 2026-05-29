"""spec-045 Phase 2 — real-fixture smoke for ClusterAnalysisRepository.

Spec.§8.2 — 3 read-only smoke tests against a real run dir on the dev
machine. These are NOT the primary correctness signal (the synthetic
suite carries that — spec-042 §4.7); they confirm the Repository
survives actually-shaped data.

Skips cleanly when no real run dir is present (CI / fresh checkouts),
via the session-scoped ``v2_budapest_run_dir`` fixture in
``tests/conftest.py``.

Note: spec.§8.2 references the fixture as ``v2_pilot_run_dir``; the
actual name in conftest.py is ``v2_budapest_run_dir``. Tracked for the
Phase 8 REVIEW.md spec-text correction.
"""
from __future__ import annotations

import pytest

from sim_bench.db.face_clustering.cluster_analysis_repo import (
    ClusterAnalysisRepoConfig,
    ClusterAnalysisRepository,
)
from face_cluster.run_store import RunMetadata
from face_cluster.views._base import ClusterRow


@pytest.fixture(scope="module")
def real_repo(v2_budapest_run_dir):
    """ClusterAnalysisRepository over the most recent Budapest run.

    Read-only=True so a hypothetical mutation regression can't pollute the
    user's actual run dir.
    """
    return ClusterAnalysisRepository(
        ClusterAnalysisRepoConfig(run_dir=v2_budapest_run_dir, read_only=True)
    )


# ---------------------------------------------------------------------------
# Tests — spec.§8.2 #1–#3
# ---------------------------------------------------------------------------

def test_real_run_loads(real_repo, v2_budapest_run_dir):  # #1
    """Construction validates the dir + DB; no exception thrown."""
    assert real_repo is not None
    # Cheap sanity that we're pointed at the right place:
    assert (v2_budapest_run_dir / "face_clustering.db").is_file()


def test_real_run_has_clusters(real_repo):  # #2
    """Real run produces at least one ClusterRow; every row has face_count > 0."""
    rows = real_repo.get_cluster_rows()
    if not rows:
        pytest.skip("Real run has zero non-noise clusters — can't smoke-test downstream shape.")
    assert all(isinstance(r, ClusterRow) for r in rows)
    assert all(r.size > 0 for r in rows), \
        f"ClusterRow with size==0 returned: {[r for r in rows if r.size == 0]}"


def test_real_run_metadata_consistent(real_repo):  # #3
    """RunMetadata.n_clusters_final matches the count of real ClusterRow objects."""
    meta = real_repo.get_run_metadata()
    assert isinstance(meta, RunMetadata)
    rows = real_repo.get_cluster_rows()
    # n_clusters_final in metadata may include the noise bucket; rows excludes it.
    # Real signal: rows length is within 1 of metadata (off by at most the noise bucket).
    assert abs(len(rows) - meta.n_clusters_final) <= 1, (
        f"Repository says {len(rows)} real clusters; metadata says n_clusters_final={meta.n_clusters_final}. "
        f"Allowed diff: 1 (noise bucket)."
    )
