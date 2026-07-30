"""spec-048 Phase 2 — repo construction stays cheap after _engine_cache removal.

The cache existed only to amortize the subprocess(alembic.exe) cost in
spec-046's _ensure_schema. Now that ensure_schema is in-process and
fast, construction does not need a global cache. This test sets a
loose ceiling so a future regression (e.g., heavy work in __init__)
is caught.
"""
from __future__ import annotations

import time

from face_cluster.repositories.run_history_repo import (
    RunHistoryRepoConfig,
    RunHistoryRepository,
)


def test_repository_construction_is_fast(tmp_path):
    db = tmp_path / "perf.db"
    cfg = RunHistoryRepoConfig(db_path=db)

    # First construction includes schema upgrade — allow more headroom.
    t0 = time.perf_counter()
    repo = RunHistoryRepository(cfg)
    first_ms = (time.perf_counter() - t0) * 1000
    del repo
    assert first_ms < 2000, f"First construction took {first_ms:.0f} ms (limit: 2000 ms)"

    # Subsequent constructions hit the already-upgraded DB.
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        repo = RunHistoryRepository(cfg)
        samples.append((time.perf_counter() - t0) * 1000)
        del repo
    mean_ms = sum(samples) / len(samples)
    assert mean_ms < 200, (
        f"Mean subsequent construction took {mean_ms:.0f} ms over 20 iterations "
        f"(limit: 200 ms). Samples: {[round(s, 1) for s in samples]}"
    )
