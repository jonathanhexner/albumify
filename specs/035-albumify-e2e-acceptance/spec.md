# Feature Specification: Albumify Face-Clustering E2E Acceptance Test

**Created**: 2026-05-15
**Status**: Draft
**Resolves**: FR-033-1 (spec-033 review follow-up); same root cause as SIGHTING-061

## Problem Statement

The spec-033 refactor shipped 40 architecture tests but **zero tests that exercise the full Albumify face-clustering pipeline on real image data**. The consequence: SIGHTING-061 (a regression that rejected 100% of faces and crashed `identity_refinement`) went undetected through every gate. Architecture tests inspect code shape; unit tests construct objects in isolation; synthetic-data tests build fake DBs. None of them simulate "config knob references upstream field nobody computes."

Without an E2E acceptance test on real data, every subsequent change to the face-clustering pipeline ships with the same blind spot.

## User Stories

### US1 — CI catches whole-pipeline regressions (Priority: P1)
As a developer changing any module touched by the face-clustering pipeline, I want CI to run the full Albumify pipeline on a small fixture and assert non-trivial output, so that runtime-semantic bugs (not just shape bugs) are caught before merge.

**Acceptance Criteria**:
- Given a 5-image fixture under `tests/fixtures/face_clustering_e2e/`, when the default Albumify pipeline runs, then `context.people_clusters` contains ≥1 cluster with ≥1 face.
- Given the same fixture, when the pipeline completes, then `RunStore(output_dir).image_detail(path)` for at least one image returns a non-empty `faces` list with `is_core=True` for at least one face.
- The test fails loudly (not by skip / xfail) if any of the above is violated.

### US2 — Test runs in reasonable time locally (Priority: P1)
The fixture is small enough (5 images) that a developer can run it before pushing. CPU-only, no GPU dependency.

**Acceptance Criteria**:
- Local run with `.venv/Scripts/python -m pytest tests/face_clustering/test_albumify_e2e.py` completes in under ~2 minutes on a developer laptop (CPU only). Real measurement on the reference machine: 117s. Most time is in InsightFace SCRFD detection + ArcFace embedding + DINOv2 scene embedding inference.
- Test does not require network access (model weights are cached locally per the existing setup).
- The test does not currently engage `UniversalCacheHandler` — fresh inference every run is intentional (so the test still catches regressions when the cache is empty). A "cached" mode could be added in a follow-up if iteration speed matters.

### US3 — Regression reference for SIGHTING-061 class (Priority: P2)
The test serves as a documented reference for "what would have caught SIGHTING-061." When a future change unblocks a quality gate, this test verifies that some face still passes the gate end-to-end.

**Acceptance Criteria**:
- Test docstring names SIGHTING-061 and explains the regression class.
- Test asserts that at least one face passes the blur, pose, and det_score gates (regardless of which is enabled today).

## Edge Cases

- All faces filtered → test fails with a message naming the gate that rejected.
- Embeddings extraction fails → test fails with a message naming the empty `face_embeddings`.
- Image fixture missing → test errors at setup (not skips); helpful message to regenerate.
- Pipeline takes >60s → CI sees a timeout failure (intentional pressure on test size).

## Requirements (brief)

- FR-001: 5-image fixture under `tests/fixtures/face_clustering_e2e/` with diverse faces (at least 2 distinct people for cluster formation).
- FR-002: Test file at `tests/face_clustering/test_albumify_e2e.py`.
- FR-003: Test invokes the production pipeline registry (`default_pipeline` from `configs/pipeline.yaml`), not a hand-built one.
- FR-004: Test runs CPU-only; model weights resolved from the existing `models/album_app/` cache or downloaded once during setup.
- FR-005: Test asserts on `RunStore.image_detail()` output to also exercise the read side (P-D acceptance).
- FR-006: Test docstring cites SIGHTING-061 and FR-033-1.
- FR-007: Test marked `@pytest.mark.face_e2e` so it can be opted out of in fast loops; included in default CI.

## Non-Goals

- Not a full Playwright UI test (out of scope; covered by a separate E2E flow).
- Not parameterized across pipelines — just the default Albumify pipeline.
- Not asserting on cluster *quality* (precision/recall) — just on non-empty, non-crashing output.

## Open Questions

- [NEEDS CLARIFICATION] Should the fixture images be committed binary, or generated at setup time from a documented Pexels/Unsplash CC0 set?
