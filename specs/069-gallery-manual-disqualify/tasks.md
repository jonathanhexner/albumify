# Tasks — Manual face disqualification (069)

> Blocked on spec-066 landing (Gallery surfaces the button + the `⚠` flag) and on
> Open decisions Q1–Q4 in `spec.md`. Status: **Draft** — do not start until promoted.

## Phase 0 — Resolve open decisions (~15 min)
- [ ] T000 Confirm Q1 (mark-only vs auto-recluster), Q2 (bulk), Q3 (dim vs hide), Q4 (button location).

## Phase 1 — Storage + contract (~1.5 h)
- [ ] T001 Alembic migration: `face_exclusions` table in per-run `face_clustering.db`.
- [ ] T002 ORM model + Pandera schema; update `docs/architecture/db_schemas.html`.
- [ ] T003 Repository methods: `add_exclusion / remove_exclusion / list_exclusions` on the
      per-run repo (extends the spec-045 `ClusterAnalysisRepository` or a sibling).
- [ ] T004 Arch tests stay green: `test_orm_models_in_sync_with_alembic.py`, `test_pandera_schemas.py`.

## Phase 2 — Service (~2 h)
- [ ] T010 NEW `face_cluster/views/disqualify.py`: `DisqualificationService` + `FaceExclusion` +
      `ExclusionResult` dataclasses. Methods: `disqualify / restore / list_exclusions / suggest`.
- [ ] T011 `suggest()` reads blur/outlier threshold from run metadata (no UI literal — spec-053).
- [ ] T012 Audit: each mutation writes an `action_log` row (`action_type=face_disqualify`).
- [ ] T013 NEW `test_disqualify_service_synthetic.py` (≥6 cases incl. soft-delete + restore idempotency).

## Phase 3 — Downstream honoring (~1.5 h)
- [ ] T020 Recluster step `Inputs` assembly filters `active=1` exclusions before kNN graph.
- [ ] T021 Export drops excluded faces.
- [ ] T022 Integration test on the **labeled golden set** (`tests.conftest.get_test_data_dir()`):
      disqualify → recluster → face absent (AC3) **and** purity/completeness hold (AC7),
      reusing `test_pipeline_e2e.py`'s `test_cluster_purity` / `test_cluster_completeness` helpers.

## Phase 4 — UI wiring (~1 h)
- [ ] T030 `cluster_strip` (Gallery): per-face disqualify/restore button; dim excluded (Q3/Q4).
- [ ] T031 `face_detail_panel` (Face Analysis): disqualify/restore for the single face.
- [ ] T032 Optional "disqualify all suggested" bulk button (Q2).
- [ ] T033 Tabs stay ≤80 LOC, no SQL (arch test).

## Phase 5 — e2e + close-out (~1 h)
- [ ] T040 Budapest `slow` scenario: disqualify a known face_id → recluster → assert gone (AC6).
- [ ] T041 `/code-review` → REVIEW.md; CHANGES_LOG; LEARNINGS if any; status → Implemented.

**Total: ~5-7 h.**
