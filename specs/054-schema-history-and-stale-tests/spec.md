# spec-054 — Schema history + clean up 2 stale tests (spec-052 #3, #4)

**Created**: 2026-05-29
**Status**: Implemented
**Predecessors**: spec-052 (current 5 red tests)
**Linked sightings**: SIGHTING-072, SIGHTING-073

---

## Problem

Two of the 5 failures in spec-052 are 5-minute fixes that became
clear under direct investigation:

**#3 `test_adaptive_threshold_fields_removed`** asserts that 5 fields
do NOT exist on `PipelineConfig` (`merge_threshold_alpha`, `_beta`,
`use_adaptive_merge_threshold`, `merge_exemplar_percentile`,
`merge_global_percentile`). Grep confirms all 5 are read in
production: `face_cluster/analysis.py`, `app/shared/merge_controls.py`,
`face_cluster/config.py`, etc. The test was written ahead of a cleanup
that never finished; the cleanup is the wrong direction. **Delete the
test.**

**#4 `test_v4_full_round_trip_real_images`** asserts
`meta.schema_version == 4`. Production correctly writes version 5
(spec-040 Phase 4 bumped it). The data round-trip itself works fine.
**Fix: 1-line update to `== 5`.**

While doing #4, address the user's correct request: we currently have
**no central record of what each schema version contains**. Add a
`SCHEMA_HISTORY` dict so future bumps document themselves.

---

## What we build

| Module | Role |
|---|---|
| `face_cluster/run_exporter.py` | Add `SCHEMA_HISTORY: dict[int, str]` documenting v3, v4, v5. Derive `SCHEMA_VERSION` from `max(SCHEMA_HISTORY)`. Keep the constant for backward import-compat. |
| `tests/face_clustering/test_merge_stage.py` | Update `assert meta.schema_version == 4` → `== 5`. |
| `tests/face_clustering/test_merge.py` | Delete `test_adaptive_threshold_fields_removed` (stale; fields are live production code). |
| `tests/architecture/test_schema_history.py` | NEW (1 case): `SCHEMA_VERSION` has a `SCHEMA_HISTORY` entry. Forces the next bump to add a description. |

---

## What we don't build

- Per-version fixture DBs for migration testing. Deferred per earlier
  discussion (was option B in my recommendation); revisit when first
  real cross-version migration code is needed.
- Refactor of `ConservativeMerger.merge_threshold_alpha` and friends.
  They're live; leave them alone.

---

## Locked decisions

1. **`SCHEMA_HISTORY` lives in `face_cluster/run_exporter.py`** next to
   `SCHEMA_VERSION` and `SCHEMA_DDL`. Same file, one source of truth.
2. **Format**: `dict[int, str]`, one-line description per version.
   Short and human-readable. No structured metadata.
3. **Arch test enforces** that any future `SCHEMA_VERSION` bump has a
   matching `SCHEMA_HISTORY` entry. The test that breaks on the next
   bump is the prompt to update the history.
4. **Delete `test_adaptive_threshold_fields_removed`** rather than
   move to `tests/architecture/`. The test's premise was that the
   fields would be removed — they weren't, and grep shows they're
   in active production use. Encoding "should remove" in a test when
   the codebase chose not to is just noise.
5. **No change to merge code.** The adaptive-threshold path stays
   exactly as is. This spec doesn't make a product decision about it.

---

## Data contracts

```python
# face_cluster/run_exporter.py

SCHEMA_HISTORY: dict[int, str] = {
    3: "Initial v3 layout: faces, clusters, cluster_assignments, "
       "run_metadata.",
    4: "Added merge_decisions table; run_metadata gained "
       "parent_run_id + iteration counts.",
    5: "Added images table; faces gained area_ratio + "
       "scene_cluster_id; cluster_assignments gained scene linkage.",
}

SCHEMA_VERSION: int = max(SCHEMA_HISTORY)
```

---

## Acceptance criteria

| # | Criterion | How verified |
|---|---|---|
| AC1 | `SCHEMA_HISTORY` exists with entries for v3, v4, v5 | Phase 1 — import test |
| AC2 | `SCHEMA_VERSION == max(SCHEMA_HISTORY)` and equals 5 | Phase 1 — same test |
| AC3 | Arch test catches a `SCHEMA_VERSION` bump that lacks a `SCHEMA_HISTORY` entry | Phase 1 — arch test |
| AC4 | `test_v4_full_round_trip_real_images` passes | Phase 2 |
| AC5 | `test_adaptive_threshold_fields_removed` no longer exists | Phase 3 |
| AC6 | `pytest tests/face_clustering tests/architecture` — **3 failed** instead of 5 (the 2 we touched are fixed/deleted; remaining 0 from this work) | Phase 4 |
| AC7 | `/code-review` REVIEW.md zero high-severity findings | Phase 5 |

---

## Risks

| Risk | Mitigation |
|---|---|
| Renaming `SCHEMA_VERSION` to a derived constant breaks an existing import | None: we KEEP the name and value (5); only its derivation changes. |
| Someone adds a v6 migration without a `SCHEMA_HISTORY` entry | The new arch test fails. |
| The deleted test was the only thing tracking the abandoned cleanup intent | If the team ever does decide to remove the adaptive-threshold code, they'll do it in its own spec at that point. No history is lost — the spec-052 PRD already documents the test's intent. |

---

## Definition of Done

- All 7 ACs green.
- Full suite drops from 5 failures to 3 (per spec-052: 2 left are #3+#4 from this spec; remaining 3 are pre-existing pre-spec-052 issues tracked separately).
- SIGHTING-072 closed; SIGHTING-073 closed.
- REVIEW.md filed.
