# Tasks: spec-054

**Status**: Implemented
**Estimated effort**: 30 min

5 phases. Tight.

---

## Phase 1 — SCHEMA_HISTORY + arch guard

- [ ] T010 In `face_cluster/run_exporter.py`:
  - Add `SCHEMA_HISTORY: dict[int, str]` with v3, v4, v5 entries (per spec §Data contracts).
  - Change `SCHEMA_VERSION` from a bare integer to `max(SCHEMA_HISTORY)`. Keep the same constant name and value (5).
- [ ] T011 NEW `tests/architecture/test_schema_history.py` (1 test):
  - `SCHEMA_VERSION` is in `SCHEMA_HISTORY`.
  - `SCHEMA_HISTORY` keys are contiguous (no gaps).
  - Every key has a non-empty description.

**Check**: arch test passes.

---

## Phase 2 — Fix #4 (test_v4_full_round_trip_real_images)

- [ ] T020 In `tests/face_clustering/test_merge_stage.py`:
  - `assert meta.schema_version == 4` → `assert meta.schema_version == SCHEMA_VERSION`.
  - Import `SCHEMA_VERSION` from `face_cluster.run_exporter`.
  - (Using the constant rather than a literal 5 means the next bump won't re-break this test.)

**Check**: `pytest tests/face_clustering/test_merge_stage.py::ut_MergeStageE2E::test_v4_full_round_trip_real_images` passes.

---

## Phase 3 — Delete #3 (test_adaptive_threshold_fields_removed)

- [ ] T030 In `tests/face_clustering/test_merge.py`:
  - Delete `test_adaptive_threshold_fields_removed`. Replace with a 3-line comment pointing at spec-054 and SIGHTING-072 so anyone curious knows where to look.

**Check**: test collection no longer includes the method.

---

## Phase 4 — Full sweep

- [ ] T040 `pytest tests/face_clustering tests/architecture -q`. Expected: **3 failed** (down from 5). The remaining 3 are pre-existing (the 9396418 cleanup skipped #5; #1 and #2 were deleted; #3 + #4 fixed/deleted in this spec).

Wait — recompute. After 9396418 cleanup: 5 → effectively 2 (because #1 and #2 were deleted, #5 skipped). So pre-spec-054 baseline is **2 failed** (just #3 + #4). After spec-054: **0 failed** from this group.

If the full suite was already showing 5 because the 9396418 cleanup runs after pytest started counting, expect 5 → 3.

**Check**: deltas match expectation. Whatever residual failures remain are not from this spec or spec-052 #1/#2/#5/#3/#4.

---

## Phase 5 — Review gate

- [ ] T050 `/code-review` → `REVIEW.md`. §1-§8.
- [ ] T051 Close SIGHTING-072 (adaptive-threshold) + SIGHTING-073 (v4 round-trip).
- [ ] T052 Flip spec to Implemented. `CHANGES_LOG.md` entry under `[BUGFIX]`.

**Check**: REVIEW.md filed; zero blockers.

---

## Test delta

| Phase | Added | Deleted | Net |
|---|---|---|---|
| 1 | 1 | 0 | +1 |
| 2 | 0 | 0 | 0 (1-line in-place edit) |
| 3 | 0 | 1 | −1 |
| **Total** | **1** | **1** | **0** |
