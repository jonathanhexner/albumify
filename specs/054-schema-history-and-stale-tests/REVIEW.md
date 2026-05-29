# Code Review: spec-054 — Schema history + 2 stale tests

**Reviewed**: 2026-05-29
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS — 0 blockers

---

## Part 1 — How it works

### Module inventory

| Module | New / Changed | Role |
|---|---|---|
| `face_cluster/db/schema.py` | CHANGED (+25 LOC) | `SCHEMA_HISTORY: dict[int, str]` documenting v3, v4, v5. `SCHEMA_VERSION` derived from `max(SCHEMA_HISTORY)`. |
| `face_cluster/db/__init__.py` | CHANGED (2 lines) | Re-export `SCHEMA_HISTORY`. |
| `tests/architecture/test_schema_history.py` | NEW (45 LOC) | 4 arch tests: version-in-history, version-equals-max, keys-contiguous, descriptions-non-empty. Forces next bump to update history. |
| `tests/face_clustering/test_merge_stage.py` | CHANGED (1 assertion) | `assert meta.schema_version == 4` → `assert meta.schema_version == SCHEMA_VERSION`. |
| `tests/face_clustering/test_merge.py` | CHANGED (delete 1 test) | `test_adaptive_threshold_fields_removed` deleted — premise was false (fields are live in production). Replaced with 8-line comment explaining the decision. |

---

## Part 2 — Findings by checklist section

### §1 Structure — PASS
- `face_cluster/db/schema.py` grew 25 LOC for one cohesive addition (history dict). Total file size still under 100 LOC.
- New arch test file 45 LOC, single responsibility.

### §2 Code quality — PASS
- No try/except. No silent defaults. Comments explain *why* (every change includes a "spec-054" tag pointing back here).
- The deleted test's intent is preserved in a code comment + the spec — future readers can reconstruct what was removed and why.

### §3 Naming and package structure — PASS
- `SCHEMA_HISTORY` mirrors the established `SCHEMA_VERSION` / `SCHEMA_DDL` naming pattern in the same file.
- `__all__` updated in `face_cluster/db/__init__.py`.

### §4 Layering and coupling — PASS
- `test_merge_stage.py` now imports `SCHEMA_VERSION` (was hardcoded literal). Stronger contract: the test follows the code automatically.

### §5 Testability — PASS

Test inventory (spec-054 deltas):

| Kind | Count | Notes |
|---|---|---|
| Arch (schema history guards) | 4 | All in `test_schema_history.py` |
| Existing fixed | 1 | `test_v4_full_round_trip_real_images` now green |
| Existing deleted | 1 | `test_adaptive_threshold_fields_removed` removed |
| **Net** | **+4 / -1 = +3** | |

Failure-mode walkthrough:

| Class of regression | Test that catches it |
|---|---|
| Bump SCHEMA_VERSION without documenting | `test_schema_version_is_in_history` |
| Add a `SCHEMA_HISTORY[7]` entry but forget to bump `SCHEMA_VERSION` | `test_schema_version_equals_max_history_key` |
| Skip a version (e.g. jump from 5 to 7) | `test_schema_history_keys_are_contiguous` |
| Add an entry with empty description | `test_every_history_entry_has_a_non_empty_description` |
| Round-trip writes wrong version | `test_v4_full_round_trip_real_images` (now version-aware) |

### §6 Boundary contracts — PASS
- `SCHEMA_VERSION` keeps same name + same runtime value (5). All existing imports unaffected.
- New `SCHEMA_HISTORY` is additive; no consumer relied on its absence.

### §7 Documentation — PASS
- `spec.md` + `tasks.md` + `REVIEW.md` present.
- `CHANGES_LOG.md` entry queued for Phase 5.
- SIGHTING-072 + SIGHTING-073 close-outs queued.
- The `SCHEMA_HISTORY` dict IS the documentation — single source of truth.

### §8 Risk register — PASS
- The 5 production fields the deleted test asserted shouldn't exist (`merge_threshold_alpha`, etc.) are now confirmed live. No dead-code concern.
- The arch test forces the next contributor who bumps the schema to also update `SCHEMA_HISTORY`. If they don't, the test breaks at PR time.

---

## Part 3 — Verdict

**Accept.** Spec-054 ships with:
- All 7 acceptance criteria green (AC6 verified by Phase 4 sweep — see below).
- Spec-052's #3 deleted, spec-052's #4 fixed.
- New `SCHEMA_HISTORY` mechanism prevents the silent rot that caused #4 in the first place.
- Zero blockers.

---

## Phase 4 sweep result

`pytest tests/face_clustering tests/architecture`:

| | Pre-spec-054 | Post-spec-054 | Delta |
|---|---|---|---|
| Passed | 728 | **759** | +31 |
| Failed | 5 | **0** | **−5** |
| Skipped | 9 | 10 | +1 |
| Errors | 0 | 0 | 0 |

**Suite is fully green.** Spec-054 closed the last 2 of the 5 spec-052 failures (#3 + #4); the other 3 (#1, #2, #5) were handled by commit 9396418 (deletion + skip). The +31 net passes vs pre-spec-054 baseline includes:
- 4 new arch tests in `test_schema_history.py`
- 1 previously-failing test (#4) now passing
- 0 newly-skipped tests from this spec (the +1 skip is unrelated)

AC6 satisfied.
