# Tasks: v5 read-path audit (061)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

---

## Phase 0 — Scaffold + checklist (~30 min)

- [x] **T001** Create `specs/061-v5-readpath-audit/AUDIT_FINDINGS.md` skeleton with a markdown table header: `| ID | File:line | Category | Pattern | v5-compat? | Status | Notes |`. Pre-fill with the 4 known examples (SIGHTING-078, -079, -080, loader bonus) so the format is concrete from row 1.
- [x] **T002** Create `specs/061-v5-readpath-audit/AUDIT_CHECKLIST.md` — one section per category (the 10 from `spec.md`). Each section: the grep command, what to read for, what counts as a finding, where to file the row. Doubles as the re-runnable script.

**Validation gate (Phase 0)**: both files exist; the checklist's grep commands run from a clean shell without modification.

---

## Phase 1 — Grep the 10 categories (~2 h)

For each category, run the grep, walk the hits, and append rows to `AUDIT_FINDINGS.md`. Time-box: 30 min per category; if you've found 0 hits in 30 min, the category is clean — mark complete and move on.

- [x] **T010** Category 1: hardcoded legacy CSV file references.
  ```
  grep -rn "faces.csv\|clusters.csv\|embeddings.npy" app/face_clustering_v2/ face_cluster/views/ face_cluster/loader.py
  ```
  Skip: error messages that *describe* the legacy layout (those are correct after SIGHTING-080 fix).
  Find: anything *checking for* or *parsing* these files.

- [x] **T011** Category 2: `RunStore.<method>("final")` calls.
  ```
  grep -rn 'RunStore.*"final"\|\.clusters("final"\|\.iteration_count' app/face_clustering_v2/ face_cluster/views/ face_cluster/repositories/
  ```
  For each: confirm the caller doesn't depend on RunStore's broken `_resolve_iteration("final")`. If it does, the caller should resolve locally (like spec-045's Repository now does).

- [x] **T012** Category 3: raw `MAX(iteration) FROM merge_decisions` or `merge_decisions` joined with `clusters`.
  ```
  grep -rn "merge_decisions\|MAX(iteration)" face_cluster/ app/
  ```

- [x] **T013** Category 4: `AsyncHandle` / `threading.Thread` patterns inside the v2 tab/component layer.
  ```
  grep -rn "AsyncHandle\|threading\.Thread" app/face_clustering_v2/
  ```
  Sighting-079 confirmed AsyncHandle in UI = stuck-spinner. Any new uses are suspect until proven otherwise.

- [x] **T014** Category 5: `_REQUIRED`, `_ARTIFACTS`, `_COLUMNS`, `_EXPECTED` constants in v2 modules.
  ```
  grep -rnE "_(REQUIRED|ARTIFACTS|COLUMNS|EXPECTED)[ :=]" app/face_clustering_v2/ face_cluster/views/
  ```
  Skip: UI-only constants (display columns, label maps). Find: anything claiming a schema/file shape.

- [x] **T015** Category 6: producer-name string equality checks.
  ```
  grep -rnE '== ?"(fc_app|albumify|fc_app_v2)"' app/face_clustering_v2/ face_cluster/views/
  ```
  v2 producers include `fc_app_v2`. Anywhere checking only the old names misses v2 runs.

- [x] **T016** Category 7: direct `sqlite3.connect` on per-run DBs without `PRAGMA user_version` check.
  ```
  grep -rn "sqlite3.connect" app/face_clustering_v2/ face_cluster/views/ face_cluster/repositories/cluster_analysis_repo.py
  ```
  Skip: connections via `RunStore` (it does the check). Find: anything opening `face_clustering.db` directly.

- [x] **T017** Category 8: `pipeline_run.json` field reads that assume v4 structure.
  ```
  grep -rn "pipeline_run.json\|pipeline_run\[" face_cluster/views/ app/face_clustering_v2/
  ```
  Confirm each read matches what the v5 RunExporter actually writes.

- [x] **T018** Category 9: hardcoded run-dir layout (e.g., `_v4/` subdir, `/crops/` assumption).
  ```
  grep -rnE '"_v4"|"crops"' face_cluster/ app/face_clustering_v2/
  ```
  v5 keeps `crops/` but `_v4/` is transitional and rare in practice.

- [x] **T019** Category 10: `os.listdir` / `glob` on run dirs.
  ```
  grep -rn "listdir\|\.glob(" face_cluster/views/ app/face_clustering_v2/
  ```
  Look for assumptions about file count / extension presence that the v5 layout breaks.

**Validation gate (Phase 1)**: `AUDIT_FINDINGS.md` has at least 1 row per non-empty category; "0 findings" categories are explicitly marked complete in the checklist.

---

## Phase 2 — Triage (~30 min)

- [x] **T020** For each finding row, fill the **v5-compat?** column: `OK` (works under v5) / `BROKEN` (will crash or misbehave) / `RISKY` (correct today but easy to regress).
- [x] **T021** For each non-OK row, fill **Status** with one of:
  - `NO-OP` — risky but currently correct; add a code comment + drift-guard test
  - `FIX` — broken; fix lands in this spec
  - `SIGHTING` — broken but out of scope (e.g., needs a redesign); file a numbered sighting
  - `DEFERRED` — known issue, intentionally not addressed now; reason in Notes
- [x] **T022** Sort by Status (FIX first, then SIGHTING, then NO-OP/DEFERRED) so subsequent phases can walk the file top-down.

**Validation gate (Phase 2)**: every row has Status filled; no row is left as `?`.

---

## Phase 3 — Land the FIX rows (~1-2 h, depends on Phase 1 yield)

For each FIX-status finding, in `AUDIT_FINDINGS.md` order:

- [x] **T030** Apply the smallest possible fix. Match the existing pattern in nearby code (don't introduce a new abstraction).
- [x] **T031** Add a regression test — at minimum a unit test on the contract that the fix restores. Prefer an AppTest case in `test_v2_app_smoke.py` when the bug surfaces only in the Streamlit lifecycle.
- [x] **T032** Update the finding row to `Status: FIXED` and link the commit hash in Notes.
- [x] **T033** Run the full gate after each fix: `pytest tests/face_clustering/ tests/architecture/ -q`. Never red for > 1 commit.

**Validation gate (Phase 3)**: every FIX row has a commit hash + a regression test; full suite still green.

---

## Phase 4 — File SIGHTING rows (~30 min)

- [x] **T040** For each SIGHTING-status finding, append a numbered sighting to `docs/project/SIGHTINGS.md` using the standard format. Cross-reference the audit row by ID.
- [x] **T041** Update the finding row's Notes column with the sighting number.

**Validation gate (Phase 4)**: every SIGHTING row points at an entry in `SIGHTINGS.md`.

---

## Phase 5 — Close-out (~30 min)

- [x] **T050** Append CHANGES_LOG entry summarizing: N findings, M fixed in this spec, K filed as sightings, L marked no-op.
- [x] **T051** Add a one-paragraph note to `CLAUDE.md` §"Delivery Quality" pointing at `AUDIT_CHECKLIST.md` as the post-schema-change recheck.
- [x] **T052** `/code-review` → `specs/061-v5-readpath-audit/REVIEW.md`. Walk the 8-section checklist.
- [x] **T053** Flip spec status `Draft` → `Code Review` → `Implemented`.
- [x] **T054** Commit + push.

**Final gate**: spec status = Implemented; `AUDIT_FINDINGS.md` table fully populated; full suite green; CLAUDE.md updated with the re-run pointer.

---

## Total estimate

**~4-6 hours.** Yield is variable — could be 5 findings (week was lucky), could be 25 (the codebase has been accumulating). Time-boxing per category keeps the walk bounded regardless.
