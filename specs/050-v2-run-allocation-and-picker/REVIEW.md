# Code Review: spec-050 — v2 App Run Allocation + Picker

**Reviewed**: 2026-05-28
**Reviewer**: Claude (self-review against `docs/guides/CODE_REVIEW_CHECKLIST.md`)
**Status**: PASS — 0 blockers, 0 follow-ups

---

## Part 1 — How it works

### Module inventory

| Module | New / Changed | Role |
|---|---|---|
| `face_cluster/run_layout.py` | NEW (45 LOC) | `allocate_run_dir(base, album) -> (run_dir, run_id)`. Fresh `<base>/<uuid4-hex>/` per run. |
| `app/face_clustering_v2/components/run_picker.py` | NEW (~110 LOC) | `RunPickerEntry` dataclass + `render_run_picker()` reading the 20 newest v2 rows from `action_log`. |
| `app/face_clustering_v2/tabs/run_tab.py` | CHANGED | Adds required Album input; allocates dir; writes `v2_last_run_dir` BEFORE pipeline runs. |
| `app/face_clustering_v2/pipeline.py` | CHANGED (signature) | New params: `run_dir`, `run_id`, `album`. Removed `output_dir`. Album persists to `action_log.source_album`. |
| `app/face_clustering_v2/tabs/clusters_tab.py` | REWRITTEN | Uses real `RunStore` API (`clusters("latest")`, `faces()`, `crop_path()`). Picker + Advanced override. |
| `scripts/run_v2.py` | CHANGED | Required `--album`; generates `run_id` internally. |
| 3 existing test files | CHANGED | Updated to new pipeline signature; monkeypatch moved to `_paths.default_db_path` (spec-048 churn). |
| 4 NEW test files | NEW | 12 new tests covering allocator, pipeline wiring, picker, end-to-end AppTest. |

### Data flow

```
Run tab
  → allocate_run_dir(base, album)         # returns (<base>/<uuid>/, uuid)
  → st.session_state.v2_last_run_dir = run_dir   # set BEFORE pipeline
  → run_v2_pipeline(src, run_dir, run_id=uuid, album=...)
      → RunHistoryRepository().start_action("fc_app_v2_run", payload={
           "run_id": uuid, "output_dir": str(run_dir),
           "source_album": album, "producer": "fc_app_v2", ...
        })
      → ... pipeline ...
      → complete_action(action_id, result_fields={n_faces, n_clusters, n_noise})

Clusters tab
  → render_run_picker(label="Pick a run from history")
      → RunHistoryRepository().find(RunHistoryCriteria(producer="fc_app_v2", limit=20))
      → format: "{started} · {album} · {n_faces}f/{n_clusters}c · {status} ({run_id[:8]})"
      → default-select the entry whose output_dir matches session.v2_last_run_dir
  → user picks; advanced override wins if non-empty
  → RunStore(run_dir) → clusters("latest"), faces(), crop_path(face_id)
  → one expander per cluster, sorted by face count desc
```

---

## Part 2 — Findings by checklist section

### §1 Structure — **PASS**
- `run_layout.py` 45 LOC, single function. `run_picker.py` 110 LOC, single component. `clusters_tab.py` 115 LOC (was 72; growth is one helper + per-face guards). No file > 200 LOC.
- Pipeline signature change is non-trivial but mechanical; all 3 callers updated.

### §2 Code quality — **PASS**
- No new try/except suppressing errors. The Clusters tab wraps `RunStore` construction in try/except that surfaces a *user-readable* `st.error` with the exception type — surface message, never silence.
- No silent defaults at boundaries — Album is required at the UI; pipeline signature requires `run_id` and `album` (no defaults).
- Comments answer "why" (the spec-048 monkeypatch move, the picker's default-selection rule, the early-session-state write rationale).

### §3 Naming and package structure — **PASS**
- `allocate_run_dir` describes what it does. `RunPickerEntry` is the data carrier; `render_run_picker` is the renderer — symmetric naming.
- `__all__` declared on both new modules.
- v2 `components/` directory grows from 5 to 6 files — still flat, no cluster-of-cluster-of-files.

### §4 Layering and coupling — **PASS**
- Picker depends on `face_cluster.repositories.RunHistoryRepository`. clusters_tab depends on `RunStore` + the picker component. No reverse imports.
- Single writer per piece of state: the Run tab is the sole writer to `v2_last_run_dir`, `v2_last_run_id`, `v2_album`. The Clusters tab reads but never writes them.

### §5 Testability — **PASS** (this is the section that mattered most this spec)

Test inventory:

| Kind | Count | Files |
|---|---|---|
| Unit (allocator) | 4 | `test_run_layout.py` |
| Unit (picker data layer) | 4 | `test_v2_run_picker.py` |
| Integration (pipeline + action_log) | 1 | `test_v2_pipeline_run_allocation.py` |
| AppTest (Clusters tab E2E) | 3 | `test_v2_run_picker_e2e.py` |
| **Total new** | **12** | |

Failure-mode walkthrough:

| Class of regression | Test that catches it |
|---|---|
| A `RunStore` method gets renamed and the tab calls the old name | `test_clusters_tab_renders_seeded_run_without_error` (the tab actually imports + calls RunStore methods through the stub) |
| Album silently defaults to `src_dir.name` instead of the typed value | `test_pipeline_records_caller_supplied_run_id_album_output_dir` — asserts `source_album == "phase2_test"` exactly |
| `run_id` reverts to a timestamp instead of the UUID | Same test — `assert row["run_id"] == run_dir.name` (UUID-shaped) |
| Picker shows non-v2 producers | `test_entries_filters_to_v2_producer` |
| Picker ordering regresses (oldest first) | `test_entries_ordered_newest_first` |
| Run tab allocates the same dir twice | `test_returns_unique_paths` (10 successive calls, all distinct) |

Mock usage: only `_StubRunStore` in the E2E test, justified per-test as "RunStore behaviour is covered by `test_run_store.py`; this test is about the tab's wiring".

### §6 Boundary contracts — **PASS**
- `run_v2_pipeline` signature now declares `run_id` and `album` as required keyword-only — typed boundary inputs, not floating dict keys.
- `RunPickerEntry` is `@dataclass(frozen=True, slots=True)` — immutable, typed.
- No new Pydantic models; no Pandera schemas changed.
- Config knob → producer check: the picker filters by `producer='fc_app_v2'`; the v2 pipeline writes `producer='fc_app_v2'` to action_log. The two strings are connected by the producer tag itself — adding an arch test for this would be over-engineering for a single-string contract.

### §7 Documentation — **PASS**
- `spec.md` — present, tight.
- `tasks.md` — present, tight, 7 phases.
- `REVIEW.md` — this file.
- `CHANGES_LOG.md` — entry filed under `[FEATURE]`.
- `SIGHTINGS.md` — SIGHTING-075 filed and immediately marked RESOLVED with the spec-050 cross-reference.
- `docs/architecture/` — no class/schema/pipeline-step changes that would require an architecture HTML update.

### §8 Risk register — **PASS**
- **Backwards compat for `runs/v2_latest/`**: pre-existing runs in that fixed dir are not migrated, but they remain loadable via the Advanced override. Documented as a non-issue in spec.md §"What we don't build."
- **Picker latency**: `RunHistoryRepository().find(...)` hits the `idx_action_log_type` index; spec-048 made `ensure_schema` ~1ms; total call ~5ms. No `st.cache_data` wrapper needed.
- **Failed runs in picker**: label includes `status`; clusters_tab's try/except around `RunStore` construction emits a user-readable error and returns early, rather than tracebacking.
- **UUID collisions**: negligible (uuid4).

---

## Part 3 — Verdict

**Accept.** Spec-050 ships with:

- All 9 acceptance criteria met (AC1–AC9 verified by the new test files + manual smoke during Phase 6).
- All 3 reported symptoms (SIGHTING-075) resolved.
- 12 new tests; coverage gap that allowed the bugs to ship is closed by `test_v2_run_picker_e2e` specifically.
- 2 incidental fixes carried in: the spec-048 monkeypatch move (`run_history_db.get_db_path` → `_paths.default_db_path`) in 3 pre-existing test files. These were broken on `main` before this spec; now fixed.
- Zero blocker findings, zero follow-ups.

The wider test suite still reports pre-existing failures unrelated to spec-050 (clustering equivalence partially recovered via the parallel spec-040 Phase 6 BaseStep.validate fix; merge-stage v4 round-trip and adaptive-threshold removal remain — covered by spec-049's tracking).
