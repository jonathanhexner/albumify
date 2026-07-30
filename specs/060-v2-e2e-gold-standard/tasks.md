# Tasks: v2 end-to-end gold-standard gate (060)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

---

## Phase 0 — Fixture wiring + skip semantics (~45 min)

- [ ] **T001** Add `v2_e2e_album_dir` session-scoped fixture to `tests/conftest.py`. Reads `SIM_BENCH_E2E_ALBUM_DIR` env var; resolves to Path; skips with a clear message when unset or path missing. Caches resolution.
- [ ] **T002** Add `v2_e2e_profile_path` session-scoped fixture. Reads `SIM_BENCH_E2E_PROFILE` env var (default `profile_4.json`); resolves under the v2 profiles dir; skips if missing.
- [ ] **T003** Add `v2_e2e_subset` fixture: takes the album dir, returns the first 50 jpgs sorted by name (deterministic). Skips with "need at least 50 jpgs" if the album has fewer.
- [ ] **T004** Create `tests/face_clustering/test_v2_e2e_smoke.py` with a single skipping placeholder test that pulls all 3 fixtures. Confirms the skip story works end-to-end before any real assertions land.

**Validation gate (Phase 0)**:
```
.venv/Scripts/python -m pytest -m slow tests/face_clustering/test_v2_e2e_smoke.py -v --no-header
```
Expected: **1 skipped** when env vars unset; **1 collected** + skip message naming the missing var. When env vars set + fixture exists, the placeholder test passes vacuously.

---

## Phase 1 — Pipeline smoke (~2.5 h)

- [ ] **T010** In `test_v2_e2e_smoke.py`, add `test_v2_pipeline_smoke` (parametrized `@pytest.mark.slow`):
  - Stage the 50-photo subset into a `tmp_path` source dir.
  - Allocate a `tmp_path` output dir (no `~/.sim_bench/runs/` pollution).
  - Load `FCParams` from `v2_e2e_profile_path`.
  - Run the full producer chain (detect_persons → insightface_detect_faces → detect_face_orientation → align_faces → extract_face_embeddings) followed by `FCAppRunner().run(context, step_configs=params.to_step_configs())`.
  - Assert `result.success is True`, `result.error_message is None`.
  - Assert `(output_dir / "face_clustering.db").is_file()`.
  - Assert `result.n_faces > 0`, `result.n_clusters > 0`, `result.n_faces_assigned > 0`.
  - Assert `(output_dir / "pipeline_run.json").is_file()` and parses as JSON with `schema_version == 5`.

- [ ] **T011** Add per-step timing log: print `{step_name}: {ms}` for each step result. Helps diagnose slowness without adding assertions.

- [ ] **T012** Negative-case sanity: temporarily flip the spec-045 resolver fix off; confirm `test_v2_pipeline_smoke` still passes (this test exercises pipeline, not UI). This proves Phase 1 isn't accidentally testing the UI resolver fix.

**Validation gate (Phase 1)**:
```
SIM_BENCH_E2E_ALBUM_DIR=D:\Budapest2025_Google `
SIM_BENCH_E2E_PROFILE=profile_4.json `
.venv/Scripts/python -m pytest -m slow tests/face_clustering/test_v2_e2e_smoke.py::test_v2_pipeline_smoke -v
```
Expected: **1 passed** in ≤ 120s on the dev machine. `output_dir/face_clustering.db` exists; `n_clusters > 0`.

---

## Phase 2 — UI smoke via Streamlit AppTest (~2.5 h)

- [ ] **T020** Add `_seed_session_state(at, run_dir)` helper — programmatically writes `current_run_dir`, `v2_last_run_dir`, `active_run_dir`, and the resolved `pipeline_result` into the AppTest's session state so tabs see a loaded run without going through History → Load Run.

- [ ] **T021** Add `test_v2_ui_smoke_all_tabs` (depends on `test_v2_pipeline_smoke`'s output via a session-scoped fixture that runs it once and yields the run_dir):
  - For each tab in the v2 app (Run, Cluster Analysis, History — plus any tabs spec-042 ships before this lands):
    - `at = AppTest.from_file("app/face_clustering_v2/main.py").run(timeout=30)`
    - Seed session state for a loaded run.
    - Click the tab.
    - Assert `len(at.exception) == 0` — no Streamlit exception nodes on the tab.
    - Assert at least one expected widget renders (per tab): Run → "Run pipeline" button visible; Cluster Analysis → cluster picker has >=1 option; History → run table renders.

- [ ] **T022** Add the spec-045 regression case explicitly: pre-seed `v2_last_run_dir` to a tmp empty dir (no `face_clustering.db`); navigate to Cluster Analysis; assert the friendly empty-state message renders (NOT a Streamlit traceback). This is the exact bug from 2026-05-29.

- [ ] **T023** Document AppTest limitations in a comment block at the top of the file: which widgets don't render, what workarounds we use, when to graduate to Playwright if AppTest can't cover something.

**Validation gate (Phase 2)**:
```
SIM_BENCH_E2E_ALBUM_DIR=D:\Budapest2025_Google `
.venv/Scripts/python -m pytest -m slow tests/face_clustering/test_v2_e2e_smoke.py -v
```
Expected: **3+ passed** (pipeline smoke + UI smoke per tab + spec-045 regression case). Total runtime ≤ 3 min.

---

## Phase 3 — Wire into the workflow (~1 h)

- [ ] **T030** Update `CLAUDE.md` §"Delivery Quality": add subsection **"E2E gate for considerable changes"** defining what counts as a considerable change (changes to `face_cluster/`, `sim_bench/pipeline/`, `app/face_clustering_v2/`, or per-run DB schema) and naming `pytest -m slow tests/face_clustering/test_v2_e2e_smoke.py` as the required pre-handoff check.

- [ ] **T031** Find the spec-implementer agent prompt (likely `.claude/agents/spec-implementer.md` based on commit `686b635`). If editable from this repo, update its Implementation gate section to run the E2E gate before flipping any "considerable change" spec to Implemented. If it's a user-config-level agent, document the manual invocation in CLAUDE.md instead and call out the gap.

- [ ] **T032** Update `specs/049-test-suite-recovery/spec.md` (or its successor) — note the new opt-in test exists, what it covers, and how to invoke it. Closes the "no E2E gate" finding spec-049 surfaced.

- [ ] **T033** Optional: add `.git/hooks/pre-push` template under `scripts/hooks/pre-push.sample` that runs the gate on `unification/spec-040`. Document in CLAUDE.md as opt-in for the maintainer.

**Validation gate (Phase 3)**:
- CLAUDE.md diff includes the new subsection.
- spec-implementer prompt (or CLAUDE.md fallback) names the gate.
- One smoke run end-to-end through the spec-implementer agent (manual): apply a trivial change to `face_cluster/`, ask the agent to ship it, observe it invoke the gate before claiming Implemented.

---

## Phase 4 — Close-out (~30 min)

- [ ] **T040** CHANGES_LOG entry tagged `[TEST]` summarizing the gate + how to invoke.
- [ ] **T041** `/code-review` → `specs/060-v2-e2e-gold-standard/REVIEW.md`. Address any high-severity findings.
- [ ] **T042** Flip status `Draft` → `Code Review` → `Implemented`.
- [ ] **T043** Commit + push.
- [ ] **T044** File `specs/061-v2-quality-band/` as a successor stub: 3-4 baseline runs to characterize variance, then assert `n_clusters in [low, high]` band. Deferred until Phase 1+2 stabilize on the dev machine.

**Final gate (Phase 4)**:
- spec status = Implemented.
- spec-061 stub exists with a one-paragraph problem statement.
- The gate has been invoked successfully at least once on a real change (T031's smoke run counts).

---

## Total estimate

**~5-7 hours** across Phases 0-4. Phase 1 is the load-bearing piece; Phase 2 dominates the long tail (AppTest learning curve).
