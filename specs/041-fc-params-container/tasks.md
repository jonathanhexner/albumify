# Tasks: FC App v2 Configuration Container & Run Tab Parity (041)

Predecessor: spec-040 (Unified Pipeline Framework, ships `FCAppRunner` + v2 app + `_build_fc_config`)
Target: replace per-step `.get(default)` translator with one Pydantic container; v2 Run tab reaches knob-parity with the original FC App.

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped (rationale required)

---

## Design notes

- **D1**: `FCParams` is the **UI-facing** contract. Algorithm code keeps consuming `face_cluster.config.PipelineConfig` (`FCConfig`). The one boundary translator is `FCParams.to_fc_config()`. We do NOT replace `FCConfig` — locked decision from spec-040.
- **D2**: Field defaults live in `FCParams` only. Today they're duplicated across `_build_fc_config`, UI literals, and `FCConfig.__init__`. After this spec there is exactly one source.
- **D3**: Field ranges (`Field(ge=..., le=...)`) match the Streamlit slider min/max in the original FC App's Run tab. Profiles outside those ranges fail loudly with `ValidationError` — preferred over silent clamping.
- **D4**: `step_configs=` kwarg on `run_v2_pipeline` stays for one release alongside the new `params=`. Deprecation warning fires on the dict path. Removed in a follow-up spec.
- **D5**: Drift between `FCParams` and `FCConfig` is caught at import time by an architecture test, not at run time as a crash.
- **D6**: No new Streamlit machinery — Run tab parity is just more widget bindings into the same `FCParams(**widget_values)` constructor.

---

## Phase 1 — Container + drift guard

- [ ] **T001** Create `face_cluster/fc_params.py`:
  - Pydantic v2 `BaseModel` with `ConfigDict(extra='forbid')`.
  - 35 fields grouped: Cluster (3), Quality (8), Exemplars (3), Optional toggles (3), Merge (16), Cap (3).
  - Each numeric field uses `Field(default, ge=..., le=...)` matching original slider bounds in spec.md §4.1.
  - `to_fc_config() -> FCConfig` — `FCConfig(**self.model_dump())`.
  - `to_step_configs() -> dict[str, dict]` — broadcast over `UNIFIED_CLUSTERING_STEPS`.
  - `classmethod load(path)` and `save(path)` for JSON round-trip.
- [ ] **T002** Add `tests/face_clustering/test_fcparams.py` covering:
  - default values produce a usable `FCConfig`
  - `extra='forbid'` raises `ValidationError` on unknown key
  - out-of-range raises `ValidationError` (one example per field group)
  - `model_dump_json` → `model_validate_json` round-trip preserves equality
  - `to_step_configs()` returns a dict whose keys equal `UNIFIED_CLUSTERING_STEPS`
  - `load`/`save` round-trip via `tmp_path`
- [ ] **T003** Add `tests/architecture/test_fcparams_fcconfig_parity.py`:
  - `set(FCParams.model_fields) == {f.name for f in dataclasses.fields(FCConfig)}` with descriptive failure listing both deltas.
- [ ] **T004** Run `pytest tests/face_clustering/test_fcparams.py tests/architecture/test_fcparams_fcconfig_parity.py`; green required before Phase 2.

**Checkpoint**: New module compiles; field set matches `FCConfig` exactly; defaults round-trip; out-of-range and unknown-key inputs raise.

---

## Phase 2 — `run_v2_pipeline` accepts `params=`

- [ ] **T010** Edit `app/face_clustering_v2/pipeline.py`:
  - Add `params: Optional[FCParams] = None` kwarg to `run_v2_pipeline`.
  - If `params is not None`: derive `step_configs = params.to_step_configs()`; emit no warning.
  - If `step_configs is not None` and `params is None`: emit `DeprecationWarning` once per process pointing at `params=`.
  - If both: raise `ValueError("pass either params or step_configs, not both")`.
- [ ] **T011** Update `app/face_clustering_v2/pipeline.py` docstring to lead with the `params=` example.
- [ ] **T012** Update `tests/face_clustering/test_fc_app_v2_e2e.py`:
  - Replace the `cfg = {...}` dict in the module fixture with an `FCParams(...)` literal.
  - Call `run_v2_pipeline(params=params)`.
- [ ] **T013** Add `tests/face_clustering/test_run_v2_pipeline_kwargs.py`:
  - both kwargs given → `ValueError`
  - neither kwarg given → succeeds with `FCParams()` defaults applied
  - legacy `step_configs=` path emits one `DeprecationWarning`
- [ ] **T014** Run `pytest tests/face_clustering/test_fc_app_v2_e2e.py tests/face_clustering/test_run_v2_pipeline_kwargs.py`; green required.

**Checkpoint**: All 4 existing e2e tests pass via the new param path; legacy path still works and warns.

---

## Phase 3 — Delete `_build_fc_config`

- [ ] **T020** Edit `sim_bench/pipeline/steps/face_clustering_steps.py`:
  - In each of the 8 step classes' `process()`, replace `fc_cfg = _build_fc_config(config)` with `fc_cfg = FCConfig(**config)`.
  - Delete the `_build_fc_config` helper function.
  - Add module-level docstring note: "Each step receives a full FCConfig-shaped dict via FCParams.to_step_configs() broadcast — no per-step translator."
- [ ] **T021** Verify no other module imports `_build_fc_config` (`grep -r _build_fc_config sim_bench/ face_cluster/ app/`).
- [ ] **T022** Run `pytest tests/face_clustering/ tests/architecture/ -k "not slow"`; full green required.

**Checkpoint**: `face_clustering_steps.py` shrinks by ~60 LOC; all step unit tests pass; no caller broken.

---

## Phase 4 — Equivalence sweep on `FCParams`

- [ ] **T030** Edit `tests/face_clustering/test_legacy_vs_v2_equivalence.py`:
  - Replace the 4 `@pytest.mark.parametrize` dict entries with 4 `FCParams(...)` literals.
  - Legacy side calls `params.to_fc_config()`; v2 side calls `params.to_step_configs()`.
  - Use the same `CANONICAL_CONFIG` shape; surface knob differences (default vs merge_on vs tighter vs larger_K) at the `FCParams(...)` literal level.
- [ ] **T031** Run `pytest -m slow tests/face_clustering/test_legacy_vs_v2_equivalence.py`; ≥ 95% pairwise agreement on all 4 configs required.

**Checkpoint**: Equivalence sweep green on real 50-image fixture; if any config drops below 95% post-refactor, halt and root-cause before Phase 5.

---

## Phase 5 — Run tab parity (35 knobs)

- [ ] **T040** Edit `app/face_clustering_v2/tabs/run_tab.py`:
  - Replace today's 7-widget block with a structured layout matching the original's section dividers:
    - "Stage 3 · Quality Gate" expander — 11 widgets (blur, max_faces, min_face_area, det_score_min, yaw/pitch/roll, require_pose).
    - "Stage 5 · Cluster" — K, distance_threshold, min_cluster_size (already present).
    - "Stage 6 · Exemplars" — 3 widgets.
    - "Optional Stages" — split_enabled, merge_enabled, attach_enabled.
    - Merge sub-panel — reuse `app/shared/merge_controls.render_merge_params(key_prefix="v2_run_")`.
  - At click time, construct `params = FCParams(**widget_values)` inside a `try/except ValidationError as e: st.error(...)`.
  - Replace `run_v2_pipeline(step_configs=...)` with `run_v2_pipeline(params=params)`.
- [ ] **T041** Manual Playwright smoke (per CLAUDE.md "Delivery Quality"):
  - Start v2 Streamlit on port 8888.
  - Navigate to Run tab.
  - Tweak at least one knob per section.
  - Run on the 6-jpg fixture; confirm success message and cluster count visible.
  - Verify the resulting v5 DB has the expected face/cluster counts.
- [ ] **T042** Update `tests/face_clustering/test_fc_app_v2_e2e.py` — add one extra test that constructs a non-default `FCParams` (e.g., `K=3, merge_enabled=True`) and verifies the run completes with corresponding `step_configs` reaching the steps.

**Checkpoint**: v2 Run tab matches original's knob set (35 fields); user can drive a non-default run end-to-end; validation errors render via `st.error`.

---

## Phase 6 — Profile script collapse

- [ ] **T050** Rewrite `scripts/migrate_fc_profiles.py`:
  - `migrate_one(path, dry_run=False)` becomes ~10 lines using `FCParams.model_validate(flat).save(path)`.
  - Detect "already v2 shape" via presence of valid `FCParams` field set with no extras; collapse `is_v2_shape`/`reshape_v1_to_v2` if no longer needed.
  - Preserve `.v1.json` backup behaviour.
- [ ] **T051** Trim `tests/face_clustering/test_profile_migration.py`:
  - Drop tests for `is_v2_shape`/`reshape_v1_to_v2` if those helpers are deleted.
  - Keep: round-trip, idempotence, dry-run, invalid-JSON, backup written.
- [ ] **T052** Run `pytest tests/face_clustering/test_profile_migration.py`; green required.

**Checkpoint**: Profile migration logic is now a thin wrapper over `FCParams`; tests still cover idempotence, backup, dry-run, invalid JSON.

---

## Phase 7 — Headless CLI runner

- [ ] **T060** Add `scripts/run_v2.py`:
  - Args: `--src PATH`, `--out PATH`, `--profile PATH` (optional), per-knob overrides (`--K`, `--distance_threshold`, ...; only the Tier-1 most-used ones — full set comes through `--profile`).
  - `--save-profile PATH` writes the resolved `FCParams` to JSON.
  - Construct `FCParams`, call `run_v2_pipeline(params=params)`.
  - Log result counts; exit 0 on success, 1 on failure.
- [ ] **T061** Add `tests/face_clustering/test_run_v2_script.py`:
  - subprocess-call against the 6-jpg fixture with `--profile`.
  - subprocess-call with `--K --distance_threshold` flags.
  - verify exit code, stdout contains "Run complete", and the v5 DB exists.
- [ ] **T062** Run `pytest tests/face_clustering/test_run_v2_script.py`; green required.

**Checkpoint**: User can repro a UI run from the shell; equivalence runs from a Makefile become trivial.

---

## Phase 8 — Code review + close-out

- [ ] **T070** Run `/code-review` slash command; output to `specs/041-fc-params-container/REVIEW.md`.
- [ ] **T071** Address any high-severity findings; document Major/Minor deferrals in REVIEW.md.
- [ ] **T072** Update `docs/architecture/classes.html` — add `FCParams` node under `face_cluster/`; link to `FCConfig` with edge labeled `to_fc_config()`.
- [ ] **T073** Update `docs/architecture/data_flow.html` if it diagrams the UI→pipeline path — replace `dict` arrow with `FCParams` arrow.
- [ ] **T074** Append a `CHANGES_LOG.md` entry per landed commit (CLAUDE.md mandate).
- [ ] **T075** Flip `specs/041-fc-params-container/spec.md` status `Draft` → `Code Review` → `Implemented` once REVIEW.md is clean.

**Checkpoint**: Review clean, docs updated, spec marked Implemented; ready for `cluster_people` rewire in a later spec (spec-040 Phase 7 follow-up).

---

## Out of scope (explicit deferrals)

- The 5 remaining FC App tabs (Recluster, Merge Analysis, Merge ML, Quality, Gallery). Tracked separately.
- Async worker + live-log machinery in v2 Run tab.
- Replacing `FCConfig` algorithm-layer dataclass with `FCParams`.
- Rewiring Albumify's `cluster_people` step to call `FCAppRunner` directly (spec-040 Phase 7).
- Removing the legacy `step_configs=` kwarg from `run_v2_pipeline` (one-release deprecation window).
