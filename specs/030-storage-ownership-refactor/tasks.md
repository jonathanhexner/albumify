# Tasks: Storage Ownership Refactor (030)

**Spec**: `specs/030-storage-ownership-refactor/spec.md`
**Architecture audit**: [`architecture_audit.html`](architecture_audit.html)
**Resolves**: SIGHTING-058

## Design Notes

- **D1 · Two new modules, replace four old ones.** `face_cluster/run_store.py` (read) and `face_cluster/run_exporter.py` (write) replace the responsibilities currently scattered across `face_cluster/loader.py`, `face_cluster/export.py`, `face_cluster/result_db.py`, and `sim_bench/pipeline/steps/face_cluster_export.py`. The four old modules don't disappear in one PR — they shrink phase by phase.

- **D2 · DB is the single relational store; .npy is the single bulk-numeric store.** Embeddings come out of the DB entirely (FR-006). The `embeddings` table is dropped. This is a real schema change, not just an unused column.

- **D3 · `merge_decisions` schema is the contract.** All 28 fields of `merge_log` get a column. Writing an unknown key raises; reading a missing column raises. No `dict.get(..., default)` patterns.

- **D4 · Schema versioning lives in `pipeline_run.json`.** A small JSON file at the run root carries `schema_version: 4`. `RunStore` reads it first; if it doesn't match, raises a clear error pointing at the migration script. The DB itself uses `PRAGMA user_version` as a redundant check.

- **D5 · No fallbacks; fail loud.** `RunStore.__init__` validates every required artifact exists. Missing → `RunStoreError("required artifact 'X' not found in run dir 'Y'")`. No `if path.exists()` branches.

- **D6 · Producer is metadata, not directory naming.** Today's prefixes (`merge_snap_`, `remerge_`, `manual_*`) become a `producer` column in `run_metadata`. Directory layout is purely `<album>/face_clustering/<run_id>/`. Lineage (parent_run_id, merge_round) is also a column.

- **D7 · UI semantics are a thin layer.** Three-state outcome labels and margin-disabled rendering are local to `_merge_decisions_panel.py`. They depend on FR-004 (full schema) being shipped first.

- **D8 · Migration script is a one-shot, not a runtime fallback.** `scripts/migrate_run_to_v4.py <run_dir>` rebuilds the new layout from whatever's on disk. Loaders never call migration code at read time. Old runs that aren't migrated simply fail to open with a pointer to the script.

- **D9 · Static checks are tests.** "No `.exists()` chain in `RunStore`" and "no direct file reads in `app/`" are pytest tests that grep the source. They're cheap, fast, and prevent regression of the architecture itself.

- **D10 · Phasing avoids a big-bang migration.** Phase 1 ships the writer alongside the legacy writers (dual-write, parallel). Phase 2 ships the reader alongside the legacy loader (parallel). Phase 3 swaps UI to the reader. Phase 4 deletes legacy writers. Phase 5 deletes legacy readers and CSVs. Each phase is independently shippable and revertable.

---

## Phase 1: Schema fix + new writer (alongside existing)

- [x] **T001** SIGHTING-058 filed in `docs/project/SIGHTINGS.md` linking to this spec.
- [x] **T002** Define `MergeDecisionRow` dataclass with all 28 fields in `face_cluster/types.py`. (Verified: dataclass field set equals on-disk `merge_log.json` field set on `face_clustering_20260508_000446`.)
- [x] **T003** `face_cluster/run_exporter.py` created with `RunExporter` class and `_write_*` private methods. Public entry: `RunExporter(output_dir).export(...)`.
- [x] **T004** SQL schema in `_SCHEMA`. `merge_decisions` has all 28 columns matching `MergeDecisionRow.field_names()`. `run_metadata` absorbs `merge_thresholds_json`, `merge_iter_summary_json`, `producer`, `parent_run_id`. `embeddings` table removed. `PRAGMA user_version = 4`.
- [x] **T005** Strict-write — `_strict_validate_merge_log` raises `RunExporterError` on unknown / missing keys. `_to_sql` raises on unsupported types. No `dict.get` defaults anywhere in the writer.
- [x] **T006** `pipeline_run.json` contains only pointer fields (run_id, source_album, producer, parent_run_id, started_at, finished_at, status, schema_version: 4, db_path). No config blob.
- [x] **T007** `RunExporter` wired into `face_cluster/pipeline.py:_export()` — writes to `<output_dir>/_v4/`. Failure is non-fatal during Phase 1 (logged warning, legacy export continues).
- [x] **T008** `RunExporter` wired into `sim_bench/pipeline/steps/face_cluster_export.py:export_for_analysis()` — writes to `<output_dir>/_v4/`. Producer tagged "albumify".
- [x] **T009** Tests in `tests/face_clustering/test_run_exporter.py` (12 tests, all passing): listdir allow-list, full-fidelity round-trip, strict-write rejects unknown / missing keys, identical layout across producers, schema-version pragma, npy alignment, no embeddings table, column order matches dataclass. Plus `test_v4_dual_write_present` and `test_v4_db_merge_decisions_match_legacy_log` in `test_merge_stage.py` exercise the wiring against the real pipeline.
- [x] **Bonus** Producer-side fix in `merge.py`: terminal-iteration rows now stamp `actually_merged=False` (previously the early-return path skipped it, leaving a contract gap). Regression test `test_every_merge_log_row_has_actually_merged` added.

**Checkpoint**: ✅ All Phase 1 task tests green. Real pipeline runs produce both legacy artifacts AND a `_v4/` subdir with the exact 5-artifact set. Smoke test on broken run `face_clustering_20260508_000446` confirms the new schema preserves every field that was previously dropped.

---

## Phase 2: New reader (alongside existing loader)

- [x] **T010** `face_cluster/run_store.py` created. Constructor validates `pipeline_run.json` exists & parses, `schema_version == 4`, all 5 expected artifacts present, DB `PRAGMA user_version` agrees with the JSON. Raises `RunStoreError` with the offending artifact name on any miss.
- [x] **T011** Read methods implemented: `.faces()`, `.metadata()`, `.merge_log()`, `.embeddings()`, `.crop_path(face_id)`, `.clusters(iteration)` (accepts int or `"base"`/`"final"`), `.iteration_count()`.
- [x] **T012** `.merge_log()` returns `List[MergeDecisionRow]`. SQLite int → Python `bool` conversion handled explicitly for the 6 boolean columns. Column order asserted to match `MergeDecisionRow.field_names()` via positional materialization.
- [x] **T013** `.embeddings()` returns `EmbeddingMatrix(matrix, face_ids)` NamedTuple. Both `.npy` files loaded atomically; length mismatch raises.
- [x] **T014** `tests/face_clustering/test_run_store.py` — 20 unit tests, all passing:
  - 9 construction-validation tests (missing dir / json / db / npy / crops / schema_version mismatch / db user_version mismatch / corrupt json)
  - 8 read-method round-trip tests (metadata, merge_log full fidelity incl. `inf` margin_gap and bool conversion, embeddings, faces, clusters base+final, iteration_count, crop_path resolves & raises on unknown / missing)
  - 1 architecture invariant test (`test_no_existence_chain_in_run_store`) — static-checks `run_store.py` source for forbidden `elif x.exists():` patterns and unguarded `if x.exists():` fallbacks

**Checkpoint**: ✅ End-to-end smoke test on `face_clustering_20260508_000446`: `RunExporter` writes → `RunStore` reads back. C0 vs C1 iter 2 row correctly reads `action=merged actually_merged=YES support=192/2 diameter=0.972/2.294`. Cross-table consistency holds: 4 rows with `actually_merged=True` matches `run_metadata.n_merges=4`.

---

## Phase 3: UI cutover to RunStore

User-visible bug fix landed in Phase 3a (this commit).  Remaining cleanup
(T015–T019, T022) is architectural hygiene and ships as Phase 3b.

### Phase 3a — bug fix (shipped)

- [x] **T020** `face_cluster/loader.py:load_pipeline_result()` now prefers `<run_dir>/_v4/` via `RunStore`, falls back to legacy DB / CSV for runs that pre-date Phase 1.  `_load_via_run_store` translates `RunStore` reads into a `PipelineResult` so existing UI code is unchanged.  Matched legacy semantics: `merged_cluster_result` is non-None whenever the merge stage ran, even with zero merges.
- [x] **T021** `_outcome_label_color(pair)` and `_format_margin_value(pair)` helpers added to `_merge_decisions_panel.py`.  Three-state outcome label (MERGED / PASSED / REJECTED) replaces the binary MERGED-or-REJECTED at three render sites.  Margin badge prints "disabled" when `merge_margin == 0` (margin_gap=inf) instead of the literal "inf".  `MergeDecisionRow` (the views' UI dataclass in `face_cluster/views/merge_view.py`) gained `actually_merged: bool = False` so the helpers can distinguish MERGED from PASSED.
- [x] **T021 tests** `tests/face_clustering/test_merge_panel_semantics.py` — 8 unit tests pinning the helpers' behaviour across all four outcome shapes and the inf-margin case.
- [x] **T023** `tests/face_clustering/test_sighting_058_regression.py` — 7 tests reproducing the user's exact bug scenario (iter-1 passed + iter-2 merged on the same pair, `merge_margin=0` in config) and asserting: loader finds v4, view reports correct `len(merges)`, iter-1 row labels PASSED (not REJECTED), iter-2 row labels MERGED, margin badge is "disabled", `support 191/2` and `diameter 0.972/2.294` (no more `/0` or `/n/a`), cross-table consistency holds.

**Checkpoint**: ✅ Full spec-030 suite 65/65 green (RunExporter + RunStore + panel semantics + SIGHTING-058 regression + merge_stage E2E).  The user-reported bug is structurally fixed end-to-end on the new path.

### Phase 3b — architecture hygiene (shipped)

- [x] **T016** `app/face_clustering/tabs/cluster_analysis_tab.py:268-273` — direct `pd.read_csv(clusters.csv)` for cluster provenance replaced with a `cluster_stats` lookup (origin / parent_ids now flow through `RunStore.clusters()` → `ClusterResult.cluster_stats`).
- [x] **T018** `app/face_clustering/_merge_decisions_panel.py:211-214` — direct `json.load(pipeline_run.json)` replaced with `result.summary["source_album"]` (already populated by the loader).
- [x] **T022** `tests/architecture/test_no_direct_run_artifact_reads.py` — static-check pytest greps `app/face_clustering/` for `pd.read_csv`, `np.load`, `json.load(`, `json.loads(...read_text)`, `sqlite3.connect`. Allow-list documents the four deliberate exceptions (`run_panels.py` and `tabs/history_tab.py` are filesystem inspectors; `tabs/labeling_review_tab.py` and `face_popup.py` read user-side sidecars). Plus `test_allow_list_entries_still_exist` to prevent zombie entries.

### Phase 3b — deferred to Phase 4

- [ ] **T015** `app/face_clustering/cache_helpers.py` — reads `crop_manifest.json` and `faces.csv` from the legacy layout. Cutover requires removing the legacy writer first (Phase 4 / T024–T025), so this stays as a transitional dependency. Documented in `_DELIBERATE_DIRECT_READS` allow-list.
- [ ] **T017** `app/face_clustering/tabs/history_tab.py:106-109` — lightweight per-run summary that deliberately reads JSON sidecars to avoid a full `PipelineResult` load. Re-evaluate once Phase 4 collapses the on-disk layout.
- [ ] **T019** `app/face_clustering/tabs/labeling_review_tab.py:23` — `merge_decisions.json` is a user-approval sidecar, not run-internal data. Defer; may stay as a sidecar exception permanently.

---

## Phase 4: Stop writing legacy artifacts

- [ ] **T024** Remove `export_results` and `export_merged_results` calls from `face_cluster/pipeline.py:_export()`. The functions remain in `face_cluster/export.py` but are unused.
- [ ] **T025** Remove `write_results_db` and `export_results` calls from `sim_bench/pipeline/steps/face_cluster_export.py`. Migrate the crop-generation logic into `RunExporter._write_crops` (it's the only piece that has no equivalent yet).
- [ ] **T026** Remove `_v4/` subdir convention from Phase 1 — `RunExporter` now writes directly to the run root.
- [ ] **T027** Update `face_cluster/manual_merge_snapshot.py` to use `RunExporter` instead of building its own CSVs.
- [ ] **T028** Switch run-directory layout to `results/<album>/face_clustering/<run_id>/` (FR-009). Update `face_cluster/run_naming.py:allocate_run_dir`. Producer recorded in `run_metadata.producer`, not in directory prefix.
- [ ] **T029** [P] Static-check test `tests/architecture/test_run_directory_listdir.py` — after any export, `set(os.listdir(run_dir)) == {"face_clustering.db", "embeddings.npy", "embedding_face_ids.npy", "pipeline_run.json", "crops"}`.

**Checkpoint**: Fresh runs from both apps produce only the 5-artifact layout. Legacy CSV/JSON files no longer appear.

---

## Phase 5: Migration + deletion

- [ ] **T030** Create `scripts/migrate_run_to_v4.py <run_dir>`:
  - Detects legacy layout (any of: CSV files, `merge_log.json`, old DB schema).
  - Rebuilds the v4 layout in-place (writes new DB with full schema, copies/regenerates `embeddings.npy` if absent, removes obsolete files).
  - Idempotent: re-running on a v4 dir is a no-op.
- [ ] **T031** [P] Test `tests/test_migration_v4.py` — fixtures of legacy run dirs (CSV-only, DB-only, mixed); migrate; assert new layout passes `RunStore` validation.
- [ ] **T032** Run migration on all existing runs under `results/`. Document outcomes in `CHANGES_LOG.md`.
- [ ] **T033** Delete `face_cluster/export.py` (after confirming no callers via grep).
- [ ] **T034** Delete `face_cluster/result_db.py` (its schema is now in `run_exporter.py`).
- [ ] **T035** Delete `_load_from_csvs` and `_load_from_db` private functions in `face_cluster/loader.py`. The public `load_pipeline_result` wrapper either delegates to `RunStore` or is removed entirely.
- [ ] **T036** Delete CSV-reading sites still found in `app/face_clustering_labeling.py`, `app/simple_face_labeling.py`, `app/face_clustering_workbench.py`, `app/face_clustering_comparison.py` if they are still referenced. (These are older sibling apps; check for dead code first.)
- [ ] **T037** Update `docs/architecture/overview.md` with the new storage model. Update `docs/README.md` if file placement rules reference any deleted artifact.
- [ ] **T038** Append `CHANGES_LOG.md` entry (`[REFACTOR]`).
- [ ] **T039** Append `LEARNINGS.md` entries for L1–L4 from the design doc.

**Checkpoint**: `git grep -E "merge_log\.json|merge_metadata\.json|crop_manifest\.json|export_summary\.json|faces_merged\.csv|clusters_merged\.csv|clusters_stage_base\.csv"` returns zero matches in production code (allow-listed in migration script and tests only).

---

## Verification (cross-phase)

- [ ] **V1** Acceptance criteria from spec.md US1: open `face_clustering_20260508_000446`, all four assertions pass.
- [ ] **V2** US2 byte-identical layout: `diff -r` of Albumify-produced and FC-App-produced runs on same input shows zero differences in file set; only timestamps and run_id differ.
- [ ] **V3** US3 single read interface: zero matches for `pd.read_csv|np.load|json.load|sqlite3.connect` against run artifacts in `app/**`.
- [ ] **V4** US4 layout: every run path is `results/<album>/face_clustering/<run_id>/`; no orphan reclusters at `results/` root.
- [ ] **V5** US5 fail-loud: introduce a deliberate schema mismatch in a fixture; `RunStore` raises with a clear message.
- [ ] **V6** Performance: `RunStore.merge_log()` on a 200-cluster run is no slower than the previous JSON-load path (target: ≤ 100ms).
- [ ] **V7** Manual: walk through all 11 FC App tabs on a real run, capture screenshots in `specs/030-storage-ownership-refactor/post_refactor_screens/`.

## Out of scope

- ML training data store (`training_db.py`) is untouched. It serves a different purpose (cross-run labels) and has its own lifecycle.
- `UniversalCache` (cross-run input caching) is untouched.
- Top-level `results/` layout cleanup beyond face-clustering subtrees (e.g., quarantining stray log files) — handle in a separate spec.

## Rollout & risk

- **Phases 1–2** are additive and reversible. Ship them independently; revert by removing imports.
- **Phase 3** is the cutover; test thoroughly. Have a flag-off path: `RunStore` falls back to legacy via env var `SIM_BENCH_USE_LEGACY_LOADER=1` for one release cycle.
- **Phase 4** stops writing legacy artifacts but doesn't delete code. Old runs still openable via migration script.
- **Phase 5** is purely deletion of dead code; safe once Phase 4 has soaked in production for one week.
- **Risk**: external notebooks (`notebooks/`) read CSVs directly. T032 must include a grep-and-document pass; users may need to update notebooks. Document in `LEARNINGS.md`.
