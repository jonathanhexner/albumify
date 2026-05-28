# Tasks: Cluster Analysis Tab (045)

Predecessors: spec-042 (v2 tab parity umbrella; History pilot), spec-043 (Repository pattern B0), spec-044 (Column Registry B0.1)
Standards reference: [`docs/architecture/architecture_standards.md`](../../docs/architecture/architecture_standards.md) §B0 (Repository), §B2 (error model), §B3 (service vocabulary), §A6 (per-tab migration discipline)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped (rationale required)

---

## Design notes (binding for every task)

- **D1**: **Build → test → migrate**, never out of order. Repository ships first (Phases 1–2). Service ships second (Phases 3–5). Tab + components ship last (Phase 6). Each phase ends with all earlier tests still green.
- **D2**: **Tab → Service → Repository → DB layering** is enforced by arch tests (Phase 7). No DB access, JSON parsing, or filesystem reads inside `app/face_clustering_v2/tabs/` or `components/`.
- **D3**: `ClusterAnalysisRepository` is a **query-shape Repository (B0b)** — it composes RunStore for schema validation and low-level reads. It does NOT own `face_clustering.db`'s schema. Therefore B0.1 (Column Registry) does NOT apply.
- **D4**: F-3 (extract `ColumnDef` to shared module) does NOT fire from this spec — the new Repository doesn't use `ColumnDef`. F-3 remains deferred to the next schema-owning Repository.
- **D5**: `ClusterView.compute` / `ClusterDebugView.compute` keep their current `PipelineResult`-shaped signatures. The Service assembles a minimal `PipelineResult` from Repository reads to feed them. Refactoring those classmethods to take typed inputs is tracked as Open Question 5 in `spec.md`, deferred to a follow-up.
- **D6**: Async compute scaffolding (the `_AsyncState` polling loop from `app/face_clustering/state.py`) is extracted to `face_cluster/views/_async.py` as `AsyncHandle[T]`. The tab never sees raw `Thread` objects.
- **D7**: Force-merge mutations create a **new run dir** (`merge_snap_{round}/`), not in-place modifications. The Repository writes the snapshot; the tab is responsible for switching `current_run_dir` to the new dir + invalidating compute workers.

---

## Phase 1 — Repository: typed types + reads (~75 min)

Repository skeleton + read methods. Composes RunStore. No mutations yet.

- [ ] **T001** Create `face_cluster/repositories/cluster_analysis_repo.py`. Add `ClusterAnalysisRepoConfig` (frozen-slotted; fields: `run_dir`, `read_only`, `log_queries`) per spec.§5.3.
- [ ] **T002** Add `ClusterAnalysisCriteria` (frozen-slotted; fields: `cluster_id`, `face_ids`, `iteration`, `exemplars_only`, `include_noise`) per spec.§5.4. `include_noise` defaults to `False`; comparisons use `NOISE_LABEL` / `is_noise()` from `sim_bench.pipeline.clustering_labels` (no bare `-1` literal anywhere in this file or the Repository).
- [ ] **T003** Add `Assignment` dataclass to `face_cluster/views/_base.py` (small typed row: `face_id: int, cluster_id: int, is_exemplar: bool, iteration: int`).
- [ ] **T004** Implement `ClusterAnalysisRepository.__init__(config)`. Validates `run_dir` is provided. Constructs an internal `RunStore(config.run_dir)` — RunStore's existing validation gives us schema-version + artifact-presence checks for free.
- [ ] **T005** Implement read methods (all delegating to the internal RunStore + shaping output):
  - `get_cluster_rows(iteration="final")` → `List[ClusterRow]`
  - `get_cluster_ids(iteration="final")` → `List[int]`
  - `find_assignments(criteria)` → `List[Assignment]`
  - `get_face_records(face_ids)` → `List[FaceRecord]` (RunStore.faces filtered)
  - `get_run_metadata()` → `RunMetadata`
  - `get_merge_log()` → `List[MergeDecisionRow]`
  - `get_cluster_result(iteration="final")` → `ClusterResult`
- [ ] **T006** Create `tests/face_clustering/repositories/test_cluster_analysis_repo_synthetic.py`. Build a tmp-dir synthetic `face_clustering.db` + `embeddings.npy` (3 clusters, ~30 faces) fixture. Tests #1–#10 from spec.§8.1.
- [ ] **T007** Run synthetic test suite: `pytest tests/face_clustering/repositories/test_cluster_analysis_repo_synthetic.py -q`. All 10 pass.

**Validation gate (Phase 1)**:

```
.venv/Scripts/python -m pytest \
  tests/face_clustering/repositories/test_cluster_analysis_repo_synthetic.py \
  tests/face_clustering/repositories/ -q
```

Expected: **+10 new tests pass** (spec §8.1 #1–#10, mutation #11/#12 added in Phase 2). Total ≥ 85 in `tests/face_clustering/repositories/` (75 from spec-044 + 10 new). Zero existing tests turn red.

---

## Phase 2 — Repository: real fixture + mutations (~45 min)

Real-data integration + snapshot-writing mutation.

- [ ] **T010** Create `tests/face_clustering/repositories/test_cluster_analysis_repo_real.py`. Tests #1–#3 from spec.§8.2 against the History-pilot fixture run dir.
- [ ] **T011** Run real-fixture tests. All 3 pass.
- [ ] **T012** Add `ForceMergeResult` dataclass (frozen-slotted; fields per spec.§4.2) to `face_cluster/views/cluster_analysis.py`. (The file doesn't exist yet — create with the dataclass; Service class added in Phase 3.)
- [ ] **T013** Implement `ClusterAnalysisRepository.save_manual_merge_snapshot(*, cluster_a, cluster_b, merge_round, config)`. Internals: validate cluster ids; raise `ValidationError` on read-only mode; delegate the on-disk write to the existing `face_cluster.merge.save_manual_merge_snapshot` callable (keeping the legacy snapshot-writer code as the implementation detail). Return typed `ForceMergeResult`.
- [ ] **T014** Add synthetic mutation tests #11–#12 from spec.§8.1.
- [ ] **T015** Run: `pytest tests/face_clustering/repositories/ -q`. All 12 synthetic + 3 real-fixture pass.

**Validation gate (Phase 2)**:

```
.venv/Scripts/python -m pytest tests/face_clustering/repositories/ -q
```

Expected: **+5 new tests pass** (3 real-fixture from spec §8.2 + 2 mutation from §8.1). Total = 12 synthetic + 3 real = **15 cluster-analysis Repository tests green**, plus the spec-044 baseline still green.

```
.venv/Scripts/python -m pytest tests/architecture/ -q
```

Expected: no architecture-test regressions (spec-043/044 guards still hold).

---

## Phase 3 — Service: skeleton + queries (~30 min)

Service composes Repository; cheap queries only.

- [ ] **T020** Extend `face_cluster/views/cluster_analysis.py`: add `ClusterAnalysisService` class with `__init__(repo)`, `list_clusters()`, `get_cluster_ids()` per spec.§6.1. Each method is a thin pass-through to the Repository.
- [ ] **T021** Add `ForceMergePreview` dataclass (frozen-slotted; fields per spec.§4.2) to `face_cluster/views/cluster_analysis.py`. Not consumed yet — wired in Phase 5.
- [ ] **T022** Create `tests/face_clustering/views/test_cluster_analysis_service_synthetic.py`. Tests #1–#3 from spec.§8.3.
- [ ] **T023** Run: `pytest tests/face_clustering/views/test_cluster_analysis_service_synthetic.py -q`. All 3 pass.

**Validation gate (Phase 3)**:

```
.venv/Scripts/python -m pytest tests/face_clustering/views/test_cluster_analysis_service_synthetic.py -q
```

Expected: **3 tests pass** (spec §8.3 #1–#3). Service constructible from a Repository; pass-through queries return typed objects identical to the Repository.

---

## Phase 4 — Service: async wrappers (~45 min)

Heavy compute behind `AsyncHandle[T]`.

- [ ] **T030** Extract the async-worker pattern from `app/face_clustering/state.py::_AsyncState` into `face_cluster/views/_async.py` as `AsyncHandle[T]` (generic dataclass; fields: `is_running`, `has_error`, `error`, `result`; methods: `start(fn, *args, **kwargs)`, `poll()`). Keep the same background-thread + polling mechanics under the hood; just give it a typed surface.
- [ ] **T031** Add `ClusterAnalysisService.compute_detail_async(cluster_id) -> AsyncHandle[ClusterView]` and `compute_debug_async(cluster_id) -> AsyncHandle[ClusterDebugView]`. Each method: build a minimal `PipelineResult` from Repository reads (per D5), then start `ClusterView.compute` / `ClusterDebugView.compute` on the handle.
- [ ] **T032** Add tests #4–#6 from spec.§8.3 (assert handle returns the expected `ClusterView` / `ClusterDebugView`; assert unknown cluster surfaces via `handle.error`).
- [ ] **T033** Run: full Service test suite. All 6 synthetic pass.

**Validation gate (Phase 4)**:

```
.venv/Scripts/python -m pytest tests/face_clustering/views/test_cluster_analysis_service_synthetic.py -q
```

Expected: **6 tests pass** (spec §8.3 #1–#6). Async handles transition to `done` (with typed `ClusterView` / `ClusterDebugView` results) or `failed` (with `error` set) within the test timeout. `AsyncHandle[T]` has its own unit test covering `start → poll → done` and `start → cancel → cancelled` transitions.

```
.venv/Scripts/python -m pytest tests/face_clustering/views/ -q
```

Expected: no regression in earlier views tests (history-pilot tests still green).

---

## Phase 5 — Service: force merge end-to-end (~60 min)

Preview + apply mutations.

- [ ] **T040** Implement `ClusterAnalysisService.preview_force_merge(cluster_a, cluster_b)` — pure compute. Mirror the legacy `_compute_force_merge_preview` body (gates: exemplar / support / post-merge-diameter); return typed `ForceMergePreview`. Reads `merge_candidate_threshold` from `repo.get_run_metadata().config` (per Open Question 3 in spec.md — confirm before this task).
- [ ] **T041** Implement `ClusterAnalysisService.apply_force_merge(cluster_a, cluster_b, *, merge_round)` — delegates to `repo.save_manual_merge_snapshot`, returns the typed `ForceMergeResult`. No session-state touching.
- [ ] **T042** Synthetic Service tests #7–#12 from spec.§8.3.
- [ ] **T043** Create `tests/face_clustering/views/test_cluster_analysis_service_real.py`. Tests #1–#4 from spec.§8.4 against the real fixture.
- [ ] **T044** Run: `pytest tests/face_clustering/views/ tests/face_clustering/repositories/ -q`. All 12 synthetic + 4 real Service + 12 + 3 Repository = 31 pass.

**Validation gate (Phase 5)**:

```
.venv/Scripts/python -m pytest tests/face_clustering/views/ tests/face_clustering/repositories/ -q
```

Expected: **31 tests pass** (12 Service synthetic + 4 Service real + 12 Repository synthetic + 3 Repository real). Force-merge mutation writes a fresh `merge_snap_{round}/` dir under the parent run; parent run dir hashes unchanged before/after.

```
.venv/Scripts/python -c "import streamlit"  # baseline sanity
.venv/Scripts/python -c "from face_cluster.views import cluster_analysis"
```

Expected: the second import succeeds **without importing streamlit anywhere in its module graph** (verified by Phase 7 arch test #1 — preview here for early signal).

---

## Phase 6 — Components + tab (~90 min)

Streamlit UI rebuilt against the Service.

- [ ] **T050** Add `current_run_dir` to History's Load Run path. Edit `app/face_clustering_v2/components/load_button.py` to write `st.session_state['current_run_dir'] = str(run_dir)` alongside the existing `pipeline_result` write. Cluster Analysis reads this via the resolver in spec.§7.2 (priority: `current_run_dir` → `v2_last_run_dir` → None). No new picker in the tab — History remains the single load surface.
- [ ] **T051** Build 6 components under `app/face_clustering_v2/components/`:
  - `cluster_picker.py` — selectbox + state hooks (~20 LOC)
  - `cluster_metrics.py` — 5-metric strip + split-signal banner + provenance row (~40 LOC)
  - `face_grid.py` — 8-col thumbnail grid (~35 LOC; reused by future Gallery)
  - `nearest_clusters.py` — nearest-clusters list with Go-To button (~40 LOC)
  - `force_merge.py` — Force Merge expander (~50 LOC; consumes Service + ForceMergePreview)
  - `cluster_debug.py` — Graph Debug section (~40 LOC; metrics + heatmap + 3 expanders)
- [ ] **T052** Rewrite `app/face_clustering_v2/tabs/cluster_analysis_tab.py` as the orchestrator from spec.§7.1. Target ≤80 LOC. No DB / JSON / filesystem access; no `cfg.get` literals.
- [ ] **T053** Wire into `app/face_clustering_v2/main.py`. Replace the existing minimal "Clusters" tab registration with `render_cluster_analysis_tab`. Keep `clusters_tab.py` in the file system for now (deletion is Phase 8); don't reference it from `main.py`.
- [ ] **T054** Manual smoke: start the v2 app, open History → Load fixture run → switch to Cluster Analysis. Confirm: dropdown populated, metrics + thumbnails render, no exceptions. Document any rough edges as new TODO items, not stop-the-line bugs.

**Validation gate (Phase 6)**:

```
.venv/Scripts/python -c "import ast, pathlib; src=pathlib.Path('app/face_clustering_v2/tabs/cluster_analysis_tab.py').read_text(); print('LOC', sum(1 for l in src.splitlines() if l.strip() and not l.strip().startswith('#')))"
```

Expected: **≤ 80 non-blank/non-comment LOC**. (Same one-liner repeated for each new file under `app/face_clustering_v2/components/cluster_*.py`; sum ≤ 200 LOC.)

```
.venv/Scripts/streamlit run app/face_clustering_v2/main.py --server.port 8889 --server.headless true
```

Expected: server starts; navigate History → Load fixture run → Cluster Analysis tab → at least one cluster card renders. **No `st.exception` on any path exercised.** Manual log of any rough edges goes to `TODO.md` (not stop-the-line).

---

## Phase 7 — Arch tests + Playwright smoke (~30 min)

Pattern locked in by arch tests; UI verified visually.

- [ ] **T060** Create `tests/architecture/test_cluster_analysis_tab.py`. Add tests #1–#5 from spec.§8.5:
  - `test_tab_has_no_direct_db_or_filesystem_access` (AST or grep on `cluster_analysis_tab.py` + `components/`)
  - `test_tab_has_no_cfg_get_literals` (grep on the v2 tab dir)
  - `test_service_returns_typed_objects` (`inspect.signature(...).return_annotation`)
  - `test_repository_takes_typed_config` (extend the spec-043 arch test pattern to `ClusterAnalysisRepository`)
  - `test_force_merge_preview_fields_match_writer` (drift guard between `ForceMergePreview` field set and what `apply_force_merge` actually populates)
- [ ] **T061** Create `tests/manual/_v2_cluster_analysis_smoke.py`. Playwright headless script per spec.§8.6 — 5-step sequence against a live `streamlit run` on port 8889.
- [ ] **T062** Run: `pytest tests/architecture/test_cluster_analysis_tab.py -v`. All 5 pass.
- [ ] **T063** Run the Playwright smoke against a live server. Document expected behavior; tolerate timing differences.

**Validation gate (Phase 7)**:

```
.venv/Scripts/python -m pytest tests/architecture/test_cluster_analysis_tab.py -v
```

Expected: **5 tests pass** (spec §8.5 #1–#5). Drift guards lock the Tab → Service → Repository layering.

```
.venv/Scripts/python -m pytest tests/architecture/ -q
```

Expected: full architecture suite still green (no regression in spec-043/044/048/050 guards).

```
.venv/Scripts/python tests/manual/_v2_cluster_analysis_smoke.py
```

Expected: Playwright run completes 5 steps without exception; screenshots saved under `tests/manual/`. Documented as a manual gate, not CI.

---

## Phase 8 — Cleanup + close-out (~45 min)

- [ ] **T070** Delete `app/face_clustering_v2/tabs/clusters_tab.py` (the 71-LOC minimal list view; replaced by Cluster Analysis tab).
- [ ] **T071** Update `docs/architecture/classes.html`. Add entries for:
  - `ClusterAnalysisRepository` (§5 Writers/readers)
  - `ClusterAnalysisRepoConfig` + `ClusterAnalysisCriteria` (§4 Internal types)
  - `ClusterAnalysisService` (§5 Writers/readers)
  - `ForceMergePreview` + `ForceMergeResult` + `Assignment` (§4 Internal types)
  - `AsyncHandle[T]` (§4 Internal types; cross-link from Cluster Analysis Service row)
- [ ] **T072** Update `docs/architecture/data_flow.html` — add a Cluster Analysis read-path sub-section.
- [ ] **T073** Add `CHANGES_LOG.md` entry summarizing spec-045 (Repository B0b + AsyncHandle + Force Merge + ≤80 LOC tab).
- [ ] **T074** Add a one-paragraph clarification to `docs/architecture/architecture_standards.md` §B0 about the schema-owning (B0a) vs query-shape (B0b) Repository distinction, citing this spec as the first B0b example. (See `TAB_DESIGN_COMPARISON.html` §"Implications for the framework".)
- [ ] **T075** Run `/code-review` to produce `specs/045-cluster-analysis-tab/REVIEW.md`. Flip spec status `Draft` → `Code Review`.
- [ ] **T076** Address any high-severity REVIEW.md findings. Flip status `Code Review` → `Implemented`.
- [ ] **T077** Commit + push.

**Validation gate (Phase 8 — close-out)**:

```
.venv/Scripts/python -m pytest tests/face_clustering/ tests/architecture/ -q
```

Expected: **full spec-045-touched suite green** (≈ 36 new tests across Repository, Service, architecture). No pre-existing tests turn red. REVIEW.md filed; no findings at severity ≥ High. Spec status flipped Draft → Code Review → Implemented after T076.

---

## Out of scope (explicit deferrals)

- **Refactoring `ClusterView.compute` / `ClusterDebugView.compute` to take typed inputs** instead of `PipelineResult`. Cleanest long-term shape; deferred to a separate compute-layer spec. Tracked as Open Question 5 in `spec.md`.
- **Promoting Cluster Analysis's `current_run_dir` pattern to a typed app-state object** (instead of free-floating `st.session_state` keys). Strong candidate for a future "app-state contract" spec.
- **Removing legacy `app/face_clustering/tabs/cluster_analysis_tab.py`**. Legacy app stays runnable for a burn-in period. Deletion is a follow-up spec.
- **Memoization of `compute_detail_async` / `compute_debug_async`**. Tracked as Open Question 2 in `spec.md`. Recommendation: Service-level memoization keyed on `(run_dir, cluster_id)` once the basic path ships.

## Total effort

**~6.5 hours** across 8 phases. Each phase independently revertible; checkpoints validated by named tests.
