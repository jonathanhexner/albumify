# Tasks: RunStore + ClusterAnalysisRepository on SQLAlchemy (059)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

**Hard prerequisite**: spec-058 lands first. ORM models under `face_cluster/db/models/` must exist with green drift-guard tests.

## Phase 0 — Performance baseline (~30 min)
- [ ] **T001** Create `tests/face_clustering/repositories/test_cluster_analysis_repo_perf.py`. Micro-bench: 1000 `get_cluster_rows()` calls against the synthetic fixture; record mean time. Commit the baseline number to the test file as `BASELINE_MS = <today's mean>`. AC7 bounds the refactor to 1.2× this baseline.

## Phase 1 — Per-run-DB session infrastructure (~30 min)
- [ ] **T010** `sim_bench/run_db/_session.py` — `make_run_db_sessionmaker(run_dir)`: builds an Engine on the per-run DB, returns a sessionmaker. Per-call connections; no pooling.
- [ ] **T011** Unit test: open + close 100 sessions in a loop, assert no connection leaks (verify via `engine.dispose()` semantics).

**Gate**: T011 passes.

## Phase 2 — ClusterAnalysisRepository migration (~1 h)
- [ ] **T020** Replace the 4 raw SQL statements in `sim_bench/db/face_clustering/cluster_analysis_repo.py` with `select(...)` calls against spec-058's ORM models. Method signatures unchanged.
- [ ] **T021** Run spec-045's full test suite (`pytest tests/face_clustering/repositories/test_cluster_analysis_repo_*.py tests/face_clustering/views/test_cluster_analysis_service_*.py -q`). All 31 tests pass without modification.
- [ ] **T022** Run the perf benchmark from T001. Within 1.2× baseline.

**Gate**: 31 tests green + perf within 1.2× baseline.

## Phase 3 — RunStore migration (~3 h)
- [ ] **T030** `RunStore.faces()` — convert to `select(Face).order_by(Face.face_id)` + hydration to `FaceRecord`. Preserve None-handling on nullable columns.
- [ ] **T031** `RunStore.clusters(iteration)` — convert. Preserve the noise-cluster handling.
- [ ] **T032** `RunStore.metadata()` — convert. JSON deserialization on `config_json` / `merge_thresholds_json` stays as today.
- [ ] **T033** `RunStore.merge_log()` — convert. `_BOOL_MERGE_FIELDS` int→bool conversion handled by Mapped types (verify in test).
- [ ] **T034** `RunStore.image_detail(path)` — convert. This is the heaviest method (joins faces + filter_decisions + cluster_assignments). Take it last.
- [ ] **T035** `RunStore.embeddings()`, `RunStore.filter_decisions()` — convert.

**Gate after each task**: `pytest tests/face_clustering/run_store/ -q` → green. Never red for > 1 commit.

## Phase 4 — Arch guards (~30 min)
- [ ] **T040** Arch test: `tests/architecture/test_no_raw_sql_in_run_store_or_sim_bench/db/face_clustering/cluster_analysis_repo.py`. Greps for `SELECT|INSERT INTO|UPDATE |DELETE FROM ` (uppercase, with trailing space) in the two files. Both must be empty.
- [ ] **T041** LOC arch test extension: `RunStore` ≤ 450 LOC; `ClusterAnalysisRepository` ≤ 200 LOC.

**Gate**: T040 + T041 green.

## Phase 5 — Cleanup + close-out (~1 h)
- [ ] **T050** Update `docs/architecture/classes.html` — `RunStore` and `ClusterAnalysisRepository` rows mention SQLAlchemy backing.
- [ ] **T051** Update `docs/architecture/data_flow.html` — read path nodes show ORM models instead of raw SQL.
- [ ] **T052** Update `docs/architecture/architecture_standards.md` §B0.2.1 — note that B0b Repositories now share spec-058's ORM models.
- [ ] **T053** spec-045's `CODE_REVIEW_SUMMARY.html` SMELL-1 entry: mark resolved with cross-reference to this spec.
- [ ] **T054** CHANGES_LOG entry.
- [ ] **T055** `/code-review` → REVIEW.md. Address any high-severity findings.
- [ ] **T056** Spec status → `Implemented`. Commit + push.

**Final gate**: full suite (`pytest tests/face_clustering/ tests/architecture/ -q`) green; perf within 1.2× baseline.

## Total estimate
**~5-6 hours.** Phase 3 dominates (one RunStore method per ~30 min, including the test cycle).
