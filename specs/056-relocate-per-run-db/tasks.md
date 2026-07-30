# Tasks: Relocate per-run-DB layer (056)

Legend: `[ ]` open · `[>]` in progress · `[x]` done · `[~]` skipped

---

## Phase 0 — Audit (~30 min)

- [ ] **T001** Grep the working tree for all imports of the four target files. Record hit counts per file:
  - `from face_cluster.run_exporter` / `import face_cluster.run_exporter`
  - `from face_cluster.run_store` / `import face_cluster.run_store`
  - `from face_cluster.db.schema` / `from face_cluster.db import schema`
  - `from face_cluster.repositories.cluster_analysis_repo` / `from face_cluster.repositories import cluster_analysis_repo`

  Expected: ~35-55 total hits.

- [ ] **T002** Grep for monkeypatch targets pinning old module paths (e.g., `monkeypatch.setattr("face_cluster.run_store..."`)). Add to migration list.

- [ ] **T003** Create arch test `tests/architecture/test_no_face_cluster_per_run_db_paths.py` with allow-list initially containing the four target paths. PASSES today.

**Gate**: T003 passes against current tree.

---

## Phase 1 — PR 1: move `db/schema.py` (~20 min)

- [ ] **T010** `git mv face_cluster/db/schema.py sim_bench/run_db/_schema.py`. Create `sim_bench/run_db/__init__.py` (empty) if it doesn't exist.
- [ ] **T011** Update every T001-listed importer of `face_cluster.db.schema` → `sim_bench.run_db._schema`. Handles both `from X import …` and `from … import X` forms.
- [ ] **T012** Remove `face_cluster.db.schema` from arch test allow-list.
- [ ] **T013** `pytest tests/ -q`. All green. PR-1 mergeable.

---

## Phase 2 — PR 2: move `repositories/cluster_analysis_repo.py` (~20 min)

- [ ] **T020** Create `sim_bench/db/__init__.py` (empty) and `sim_bench/db/face_clustering/__init__.py` (empty). `git mv face_cluster/repositories/cluster_analysis_repo.py sim_bench/db/face_clustering/cluster_analysis_repo.py`.
- [ ] **T021** Update importers (~5 hits).
- [ ] **T022** Remove `face_cluster.repositories.cluster_analysis_repo` from arch test allow-list.
- [ ] **T023** `pytest tests/ -q`. All green.

---

## Phase 3 — PR 3: move `run_store.py` (~60-90 min, biggest blast radius)

- [ ] **T030** `git mv face_cluster/run_store.py sim_bench/run_db/store.py`.
- [ ] **T031** Update importers (~15-25). Hot spots:
  - `app/face_clustering_v2/tabs/` (multiple)
  - `app/face_clustering_v2/components/` (multiple)
  - `app/streamlit/` (history + run views)
  - `sim_bench/pipeline/steps/` (face_cluster_bridge, others)
  - `face_cluster/fc_app_runner.py`
  - `tests/face_clustering/run_store/`
  - `tests/architecture/`
- [ ] **T032** Update monkeypatch string literals from T002.
- [ ] **T033** Remove `face_cluster.run_store` from arch test allow-list.
- [ ] **T034** `pytest tests/ -q`. All green.
- [ ] **T035** Restart Streamlit; smoke-test Cluster Analysis tab + History tab in `face_clustering_v2` (heavy RunStore users). No console errors.

---

## Phase 4 — PR 4: move `run_exporter.py` (~40-60 min)

- [ ] **T040** `git mv face_cluster/run_exporter.py sim_bench/run_db/exporter.py`.
- [ ] **T041** Update importers (~10-15). Hot spots:
  - `sim_bench/pipeline/steps/face_cluster_export.py`
  - `face_cluster/fc_app_runner.py`
  - `tests/face_clustering/` (multiple)
- [ ] **T042** Remove `face_cluster.run_exporter` from arch test allow-list (now empty).
- [ ] **T043** `pytest tests/ -q`. All green.
- [ ] **T044** Create equivalence test `tests/run_db/test_relocate_equivalence.py`: hash the output of `RunExporter.export()` against the synthetic fixture; assert equal to a pre-move recorded baseline.

---

## Phase 5 — Notebooks + docs sweep (~30 min)

- [ ] **T050** Grep `notebooks/` for any of the four old paths. Update actively-used ones (`notebooks/face_clustering/eda_merge_explore.ipynb`, `eda_merge_ml.ipynb`).
- [ ] **T051** Update `docs/architecture/{classes.html, data_flow.html, db_schemas.html, index.html, overview.md}` — every mention of the old paths becomes the new path.
- [ ] **T052** Verify `specs/057/EXECUTIVE_REVIEW_057_059.html` package tree is already updated (done in this work).

**Gate**: visual sweep. No old paths remain.

---

## Phase 6 — Close-out (~30 min)

- [ ] **T060** Mutation test the arch guard: temporarily reintroduce one old path in a throwaway file; confirm arch test red; revert.
- [ ] **T061** CHANGES_LOG entry under `[REFACTOR]`. Note the 4 PRs and the import-path migration.
- [ ] **T062** `/code-review` → `REVIEW.md`. Address any high-severity findings.
- [ ] **T063** Spec-056 status → `Implemented`. Commit + push.

**Final gate**: 7 ACs from spec.md all green; full suite green; arch test green with empty allow-list.

---

## Test delta

| Phase | Added | Deleted | Net |
|---|---|---|---|
| 0 | 1 (arch guard) | 0 | +1 |
| 1-3 | 0 | 0 | 0 (pure migration) |
| 4 | 1 (relocate equivalence) | 0 | +1 |
| **Total** | **2** | **0** | **+2** |

---

## Total estimate

**~2-3 hours** across 4 PRs. Pure grep+replace; biggest cost is running tests after each PR.
