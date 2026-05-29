# spec-056 — Relocate per-run-DB layer to `sim_bench/`

**Created**: 2026-05-29
**Status**: Implemented (2026-05-29)
**Predecessors**: spec-053 (RunExportInputs boundary), spec-054 (SCHEMA_HISTORY)
**Successors**: spec-057 (RunExporter split), spec-058 (per-run ORM models), spec-059 (RunStore + CARepo on SQLAlchemy)
**Trigger**: Review of specs 057-059 surfaced that the per-run-DB layer (which holds `images` + `scene_clusters` + `run_metadata` in addition to face-clustering tables) is broader than face-clustering — it logically belongs under `sim_bench/`, and albumify will eventually consume it. Relocating BEFORE the refactor work avoids a two-valid-paths window and keeps each subsequent spec single-purpose.

---

## Problem

Four files in `face_cluster/` implement the per-run-DB layer:

| File | LOC | Role |
|---|---|---|
| `face_cluster/run_exporter.py` | 826 | Write path — produces `face_clustering.db` |
| `face_cluster/run_store.py` | 546 | Read path |
| `face_cluster/db/schema.py` | 256 | DDL constants + `SCHEMA_VERSION` + `SCHEMA_HISTORY` |
| `face_cluster/repositories/cluster_analysis_repo.py` | 305 | Clustering-domain repository |

Of the 9 per-run-DB tables, only 5 are face-clustering (`faces`, `clusters`, `cluster_assignments`, `merge_decisions`, `filter_decisions`). Four are broader (`images`, `scene_clusters`, `scene_cluster_assignments`, `run_metadata`). Albumify will consume `images` + `scene_clusters` directly. The `face_cluster/` package name overpromises ownership.

Doing this move BEFORE spec-057/058/059 means each subsequent spec operates at the canonical path. No shim window, no two-valid-paths confusion, no `face_cluster.X` vs `sim_bench.run_db.X` ambiguity during development.

---

## What we build

Pure file relocation + importer migration. **Zero behavior change.** Byte-equivalent output dirs; byte-equivalent test results.

Four moves:

| From | To |
|---|---|
| `face_cluster/run_exporter.py` | `sim_bench/run_db/exporter.py` |
| `face_cluster/run_store.py` | `sim_bench/run_db/store.py` |
| `face_cluster/db/schema.py` | `sim_bench/run_db/_schema.py` |
| `face_cluster/repositories/cluster_analysis_repo.py` | `sim_bench/db/face_clustering/cluster_analysis_repo.py` |

Plus the importer sweep across ~35-55 call sites in `sim_bench/pipeline/`, `app/face_clustering*/`, `app/streamlit/`, `face_cluster/` (non-moved code), `tests/`, and `scripts/`.

---

## What we don't build

- **No refactor.** The monolithic 826-LOC RunExporter stays monolithic. Spec-057 splits it AT THE NEW LOCATION.
- **No ORM.** Raw SQL stays raw SQL. Specs 058/059 add ORM.
- **No subdomain split.** All ORM models will land in `sim_bench/run_db/models/` (flat namespace) per spec-058. Future albumify spec may revisit.
- **No deletion of `face_cluster/` algorithm code.** Only the four files above move. `face_cluster/repositories/_base_repository.py`, `_session.py`, `_engine.py`, `_errors.py`, `_orm_base.py`, `_schema.py`, `run_history_repo.py` all stay — they belong to the `sim_bench.db` scope, not the per-run DB.
- **No shim files.** The whole point is to avoid the shim window.

---

## Locked decisions

1. **Atomic move per file.** Four PRs, one per file move, ordered smallest-blast-radius first: `schema.py` → `cluster_analysis_repo.py` → `run_store.py` → `run_exporter.py`. Each PR: move + sweep + verify green.
2. **No deprecation shim.** If a PR breaks the build, fix it in the PR; don't paper over with a shim.
3. **Test imports follow code imports.** Every test file that imports the moved module gets updated in the same PR.
4. **Schema values keep their values.** `SCHEMA_VERSION=5` stays. `SCHEMA_HISTORY` content stays. `EXPECTED_ARTIFACTS` stays. This is import-path migration only.
5. **`sim_bench/db/face_clustering/` exists pre-emptively.** Spec-059's ClusterAnalysisRepository goes there. Creating the package now (with `__init__.py`) is cheaper than at 059.
6. **Arch test is a deny-list.** `tests/architecture/test_no_face_cluster_per_run_db_paths.py` greps for any of the four old paths. Cheaper than `__getattr__`-based deprecation warnings.

---

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `sim_bench/run_db/{exporter.py, store.py, _schema.py}` exist with file-identical content to the originals | `git diff --no-index` |
| AC2 | `sim_bench/db/face_clustering/cluster_analysis_repo.py` exists likewise | `git diff --no-index` |
| AC3 | All four original `face_cluster` files deleted | `git ls-files` |
| AC4 | Zero hits in the working tree for the old import paths | arch test `test_no_face_cluster_per_run_db_paths.py` |
| AC5 | `pytest tests/ -q` — all green; spec-040/045/049/053/054 baselines unchanged | run |
| AC6 | Output of `RunExporter.export()` is byte-identical pre- and post-move on the synthetic fixture (spec-053 fixture) | new test `tests/run_db/test_relocate_equivalence.py` |
| AC7 | Architecture HTMLs (`classes.html`, `data_flow.html`, `db_schemas.html`, `overview.md`) updated to reference new paths | grep + visual |

---

## Risks

| Risk | Mitigation |
|---|---|
| Importer sweep misses a call site; runtime `ImportError` at first user touch | AC4 arch test runs in CI; greps the entire tree for old paths. Cannot land otherwise. |
| Notebooks under `notebooks/` import via old paths and break silently | Phase 5 sweeps `notebooks/`. Annotate or update actively-used ones (`notebooks/face_clustering/`). |
| Circular import during move (e.g., `sim_bench.run_db.exporter` imports `face_cluster.something` which transitively imports back) | Per-PR atomicity bounds blast radius. T001 audit checks for this pre-move. Revert PR if it fires. |
| Test fixtures pin old module paths (e.g., `monkeypatch.setattr("face_cluster.run_store.RunStore._connect", ...)`) | T002 greps for monkeypatch string literals and adds them to the migration list. |

---

## Effort estimate

**~2-3 hours total** across 4 PRs. Pure grep+replace; biggest cost is running tests after each PR.

- PR 1 (`schema.py`): ~20 min
- PR 2 (`cluster_analysis_repo.py`): ~20 min
- PR 3 (`run_store.py`): ~60-90 min — biggest blast radius
- PR 4 (`run_exporter.py`): ~40-60 min

---

## Final binding structure (post-056)

**Binding.** Mirrored in `specs/057/EXECUTIVE_REVIEW_057_059.html` §9 and in specs 057/058/059. Drift = Code Review §1 fail.

### Package layout
```
sim_bench/run_db/                             # NEW per-run-DB layer
├─ exporter.py                                # ← face_cluster/run_exporter.py (intact, 826 LOC)
├─ store.py                                   # ← face_cluster/run_store.py (intact, 546 LOC)
└─ _schema.py                                 # ← face_cluster/db/schema.py (intact, 256 LOC)

sim_bench/db/face_clustering/                 # NEW clustering-domain scope
├─ __init__.py                                # empty marker
└─ cluster_analysis_repo.py                   # ← face_cluster/repositories/cluster_analysis_repo.py (intact, 305 LOC)

face_cluster/                                 # algorithm-only after this spec
├─ db/
│  ├─ __init__.py
│  └─ validators.py                           # unchanged (face-clustering validators only)
│  # schema.py — DELETED (moved to sim_bench/run_db/_schema.py)
├─ repositories/
│  ├─ _base_repository.py · _session.py · _engine.py · _errors.py
│  ├─ _orm_base.py · _schema.py
│  └─ run_history_repo.py
│  # cluster_analysis_repo.py — DELETED (moved to sim_bench/db/face_clustering/)
│ # run_exporter.py — DELETED (moved to sim_bench/run_db/exporter.py)
│ # run_store.py — DELETED (moved to sim_bench/run_db/store.py)
├─ fc_params.py · fc_app_runner.py · pipeline.py · run_layout.py
├─ analysis.py · exemplar/ · clustering algorithms...
```

### Classes
No new classes. Every class keeps its name and signature:

| Class | New module |
|---|---|
| `RunExporter` (+ `RunExportInputs`, `RunExportResult`, `RunExporterError`, `_VALID_PRODUCERS`) | `sim_bench/run_db/exporter.py` |
| `RunStore` | `sim_bench/run_db/store.py` |
| `ClusterAnalysisRepository` (+ `ClusterAnalysisRepoConfig`) | `sim_bench/db/face_clustering/cluster_analysis_repo.py` |

DDL constants, `SCHEMA_VERSION`, `SCHEMA_HISTORY`, `EXPECTED_ARTIFACTS` keep their names; only their module path changes.

### Cross-spec invariants
1. **One canonical path per class.** No two-valid-paths window. After 056 lands, importing from `face_cluster.{run_exporter, run_store, db.schema, repositories.cluster_analysis_repo}` fails with `ImportError`. That IS the contract.
2. **Zero behavior change.** Byte-equivalent test results. File content unchanged.
3. **Subsequent specs operate at the new location.** Spec-057 splits `sim_bench/run_db/exporter.py`; spec-058 adds ORM at `sim_bench/run_db/{_base.py, _session.py, models/}`; spec-059 flips `sim_bench/run_db/store.py` and `sim_bench/db/face_clustering/cluster_analysis_repo.py` to ORM.
4. **`face_cluster/` is algorithm-only going forward.** Per-run-DB additions go to `sim_bench/run_db/` (entities + writers + RunStore) or `sim_bench/db/face_clustering/` (clustering-domain repositories).

---

## Definition of Done

- All 7 ACs green
- 4 PRs merged
- Full test suite green
- Spec-056 status → `Implemented`
- CHANGES_LOG entry under `[REFACTOR]`
