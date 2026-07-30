# spec-056 — Code Review

**Reviewer**: Claude Opus 4.7 (1M context)
**Date**: 2026-05-29
**Spec status going in**: In Progress (4 PRs landed: f4e5dbb, ee7d8fd, a294899, d91e881)
**Checklist**: `docs/guides/CODE_REVIEW_CHECKLIST.md`

---

## Part 1 — How it works

Spec-056 was a four-PR relocation of the per-run-DB layer from
`face_cluster/` (algorithm code) to `sim_bench/` (infrastructure code).

```
BEFORE                                            AFTER
face_cluster/                                     face_cluster/
├─ db/schema.py             ──── PR1 ────►        ├─ db/__init__.py    (validators only)
├─ run_store.py             ──── PR3 ────►        └─ db/validators.py  (unchanged)
├─ run_exporter.py          ──── PR4 ────►
└─ repositories/                                  sim_bench/
   └─ cluster_analysis_repo.py PR2 ────►          ├─ run_db/
                                                  │  ├─ _schema.py   (was face_cluster/db/schema.py)
                                                  │  ├─ store.py     (was face_cluster/run_store.py)
                                                  │  └─ exporter.py  (was face_cluster/run_exporter.py)
                                                  └─ db/face_clustering/
                                                     └─ cluster_analysis_repo.py
```

PRs landed smallest-blast-radius first to de-risk the migration:

| PR | Commit | Files moved | Importers updated |
|---|---|---|---|
| PR1 | f4e5dbb | db/schema.py | 5 direct + 4 indirect (face_cluster.db re-export) |
| PR2 | ee7d8fd | cluster_analysis_repo.py | 6 |
| PR3 | a294899 | run_store.py | 10 (incl. test_no_direct_run_artifact_reads error text) |
| PR4 | d91e881 | run_exporter.py | 11 (incl. inline EXPECTED_ARTIFACTS migrations) |

## Part 2 — Findings

### §1 Structure (file org / single-step files / no fan-out churn)
- **PASS.** Each PR is a single-purpose file rename. No file gained or lost responsibilities. `face_cluster/db/__init__.py` was simplified (dropped schema re-exports; validators stay) which is a net structural improvement.

### §2 Naming + conventions
- **PASS.** New file names follow established conventions: `_schema.py` (private to package) mirrors the prior file role; `store.py` and `exporter.py` drop the `run_` prefix because they live under `run_db/` (DRY). `sim_bench/db/face_clustering/` mirrors a future `sim_bench/db/image_clustering/`.

### §3 Error handling + edge cases
- **PASS.** No new error paths were introduced. All `RunStoreError`, `RunExporterError`, `ValidationError` paths preserved verbatim.

### §4 Tests
- **PASS.** 273 tests green across the touched suites after PR4 (`tests/architecture` + `tests/face_clustering/{test_run_store, test_run_exporter, test_filter_context_p2_export, test_helpers_calc_equivalence, test_sighting_058_regression, test_schema_v5_writes, test_v2_app_smoke, repositories, views}`). The new arch test (`tests/architecture/test_no_face_cluster_per_run_db_paths.py`) parametrizes over all 4 legacy paths and fails the build if any reappear; allow-list ends empty.
- **PASS.** spec-056 explicitly is a behavior-preserving refactor; no new behavior tests needed.

### §5 Testability
- **PASS.** Pure rename, so the testable surface is unchanged. Monkeypatch fixtures that referred to module attributes by reference (e.g., `monkeypatch.setattr(run_store_module, "RunStore", ...)`) keep working because Python's import binding is unchanged; only the dotted path changed.

### §6 Boundary contracts
- **PASS.** Public surface frozen: `RunStore`, `RunStoreError`, `RunMetadata`, `EmbeddingMatrix`, `FilterDecisionRow`, `RunExporter`, `RunExporterError`, `RunExportInputs`, `RunExportResult`, `ClusterAnalysisRepository`, `ClusterAnalysisRepoConfig`, `ClusterAnalysisCriteria` — all keep their signatures and return shapes. Tests imported them from the new paths without source changes other than the `from X import …` line.
- **PASS-WITH-FOLLOWUP.** `sim_bench/run_db/exporter.py` still re-exports `EXPECTED_ARTIFACTS` and `SCHEMA_VERSION` via its `__all__`. Callers that historically reached these via `face_cluster.run_exporter` were migrated to `sim_bench.run_db._schema` directly in PR4; the re-export now has no external consumers. Spec-057 should drop the re-export when it slims the facade.

### §7 Documentation
- **PASS.** `docs/architecture/{classes.html, data_flow.html, db_schemas.html, architecture_standards.md}` all updated to the new paths. The HTML docs auto-search-and-replace cleanly (no semantic drift in surrounding prose).

### §8 Performance / observability
- **N/A.** Pure rename; no runtime behavior changed.

## Part 3 — Verdict

**Accept.** All 7 acceptance criteria from `spec.md` are met:
1. ✅ Four files relocated by `git mv`
2. ✅ All importers updated; no `face_cluster.{run_store,run_exporter,db.schema,repositories.cluster_analysis_repo}` paths remain in source (arch test enforces)
3. ✅ Public surface unchanged
4. ✅ Test suite green
5. ✅ Architecture docs updated
6. ✅ Per-PR commits with mergeable boundaries
7. ✅ Allow-list arch test shrinks to empty after PR4

**No high-severity findings. One pass-with-followup** in §6 (legacy re-exports
in `exporter.py`) is tracked into spec-057 scope (RunExporter facade slim-down).

**Spec status flip authorized**: `In Progress` → `Implemented`.
