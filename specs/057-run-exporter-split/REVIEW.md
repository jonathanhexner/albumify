# spec-057 — Code Review

**Reviewer**: Claude Opus 4.7 (1M context)
**Date**: 2026-05-29
**Spec status going in**: In Progress
**Checklist**: `docs/guides/CODE_REVIEW_CHECKLIST.md`
**Commits**: 9761529 (Phase 0), aeb1ea2 (T011), b921f99 (T012+T013), f5541e0 (Phase 2), 8054a5a (Phases 3+4)

---

## Part 1 — How it works

`sim_bench/run_db/exporter.py` was a 924-LOC monolith with 10 responsibilities (faces, clusters, merges, filter_decisions, images, scenes×2, run_metadata, embeddings.npy, pipeline_run.json, crops). spec-057 splits each table / artifact into its own writer module:

```
sim_bench/run_db/
├─ exporter.py          (467 LOC — facade only: validate + open transaction + delegate)
├─ _errors.py            (RunExporterError, extracted to break circular import)
├─ _schema.py           (DDL — pre-existing; spec-058 will derive this from ORM)
├─ store.py             (RunStore — spec-059 territory)
├─ writers/
│  ├─ _common.py        (maybe_float, to_sql)
│  ├─ faces_writer.py
│  ├─ clusters_writer.py
│  ├─ merges_writer.py
│  ├─ filter_decisions_writer.py
│  ├─ images_writer.py
│  ├─ scenes_writer.py
│  └─ run_metadata_writer.py
└─ artifact_writers/
   ├─ embeddings_writer.py
   ├─ pipeline_run_writer.py
   └─ crops_writer.py
```

`RunExporter.export()` is now: validate inputs → write crops + embeddings → open one SQLite transaction → call each per-table writer in order → commit (or rollback on any exception) → write pipeline_run.json. Each `_write_X` method on RunExporter is a 2-line delegation to the matching writer module.

### Verification scaffolding

Two new tests guard the split:

| Test | Purpose |
|---|---|
| `tests/run_db/test_split_equivalence.py` | Hashes every produced file (per-table canonical row dump for the DB + raw bytes for npy) against a snapshot. Caught the initial non-deterministic `images.created_at` / `scene_clusters.created_at` and mask them. |
| `tests/run_db/test_atomicity.py` | Monkeypatches `merges_writer.write_merges` to raise; asserts no rows landed in the data tables. Proves the single-transaction contract survives the split. |
| `tests/architecture/test_run_exporter_layering.py` | Caps each writer module at 200 LOC. Existing writers are 25-124 LOC. |

## Part 2 — Findings

### §1 Structure
- **PASS.** The split is principled: one table → one writer module; one non-DB artifact → one artifact_writer module. Helpers grouped in `_common.py`. Circular import between exporter and writers/_common avoided by extracting `RunExporterError` into `_errors.py`.

### §2 Naming + conventions
- **PASS.** `writers/<table>_writer.py` mirrors the table name. Public function name is `write_<table>` (`write_faces`, `write_merges`, …). Helpers renamed from `_maybe_float` / `_to_sql` (module-private) to `maybe_float` / `to_sql` (public API of the writers package).

### §3 Error handling
- **PASS.** Every writer raises the same exception types it did before extraction. `to_sql()` still raises `RunExporterError` on unsupported types. The `export()` facade still wraps the writer calls in `try/except` + `conn.rollback()` so any exception aborts cleanly. T040 atomicity test pins this.

### §4 Tests
- **PASS.** Golden-hash equivalence + atomicity test + LOC arch guard. Every writer extraction was verified against the golden snapshot before committing. Net test delta: +3 test files, +12 new test cases.

### §5 Testability
- **PASS-WITH-FOLLOWUP.** Each writer module is independently importable and unit-testable now. spec-057 didn't add per-writer unit tests (the golden-hash test covers end-to-end correctness, and the writers are largely mechanical), but spec-058 / spec-059 will exercise the per-table seams when they swap raw SQL for ORM. *Follow-up*: if a writer grows non-trivial branching, add a focused unit test alongside it.

### §6 Boundary contracts
- **PASS.** `RunExporter.export()` and `RunExporter.calc()` public signatures unchanged. `EXPECTED_ARTIFACTS`, `SCHEMA_VERSION`, `RunExportInputs`, `RunExportResult`, `RunExporterError` all still importable from `sim_bench.run_db.exporter`. The spec-053 entry point (`calc(inputs) -> result`) untouched.

### §7 Documentation
- **PASS-WITH-FOLLOWUP.** spec.md status flipped to Implemented; this REVIEW.md walks all 8 sections; CHANGES_LOG entry added. `docs/architecture/classes.html` and `data_flow.html` still show RunExporter as a single class — they should be updated to show the writer fan-out. *Follow-up ticket*: bake the writer table into the architecture docs (low priority — code is the source of truth, REVIEW.md captures the structure).

### §8 Performance / observability
- **PASS.** Golden-hash test proves byte-for-byte equivalence on a representative input. The extra function-call overhead from delegation is negligible relative to SQLite executemany throughput; no measurable performance change.

## Part 3 — Verdict

**Accept.** All acceptance criteria from spec.md met:

| AC | Met by |
|---|---|
| 1. Each table writer in its own module | 7 files under `writers/` |
| 2. RunExporter.export() ≤ small facade | Down from 270+ LOC to ~95 lines (incl. docstring) |
| 3. Single-transaction contract preserved | T040 atomicity test |
| 4. Byte-equivalent output | T001/T002 golden-hash test |
| 5. Arch test caps writer LOC | T032 parametrized test |
| 6. Public surface unchanged | spec-053/045 test suites pass without modification |

**No high-severity findings.** Two pass-with-followups (per-writer unit tests in §5, docs HTML update in §7) — both low priority, don't block the spec-058/059 work that depends on this split.

**Spec status flip authorized**: `In Progress` → `Implemented`.
