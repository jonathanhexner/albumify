# Feature Specification: Storage Ownership Refactor (RunStore / RunExporter)

**Created**: 2026-05-09
**Status**: In Progress
**Resolves**: SIGHTING-058 (merge log data loss); completes spec-026 Phase 3
**Architecture audit**: [`architecture_audit.html`](architecture_audit.html) (in this spec directory)

## Problem Statement

Face clustering runs persist the same logical information to **2–4 different artifacts on disk**, with no declared owner per fact. Six information types (face↔cluster, cluster metadata, embeddings, crop paths, merge decisions, run config) are spread across **21 storage locations** in **14 files**. This causes three structural failure modes:

1. **Silent schema drift between writers.** `face_cluster.db` `merge_decisions` table has 12 columns; `merge_log.json` has 28 fields. The DB is preferred by the loader, so 5 fields are silently dropped. Result: the FC App shows `"4/4 REJECTED"`, `"Support 191/0"`, `"Diameter 0.972/n/a"`, `"Actual merges: 0"` for a run that actually merged 4 cluster pairs correctly. Reported on run `face_clustering_20260508_000446`.
2. **Producer asymmetry.** Albumify (`sim_bench/pipeline/steps/face_cluster_export.py`) writes both JSON and SQLite. Standalone FC App (`face_cluster/pipeline.py:_export`) writes JSON only. Same logical operation, two implementations, different on-disk layouts.
3. **Existence-check fallbacks as architecture.** `face_cluster/loader.py` contains 9 separate `if x.exists()` branches. Readers in `app/face_clustering/` bypass the loader entirely and `pd.read_csv` files directly. Renaming any artifact requires grep-and-replace across the codebase.

The previous "prefer JSON when DB exists" patch was rejected as a code smell that perpetuates the duplication. This spec eliminates duplication at the source.

## User Stories

### US1 — Open an Albumify-produced run in the FC App without data corruption (Priority: P1)
A user runs the pipeline in Albumify, then opens the resulting run in the standalone Face Clustering App via deep-link.
**Acceptance Criteria**:
- Given an Albumify-produced run with merges, when the FC App loads it, then "Actual merges" matches `run_metadata.n_merges`.
- Given the same run, when viewing C0+C1 in the Merge Analysis tab, then the row shows `MERGED` (not `REJECTED`), `Support` shows the actual `support / required_support`, and `Diameter` shows `post_diameter / max_allowed_diameter` (no `"n/a"`).
- Given the merger config has `merge_margin = 0.0`, when the Margin badge renders, then it displays `"disabled"` instead of `"inf"`.
- Given a pair with `action="passed"` (passed gates but lost iteration tiebreak), when rendered, then the outcome label is `PASSED` (not `REJECTED`).

### US2 — Albumify and FC App produce byte-identical run layouts (Priority: P1)
A run produced by either app is indistinguishable on disk from a run produced by the other.
**Acceptance Criteria**:
- Given the same input (album, config), when run by Albumify and by FC App, then `os.listdir(run_dir)` returns identical sorted file lists.
- Given a run, when checked, then exactly one of these artifacts exists for each information type: faces (DB), clusters (DB), embeddings (.npy), crop images (files), merge decisions (DB), run metadata (DB + small JSON pointer).
- Given any run directory, when listing top-level files, then the names belong to a fixed allow-list: `face_clustering.db`, `embeddings.npy`, `embedding_face_ids.npy`, `pipeline_run.json`, `crops/`. No legacy CSVs, no `merge_log.json`, no `merge_metadata.json`, no `crop_manifest.json`, no `export_summary.json`.

### US3 — All UI tabs go through one read interface (Priority: P1)
Every UI component in `app/` and `app/face_clustering/` retrieves run data through a single `RunStore` class.
**Acceptance Criteria**:
- Given the codebase, when grepping `app/` for `pd.read_csv`, `np.load`, `json.load(`, `sqlite3.connect`, then no matches resolve to a run-directory artifact.
- Given a missing required artifact in a run directory, when `RunStore` is instantiated, then it raises `RunStoreError` with a clear message naming the missing file.
- Given the FC App's 11 tabs, when all are exercised on a fresh run, then every tab renders without falling back to "no data" placeholders.

### US4 — Album-scoped run directory hygiene (Priority: P2)
All face-clustering runs live under their owning album, never as siblings of albums.
**Acceptance Criteria**:
- Given a new run, when written, then its path is `results/<album>/face_clustering/<run_id>/` (not `results/<run_id>/` and not `results/<album>/face_clustering_<run_id>/`).
- Given an old run created before this spec, when opened by either app, then it loads correctly via a backfill migration step (one-time script).
- Given a run produced by remerge or manual merge, when written, then `producer` is recorded as a column in `run_metadata`, not encoded in the directory name.

### US5 — Schema is the contract, fail loud on drift (Priority: P2)
The DB schema is the single source of truth for what fields exist; writers and readers cannot disagree.
**Acceptance Criteria**:
- Given a writer that tries to write an unknown field, when called, then the writer raises (no silent drop).
- Given a reader expecting a field, when the column is missing in DB, then it raises (no silent default to `0`/`None`).
- Given the schema, when bumped, then `pipeline_run.json:schema_version` reflects the new version and `RunStore` refuses to load runs from incompatible older versions without explicit migration.

## Edge Cases

- Old runs written before this spec exist on disk in the legacy layout. A migration script must be able to rebuild them from existing files.
- Apply+Remerge in the FC App writes a child run with a `parent_run_id`. Format must support this, and `RunStore` must resolve cross-run lineage queries.
- Manual-merge snapshot has its own writer (`face_cluster/manual_merge_snapshot.py`); it must use `RunExporter` too.
- During a pipeline run, the DB does not yet exist. UI components that watch a run-in-progress must tolerate this without falling back to scanning files.
- Schema migrations on existing DBs (ADD COLUMN) must be idempotent — opening a run multiple times must not double-apply.
- `embeddings.npy` and `embedding_face_ids.npy` must be loaded as a pair; loading one without the other is an error.

## Requirements (brief)

- **FR-001**: Single class `RunExporter` with method `export(...)` is the only writer of run-directory artifacts. No other module writes to a run directory.
- **FR-002**: Single class `RunStore` is the only reader. Every UI/CLI/notebook accesses run data through it.
- **FR-003**: Run directory contains exactly: `face_clustering.db`, `embeddings.npy`, `embedding_face_ids.npy`, `pipeline_run.json`, `crops/*.jpg`. Nothing else.
- **FR-004**: `face_clustering.db.merge_decisions` table contains all 28 fields of an in-memory merge log entry — no field is dropped on write.
- **FR-005**: `face_clustering.db.run_metadata` table absorbs all fields previously in `merge_metadata.json` and `export_summary.json`.
- **FR-006**: `face_clustering.db.embeddings` table is removed; embeddings live exclusively in `embeddings.npy`.
- **FR-007**: `pipeline_run.json` contains only run-level pointer fields (run_id, source_album, producer, started/finished, status, schema_version). It is never read by code that has DB access.
- **FR-008**: `RunStore` raises on missing artifacts; no silent fallback chains. All `if x.exists()` branches in `loader.py` are removed.
- **FR-009**: All face-clustering runs are written to `results/<album>/face_clustering/<run_id>/`. Reclusters/remerges/manual snapshots live under the same album, distinguished by a `producer` column in `run_metadata`.
- **FR-010**: One-time migration script can rebuild a run directory from any combination of legacy artifacts (CSV/JSON/old DB) into the new layout.
- **FR-011**: UI semantics — outcome labels are MERGED (actually_merged=true), PASSED (action=passed), REJECTED (action=rejected). Margin badge says "disabled" when `merge_margin == 0`.
- **FR-012**: Static guards (tests) prevent regressions: no `.exists()` chain in `RunStore`, no direct file reads in `app/`, run directory listdir matches allow-list.
