# V5 Read-Path Audit — Findings

> Each row: a file:line that matched one of the 10 categories in
> `AUDIT_CHECKLIST.md`. Status is `NO-OP` (already correct), `FIX`
> (broken; fix in this spec), `SIGHTING` (broken but out of scope —
> file numbered sighting), `DEFERRED` (known issue, intentionally
> not addressed now).

---

## Run #1 — 2026-05-30, Phases 0-2

**Scope walked**: `app/face_clustering_v2/`, `face_cluster/views/`,
`face_cluster/repositories/`, `sim_bench/db/face_clustering/`,
`sim_bench/run_db/`, `face_cluster/loader.py`.

**Yield**: 8 findings across 10 categories. 2 NO-OP / 1 NEW SIGHTING /
1 OPEN-AT-ROOT (SIGHTING-078) / 4 SKIP (out-of-scope or display-only).

| ID | File:line | Category | Pattern | v5-compat? | Status | Notes |
|---|---|---|---|---|---|---|
| F01 | `face_cluster/views/history.py:248-250, 264, 507` + `load_button.py:43` | 1 (legacy CSV refs) | `_LEGACY_CSV_ARTIFACTS` constant + docstring + error messages mention `faces.csv` / `clusters.csv` | OK | **NO-OP** | Already correct after SIGHTING-080 fix — the constant is now used as one of three valid layouts via `_run_dir_has_loadable_artifacts()`. Reading the rows confirms each is correctly v5-aware. |
| F02 | `sim_bench/run_db/store.py:610` | 2 (`RunStore.<method>("final")`) | `RunStore._resolve_iteration` calls `iteration_count()` which queries the wrong table | BROKEN | **SIGHTING-078** (already filed, root fix open) | Workaround shipped in spec-045 (Repository resolves "final" locally). Root fix still belongs in RunStore — touches spec-056 territory. Other callers of `RunStore.clusters("final")` outside the audit scope (e.g., legacy `app/face_clustering/`) still hit this. |
| F03 | `sim_bench/run_db/store.py:597` | 3 (`MAX(iteration) FROM merge_decisions`) | The literal SQL inside `RunStore.iteration_count()` | BROKEN | **SIGHTING-078** (same finding as F02) | Root location of SIGHTING-078. One-line fix once spec-056 work settles: change to `MAX(iteration) FROM clusters` or `MAX(MAX(...)) ` across both tables. |
| F04 | `app/face_clustering_v2/components/cluster_metrics.py:7` + `cluster_analysis_tab.py:57` | 4 (AsyncHandle in v2 UI) | Comments referencing SIGHTING-079 — NO actual usage | OK | **NO-OP** | Verified: only comments, no live `AsyncHandle` / `threading.Thread` calls in the v2 UI layer after the SIGHTING-079 fix. |
| F05 | `sim_bench/db/face_clustering/cluster_analysis_repo.py:326` | 7 (direct `sqlite3.connect`) | `_connect()` opens the per-run DB without `PRAGMA user_version` check | OK | **NO-OP** | Safe because `__init__` constructs `RunStore` first, which validates `PRAGMA user_version == SCHEMA_VERSION`. Drift-guard candidate: a future arch test could enforce "every Repository that calls `sqlite3.connect` must also instantiate a RunStore in `__init__`." |
| F06 | `face_cluster/views/history.py:529, 530, 539, 540` | 8 (`pipeline_run.json` field reads) | `_summary_from_pipeline_run` reads `summary`, `stages`, `merge_metadata`, `merge_log` keys | **BROKEN (degraded UX)** | **SIGHTING (new — file SIGHTING-089)** | v5 `pipeline_run_writer.py` writes only 9 top-level keys: `run_id, source_album, producer, parent_run_id, started_at, finished_at, status, schema_version, db_path`. The 4 keys History reads are **absent** in v5 runs. Result: History run-detail panel renders a sparse RunSummary (no n_faces, no n_clusters, no stage timings, no config) for every v2 run. No crash — `.get()` returns defaults — but the user sees blanks where the legacy panel showed numbers. |
| F07 | `face_cluster/views/history.py:276` | 9 (`_v4/` reference) | Checks for `_v4/face_clustering.db` as one of 3 valid layouts | OK | **NO-OP** | Correct — SIGHTING-080 fix added this fallback intentionally. |
| F08 | `sim_bench/pipeline/steps/face_cluster_export.py:131` | 9 (`_v4/` reference) | **Albumify export step still writes face_clustering.db to `_v4/` subdir**, not v5 top-level | OUT OF SCOPE (write path) | **DEFERRED** | This is a write-side asymmetry, not a v2 read-path issue. Out of this spec's scope (read paths only). But worth a separate sighting: if Albumify pipelines land in History, the load path now handles `_v4/` so OK; the only "bug" is that Albumify and fc_app_v2 produce different on-disk shapes, which is confusing. |

### Categories with ZERO findings (clean)

| Category | Conclusion |
|---|---|
| 6 — Producer-name string equality | No v2 code filters by producer name. Either by-design or untested code path. |
| 10 — `listdir` / `glob` on run dirs | Only hit is `_profile_bar.py` (profile JSON discovery — unrelated). v2 read paths do not list run dirs. |

### Patterns that simplified the picture

- **Categories 1, 4, 7**: previous sightings (078, 079, 080) already shipped their fixes correctly. The audit confirms no regression.
- **Categories 2, 3**: both point at the same root (SIGHTING-078 in RunStore). One real bug, two grep signatures.
- **Category 5**: `_LEGACY_CSV_ARTIFACTS` is the only schema-claiming constant left in v2 code, and it's used correctly.
- **Category 6**: zero hits is the safe state — no v2 path silently excludes runs by producer.

---

## Decision summary

| Status | Count | Action |
|---|---|---|
| NO-OP (already correct) | 4 (F01, F04, F05, F07) | Document as drift-guard candidates; no code change |
| Already filed (SIGHTING-078) | 2 (F02, F03 — same root) | No new sighting; tracks in existing one |
| New sighting | 1 (F06) | **File SIGHTING-089**: v5 pipeline_run.json missing legacy summary keys; History run-detail panel sparse |
| Out of scope (write side) | 1 (F08) | Note in this audit; no separate sighting (Albumify writer asymmetry is its own concern) |

### Proposed FIX-status items for Phase 3 of this spec

**None.** The audit found one new bug (F06 → SIGHTING-089) but it's a
write-side problem (the v5 `pipeline_run.json` writer drops keys the
reader expects). Fixing it in the reader alone is brittle; the real fix
is to either (a) expand the writer to include the keys, or (b) point the
reader at `run_metadata` table instead. Both are bigger than this spec —
file SIGHTING-089 and handle separately.

### Recommended drift-guard tests (for a follow-up spec)

1. **Layout-loadability symmetry**: `_run_dir_has_loadable_artifacts(out)`
   must accept every layout `load_pipeline_result(out)` can read.
   (Would have caught SIGHTING-080.)
2. **Per-run-DB sqlite3 callers validate schema**: every file that calls
   `sqlite3.connect` on a per-run DB must either construct a `RunStore`
   first or call `PRAGMA user_version` explicitly. (Would have caught
   the latent F05 risk.)
3. **pipeline_run.json reader/writer field parity**: every key the
   reader does `prun.get("X")` on must appear in the writer's payload.
   (Would catch F06 / SIGHTING-089.)

These are pattern-level guards — not in scope for spec-061 (which is
discovery only). They belong in spec-062 (the click-every-button gate)
or its own micro-spec.

---

## Net outcome of this run

- **0 user-facing crashes** found that weren't already known.
- **1 user-facing degradation** found (F06 — sparse History detail panel for v2 runs).
- The pattern of "v2 code path assumes pre-spec-040 layout" has been
  drained to the obvious surfaces. The remaining offenders are either
  already-tracked (SIGHTING-078) or write-side (out of scope).

**Status: Phase 0-2 complete. Phase 3 (FIX items) intentionally empty
this run.** Phase 4 (file SIGHTING-089) and Phase 5 (close-out) pending
user direction.
