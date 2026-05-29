# spec-061 — Audit every v2 read path against the v5 reality

**Created**: 2026-05-29
**Status**: Draft
**Predecessors**: spec-040 (schema v5; the layout shift this audit verifies adoption of), spec-042 (v2 tab parity umbrella), spec-045 (Cluster Analysis tab — first big tab where the assumptions started biting), spec-050 (v2 run picker), spec-060 (E2E gate — Phase 2 effectively shipped via `test_v2_app_smoke.py`).
**Successors**: per-finding fix specs as discovered (one per pre-v5 assumption found, if non-trivial).
**Trigger**: 2026-05-29, three sightings filed in a single week (SIGHTING-078, -079, -080), all the same shape — a v2 code path failing because it was ported assuming a pre-spec-040 layout. A fourth (loader v5 path) was discovered while writing the SIGHTING-080 regression test. At this rate, the codebase has more of these latent.

---

## Problem

spec-040 Phase 4 (schema v5) made a sweeping shape change:
- per-run `face_clustering.db` (sqlite) replaces `faces.csv` + `clusters.csv`
- embeddings move out of the DB into `embeddings.npy`
- `merge_decisions` can have rows at iteration N even when `clusters` does not (no-op merge round)
- new producer tag `fc_app_v2` (vs legacy `fc_app` / `albumify`)
- per-run UUID dirs allocated BEFORE the pipeline runs (spec-050)

The v2 app components — History tab, Cluster Analysis tab, load_button, AsyncHandle wiring, the loader, the artifact gate — were assembled across several specs (040, 042, 045, 050). Each new component was written against *some* understanding of the v5 reality, but not all of them. The three sightings this week prove that:

| Sighting | Pre-v5 assumption that survived |
|---|---|
| **078** | `RunStore._resolve_iteration("final")` reads `MAX(iteration) FROM merge_decisions` — wrong when merger ran but merged nothing. |
| **079** | `AsyncHandle[T]` ported from legacy `_AsyncState` without porting its `time.sleep + st.rerun()` polling loop. Streamlit doesn't poll background threads. |
| **080** | `_REQUIRED_ARTIFACTS` constant in History tab kept the legacy CSV trio after spec-040 collapsed to one DB. |
| *(loader bonus)* | `load_pipeline_result` for top-level `face_clustering.db` routed through `_load_from_db` which expects an in-DB `embeddings` table that v5 retired. |

Each was a 5–20 LOC fix once found. The cost was discovery — every one was found by the user clicking through the app and reporting a crash. This spec stops that cycle by **auditing every v2 read path proactively** instead of reactively.

## What we build

**A grep-driven audit** that walks every v2 file looking for the known anti-patterns, plus a write-up of each finding with: location, what it assumes, whether it's broken under v5, and a fix or a follow-up sighting.

Concrete deliverables:

```
specs/061-v5-readpath-audit/
  AUDIT_FINDINGS.md     -- one row per finding: file:line, pattern, status, fix link
  AUDIT_CHECKLIST.md    -- the categories to grep; lives in the spec dir
                            so future contributors can re-run the audit
  spec.md (this file)
  tasks.md
```

Plus: any blocking fixes land in this spec; non-blocking ones become their own sightings.

## What we don't build

- **No code refactor unless a finding is broken.** This is a discovery spec. A pattern that "looks risky" but works correctly under v5 is *documented*, not changed.
- **No new schema or contract.** The v5 schema is the truth being audited against; this spec doesn't change it.
- **No retroactive test for every fixed sighting.** SIGHTING-078, -079, -080 already shipped their regression tests (`test_v2_app_smoke.py`); this spec only adds tests for NEW findings it surfaces.
- **No audit of the legacy app** (`app/face_clustering/`). spec-040 Phase 7 retires it; auditing dead code wastes effort.

## The audit categories (what to grep)

| # | Pattern | Where it'd live | Example sighting |
|---|---|---|---|
| 1 | Hardcoded `faces.csv` / `clusters.csv` / `embeddings.npy` references | Anywhere checking run-dir contents | SIGHTING-080 |
| 2 | `RunStore.<method>("final")` calls | Anywhere consuming a per-run DB | SIGHTING-078 |
| 3 | `MAX(iteration) FROM merge_decisions` or equivalent SQL | Direct DB queries | SIGHTING-078 root |
| 4 | `AsyncHandle` / threading patterns inside `app/face_clustering_v2/` | UI compute paths | SIGHTING-079 |
| 5 | Constants ending in `_REQUIRED`, `_ARTIFACTS`, `_COLUMNS`, `_EXPECTED` | All v2 modules | SIGHTING-080 |
| 6 | Producer-name checks (`== "fc_app"`, `== "albumify"`) | Filter logic | SIGHTING-075 was adjacent |
| 7 | Schema-version-blind paths — `face_clustering.db` opened without `PRAGMA user_version` check | Direct sqlite usage | unknown |
| 8 | `pipeline_run.json` field reads that assume v4 structure | History detail, summary builder | unknown |
| 9 | Hardcoded run-dir layout expectations (e.g., `_v4/` subdir) | Loader, RunStore composers | loader bonus |
| 10 | `os.listdir` / `glob` calls on run dirs | Artifact discovery | unknown |

The checklist lives in `AUDIT_CHECKLIST.md` so it can be re-run after future schema changes.

## Locked decisions

1. **Findings file is append-only.** Each finding gets a row; status changes update the same row in place. No deletions — the trail is the value.
2. **Severity gate.** A finding is **blocking** if it can be triggered by an action a user is likely to take in the next month. Non-blocking findings become sightings; blocking ones get fixed in this spec.
3. **One commit per fix.** Don't batch multiple findings into one commit — makes git blame useless when the next regression hits.
4. **No structural refactor inside this spec.** If a finding suggests a deeper redesign (e.g., "AsyncHandle pattern should be removed entirely"), file a separate spec; this one ships discovery + small fixes only.
5. **Re-runnable.** `AUDIT_CHECKLIST.md` doubles as a script. Future contributors run it after schema changes to catch the next wave.

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `AUDIT_FINDINGS.md` exists with one row per location matching any of the 10 patterns | grep + manual review |
| AC2 | Every finding has: location, pattern category, v5-compatibility verdict, status (NO-OP / FIX / SIGHTING / DEFERRED) | inspection |
| AC3 | Every "FIX" finding shipped a fix in this spec dir | commit log |
| AC4 | Every "SIGHTING" finding has a numbered sighting in `docs/project/SIGHTINGS.md` | grep |
| AC5 | `AUDIT_CHECKLIST.md` runs as a self-contained walk-through; a future contributor can re-execute it without reading this spec | manual try |
| AC6 | `test_v2_app_smoke.py` gains at least one new AppTest per "FIX" finding | inspection |
| AC7 | No regression in `pytest tests/face_clustering/ tests/architecture/ -q` | run |

## Risks

- **Audit-fatigue.** 10 categories × an unknown number of v2 files = a lot of greps. Mitigate: time-box each pattern to 30 min; if a pattern surfaces 0 findings in 30 min, mark it complete and move on.
- **False positives.** Some `_COLUMNS` constants are unrelated to schema (e.g., display columns). Skip annotation cheaply ("UI-only, not schema") to keep the findings file honest.
- **Scope drift.** A finding might reveal "X needs a full redesign." Don't redesign in this spec; file a follow-up.
- **The next wave.** This spec catches the *known* anti-patterns. A category we don't know about (yet) could still bite us. Mitigate: spec-060 E2E gate (when ramped to Phase 3) is the long-term net.

## Effort estimate

**~4-6 hours.** Discovery dominates; fixes are small. Concrete breakdown in `tasks.md`.

## Open questions

1. **Run AUDIT_CHECKLIST as a CI gate?** Probably not yet — too noisy. But the checklist itself should be runnable. Decision: ship as a manual walk-through; revisit if findings keep recurring.
2. **Include `face_cluster/loader.py` and `face_cluster/run_store.py`?** Yes — they're consumed by v2 tabs (loader.py was the bonus finding for SIGHTING-080). The "v2 read path" definition is anything in the dependency closure of v2 tab/component code.
3. **What about write paths?** Out of scope. SIGHTING-078/-079/-080 were all read-side. If a write-side anti-pattern surfaces during the audit, file a sighting; don't expand the spec scope.
