# spec-079 — Albumify onto the Face-Clustering-v2 Shared Core

Status: **In Progress**
Owner: Jonathan Hexner
Created: 2026-06-05

## Problem

Albumify (the `app/streamlit/` web app + FastAPI backend) and Face-Clustering v2
(`app/face_clustering_v2/`) share the **pipeline** but diverge below it:

- FC v2 reads the per-run `face_clustering.db` through a clean stack:
  `RunStore → repositories → views → typed dataclasses → thin tabs`.
- The API re-derives a **denormalized** `people` table into the central
  `sim_bench.db` and serves its own Pydantic schemas. It imports **none** of
  `run_db.store`, `face_cluster.repositories`, or `face_cluster.views`.

Result: two contract shapes for the same "face" / "cluster" / "person", a stale
duplicate of cluster membership (`people.face_instances`), and no repository for
people. See `PLAN.html` (architecture) and `EXECUTION_PLAN.html` (staged plan).

## Goal

Keep the Streamlit + FastAPI split. Make the API a **thin serializer over the
v2 view layer**, introduce a **proper PeopleRepository**, and (pending the
people-table history report) slim the `people` table to an authored-only
overlay. Hold the FC v2 gold standard on `D:\Budapest2025_Google` (profile_4)
after every stage.

## Equivalence anchor (binding)

On the Budapest album with `profile_4.json`, Albumify must land in the FC v2
reference band (`6437d335de914755bc3edb825c9591c0`):

- **IDENTITY count = the `people`-table row count** (NOT `PipelineResult.num_clusters`,
  which is *scene/image* clustering — see `CLUSTERING_EXPLAINED.html`). FC v2 ref = 15.
- `n_faces_assigned == 107`; cluster sizes `[35, 24, 14, 7, 4, 3, 3, 3, 2, 2, …]`
- bands: identities 12–18, rejected 220–240 (documented tolerance)

Captured once in Stage 0 as a golden fixture; asserted after every later stage.
Constants live in `tests/_budapest_baseline.py` (shared by both test suites).

### Clustering facts established in Stage 0 (corrected)

- Albumify and FC v2 run the **same** face-identity algorithm: `cluster_people`
  with `method: face_cluster_knn` → `face_cluster_bridge.build_fc_config` →
  `FCConfig`. The only difference is config values, so parity = **config overlay**
  (profile_4's flat knobs onto the `cluster_people` step), not an algorithm swap.
- `cluster_scenes` and `cluster_by_identity` are **image-level** (scene) steps,
  unrelated to the identity count.
- Decision: **keep `identity_refinement`** in the baseline (Albumify product
  behavior); the `people` table uses `refined_people_clusters or people_clusters`.
- `cluster_by_identity` has a known design flaw (groups by face count, not
  dominant-identity) — parked in **PRD-080**, out of scope here.

## Approach — staged, each gated by test → /code-review → commit

| Stage | Outcome | Destructive? |
|---|---|---|
| 0 | Baseline capture + both test harnesses, green on current code | no |
| 1 | `PeopleRepository` (dormant, mirrors `ClusterAnalysisRepository`) | no |
| 2 | `PeopleService` / view (membership + name overlay) | no |
| 3 | API **reads** delegate to the view; behavior preserved | no |
| 4 | Slim `people` table to authored-only (Alembic migration) | **yes** — gated on PEOPLE_TABLE_REPORT.html + user sign-off |
| 5 | Unify contracts (schemas from views) + Albumify UI_SPEC/ColumnSpec | no (schema reshape) |

## Test strategy

1. **Backend / endpoint** (`tests/api/`) — FastAPI `TestClient`, in-process,
   comprehensive: every read endpoint vs golden + the anchor numbers; mutation
   round-trips (rename / merge / split); contract guards (bbox shape, identity
   field, no dropped quality fields).
2. **Frontend / Playwright** (`tests/streamlit/e2e_albumify/`) — slim, forked
   from `tests/face_clustering/e2e_budapest/conftest.py`: People-page smoke
   (~15 cards) + rename-persists. Screenshot on every run.

## Non-goals

- Collapsing the HTTP boundary (Albumify stays a web/API client).
- Removing the central catalog DB (albums, `universal_cache`, overrides, events
  are kept — FC v2 has no equivalent).

## Open decision

Stage 4 (slim `people`) is **gated** on the parallel people-table history report
(`PEOPLE_TABLE_REPORT.html`) — if the table was meant to be a broader, unfinished
feature, we revisit scope before the destructive change. Default if the report
finds nothing broader: proceed with slim.

## Exit / Code Review gate

`/code-review` after each stage on the changed modules; no open High-severity
finding before commit. Final `REVIEW.md` before flipping to Implemented.
