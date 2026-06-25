# spec-086 — ImageRepository over run_db (slice 1: person images)

**Created**: 2026-06-25 · **Status**: Implemented · **Priority**: P1

> **DECISION (2026-06-25): Approach B — central API DB, not per-run `run_db`.**
> Investigation (T0/T1) showed reusing `run_db` is more entangled than assumed:
> `RunExporter` is face-clustering-coupled (needs crops/`cluster_result`/merge logs),
> the API has no per-run dir (`fc_export_dir` is None on API runs), and the run_db
> golden-hash contract would be touched. Approved alternative: add the normalized
> tables to the central `sim_bench.db` the API already owns, written by
> `pipeline_service` (dual-write next to the blob), read by `ImageRepository`. Same
> repository + SQL JOIN + `ImageMetrics` output; lower risk. The sections below that
> say "run_db" are superseded by this decision; the build as implemented is recorded
> in `tasks.md`.
**Predecessors**: spec-040 (unified pipeline / run_db), spec-085 (ImageMetrics SSOT), SIGHTING-104
**Source**: user — "the true solution is SQL tables + repository, for image metrics and almost
any data transfer." This spec is **slice 1** of that program: prove the pattern on one surface
(person images) end-to-end, behind the unchanged `ImageMetrics` contract, with equivalence +
Budapest E2E gates. Not a big-bang migration.

## Problem

People & Faces is starved of data because `people_service.get_person_images()` returns only
`{image_path, face_count, faces}` and never joins per-image metrics. Consequences: the
"Selected" filter shows nothing, "Sort by Score" is inert, and the Image Detail popup draws no
face boxes (all documented in `docs/architecture/people_images_missing_metrics.html`).

Root design smell (user's diagnosis, confirmed): the central DB (`sim_bench.db`) stores per-image
and per-face data as **JSON blobs** (`pipeline_results.image_metrics`, `people.face_instances`).
"Images for a person" is therefore a hand-coded Python stitch, not a SQL JOIN — so a join can
silently "forget" half its columns. A normalized schema makes that class of bug impossible.

## Key finding (the gating fact)

A **normalized schema already exists** — `sim_bench/run_db/` (per-run `face_clustering.db`):
tables `images`, `faces`, `face_scores`, `filter_decisions`, `clusters`, `scene_clusters` with
FKs, written by `RunExporter` and readable via `RunStore`.

**BUT the Albumify API never writes or reads it.** Confirmed:
- API runs via `execute_spec()` (`pipeline/run.py:32`), which does NO persistence by design.
- `RunExporter` is called only by `run_pipeline()` (`run.py:140`, CLI) and the
  `face_cluster_export` step — **neither is in the API `default_pipeline`**.
- `grep run_db sim_bench/api` → **NONE**.

So slice 1 must **first make the API write run_db**, then read person-images from it.

## Coverage: ImageMetrics field → run_db source

| ImageMetrics field | run_db source | Status |
|---|---|---|
| iqa/ava/sharpness/composite | `images.*_score` | ✅ |
| cluster_id | `images.scene_cluster_id` | ✅ |
| face_count | `images.n_faces` | ✅ |
| filter_scores (bbox+conf+passed) | `faces.bbox_*_ratio`, `det_score`, `rejection_reason` | ✅ |
| face_pose/eyes/smile_scores | `face_scores.pose/eyes/expression_score` | ✅ |
| roll_angles | `faces.roll` | ✅ |
| **is_selected** | — | ❌ GAP |
| **filter_reason** (spec-084) | `faces.rejection_reason` (per-face only) | ⚠️ partial |
| **person_detected / body_facing / person_confidence** | — | ❌ GAP |

The 3 gaps are select_best + person-detection outputs not yet persisted to run_db. Slice 1
scopes a minimal **`image_selection`** addition (is_selected + image-level reason); person-detection
fields are deferred to a later slice (they don't block the People filter/box bugs).

## What we build (slice 1, strangler-fig)

1. **Write path** — wire run_db export into the API pipeline (`pipeline_service.execute_pipeline`,
   after `execute_spec`): persist `images`, `faces`, `face_scores`, plus a new `image_selection`
   row (is_selected, reason). Reuse `RunExporter` where it fits; extend minimally for selection.
   The blob write stays (dual-write) so nothing breaks during the slice.
2. **Repository** — `ImageRepository(run_dir|session)` with
   `get_images_for_person(person_id) -> list[ImageMetrics]`, implemented as a real SQL JOIN
   `faces ⋈ images ⋈ face_scores ⋈ image_selection`. No JSON stitching.
3. **Endpoint swap** — `people_service.get_person_images()` delegates to the repository; map rows
   to the **existing `ImageMetrics` shape**. Widen `PersonImageResponse` to carry those fields
   (today it declares only 3, so FastAPI strips the rest).
4. **Frontend** — add a "Not selected" option to the People gallery filter
   (`people_browser.py`). No other UI change — the `ImageMetrics` contract is unchanged, so the
   popup boxes and scores light up automatically.

## AC

- **AC1** An API pipeline run writes a per-run `face_clustering.db` with non-empty `images`,
  `faces`, `image_selection`.
- **AC2** `GET /people/{album}/{person}/images` returns, per image, a populated `is_selected`,
  `composite_score`, and `filter_scores` (with bbox) — verified against real Budapest data.
- **AC3** People & Faces: "Selected"/"Not selected" filters partition the gallery; Image Detail
  popup draws numbered face boxes; Faces tab shows pose/eyes scores.
- **AC4** Equivalence: repository output `==` the (corrected) blob-join output for the same run
  (golden test), so the swap is provably behavior-preserving.
- **AC5** `tests/face_clustering/e2e_budapest/` green (V2 baseline gate). Existing `tests/api/`
  green.

## Non-goals (later slices)

- Migrating Results/Explore/clusters reads to the repository (slice 2+).
- Removing the blob columns / data migration of old albums (final slice, after all reads move).
- Person-detection fields in run_db (separate slice).
- A central (not per-run) normalized DB — out of scope; reuse the existing per-run store.

## Risks

- **Dual write divergence** — blob and run_db could disagree. Mitigated by AC4 equivalence test.
- **run_dir plumbing** — the API must know each run's `run_dir`. `PipelineResult.fc_export_dir`
  already carries a run dir for some runs; T1 confirms/creates the canonical path.
- **Budapest gate** — any write-path change risks the 15-cluster/340-face invariant; AC5 guards it.

## Notes

Effort: this slice is M (one focused spec). The full "almost any data transfer" migration is a
multi-slice **program** (~4-6 specs), each gated by equivalence + Budapest. This spec deliberately
does ONE vertical to de-risk and prove the repository pattern before widening.
