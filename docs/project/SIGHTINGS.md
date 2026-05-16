# Sightings

This file tracks issues that need investigation and resolution.

---

<!-- Format:
### SIGHTING-XXX: Brief title
**Status**: OPEN / IN PROGRESS / RESOLVED
**Severity**: Critical / High / Medium / Low
**Reported**: YYYY-MM-DD
**Persona**: Who should fix this (e.g., Senior SW Engineer, ML Engineer)

**Problem Description**:
What is the issue?

**Symptoms**:
- Observable behavior

**Suspicion**:
Possible root cause

**Steps to Reproduce**:
1. Step 1
2. Step 2

**Resolution**:
(filled when resolved)

**Findings**:
(what was learned - also add to LEARNINGS.md)
-->

### SIGHTING-065: Image-level fields denormalized onto every `faces` row (no `images` table)
**Status**: OPEN
**Severity**: Medium (design smell; not a data bug)
**Reported**: 2026-05-16 (during DB doc review)
**Persona**: SW Architect

**Problem Description**:
The per-run `face_clustering.db` has no dedicated `images` table. Image-level fields (`iqa_score`, `ava_score`, `sharpness_score`, `scene_cluster_id`) are denormalized onto every row of the `faces` table — added by spec-033 P-C C-3 so `RunStore.image_detail(path)` could do a single JOIN. The result: every face of the same image carries duplicate copies of these fields; updating an image score requires updating N face rows; an image with zero detected faces has nowhere to record its IQA score.

**Symptoms**:
- Redundant storage (N face rows × 4 image-level columns each).
- "Where is the image-level data?" question has no clean answer — it's distributed.
- No way to record information about images that have no detected faces (e.g., scenery photos that still got an IQA score).

**Suspicion**:
The shortcut was taken to keep `image_detail()` to one query. An `images` table + a join would be the right design.

**Recommended Fix**:
- Add an `images` table: `(image_path PK, image_id, iqa_score, ava_score, sharpness_score, scene_cluster_id, n_faces, created_at)`.
- Remove the 4 denormalized columns from `faces`.
- `RunStore.image_detail()` does one extra JOIN — still single-call, still cheap.
- Migrate as a non-additive schema change (bumps SCHEMA_VERSION).
- Update both <code>docs/architecture/db_schemas.html</code> and <code>db_global.html</code> in the same PR.

---

### SIGHTING-064: `area` column unit confusion — propose `area_ratio` canonical column
**Status**: OPEN
**Severity**: Medium (consumer confusion; SIGHTING-060 is the underlying unit-drift)
**Reported**: 2026-05-16 (during DB doc review)
**Persona**: SW Engineer

**Problem Description**:
`faces.area` (and bbox_x/y/w/h) are in mixed units across the two pipelines: px / px² on FC App standalone, fraction / fraction² on Albumify. Consumers reading the column must know which producer wrote it. SIGHTING-060 documents the underlying bug; this sighting proposes the structural fix.

**Recommended Fix**:
- Add `area_ratio REAL NOT NULL` to `faces`, defined as `bbox_w_fraction * bbox_h_fraction` ∈ [0, 1] regardless of producer.
- Keep `area` for backward compatibility, but mark it deprecated in the docs and remove it after one release.
- Same treatment for bbox: add `bbox_x_ratio, bbox_y_ratio, bbox_w_ratio, bbox_h_ratio` and deprecate the raw columns.
- Update Pandera schema with `Check.in_range(0.0, 1.0)` on the new ratio columns — catches unit confusion at write time.
- Update <code>docs/architecture/db_schemas.html</code> in the same PR.

---

### SIGHTING-063: Bridge pose-lookup operator precedence (dead branch, latent bug)
**Status**: OPEN
**Severity**: Low (no current observable failure — both lookup paths return None today)
**Reported**: 2026-05-15 (from spec-033 REVIEW.md, FR-033-7)
**Persona**: SW Engineer

**Problem Description**:
`sim_bench/pipeline/steps/face_cluster_bridge.py:80` reads:
```python
pose_scores = if_face.get("pose_scores") or if_scores.get("pose") if isinstance(if_face, dict) else None
```
The ternary binds only to the second operand: it parses as `if_face.get("pose_scores") or (if_scores.get("pose") if isinstance(if_face, dict) else None)`. Almost certainly not what was intended. Today both `.get()` calls return None (no upstream step produces a 3-tuple pose under either key), so the bug is invisible.

**Suspicion**:
Author intended `(if_face.get("pose_scores") or if_scores.get("pose")) if isinstance(if_face, dict) else None`. The `if_face.get(...)` outside the ternary will error if `if_face` is ever not a dict.

**Steps to Reproduce**: would require an `if_face` that's not a dict — doesn't happen in current code paths.

**Recommended Fix**: parenthesize correctly, or delete the dead branch entirely. The pose-lookup logic should be revisited as part of spec-037 anyway (which adds a real pose step), so this can be folded into that PR.

---

### SIGHTING-062: Duplicate `filter_decisions` rows possible on Albumify runs (spec-033 P-C C-3 + spec-032 wiring)
**Status**: OPEN
**Severity**: Medium (silent data corruption risk; protected by SQLite PRIMARY KEY which would surface as a write failure)
**Reported**: 2026-05-15 (from spec-033 REVIEW.md, FR-033-5)
**Persona**: SW Engineer

**Problem Description**:
After spec-033 P-C C-3 wired `filters=context.filters` into `RunExporter.export()`, two write paths on the Albumify pipeline now contribute to `filter_decisions`:
1. `sim_bench/pipeline/steps/filter_quality.py:79` records `image_quality` decisions via `context.filters.record(...)`.
2. `sim_bench/pipeline/steps/face_cluster_export.py:151` forwards the same `FilterContext` to `RunExporter`, which inserts every recorded decision into `filter_decisions`.

If any other step also calls `context.filters.record(...)` for the same `(item_id, filter_name)` pair, the table's `PRIMARY KEY (item_id, filter_name)` raises `IntegrityError` on INSERT — which today is caught by the try/except at `face_cluster_export.py:118-138` and logged as a non-fatal warning. The user never sees it.

**Symptoms** (potential, not observed yet):
- Silent log warning "v4 dual-write failed: UNIQUE constraint failed: filter_decisions.item_id, filter_decisions.filter_name"
- Missing v4 artifacts on affected runs

**Steps to Reproduce**:
1. Run a full Albumify pipeline on a 5-image fixture.
2. Grep the run log for "UNIQUE constraint failed".
3. Inspect `_v4/face_clustering.db` for completeness.

**Recommended Fix**:
- Short-term: tighten the try/except at `face_cluster_export.py:118` so PK violations fail loud (don't get masked as warnings).
- Medium-term: assert in `FilterContext.record(...)` that re-recording the same `(item_id, filter_name)` is explicit (the docstring says it replaces, but writes accumulate — verify).
- Resolution may be that this is benign because `FilterContext` already dedupes by replacing; the verification is the cheap part.

---

### SIGHTING-061: spec-033 P-C C-1 regression — bridge gate-unblock + missing blur step → 100% face rejection
**Status**: RESOLVED (workaround landed same day; root fix needs an `insightface_score_blur` step)
**Severity**: Critical (blocks identity_refinement on every Albumify run)
**Reported**: 2026-05-15
**Resolved**: 2026-05-15
**Persona**: ML Engineer

**Problem Description**:
After spec-033 P-C C-1 removed the bridge's hardcoded force-disables for the five quality gates, every face on an Albumify run got rejected at the blur gate, leaving `people_clusters` empty and crashing `identity_refinement` with "Required context key is empty: people_clusters".

**Symptoms** (from `logs/2026-05-15_11-18-44/api.log`):
- 428 faces, 200 candidates after top-k.
- `Quality gating: 0 core, 428 holdout faces`.
- `face_cluster_knn: no faces passed quality gating, all noise`.
- `Excluded 428 noise faces (label=-1) from people clusters`.
- Pipeline fails at `identity_refinement`.

**Root Cause**:
`face_cluster_bridge.build_fc_config` post-P-C honored `cluster_people.blur_min: 50.0` from `configs/pipeline.yaml`. But the active InsightFace pipeline has no blur-scoring step — `insightface_faces[path]["faces"][i]` has no `blur_score` field. The bridge plumbed nothing, so every `FaceRecord.blur_score` stayed at 0.0 and the gate rejected all 428 faces.

**Fix**:
`face_cluster_bridge.build_fc_config` now pins `blur_min=0.0` regardless of config, with a docstring explaining "InsightFace pipeline has no blur step yet; re-enable when `insightface_score_blur` lands and the bridge plumbs it." Architecture test `test_bridge_pose_and_det_gates_read_from_config` checks the docstring explanation is still present.

**Findings** (added to LEARNINGS.md):
- "Plumb fields through the boundary" only works if the upstream producer actually computes the field. Removing a downstream force-disable without auditing upstream computation = silent 100% rejection.
- The right pattern: bridge pins permissive defaults for any gate whose upstream signal isn't yet computed in this pipeline. Honest. Reversible the moment the signal exists.
- Bidirectional plumbing audit checklist: for every field the bridge claims to recover, verify (a) the upstream step writes it, (b) under the dict key the bridge reads.

---

### SIGHTING-060: `face.area` unit drift — three producers write three different units to the same field
**Status**: OPEN
**Severity**: High
**Reported**: 2026-05-11
**Persona**: Senior SW Engineer
**Related**: SIGHTING-059 Issue 2; spec-030

**Problem Description**:
Three independent producers in the codebase populate the `face.area` field with three different units. Downstream consumers (quality gate, UI, slider label) each assume whichever unit is convenient, with no explicit unit declaration anywhere in the data. Result: on Albumify-produced runs the UI renders "Area 0 px²" for face_26 because the column is actually a fraction (0.001) being cast to int.

**Concrete evidence — three producer sites**:
| File | Line | Code | Unit |
|---|---|---|---|
| `face_cluster/embedding.py` | 99 | `area = (x2 - x1) * (y2 - y1)` | **raw pixels** |
| `sim_bench/pipeline/steps/face_cluster_bridge.py` | 49 | `area = float(bbox.get("w", 0) * bbox.get("h", 0))` | **fraction of image area** (bbox.w/h are normalized 0..1) |
| `sim_bench/pipeline/steps/filter_quality_gate.py` | 111 | `area = w_px * h_px` | **raw pixels** |

**Concrete evidence — consumers each assume their own unit**:
- `face_cluster/quality.py:300-304` — compares `face.area >= threshold` directly; threshold comes from a slider labeled "px"
- `app/face_clustering/tabs/face_analysis_tab.py` — renders `int(area) px²` in the UI
- `app/face_clustering/tabs/run_tab.py:90` — `min_face_area px (0=off)` slider, default 0
- `face_cluster/views/face_view.py:64` — uses `face.area < 1000` as a sanity check (assumes pixels)

**Symptoms**:
- On the reference run `face_clustering_20260510_231628`, every face's area is a fraction 0.0001..0.45, but the Face Analysis tab shows "Area 0 px²" for all of them.
- The FC App's `min_face_area` slider is effectively dead unless the user happens to use the FC App's standalone runner (then area is pixels and the slider works).
- The Albumify `min_face_ratio` default (0.005) doesn't appear to be filtering face_26 (area=0.001) — separate sub-bug requiring confirmation of which config path actually ran.

**Suspicion**:
- The two pipelines (Albumify and FC standalone) were developed independently and converged on the same field name with different unit conventions.
- No place in the schema/types declares the unit, so the bug was invisible until a multi-producer scenario (Albumify exporting for FC App) made it manifest.

**Resolution direction** (to be finalized in design):
- Either: (a) standardize on ONE unit across all 3 producers, OR
- (b) add an explicit `area_unit` column (`"px"` or `"image_ratio"`) carried alongside `area`, and make every consumer dispatch on it.
- Recommended: (a) — fewer moving parts, kills the bug at the source. Pixels is the more intuitive unit and is what the user expects ("min face size in pixels").
- Either way also fix the unrelated sub-bug: figure out why `detect_faces.min_face_ratio: 0.005` did not filter face_26 on this run.

**Steps to Reproduce**:
1. Run Albumify on any album with merge_enabled (gives `mode=main_app_export`).
2. Open the resulting `face_clustering_<ts>/faces.csv`. Note `area` column max is < 1.0.
3. Open the FC App, navigate to Face Analysis tab, find any face. UI shows "0 px²".

**Findings (preliminary)**:
- This is a classic "no schema, no contract" multi-writer bug. Same shape as SIGHTING-058 (merge_log.json field divergence between Albumify and FC App writers). Pattern: when two apps write to the same data store independently, drift is the default outcome, not the exception.
- Long-term prevention: every field with a unit should carry the unit, OR every producer should be invoked through a single typed writer (the `RunExporter` pattern from spec-030).

---

### SIGHTING-059: Multiple data-integrity defects on Albumify face_clustering runs (cluster 6 has no crops, area shown as 0px², blur=0 for everyone, merge gate fields missing)
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-05-10
**Persona**: ML Engineer + Senior SW Engineer
**Reference run**: `results/album/face_clustering_20260510_231628/` (mode=`main_app_export`, source `D:\Budapest2025_Google`, 428 faces, 14 clusters, merge enabled)

**Problem Description**:
User reports four distinct defects on a fresh Albumify-produced run, suggesting multiple writers are still emitting incomplete/incorrect data despite SIGHTING-058 / spec-030 work. (Note: spec-030 Phase 3 is on a feature branch, not merged to main — so the user is hitting Phase 1+2 state where v4 is dual-written but the legacy artifacts the UI still reads remain authoritative.)

**Symptoms — verified by tracing `faces.csv`, `merge_log.json`, `embeddings.npy`**:

1. **Cluster 6 has no thumbnails in the UI.**
   - `face_46` (from `20250822_123354.jpg`) and `face_47` (from `20250822_123400.jpg`) are assigned to cluster 6, `is_core=True`, all four quality gates `pass=True`, `quality_rejection_reason=NaN`.
   - But `crop_path = NaN` in faces.csv and **no `face_0046_*` / `face_0047_*` exist in `crops/`**. Other cluster faces (e.g. face_26 → `face_0026_aligned.jpg`) do have crops.
   - So the UI is faithfully showing "no crop", but the data is missing it. Crop generation is silently dropping a subset of faces that survive quality.

2. **face_26 (from `20250822_122626.jpg`) shows "Area 0px²" in the UI but is in cluster 4.**
   - faces.csv records `area = 0.001019` — the column is a **fraction of image area** (overall column ranges 0.0001 → 0.45).
   - The Run-tab control is labeled `min_face_area px (0=off)`. Either the column unit is wrong, or the slider/label is wrong. Whichever it is, the UI rendering "0 px²" is the int-cast of a 0–1 float.
   - The filter is also disabled by default (slider value 0), so face_26 was never going to be filtered regardless.

3. **`blur_score = 0.0` for ALL 428 faces.**
   - `faces['blur_score'].describe()` → min=0, max=0, std=0. The blur step is either no-op or its results are clobbered before export.
   - `det_score` is `NaN` for every face also.

4. **Merge log is missing the four gate `*_pass` booleans.**
   - For C4–C6 row in `merge_log.json`: `support_pass`, `margin_pass`, `diameter_pass`, `distance_pass` are all absent. Same for `centroid_dist`, `d_cross_min`, `d_cross_p25` (only `p25_cross_dist` is present).
   - The UI is forced to recompute pass/fail or render "REJECTED 4/4" via fallback logic — which mis-renders cases like the user's old C0/C1 complaint.
   - The 28-field contract documented in spec-030 is **not** what's actually on disk.

5. **Recluster tab has a "Load profile" picker; Run tab does not.**
   - `app/face_clustering/tabs/recluster_tab.py:58 _render_profile_bar` — selectbox of all profiles in `~/.sim_bench/profiles/`.
   - `app/face_clustering/tabs/run_tab.py` — no equivalent. User must re-enter every parameter for a fresh run instead of loading a tuned profile.

6. **Cluster 4 (28 faces) is internally incoherent.**
   - Pairwise cosine distances inside cluster 4: **mean=0.471, max=0.867, min=0.115**. Anything > ~0.4 is "different person" — so cluster 4 is a chain-merged blob of multiple people, not one identity. (Reported separately by user as "merge distances don't seem real". Distances themselves *are* real; it's that the clustering thresholds let a chain through.)
   - C4 vs C6 cross distances: mean=0.682, min=0.499 — they're correctly far apart. The merge decision rejected them. But the rejected pair is still being surfaced because C4 itself is bloated.

**Suspicion**:
- (1) Crop step iterates a different subset than quality step — face passes quality but its `crop_path` never populates. Likely an indexing/filter mismatch in `face_cluster/pipeline.py` crop stage when faces survive quality but fail some downstream check before crop persistence.
- (2) `area` column unit drift: somewhere the column was changed from pixel² to fraction (or never was pixels), but UI label and slider unit are stale. Need to grep producers.
- (3) Blur step likely runs but its result is overwritten when faces.csv is rebuilt by a later stage, OR the blur stage was disabled in pipeline.yaml and no test caught it. det_score=NaN suggests the same — fields not being persisted from detection.
- (4) Two writers for merge_log.json with different schemas — one writes `support_pass` and one doesn't. Spec-030 Phase 3 (single RunExporter) addresses this; on main it's still split.
- (5) Pure UI gap — easy to add.
- (6) The clustering parameters at run time were too loose for this album, OR the merge gates aren't catching the chain because per-pair gates can't see global cluster cohesion.

**Steps to Reproduce**:
1. Run Albumify on `D:\Budapest2025_Google` with merge_enabled=True (run dir: `results/album/face_clustering_20260510_231628/`).
2. Open FC App, deep-link to the run.
3. Cluster Analysis tab → cluster 6: thumbnails missing.
4. Face Analysis tab → face_26: area "0 px²".
5. Merge Analysis tab → C4 vs C6: gate badges show pass-state without underlying `*_pass` booleans.

**Resolution**:
TBD — see investigation plan in this sighting.

**Findings (preliminary)**:
- v4 dual-write present (`_v4/` subdir with `face_clustering.db`, npy, crops) but legacy `merge_log.json`/`faces.csv` remain authoritative on main → defects in legacy producers still leak through.
- Need to bring Phase 3 (single-reader cutover) to main *or* fix the legacy producers in place.
- Defects (1)/(2)/(3) are **producer bugs**, not display bugs — none of spec-030's reader work fixes them.

---

### SIGHTING-058: Albumify-produced face clustering runs render with broken merge values in FC App
**Status**: SPEC READY (spec-030)
**Severity**: Critical
**Reported**: 2026-05-09
**Persona**: Senior SW Architect
**Spec**: `specs/030-storage-ownership-refactor/`
**Architecture audit**: `specs/030-storage-ownership-refactor/architecture_audit.html`

**Problem Description**:
On run `face_clustering_20260508_000446`, the FC App's Merge Analysis tab shows nonsense values for pair C0 vs C1: `gates=4/4 REJECTED`, `Support 191/0`, `Margin: inf`, `Diameter 0.972/n/a`, and "Actual merges: 0" — even though `run_metadata` on disk records 4 merges actually executed (14 → 10 clusters).

**Root Cause** (after architecture audit):
Six logical information types are written to 21 storage locations across 14 files per run. The DB `merge_decisions` table has 12 columns; `merge_log.json` has 28 fields per row. The loader prefers DB when present; Albumify writes both, FC App writes JSON only. Result: Albumify-produced runs feed the loader the lossy DB copy, which is missing `actually_merged`, `required_support`, `max_allowed_diameter`, `cluster_a_size`, `cluster_b_size`, `unique_support`, `T_a/T_b/T_global`, `p25_cross_dist`, and three `margin_*` fields. The UI prints defaults (`0`, `None`, `n/a`) for those fields and treats the missing `actually_merged` as falsy → all 4 actual merges are misclassified as REJECTED candidates.

Compounding factors:
- `face_cluster/loader.py` has 9 separate `if x.exists()` fallback branches.
- UI components in `app/face_clustering/` bypass the loader and `pd.read_csv` files directly.
- `face_cluster/export.py:193-196` renames `clusters.csv` → `clusters_stage_base.csv` mid-export and writes a new `clusters.csv` with different semantics.
- Margin gate disabled (`merge_margin=0`) renders as literal "inf" because no display contract for the disabled state.

**Symptoms**:
- "C0 vs C1 4/4 REJECTED" with all four gate badges green.
- "Support 191/0??", "Diameter 0.972/n/a", "Margin: inf".
- "Actual merges: 0" despite 4 merges on disk.
- Bug only appears on runs created by Albumify (mode `main_app_export`); FC-App-created runs render correctly because they don't write the lossy DB.

**Resolution**:
Spec-030 — eliminate duplication at the source rather than patching the reader. Single `RunExporter` writer used by both apps, single `RunStore` reader used by all consumers, full-fidelity DB schema (all 17 merge_decisions fields), no fallback chains. Legacy artifacts (`merge_log.json`, `merge_metadata.json`, `crop_manifest.json`, `export_summary.json`, `faces_merged.csv`, `clusters_merged.csv`, `clusters_stage_base.csv`, DB embeddings BLOB column) deleted. UI semantics fix: three-state outcome label (MERGED/PASSED/REJECTED), margin badge "disabled" when 0.

**Findings (preliminary, will move to LEARNINGS.md on resolution)**:
- Duplication is the disease, schema mismatch is the symptom.
- `if x.exists()` fallback chains are unrecoverable design debt.
- One owner per fact; pick the store that fits the data shape.
- Single writer + single reader interface turns layout changes into refactors instead of archaeology.

---

### SIGHTING-057: ConservativeMerger logs ALL passing candidates as "merged" instead of only the executed one
**Status**: RESOLVED
**Severity**: Critical
**Reported**: 2026-05-05
**Persona**: ML Engineer

**Problem Description**:
`merge_clusters_with_logging()` records multiple entries as `action: "merged"` per iteration. The merger is supposed to pick ONE best merge per iteration, execute it, then re-evaluate. But the log contains ALL candidates that passed the 4 gates in that iteration, not just the winner.

**Evidence**:
- Iteration 1: 5 entries with `action: "merged"` (should be exactly 1)
- C0+C1 appears as "merged" in both iter 1 AND iter 2, but with different cluster_a_size (22 vs 24)
- This means C0+C1 passed gates in iter 1 but wasn't the actual merge — C0+C11 was

**Impact**:
1. FC app shows inflated "4 actual merges" when really fewer happened
2. FC app Merge Analysis shows incorrect iteration grouping
3. Per-iteration cluster_assignments in the DB may be wrong (all passing candidates treated as merged)
4. Leads to incorrect merges being displayed (cluster 0 with faces from 2 different people)

**Root cause**: Need to check `merge.py:merge_clusters_with_logging()` — likely logs evidence for all candidates before selecting the best one, and marks all passing candidates as "merged" instead of only the winner.

---

### SIGHTING-056: Code quality — buried imports, hardcoded constants, oversized files
**Status**: OPEN
**Severity**: High
**Reported**: 2026-05-03
**Persona**: Senior SW Engineer

**Problem Description**:
Systemic code quality issues across the codebase:

1. **Buried imports**: 20+ files have `import` statements inside functions instead of at module top. Examples: `cache_handler.py` imports `sqlalchemy.or_` inside `load_from_cache()`, `cluster_people.py` imports 6 face_cluster modules inside `_run_face_cluster_knn()`.

2. **Hardcoded constants**: `BATCH_SIZE = 200`, `PAGE_SIZE = 20`, thumbnail sizes, etc. scattered across files instead of in config or central constants module.

3. **Oversized files**: `face_service.py` (758 lines), `api_client.py` (725 lines), `cluster_people.py` (600+ lines). These should be split by responsibility.

**Root cause**: Rapid feature development without refactoring. Each fix adds code at the point of need without considering module structure.

**Fix approach**:
1. Move all imports to module top (except genuine lazy imports for optional dependencies)
2. Create `sim_bench/constants.py` for shared constants (batch sizes, page sizes, thumbnail dims)
3. Split oversized files by responsibility
4. Run a linter (ruff/flake8) and enforce import ordering

---

### SIGHTING-055: Face Clustering App — deep-link from main app doesn't work properly
**Status**: OPEN
**Severity**: High
**Reported**: 2026-05-01
**Persona**: Senior SW Engineer

**Problem Description**:
When opening the Face Clustering App via deep-link (e.g., `http://localhost:8502/?load_run=results\album\face_clustering_20260501_112856`):
- Shows "No merged result available. Enable merge_enabled in Pipeline Config and re-run or recluster."
- Cluster Analysis shows "(no crop)" instead of face images
- The export from main app doesn't include crop images needed by standalone app

**Root cause**: The main app's `_export_for_analysis` in `cluster_people.py` exports `faces.csv`, `clusters.csv`, `embeddings.npy` but does NOT export face crop images. The standalone Face Clustering App expects crop images to exist for display.

---

### SIGHTING-054: Scene Clustering tab — completely unreadable, no images shown per cluster
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-05-01
**Persona**: Frontend Developer

**Problem Description**:
The Scene Clustering tab in Explore is unreadable. It should show the actual images that belong to each cluster and which ones were selected, but instead shows cluster IDs and numbers without any visual content.

**Needed**: For each scene cluster, show a grid of the images in that cluster with selected/rejected badges. The user needs to SEE why images were grouped together and which ones were chosen.

---

### SIGHTING-053: Bounding boxes still not visible anywhere in the app
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-05-01
**Persona**: Frontend Developer

**Problem Description**:
Despite `bbox_overlay.py` being created and code added to `people_browser.py`, bounding boxes are NOT appearing in the People & Faces person detail view. The bbox coordinates use normalized (0-1) values but the overlay function expects pixel coordinates. The conversion math (`int(person.thumbnail_bbox[0] * 1000)`) is wrong — it should use the actual image dimensions.

**Also needed**: Bounding boxes in Explore/Face Detection tab, in the image detail popup, and in Results gallery for multi-face images.

---

### SIGHTING-052: Explore page — all tabs useless without visual inspection
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-05-01
**Persona**: UX Designer / Senior SW Engineer

**Problem Description**:
All Explore tabs show text-only tables with filenames and numbers. No images, no thumbnails, no visual way to understand what happened. Users cannot drill down or visually inspect pipeline decisions.

**Symptoms**:
- Image Quality tab: filename + IQA number + "Selected Yes/No". No image thumbnail.
- Selection tab: filename + composite score. No image to see WHY it was selected.
- All tabs are tables of numbers with no visual context.

**Root cause**: Explore tabs were built as data tables without integrating image thumbnails or click-to-inspect functionality. The image detail popup exists but is not wired into these tables.

**Fix needed**:
- Every table row that references an image needs a thumbnail column
- Clicking a row should open the image detail popup with full metadata
- Face-related tabs need face crop thumbnails with bounding boxes
- This is the difference between "data dump" and "visual exploration tool"

---

### SIGHTING-051: Explore page — step_decisions not populated for fresh runs
**Status**: OPEN
**Severity**: High
**Reported**: 2026-05-01
**Persona**: Senior SW Engineer

**Problem Description**:
After a fresh pipeline run (Budapest2025_Google_run2), the Explore page shows "No structured decisions from pipeline" and falls back to raw image data tables.

**Symptoms**:
- Warning: "No step decisions available for this run"
- Fallback tables shown instead of decision records

**Suspicion**:
The `step_decisions` data is emitted by pipeline steps into `context.step_decisions`, serialized in `pipeline_service.py`, and stored in `PipelineResult.step_decisions` JSON column. Either:
1. The DB migration didn't run (new column not added)
2. The serialization is failing silently
3. The API isn't returning the field
4. The `result_service.py` `list_results()` method doesn't include `step_decisions`

**Investigation**: Check if `step_decisions` column exists in DB, check if it's populated after run, check API response.

---

### SIGHTING-050: People & Faces — View button does nothing
**Status**: OPEN
**Severity**: High
**Reported**: 2026-05-01
**Persona**: Frontend Developer

**Problem Description**:
In People & Faces page, clicking "View" on a person does nothing — no navigation, no expansion, no detail.

**Root cause**: The `on_person_click` callback calls `_on_person_click(person_id)` which sets `st.session_state.selected_person_id` and calls `st.rerun()`. But the page may not be checking this state correctly, or the `render_person_detail` component may not match the person_id format.

---

### SIGHTING-049: People & Faces — "Person ?" displayed as name
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-05-01
**Persona**: Frontend Developer

**Problem Description**:
In Face Clustering tab of Explore, people are listed as "Person ?" instead of showing cluster index or a meaningful identifier.

**Root cause**: Code uses `getattr(p, 'person_index', '?')` which returns '?' when `person_index` is not an attribute of the Person model, or when the attribute is None.

---

### SIGHTING-048: Explore page — Selection/Quality tabs show no images, just text tables
**Status**: OPEN (duplicate of SIGHTING-052)
**Severity**: Critical
**Reported**: 2026-05-01

---

### SIGHTING-047: Main app — Performance: laggy UI and API timeouts
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-04-30
**Persona**: Senior SW Engineer / Frontend Developer

**Problem Description**:
The main app is very laggy overall and hits API timeouts. Multiple pages are slow to load or fail to load.

**Symptoms**:
- Faces page hits timeouts
- General sluggishness across the app
- API errors from slow responses

**Investigation needed**:
- Profile which API endpoints are slow (faces, people, results?)
- Check if large data payloads (all faces for an album) are being loaded at once instead of paginated
- Check if image thumbnail loading is blocking the UI
- Check if uncached API calls are made on every rerun

---

### SIGHTING-046: Main app — Image detail popup needed (click image to see all metadata)
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
Similar to the face_clustering app, clicking on any image should show full metadata: which cluster it belongs to, IQA score, face pose, face score, distance to closest other cluster, selection reason, etc. Currently no way to drill into individual image details from the gallery.

**Symptoms**:
- No click-through from gallery to image detail
- No way to understand WHY a specific image was selected or rejected

**Design reference**: Face Clustering App's face popup shows all face metadata on click.

---

### SIGHTING-045: Main app — Pipeline step observability: no way to understand what each step did
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Senior SW Engineer / UX Designer

**Problem Description**:
For every pipeline feature (scene clustering, face clustering, image quality, image scoring, selection) there must be a way to explore what happened. Currently the user gets a final result with no ability to understand intermediate decisions.

**Needed per step**:
- **Scene clustering**: Which images were grouped together and why? Similarity scores.
- **Face clustering**: Already has deep-dive via face_clustering app — need to expose in main app or link to it.
- **Image quality (IQA/AVA)**: Score distribution, which images failed and why.
- **Face scoring**: Pose, expression, eyes — per-face breakdown.
- **Selection**: Why was each image selected or rejected? Score breakdown, comparison with cluster peers.

**Design note**: The face_clustering app already has excellent observability (11 tabs). The main app needs similar observability for ALL pipeline steps, not just face clustering.

---

### SIGHTING-044: Main app — Metrics table doesn't explain selection/rejection reasons
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
The Metrics Table tab shows per-image scores but doesn't explain WHY an image was selected or filtered. Users can't tell what threshold caused rejection or what score combination led to selection.

**Needed**:
- Column showing selection/rejection reason (e.g., "IQA below 0.2", "Best in cluster 5", "Duplicate of IMG_0142")
- Highlight which score(s) were decisive
- Filter by reason

---

### SIGHTING-043: Main app — Comparisons tab unclear purpose (6 comparisons, so what?)
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: UX Designer

**Problem Description**:
The Comparisons tab shows "6 comparisons performed" with Siamese/duplicate check results, but the user doesn't understand what actionable insight this provides. The tab exists for debugging but has no context or explanation.

**Needed**:
- Explanation of what comparisons are (tiebreakers between top candidates, duplicate detection)
- Context: which cluster/person the comparison affects
- Outcome: what decision was made based on the comparison
- Consider merging this into per-image detail rather than a standalone tab

---

### SIGHTING-042: Main app — People section shows non-people (false positive face clusters)
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: ML Engineer

**Problem Description**:
The People page shows entries that are clearly not people (e.g., patterns, objects detected as faces). Need to understand how these false positives arrive and provide a way to flag them.

**Investigation needed**:
- Are these false positive face detections (InsightFace SCRFD)?
- Or are they real faces that were clustered incorrectly?
- What det_score thresholds are being used?
- Should there be a "Not a Person" action (already exists in Face Management but not in People page)

---

### SIGHTING-041: Main app — People vs Faces pages: unclear distinction
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: UX Designer

**Problem Description**:
The app has both a "People" page and a "Faces" page (Face Management). The distinction between them is unclear to users.

**Current state**:
- **People**: Browse detected people (clustered identities), rename, merge, view photos
- **Faces**: Individual face management — assign borderline faces, fix mistakes, batch operations

**Needed**: Either merge these into one page with clear sections, or make the distinction obvious in navigation (e.g., "People" for browsing, "Face Corrections" for fixing).

---

### SIGHTING-040: Main app — Sub-Clusters page: images too large, low resolution, no bounding boxes
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
The Sub-Clusters tab shows face-based sub-clusters within scene clusters, but the images are too large, low resolution, and don't show which face in the image caused it to be placed in a particular sub-cluster.

**Needed**:
- Smaller thumbnails (consistent sizing)
- Bounding box overlay showing which face is relevant
- Better resolution (or at least consistent quality)

---

### SIGHTING-039: Main app — Face browsing needs bounding boxes to show which face is meant
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
When browsing faces (in People, Sub-Clusters, or any face-related view), the full image is shown without indicating which face in the image is being referenced. For images with multiple faces, this makes it impossible to understand the clustering.

**Needed**:
- Bounding box overlay on the relevant face
- Or face crop thumbnail alongside the full image
- Applied everywhere faces are displayed (People detail, Sub-Clusters, Face Management)

---

### SIGHTING-038: Main app — Face clustering results don't load in standalone Face Clustering App
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Senior SW Engineer

**Problem Description**:
Pipeline exports face clustering artifacts to `results/{album}/face_clustering_{timestamp}/` but there's no easy way to load this in the standalone Face Clustering App. The deep-link mechanism (fc_export_dir) was implemented but:
1. The standalone app may not support loading from arbitrary paths via URL parameter
2. The export format may not match what the standalone app expects
3. Consider embedding the face clustering analysis UI directly inside the main app instead of requiring a separate app

**Options to investigate**:
1. Fix the deep-link: ensure standalone app can load from `?load_run=path`
2. Embed face clustering analysis tabs directly in the main app (preferred — single app experience)
3. Both: embed overview in main app, link to standalone for deep analysis

---

### SIGHTING-037: Main app — face_cluster_knn produces one giant cluster (all faces = 1 person)
**Status**: RESOLVED
**Severity**: Critical
**Reported**: 2026-04-30
**Persona**: ML Engineer / Senior SW Engineer

**Problem Description**:
After running the pipeline with `face_cluster_knn` method, the View Results page shows exactly 1 person containing all faces. The clustering algorithm is grouping all faces into a single cluster instead of separating identities.

**Symptoms**:
- View Results > People shows 1 person
- All faces assigned to the same cluster
- No fc_export_dir deep-link visible (suggesting export may not have run)

**Suspicion**:
Multiple possible causes:
1. **Old run data**: Previous run with blur_min=50.0 rejected all faces → all labels=-1 → one noise cluster. Fixed noise filtering, but user may still see old results.
2. **KNN graph too connected**: If K=5 + distance_threshold=0.35 creates one giant connected component across all identities, clustering produces one cluster. Need to verify embeddings are correct and diverse.
3. **Pipeline didn't complete**: If select_best or another step fails, the result may not be saved properly, showing stale data from a previous run.

**Steps to Reproduce**:
1. Start apps with `restart_apps.bat`
2. Select album, run pipeline with face_cluster_knn (default)
3. Check View Results > People — expect multiple people, see only 1

**Investigation needed**:
- Check pipeline run logs for `face_cluster_knn: N clusters` line
- Verify embeddings are not all identical (zero vectors, same vector)
- Check if pipeline completed or failed mid-run
- Test with known dataset where expected cluster count is known

---

### SIGHTING-036: Main app — fc_export_dir deep-link not appearing in View Results
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Senior SW Engineer

**Problem Description**:
The "Open in Face Clustering App" deep-link does not appear in the View Results tab after running the pipeline with face_cluster_knn method. The full chain was implemented (context → DB → API → UI) but user does not see it.

**Symptoms**:
- No info box with "Open in Face Clustering App" link in View Results
- No export artifacts on disk

**Suspicion**:
The `_export_for_analysis` function only runs after successful clustering with core faces. If quality gating rejects all faces (blur_min bug, now fixed) or clustering produces no clusters, the export is skipped. Also the pipeline must complete successfully for results to be saved to DB — if select_best crashes, the result row may never be written with fc_export_dir.

**Steps to Reproduce**:
1. Run pipeline with face_cluster_knn + export_for_analysis checked
2. Check View Results tab for deep-link
3. Check `results/` directory for export artifacts

**Chain verification needed**:
- Does `_export_for_analysis` run? (check logs for "Exported face clustering artifacts")
- Does `pipeline_service.py` save `fc_export_dir` to DB?
- Does `result_service.py` return it in `list_results()`?
- Does `results.py` read `latest.get("fc_export_dir")`?

---

### SIGHTING-035: Main app — Merge Parameters controls cause page jump/tab collapse
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
Modifying ANY Merge Parameters option (inside the "Merge Parameters" expander within "Advanced Configuration") causes the entire tab to minimize and the page to restart/jump. This is separate from SIGHTING-032 (general slider jump) — the merge params are specifically worse, possibly because they are in a nested expander rendered by `render_merge_params()` from `app/shared/merge_controls.py`.

**Symptoms**:
- Moving any merge parameter slider → tab minimizes, page jumps to top
- Makes merge parameter tuning completely unusable
- Worse than regular Advanced Configuration sliders

**Suspicion**:
1. **Nested expander issue**: "Merge Parameters" expander is inside "Advanced Configuration" expander, inside `@st.fragment`. Nested expanders may not preserve state correctly in fragments.
2. **Widget key prefix collisions**: `render_merge_params(key_prefix="fc_")` generates keys like `fc_merge_candidate_threshold`. If these conflict with face_clustering app's `rc_` prefix keys somehow (shared module), state could be corrupted.
3. **Too many widgets in fragment**: 15+ merge params + 15+ other params = 30+ widgets in one fragment. Fragment rerun with 30 widgets may cause visible DOM churn.
4. **`@st.fragment` + conditional rendering**: The merge params expander only appears when `fc_merge_enabled` checkbox is checked. Adding/removing a large block of widgets in a fragment may cause layout instability.

**Steps to Reproduce**:
1. Go to Run Pipeline tab
2. Open Advanced Configuration
3. Check "merge_enabled" checkbox
4. Open "Merge Parameters" expander
5. Move any slider — observe tab collapse

**Investigation plan**:
- Test if issue occurs without `@st.fragment` (full page rerun)
- Test with `st.form` instead of fragment for merge params section
- Check if nested expanders in fragments is a known Streamlit issue
- Profile fragment rerun time with merge params

---

### SIGHTING-034: Main app — Save Settings not working; needs DB-backed profiles
**Status**: PARTIALLY RESOLVED (ProfileStore added for face clustering params; full pipeline config profiles still needed)
**Severity**: High
**Reported**: 2026-04-30
**Persona**: Senior SW Engineer

**Problem Description**:
The "Save Settings" button in the pipeline runner either doesn't persist settings correctly or the UI doesn't reflect saved values on reload. User wants DB-backed profiles (like face_clustering app's `ProfileStore`) with ability to save/load multiple named presets.

**Symptoms**:
- Clicking "Save Settings" — unclear if it works
- No ability to name/manage multiple presets
- No preset selector dropdown

**Suspicion**:
Current implementation uses `client.save_user_config(user_id, ...)` which calls an API endpoint. The endpoint may not exist, may not persist correctly, or may not round-trip all config values. The face_clustering app uses `ProfileStore` backed by SQLite — this pattern should be reused.

**Investigation needed**:
- Verify the `save_user_config` / `get_user_config` API endpoints exist and work
- Check if all config values round-trip correctly (especially nested dicts like merge params)
- Design: should main app use same `ProfileStore` or its own config_profiles table?

---

### SIGHTING-033: Main app — irrelevant clustering parameters shown for face_cluster_knn method
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
When `face_cluster_knn` is selected as the clustering method, the config dict still includes parameters from other methods (HDBSCAN, agglomerative, mutual_knn) with their default values. These get sent to the API and may confuse users or cause unexpected behavior.

**Symptoms**:
- Config dict always includes `cluster_selection_epsilon`, `pca_components`, `k`, `similarity_threshold` regardless of selected method
- These params are not used by face_cluster_knn but pollute the config

**Steps to Reproduce**:
1. Select face_cluster_knn method
2. Run pipeline
3. Check config dict in pipeline run — contains params from other methods

---

### SIGHTING-032: Main app UI — slider/control interaction causes page to jump/collapse
**Status**: OPEN (partially mitigated)
**Severity**: Critical
**Reported**: 2026-04-30
**Persona**: Frontend Developer

**Problem Description**:
In the main Album App pipeline runner, every time a slider is moved or a control is clicked inside the "Advanced Configuration" expander, the page jumps — the expander collapses or the view scrolls to a different level.

**Partial mitigation applied**:
- Added `@st.fragment` to isolate config UI reruns
- Cached `_load_user_settings()` to eliminate API latency on fragment reruns
- Fixed fc_K variable overwrite bug

**Remaining symptoms**:
- Merge Parameters controls still cause severe jumps (see SIGHTING-035)
- General Advanced Configuration sliders improved but occasional jumps remain

**Suspicion**:
Streamlit fragment reruns with nested expanders and many widgets cause DOM instability. May need `st.form` approach or sidebar layout.

**Steps to Reproduce**:
1. Start main app
2. Go to Results page, select an album
3. Open "Advanced Configuration" expander
4. Move any slider
5. Observe: expander may collapse / page may jump

---

### SIGHTING-031: Main app UX — pipeline runner buried in Results page, poor navigation
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-30
**Persona**: UI/UX Designer, Frontend Developer, Lead SW Engineer

**Problem Description**:
The main Album App has fundamental UX issues:
- "Run Pipeline" is located inside the "Results" page — unintuitive (you go to results to START a run?)
- Navigation doesn't follow user workflow (select album → configure → run → view results)
- Similar UX debt to what the face clustering app had before its rework

**Symptoms**:
- New users can't find how to run a pipeline
- Configuration and results are mixed in the same page
- No clear workflow progression

**Suspicion**:
Organic growth without UX planning. Needs a design review to restructure navigation around user workflows. Suggested approach: joint review between UI, frontend, and product roles.

**Steps to Reproduce**:
1. Start main app
2. Try to run a pipeline for an album
3. Notice you have to navigate to "Results" to find the pipeline runner

---

### SIGHTING-030: Exemplar recalculation after merge is incomplete — bridge nodes never discovered
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-30
**Persona**: ML Engineer

**Problem Description**:
After two clusters are merged, exemplar recalculation in `_merge_two_clusters` only considers the union of the two existing exemplar sets as candidates. It never evaluates non-exemplar nodes in the merged cluster. This means a node that would be an excellent bridge (close to a third cluster) is invisible to the exemplar gate if it was never an exemplar in its original small cluster.

**Symptoms**:
- In Austria24_4 recluster_6, C1 vs C10 fails Gate A by exactly 0.005 (exemplar_dist=0.575 vs threshold=0.570) in all 5 iterations. C1 grows from 135 → 138 → 140 → 143 but the exemplar distance to C10 never closes — because none of the newly added nodes (from C6, C12, C16) were ever exemplars, so they never enter the candidate pool.
- Generally: persistent near-miss pairs where the threshold "almost" passes across multiple iterations may indicate this blind spot.

**Suspicion**:
`_merge_two_clusters` lines 754–772: `combined_exemplars = list(set(exemplars_a + exemplars_b))`. This is a union of prior exemplars only. A proper re-exemplar pass should rank all nodes in the merged cluster by d10 and select the top N — the same logic used in the initial exemplar selection stage.

**Steps to Reproduce**:
1. Run clustering on Austria24_4, recluster_6
2. Inspect merge_log.json — find C1 vs C10 entries across iterations 1–5
3. Observe exemplar_dist=0.575 in all iterations despite C1 growing

**Resolution**:
Replaced the `combined_exemplars = exemplars_a + exemplars_b` shortcut in `_merge_two_clusters`
with `_ClusterStatHelper.select_exemplars(merged_nodes, ...)` — full d10 ranking + greedy
suppression over ALL nodes in the merged cluster. Uses the helper already present in `merge.py`.
Change is ~13 lines → 5 lines. Four unit tests added in `tests/face_clustering/test_merge.py`
(`ut_MergeTwoClusters`), all passing. Validation on Austria24_4 pending next recluster.

**Status**: RESOLVED
**Resolved**: 2026-04-30

---

### SIGHTING-029: Merge Analysis UI — iterations collapsed, hides repeated evaluations
**Status**: RESOLVED
**Severity**: Low
**Reported**: 2026-04-30
**Resolved**: 2026-04-30
**Persona**: Senior SW Engineer

**Problem Description**:
The Merge Analysis tab presents one row per cluster pair without separating by iteration. When the iterative merger re-evaluates the same pair in successive iterations (e.g., C1 vs C10 appears in 5 iterations with C1 growing 135→138→140→143), only the first-iteration entry is visible. This makes it appear the pair was only evaluated once, hiding the full decision history.

**Symptoms**:
- User sees C1(sz=135) vs C10 REJECT and concludes C1 was never updated after merging
- Iteration-by-iteration size growth of the winning cluster is invisible
- No way to see whether a rejection reason changed or persisted across iterations

**Steps to Reproduce**:
1. Open Merge Analysis tab for Austria24_4 / recluster_6
2. Find C1 vs C10 — shows size=135
3. Check merge_log.json directly — same pair appears in iterations 1–5 with sizes 135, 138, 140, 140, 143

**Design Notes for Fix**:
- Add iteration number as a visible column in the Merge Analysis table
- Allow filtering/grouping by iteration (e.g., iteration selector or expandable iteration sections)
- Or: show latest-iteration entry by default with a "show all iterations" expander per pair

**Resolution**:
- `MergeDecisionRow` gained `iteration: int` field.
- `_parse_merge_log` removes `seen_rejected` dedup — keeps ALL rejection entries across iterations.
- Added `_latest_per_pair()`, `_build_pair_history()`, `_build_iter_timeline()` helpers.
- `MergeAnalysisView` now carries `n_iterations`, `iter_timeline`, `all_rejection_rows`, `pair_history`.
- Default "Latest" view shows the last iteration entry per pair (fixes C1 showing size=135).
- New Iteration Timeline strip above the pair list shows what was merged at each step.
- "Iteration" selectbox (Latest / All / 1…N) added to gallery controls.
- Multi-iter badge (`N iters`) in pair header; history table in expander body.
- "Iterations run" metric added to summary.
- 13 unit tests in `tests/face_clustering/test_merge_iter_view.py`, all passing.

---

### SIGHTING-028: Label Verification tab UX regression — row images gone, per-row decisions gone
**Status**: OPEN
**Severity**: High
**Reported**: 2026-04-27
**Persona**: Senior SW Engineer

**Problem Description**:
The Label Verification tab was redesigned to use `st.dataframe` for performance, but this introduced two severe UX regressions that make the tab unusable for its core purpose (visually reviewing face pairs and assigning labels).

**Symptoms**:
1. **Images no longer visible per row** — face thumbnails for each cluster were removed from the table rows. The only way to see them is to click a row and scroll down to the detail panel below. This makes it impossible to browse and compare pairs efficiently.
2. **Per-row Merge/Reject/Ignore buttons are gone** — the table uses row selection + bulk action buttons. Users can no longer click a single inline button per row; they must select a row in the dataframe, scroll down, and click a button in a separate panel. The interaction flow is broken.

**Suspicion**:
Root cause is the switch from per-row `st.columns` rendering (with inline images + buttons) to `st.dataframe`. `st.dataframe` does not support embedded interactive widgets (buttons) or multi-image cells natively. The performance fix (avoiding 150+ widgets) and the UX requirement (inline images + buttons per row) are in direct conflict with the current approach.

**Steps to Reproduce**:
1. Open Label Verification tab
2. Load any dataset
3. Observe: table shows only text columns, no face thumbnails in rows
4. Observe: no Merge/Reject/Ignore buttons per row

**Design Notes for Fix**:
- The core tension: inline images + buttons per row → many widgets → slow. `st.dataframe` → fast → no inline widgets.
- Possible approaches to reconcile:
  a. Return to per-row rendering but eliminate `st.rerun()` from button handlers — let `@st.fragment` handle reruns naturally without calling `st.rerun()` at all. This was never actually tested.
  b. Hybrid: `st.dataframe` for the filter/sort/select UX, with a separate fixed-height image strip that shows thumbnails for the current page (rendered outside the dataframe).
  c. Accept that inline images are expensive and show a fixed number of rows (e.g., 10) with full per-row rendering, paginated tightly.
- Before fixing, benchmark approach (a) — removing `st.rerun()` from inside the fragment may be sufficient to get acceptable performance with per-row rendering.

**Resolution**:
(pending)

---

### SIGHTING-027: Merge decisions unreliable for small clusters — no statistical reliability gating
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-27
**Persona**: ML Engineer

**Problem Description**:
The merge pipeline evaluates all candidate pairs with the same fixed thresholds regardless of cluster size. Features like `p10_cross_dist`, `support_fraction`, and `diameter_expansion` are computed from `n_cross_pairs = size_a × size_b` face pairings. For small clusters (e.g., size 2 vs size 3 → 6 pairs), these statistics are highly noisy and unreliable. A p10 of 0.58 from 6 pairs is nearly meaningless; the same value from 400 pairs is a strong signal.

The current `ConservativeMerger` does not distinguish two fundamentally different rejection reasons:
1. **Truly far apart** — clusters are different people; hard reject, should not retry.
2. **Insufficient statistics** — clusters may be the same person but there are too few faces to be confident; soft reject, should retry after neighboring merges grow the clusters.

**Symptoms**:
- Visual inspection of cluster pairs in the Label Verification tab confirms `p10_cross_dist ≈ 0.60` is a reliable separator — but only for pairs where both clusters have enough faces for stable statistics.
- Small clusters (size 1–3) produce highly variable p10/support values; the merge pipeline treats them identically to large clusters.
- Post-merge reclusters have near-zero candidate pairs, suggesting the base merge stage was too conservative on small clusters that later grew.

**Observations**:
- `p10_cross_dist ≈ 0.60` is the empirical threshold separating merges from rejects across all inspected datasets.
- Statistical reliability (variance of p10) scales with `n_cross_pairs = size_a × size_b`. The ~10th observation (p10) requires at least 10 pairs to be meaningful; i.e., `n_cross_pairs >= ~20` for basic stability.
- The existing `ConservativeMerger` is already iterative (one merge per iteration, recomputes stats after each merge) — the infrastructure for multi-phase merging already exists.
- The support gate partially captures this (`required_support = max(size * frac, min_count)`) but doesn't prevent evaluation of unreliable statistics in the first place.

**Hypothesis**:
A two-phase merge would improve quality:
- **Phase 1** (tight + statistically reliable): merge only pairs where `n_cross_pairs >= 20` AND `p10 < ~0.40`. Grows clusters until no more high-confidence merges remain.
- **Phase 2** (borderline): re-evaluate all pairs — including Phase 1 soft-rejects — now with larger clusters providing stable statistics. Use `p10 < 0.60` or ML model score.

The `ConservativeMerger` already loops and recomputes per-cluster stats after each merge; adding a phase parameter and `n_cross_pairs` minimum gate is the minimal change needed.

**Steps to Reproduce**:
1. Load any canonical run in the Label Verification tab (e.g., Germany_12).
2. Sort pairs by cluster size (size_a × size_b ascending).
3. Observe that small-cluster pairs (n_cross_pairs < 10) show inconsistent p10 values for visually similar faces.
4. Compare against large-cluster pairs where p10 clearly separates same-person from different-person.

**Resolution**: (open)

---

### SIGHTING-026: Quality gate passes poor-quality faces — det_score unused as filter criterion
**Status**: RESOLVED — 2026-04-24
**Severity**: Medium
**Reported**: 2026-04-24
**Persona**: ML Engineer

**Problem Description**:
Poor-quality faces (e.g., face_0548 — partial, occluded, or borderline detections) were passing quality gating and entering the clustering pipeline. The InsightFace SCRFD detector provides a detection confidence score (`det_score`, range 0–1) for every detected face, already captured in `FaceRecord.det_score` and exported to `faces.csv`, but was never evaluated as a quality gate.

**Symptoms**:
- face_0548 (user-reported example) described as poor quality but not filtered out
- `QualityGater._evaluate_gates()` evaluated only: blur, pose_yaw, pose_pitch, area — `det_score` absent
- Faces with det_score < 0.6 visible in faces.csv for existing runs

**Root Cause**:
`det_score` gate was never added to `_evaluate_gates()` when spec-012 added the field to `FaceRecord`. The field was computed and persisted but the quality module was never updated to consume it.

**Resolution**:
- Added `det_score_min: Optional[float] = None` to `PipelineConfig` (default off for backward compat)
- Added `_add_det_score_gate()` to `QualityGater`; wired into `_evaluate_gates()` and `_top_k_verdict()`
- `"det_score"` is first in rejection priority order (most fundamental gate)
- Permissive pass when `face.det_score is None` (legacy data without det_score)
- Added `det_score_min` number input to Run tab UI (Stage 3 section)
- Added 6 unit tests covering all gate states (all pass)
- Files: `face_cluster/config.py`, `face_cluster/quality.py`, `app/face_clustering/tabs/run_tab.py`, `tests/face_clustering/test_quality_gating.py`

**Findings**:
- "Decisions computed then discarded" was the systemic cause — see LEARNINGS 2026-04-22 entry
- Recommended starting value for new runs: `det_score_min=0.7`
- Remaining candidate improvements (not implemented): eye-region blur, embedding norm gate, landmark geometry check

---

### SIGHTING-025: Apply + Remerge can silently overwrite an existing sibling run directory
**Status**: RESOLVED — 2026-04-23
**Severity**: Medium
**Reported**: 2026-04-22
**Persona**: Senior SW Engineer

**Problem Description**:
During investigation of `D:\sim-bench\results\Noa2_5_1\`, the directory `merge_remerge_1` was written at least twice — first as the product of "Apply + Remerge" on `base` (~8:51 PM), and again at 9:02 PM as the product of another Apply + Remerge on `merge_snap_1`. The first version was clobbered in place; its `pipeline_run.json`, `merge_log.json` (originally 41 KB, now `[]`) and intermediate state are gone. The user also has no visible warning when this happens.

**Symptoms**:
- Two consecutive sessions both produced `merge_remerge_1/` — the later one overwrote the earlier without prompting.
- `merge_log.json` shrank from ~41 KB of real per-pair decisions to `[]` (the later remerge found 0 candidates).
- `session.json` / `run_history_db` lose a node — the run graph is silently lossy.

**Suspicion**:
- The Apply + Remerge action names output directories by fixed convention (`merge_remerge_<n>`) without checking for existence or branching on conflict.
- `_invalidate_run_caches()` + overwrite-in-place are chosen over rename/suffix-bump.

**Desired behavior**:
1. Refuse to overwrite an existing run directory, or auto-rename with a `_2`, `_3`, ... suffix.
2. Show the user the final output path before writing.
3. Record both runs in the session graph so history stays lossless.

**Steps to Reproduce**:
1. Run Apply + Remerge from any base run → creates `merge_remerge_1/`.
2. From that run, do another manual merge / remerge action that would produce `merge_remerge_1` again.
3. Observe the original folder contents replaced with no prompt.

**Fix direction**:
- `session_manager` should allocate a unique output dir name before dispatching the action. Central naming helper used by all session actions.

---

### SIGHTING-024: No cluster provenance/history — unclear whether a large cluster came from base clustering or merging
**Status**: PARTIALLY RESOLVED
**Severity**: High
**Reported**: 2026-04-22
**Persona**: ML Engineer / End User

**Problem Description**:
In `D:\sim-bench\results\Noa2_5_1\merge_remerge_1`, a suspiciously large cluster was observed but there is no way to tell how it was formed — was it a single base-clustering output, the result of an auto-merge, a manual merge, or produced by a remerge pass? There is also no way to revert it.

**Symptoms**:
- Cluster appears to pool faces of multiple identities (suspected over-merge or incorrect base clustering).
- `merge_log.json` is effectively empty (2 bytes), so merge provenance is not captured in the run artifacts.
- User cannot answer: "at which stage did this cluster get this big?" (base → auto-merge → manual-merge → remerge).
- No per-cluster "undo" or split affordance in the UI.

**Suspicion**:
- Run artifacts snapshot only the final state. Intermediate cluster snapshots (post-base, post-auto-merge, post-manual-merge, post-remerge) are not persisted per run.
- `merge_log.json` writer may only append when merges happen in that specific stage, losing cross-stage history across `remerge` invocations.
- The UI shows `clusters_merged.csv` with no link back to parent cluster IDs or the merge decision that produced it.

**Desired behavior**:
1. **Provenance field per cluster**: every cluster row carries `origin ∈ {base, auto_merge, manual_merge, remerge}` plus parent cluster IDs and a reference into `merge_log.json`.
2. **Stage snapshots**: persist `clusters_stage_{base,auto_merge,manual_merge,remerge_N}.csv` so the UI can show a timeline.
3. **Split UI**: per-cluster "History" panel showing the chain of operations that built it, and a "Split back to parents" action (at least one level of undo) for manual and auto merges.
4. **Timeline view**: a run-level view listing stages, counts of clusters, and merges applied per stage.

**Steps to Reproduce**:
1. Open `D:\sim-bench\results\Noa2_5_1\merge_remerge_1` in the face clustering app.
2. Find the large cluster; observe that `merge_log.json` is empty and no stage snapshots exist.
3. Try to determine how the cluster was formed or undo it — not possible.

**Resolution** (spec 012, 2026-04-22, partial):
- `clusters.csv` now carries `origin` (`base`/`auto_merge`/`manual_merge`/`remerge`) and `parent_cluster_ids` columns.
- `clusters_stage_base.csv` snapshot written before merge overwrites `clusters.csv`.
- `manual_merge_snapshot.py` writes `origin="manual_merge"` with parent IDs.
- "Provenance" expandable panel added to Cluster Analysis tab.
- Remaining: split/undo UI and cross-run timeline view are deferred to spec 013.

**Fix direction**:
- Extend `face_cluster/export.py` (or stage writers) to emit stage snapshots + a `cluster_origin` column.
- Ensure `remerge()` appends to, rather than overwrites, `merge_log.json` and records the stage.
- Add a "Cluster History" expander in `app/face_clustering.py` and a split/undo action wired to stored parents.

---

### SIGHTING-023: Exemplars view truncates — need "Show all exemplars" option
**Status**: RESOLVED
**Severity**: Medium
**Reported**: 2026-04-22
**Persona**: Frontend Engineer

**Problem Description**:
When inspecting a cluster in the face clustering app (`app/face_clustering.py`), only a limited number of exemplar faces are rendered. For large or mixed clusters the user needs to see the full exemplar set to judge identity consistency and spot contaminants.

**Symptoms**:
- Cluster detail view shows a fixed, small number of exemplars (top-K).
- No toggle/button to expand to the full ranked exemplar list.
- Hard to audit large clusters (see also SIGHTING-024) because only a slice is visible.

**Desired behavior**:
- Add a "Show all exemplars" toggle / "Load more" button on each cluster card.
- Order by exemplar score (descending) and show score + face_id next to each thumbnail.
- Keep the default truncated view for performance, but make the full set reachable in one click.

**Steps to Reproduce**:
1. Open `D:\sim-bench\results\Noa2_5_1\merge_remerge_1`.
2. Select any cluster with many faces.
3. Observe capped exemplar grid with no way to expand.

**Fix direction**:
- In the cluster gallery component, add `show_all_exemplars` state per cluster; when true, iterate the full ranked exemplar list from the analysis view instead of slicing.

---

### SIGHTING-022: Quality gating opaque — poor-quality faces enter pipeline without visible criteria/thresholds
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-04-22
**Persona**: ML Engineer

**Problem Description**:
In `D:\sim-bench\results\Noa2_5_1\merge_remerge_1` (e.g. Cluster 8), clearly low-quality faces made it into clustering. There is no per-face breakdown of which quality criteria were evaluated, what thresholds were used, and which gate each face passed/failed, so it is impossible to diagnose why a bad face entered or tune thresholds accordingly.

**Symptoms**:
- Cluster 8 contains faces that a human would reject on sight (blur / pose / occlusion / size).
- `faces.csv` does not expose per-gate decisions or the thresholds in force for the run.
- The UI does not show, per face: blur score, pose (yaw/pitch/roll), size, landmark confidence, detector score, and the pass/fail verdict of each gate with threshold.

**Suspicion**:
- `face_cluster/quality.py` computes a verdict but does not persist the per-criterion values and thresholds alongside each face.
- Pipeline config (`configs/pipeline.yaml` quality section) is not snapshotted into the run directory in a discoverable way.
- Some gates may be effectively disabled (threshold too permissive) without surfacing this to the user.

**Desired behavior**:
1. For every face, persist every quality metric (blur, yaw, pitch, roll, size_px, det_score, landmark_conf, …) and a per-gate verdict with the threshold used, in `faces.csv` or a sidecar `quality_report.csv`.
2. Snapshot the active quality thresholds into the run directory (e.g. `quality_config.json`).
3. Add a "Quality report" panel per face in the app showing metric, threshold, and pass/fail for each gate; highlight faces that entered despite being close to a threshold.
4. Add a run-level summary: counts of faces rejected per gate, and counts of faces that passed by a thin margin.

**Steps to Reproduce**:
1. Open `merge_remerge_1`, inspect Cluster 8.
2. Observe obviously bad faces; then try to determine which quality criteria they passed and with what thresholds — not possible from current artifacts/UI.

**Fix direction**:
- Extend `face_cluster/quality.py` to return a structured `QualityReport` per face and write it to disk.
- Dump the quality config block into the run directory alongside `pipeline_run.json`.
- Add a per-face expander in `app/face_clustering.py` that renders the report.

**Resolution** (spec 012, 2026-04-22):
- `QualityVerdict` + `GateResult` dataclasses added to `face_cluster/types.py`.
- `QualityGater.select_core_set()` now returns per-face verdicts; each gate's value and pass/fail persisted to `faces.csv` as `quality_<gate>_value` / `quality_<gate>_pass` columns; `quality_rejection_reason` records the first failing gate name.
- `quality_config.json` written to every run directory with the exact thresholds in force.
- `quality_summary` (rejected-per-gate counts, near-threshold counts) merged into `pipeline_run.json`.
- "Quality Report" expandable panel added to Face Analysis tab; graceful N/A on legacy runs.

---

### SIGHTING-021: Merge Analysis shows too many pairs — no transitive reduction, impractical UX
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-04-19
**Persona**: ML Engineer / End User

**Problem Description**:
The Merge Analysis tab in the face clustering app presents ALL pairwise merge candidates for manual review. On a 753-face, 39-cluster dataset (Austria24_2), this produced **226 candidate pairs across 23 pages**. The user approved 33 and rejected 193 — a tedious session where only 1 rejection was meaningful (a cluster with no actual faces, which also raises a filtering concern).

**Symptoms**:
1. **No transitive reduction**: If user approves A+B and A+C, B+C is still shown as a separate pair. Since A,B,C will all end up in the same merged cluster, asking about B+C is redundant.
2. **Too many low-value pairs**: Most of the 193 rejections were obvious non-merges that the algorithm's multi-evidence criteria would have rejected anyway. Only near-misses (3/4 gates passing) are worth human review.
3. **Faceless cluster appeared as candidate**: At least one candidate cluster contained no recognizable faces. Unclear how it passed filtering. Could be a crop_manifest issue or quality gate issue.
4. **Iterative remerge not reducing workload**: The remerge pipeline (spec 007) is supposed to re-evaluate on post-merge state, but it crashed (SIGHTING-021a below) so the user never saw reduced candidates.

**Desired behavior**:
- **Transitive grouping**: Group merge candidates into connected components. If {A,B,C} are all proposed candidates, show ONE decision: "Merge clusters A, B, C?" instead of 3 separate pairs.
- **Auto-approve high-confidence**: Pairs where all 4 gates (exemplar, support, margin, diameter) pass should be auto-merged or shown as "recommended" with one-click bulk approve.
- **Show only near-misses for review**: Focus human attention on pairs that passed 3/4 gates — the borderline cases where human judgment matters.
- **Better ordering**: Show highest-confidence merges first, not in arbitrary order.

**Reproduction**: Run pipeline on D:\Austria24, open Merge Analysis tab, count pages.

**Fix direction**:
1. In `MergeAnalysisView.compute()`, group candidates by connected component (union-find on proposed pairs). Present component-level decisions instead of pair-level.
2. Add a "Smart Approve" mode: auto-approve all 4/4-gate-pass pairs, show 3/4-gate-pass pairs for review.
3. Investigate faceless cluster — check crop_manifest and quality gate for empty/corrupt entries.

**Resolution** (2026-04-20):
- `face_cluster/merge.py`: `CandidateGroup` + `group_merge_candidates()` — pure algorithm, union-find grouping, cohesion classification (pipeline-ready interface for future integration).
- `face_cluster/analysis_views.py`: `MergeGroup` wrapper, `_build_merge_groups()`, `MergeAnalysisView` extended with `merge_groups`, `n_auto_approve`, `n_review`, `n_auto_reject`.
- `app/face_clustering.py`: Group gallery with per-group expand, Smart Approve button, group/flat toggle, smart pre-fill on load.
- Cohesion promotion: groups with >= 80% cohesion and min_gates >= 3 promoted to auto_approve even if not all pairs are 4/4.
- 12 new unit tests, all 106 face_clustering tests pass.
- Items 1 and 2 implemented. Item 3 (faceless cluster investigation) tracked separately as needed.

---

### SIGHTING-021a: Remerge pipeline crashes with KeyError in merge stage
**Status**: IN PROGRESS
**Severity**: Critical
**Reported**: 2026-04-19
**Persona**: Senior SW Engineer

**Problem Description**:
Clicking "Apply Approved Merges" triggers `save_manual_merge_snapshot()` + `PipelineConfig.remerge()`. The remerge pipeline crashes in the merge stage with `KeyError: <cluster_id>`.

**Root Cause Identified**:
Two bugs in `_load_source_remerge()`:
1. (Fixed 2026-04-18) `cluster_stats={}` — empty dict passed to `ClusterResult`. Merge stage accesses `cluster_stats[cluster_id]` → `KeyError`. Fixed by computing stats from distance matrix.
2. (Fixed 2026-04-18) `merge.py` used `cluster_stats[cluster_id_a]` — changed to `.get(cluster_id_a, {})` for defense.

**Why user still sees the error**: Streamlit caches imported modules in-process. The user must **restart the Streamlit app** (`Ctrl-C` and relaunch) for code changes to take effect. The `.pyc` cache was also stale (cleared).

**Verification**:
- Log confirms: `results/merge_remerge_1/logs/run_20260418_235733.log` — "Proposed 30 merge candidates" then crash.
- Traceback shows the NEW `.get()` source text but EXECUTES old `[]` bytecode (Python shows current file content in tracebacks, not what was compiled).
- Tests pass: `test_pipeline_remerge.py` (8 tests) all green after the fix.

**Steps to verify fix**: Restart Streamlit, re-run Apply Approved Merges on Austria24_2.

---

### SIGHTING-020: test_heuristic_pose_never_gates conflicts with intentional quality.py change
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-18
**Persona**: Senior SW Engineer / ML Engineer

**Problem Description**:
`test_quality_gating.py::ut_QualityGater::test_heuristic_pose_never_gates` creates a face with pose=(90, 85, 0) and asserts it passes quality gating when `use_pose_estimation=False`. The test's intent: InsightFace heuristic landmark-based pose values are unreliable (claim: produce 50-85° pitch for frontal faces) and must not gate faces.

However, `quality.py:215` now hardcodes `apply_pose_angles = True` with the comment "Pose angles come from InsightFace's 1k3d68 model — reliable for gating." This was an intentional design change. The result: the test fails because pose IS applied.

**Symptoms**:
```
AssertionError: Heuristic pose must not gate faces
assert 0 in []
```

**Suspicion**:
Two conflicting design intents:
1. Test author: InsightFace heuristic pose inflates pitch (50-85° for frontal) → unreliable → don't gate
2. quality.py change author: InsightFace 1k3d68 is reliable enough for gating → always gate

The real-world evidence: person_2's test images have InsightFace poses (roll=35.8°, yaw=-30.3°) that caused them to be quality-gated. Whether these are genuine tilts or heuristic artifacts is unclear.

**Steps to Reproduce**: `pytest tests/face_clustering/test_quality_gating.py::ut_QualityGater::test_heuristic_pose_never_gates`

**Resolution needed**: Design decision — should InsightFace 1k3d68 pose values be used for quality gating when `use_pose_estimation=False`?

---

### SIGHTING-019: Manual merge approval is one-shot — no iterative re-evaluation
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-04-17
**Resolved**: 2026-04-17
**Persona**: ML Engineer / Senior SW Engineer

**Problem Description**:
After clicking "Apply Approved Merges", the app applies the user's decisions in one shot and stops. It never re-evaluates candidates on the newly merged clusters. This means transitive merges — where cluster AB (result of merging A+B) is now close enough to merge with C — are never surfaced. The user only gets one round of review, regardless of how many more merges would become apparent after the first round.

The `ConservativeMerger` handles this correctly (iterative inner loop at `merge.py:109`), but `apply_manual_merges` is a one-shot union-find with no follow-up candidate evaluation. The Merge Analysis tab also never re-computes candidates on `st.session_state.merge_approval_result`.

**Symptoms**:
- Clusters that should have been merged are never shown as candidates because their `min_exemplar_dist` only drops below `merge_candidate_threshold` after another cluster is merged first
- Pairs don't appear in Near Misses because they were never proposed — exemplar distance is above `merge_candidate_threshold` (default 0.45) before the first-round merges happen
- User must do a full Recluster with relaxed `merge_candidate_threshold` to see additional candidates, losing their manual labels in the process

**Suspicion**:
The fix requires: after "Apply Approved Merges", re-compute `MergeAnalysisView` on the post-merge `ClusterResult` (using the same distance matrix) and present a fresh round of candidates for review. This loop should repeat until no new candidates are proposed.

**Steps to Reproduce**:
1. Run pipeline, open Merge Analysis tab
2. Approve a set of merges and click "Apply Approved Merges"
3. Notice that no new candidates are offered — even if the merged clusters are now close to additional clusters

**Fix direction**:
Add a "Re-evaluate candidates" button in the Merge Analysis tab (or make "Apply Approved Merges" automatically trigger a new candidate evaluation round). Re-run `MergeAnalysisView.compute()` on `st.session_state.merge_approval_result` using the original distance matrix. Loop until stable.

**Resolution** (2026-04-17):
Implemented spec 007 (config-driven pipeline + iterative manual merge). "Apply Approved Merges" now:
1. Calls `save_manual_merge_snapshot()` to write a self-contained snapshot of the current merge state
2. Runs `PipelineConfig.remerge(snapshot_dir, ...)` in an `_AsyncState` background thread
3. The remerge pipeline re-runs ConservativeMerger on the snapshot clusters, discovering transitive merge candidates
4. On completion, the Merge Analysis tab reloads fresh candidates from the new result
This fully replaces the old one-shot `apply_manual_merges` call.

---

### SIGHTING-018: Manual merge result not visible in Clusters (Merged) tab
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-04-14
**Persona**: Senior SW Engineer

**Problem Description**:
After clicking "Apply Approved Merges" in the Merge Analysis tab, the resulting `ClusterResult` is stored only in `st.session_state.merge_approval_result`. It is never routed to the Clusters (Merged) tab, so the user cannot browse the merged clusters with face images. "Save Decisions" writes `merge_decisions.json` to disk but nothing reads it back to re-apply the merges — the file is currently inert.

**Symptoms**:
- Clicking "Apply Approved Merges" shows only a cluster size table (no face images) on the Merge Analysis tab
- Clusters (Merged) tab still shows the heuristic merge result, not the manually approved one
- Page refresh loses the applied result entirely
- No path exists to visually browse manually-merged clusters

**Suspicion**:
The Clusters (Merged) tab reads from `result.merged_cluster_result` (loaded from disk). The manual result lives only in session state and is never written to disk or passed to that tab's rendering logic.

**Fix direction**:
Wire `st.session_state.merge_approval_result` into the Clusters (Merged) tab so it takes precedence over `result.merged_cluster_result` when present. Show a banner indicating the displayed result is the manually approved version. Optionally persist by writing the result to disk on "Save Decisions".

**Steps to Reproduce**:
1. Load a run with merge results in Merge Analysis tab
2. Approve some merge pairs and click "Apply Approved Merges"
3. Navigate to Clusters (Merged) tab — heuristic result still shown, not manual result

---

### SIGHTING-017: Embed cache tied to output_dir -- missed on every new run
**Status**: OPEN
**Severity**: Medium
**Reported**: 2026-04-14
**Persona**: Senior SW Engineer

**Problem Description**:
The embed cache (InsightFace detection + 512-d ArcFace embeddings) is stored at `{output_dir}/.embed_cache/`. Since `output_dir` changes per experiment run, the cache is never reused -- embedding re-runs from scratch every time even when the source images are identical.

**Symptoms**:
- Every new run with a different output directory repeats the full embed stage (~minutes for large albums)
- Cache only hits if the user manually re-enters the exact same output directory

**Suspicion**:
Cache location should be keyed off the **image directory** (which is stable across runs), not the output directory (which changes per experiment). Options: `{image_dir}/.face_embed_cache/`, or a shared location like `~/.sim_bench/embed_cache/{fingerprint}/`.

**Steps to Reproduce**:
1. Run pipeline with image_dir=`D:\photos`, output_dir=`results\run_001`
2. Run pipeline again with same image_dir, output_dir=`results\run_002`
3. Observe embed stage runs again despite identical source images

**Resolution**:
(pending)

---

### SIGHTING-016: exemplar_face_ids in clusters.csv are core-set indices, not face_ids
**Status**: RESOLVED
**Severity**: Critical
**Reported**: 2026-04-08
**Persona**: Senior SW Engineer

**Problem Description**:
`clusters.csv` writes `exemplar_face_ids` as raw values from `cluster_result.exemplars`, which are graph-local node indices (0..n_core-1). The export function correctly remaps cluster membership through `core_indices` (lines 52-62) but does NOT remap exemplar indices (line 94/99). This means the exemplar_face_ids column contains meaningless numbers that happen to look like face_ids.

**Symptoms**:
- Cluster Analysis tab shows exemplar crops that don't belong to the cluster
- Cluster 5 in Germany_8: actual members are face_id 47, 458. Exemplars in CSV: 26, 292 (which are core-set indices, not face_ids)
- Confirmed across all clusters in the run: `written_in_cluster? False` for 8/10 tested clusters

**Suspicion**:
`export.py` line 94: `exemplar_ids = cluster_result.exemplars.get(cluster_id, [])` — these are graph-local indices. Line 99: `json.dumps(exemplar_ids)` — writes them without mapping through `core_indices` then to `face.face_id`.

**Steps to Reproduce**:
1. Run pipeline with K=15 on Google_Germany (output: results/Germany_8)
2. Open clusters.csv, look at cluster 5: exemplar_face_ids = [26, 292]
3. Open faces.csv, filter cluster_id=5: actual members are face_id 47, 458
4. Check: core_faces[26].face_id = 47, core_faces[292].face_id = 458 — confirms index-space mismatch

**Root cause**:
Same class of bug as SIGHTING-015. The `cluster_result_for_export` snapshot is in graph-local coordinates. Export remaps the cluster membership mapping but not the exemplar list.

### SIGHTING-015: Live-run clusters contain random faces — graph-local index not remapped to face-list index
**Status**: RESOLVED
**Severity**: Critical — all live-run cluster assignments are wrong
**Reported**: 2026-04-07
**Persona**: Senior SW Engineer

**Problem Description**:
When running the pipeline from the app (not loading from history), `cluster_result.clusters` maps cluster_id to graph-local node indices (0..n_core-1) instead of indices into the full `faces` list. The app and all analysis views (`ClusterView`, `FaceView`, `ClusterDebugView`) treat these as face-list indices, causing every cluster to display the wrong faces.

**Symptoms**:
- Cluster 6 (user-reported example): shows face_0292 and face_0026 with distance 0.96 — impossible given threshold 0.35
- Face Analysis tab shows neighbors from wrong clusters
- Cluster metrics (diameter, distances) are computed on wrong faces
- Bug only affects live runs; history loads are correct because `loader.py` remaps correctly

**Root Cause**:
`ConnectedComponentsClusterer.cluster()` stores graph-node indices in `clusters` dict. `export.py` converts via `core_indices[graph_node]` for CSV output (correct). `loader.py` rebuilds from CSV (correct). But `pipeline.py` returns the raw `ClusterResult` to the app without converting — the two code paths (live vs loaded) produce different index semantics.

**Fix**:
In `pipeline.py`, after exemplar selection (which needs graph-local indices for the distance matrix) and before export, remap:
- `cluster_result.clusters`: `core_indices[node]` for each node
- `cluster_result.exemplars`: same remapping
- `cluster_result.labels`: rebuild as full-length array indexed by face-list position

Export receives a separate copy with raw indices since it does its own `core_indices` mapping.

**Reproduction**:
1. Run pipeline from app (not history load)
2. Open Cluster Analysis for any cluster
3. Check if face crops match — they won't (random faces from other identities)

**Verification**:
After fix, `cluster_result.clusters[cid]` contains face-list indices matching what `loader.py` produces. Both live-run and history-load paths now use the same index convention.

---

### SIGHTING-014: ModuleNotFoundError face_cluster in Streamlit despite venv active [CLOSED]
**Status**: CLOSED
**Severity**: High — blocks all app usage
**Reported**: 2026-04-04
**Persona**: Senior SW Engineer

**Problem Description**:
`streamlit run app/face_clustering.py` raises `ModuleNotFoundError: No module named 'face_cluster'` even with the venv activated. `python -c "import face_cluster"` succeeds from the project root.

**Root Cause**:
The editable install finder (`__editable___sim_bench_0_1_0_finder.py`) was generated when `face_cluster/` did not yet exist (or `pip install -e .` was not re-run after `face_cluster/` was added). Its `MAPPING` only contains `{'sim_bench': '...'}`. `face_cluster` is absent.

`python` from the project root works because Python adds the current working directory to `sys.path`. Streamlit adds the script directory (`app/`) to `sys.path`, not the project root — so `face_cluster` is not found.

**Fix**:
1. Add `face_cluster` explicitly to `setup.cfg` `packages` so it survives future `find:` misses
2. Re-run `pip install -e .` to regenerate the editable finder with updated MAPPING
3. Add post-install check to CI / dev setup docs

**Resolution**:
`setup.cfg` updated to list packages explicitly; `pip install -e .` re-run; CLAUDE.md updated with rule: re-run `pip install -e .` whenever a new top-level package directory is added.

---

### SIGHTING-013: App Crashes on Cluster Analysis — crop_manifest Format Assumption Wrong [CLOSED]
**Status**: CLOSED
**Severity**: High
**Reported**: 2026-04-04
**Persona**: Senior SW Engineer

**Problem Description**:
`_crop_for_face()` in `app/face_clustering.py` called `entry.get("crop_path", "")` on the manifest entry, assuming the format was `{face_id: {"crop_path": "..."}}`. The actual format written by `face_cluster/crops.py` is `{face_id: "relative/path/string"}`. This caused `AttributeError: 'str' object has no attribute 'get'` on every render of the Cluster Analysis and Face Analysis tabs.

**Symptoms**:
- App crashed with `AttributeError` immediately on switching to Cluster Analysis after a run
- Traceback pointed to `_crop_for_face` → `entry.get("crop_path", "")`

**Root Cause**:
`_crop_for_face` was written by assuming the manifest schema without reading `face_cluster/crops.py`. The two modules had an implicit, undocumented contract. No test exercised the actual read path from manifest → PIL Image.

**Resolution**:
- Fixed `_crop_for_face`: `return _load_crop(str(output_dir / entry))`
- Fixed same assumption in `_load_result_from_dir` (History tab)
- Added `test_crop_manifest_format_is_flat_string` regression test — asserts entry is `str`, resolves to existing file
- 19/19 tests passing

---

### SIGHTING-012: face_id Not Globally Unique — Crops Overwrite, Browse Shows Wrong Images [CLOSED]
**Status**: CLOSED
**Resolved**: 2026-04-03
**Severity**: Critical
**Reported**: 2026-04-03
**Persona**: Senior SW Engineer

**Problem Description**:
`face_id` resets to 0 on every call to `InsightFaceEmbedder.detect_and_embed()`. The pipeline calls this method once per image. Face IDs are therefore only unique within a single image — not across the run. With 1230 images the maximum face_id equals the maximum number of faces detected in any single image (11 in Google_Germany). Result: 1103 faces share only 12 unique IDs, crops overwrite each other, and the cluster browser shows the same 12 images across all 125 clusters.

**Symptoms**:
- `crop_manifest.json` has 12 entries for a 1230-image album
- `faces.csv` has 1103 rows but `face_id` only ranges 0–11
- Cluster browser shows repetitive images (same 12 crops across all clusters)
- `face_id` value counts are heavily skewed: face_id=0 appears 543 times

**Root Cause**:
`face_id_counter = 0` is declared as a local variable inside `detect_and_embed()` in `face_cluster/embedding.py:70`. It resets to 0 on every invocation. The pipeline at `face_cluster/pipeline.py:188` calls `embedder.detect_and_embed([str(img_path)])` once per image in a loop.

**Why E2E Test Didn't Catch It**:
- Test data is 9 solo-portrait images (1 face per image) — all face_ids=0
- Clustering uses list index, not face_id, so clusters are correct
- Purity/completeness tests use `image_path` for identity lookup, not face_id
- No test asserts face_id uniqueness

**Fix**:
In `face_cluster/pipeline.py`, after the embed loop, reassign face_ids sequentially:
```python
for new_id, face in enumerate(faces):
    face.face_id = new_id
```
Add a test asserting `len({f.face_id for f in result.faces}) == len(result.faces)`.

**Resolution**: Fixed in `face_cluster/pipeline.py` — after embed loop, reassign face_ids sequentially with `for new_id, face in enumerate(faces): face.face_id = new_id`. Added `test_face_ids_are_globally_unique` regression test in `tests/face_clustering/test_pipeline_e2e.py`. 24/24 tests passing.

**Steps to Reproduce**:
1. Run `FaceClusteringPipeline` on any album with more than 1 image
2. Check `faces.csv` — face_id column has repeated values

---

### SIGHTING-011: Playwright E2E Tests Connect to Wrong Port [RESOLVED]
**Status**: ✅ RESOLVED
**Severity**: Low
**Reported**: 2026-04-02
**Persona**: Senior SW Engineer

**Problem Description**: Playwright tests hardcode `http://localhost:8501` but Streamlit picks the next free port on each launch. When another Streamlit session is already running (or port 8501 is in use), the app starts on 8502 and tests fail with `ERR_CONNECTION_REFUSED`.

**Symptoms**: `playwright._impl._errors.Error: Page.goto: net::ERR_CONNECTION_REFUSED at http://localhost:8501/`

**Suspicion**: Hardcoded `APP_URL = "http://localhost:8501"` in `tests/face_clustering/test_streamlit_e2e.py`.

**Steps to Reproduce**: Start any Streamlit app on 8501, then run `pytest tests/face_clustering/test_streamlit_e2e.py -m e2e`.

**Resolution**: Fixed — `APP_PORT = 8502` updated. Long-term: make configurable via `--base-url` or env var `STREAMLIT_TEST_PORT`.

---

### SIGHTING-010: Empty clusters.csv — Core Index Mapping Bug in export.py [RESOLVED]
**Status**: ✅ RESOLVED
**Severity**: Critical
**Reported**: 2026-04-02
**Resolved**: 2026-04-07 (SIGHTING-015 fix)
**Persona**: Senior SW Engineer

**Problem Description**: Running the pipeline on `D:\Google_Germany` produces an empty `clusters.csv`. Even when clustering succeeds, all faces in `faces.csv` show `cluster_id = -1`.

**Root Cause Identified**: `cluster_result.clusters` stores **graph-node indices** (0 to n_core-1), but `export.py` maps them directly against the full `faces` list index. When not all faces pass quality gating, `core_indices = [3, 7, 11, ...]` — graph node 0 corresponds to `faces[3]`, not `faces[0]`. The export uses `face_to_cluster.get(i, -1)` where `i` is the full-list index, so all real cluster assignments are missed.

**Why E2E test passed**: The test used `blur_min=10.0`, causing all/nearly-all faces to be core, making `core_indices ≈ [0,1,2,...]`. Graph node index i == face list index i by coincidence. The bug was invisible.

**Steps to Reproduce**: Run pipeline on any dataset where quality gating rejects some faces (default `blur_min=50.0`). Check clusters.csv — empty despite valid clustering output in logs.

**Fix**: Pass `core_indices` to `export_results()`. Map graph node index `g` → `core_indices[g]` → full face list index.

---

### SIGHTING-009: CLI Script Crashes on Windows with Unicode Progress Bar [RESOLVED]
**Status**: ✅ RESOLVED
**Severity**: High
**Reported**: 2026-04-02
**Persona**: Senior SW Engineer

**Problem Description**: `scripts/run_face_clustering.py` crashes immediately at the discover stage with `'charmap' codec can't encode characters in position 15-34`.

**Root Cause**: `on_progress()` in the script uses `█` (U+2588) and `░` (U+2591) for a progress bar. Windows console uses `cp1252` (charmap) encoding by default, which cannot encode these Unicode block characters. The exception propagates into the `try/except` block inside `pipeline.run()`, which wraps it as `PipelineStageError("discover", ...)`.

**Steps to Reproduce**: `python scripts/run_face_clustering.py --images test_data/face_clustering/source_images --output results/test` on Windows.

**Fix**: Replace Unicode block chars with ASCII `#` and `-` in the progress bar.

---

### SIGHTING-008: Face Clustering Has No Cohesive Pipeline — Collection of Disconnected Scripts [IN PROGRESS]
**Status**: 🟡 IN PROGRESS — core implementation complete (17/17 tests pass), Playwright E2E on D:\Google_Germany pending
**Severity**: Critical
**Reported**: 2026-04-01
**Persona**: Senior SW Engineer + ML Engineer

**Problem Description**:
The face clustering subsystem is a collection of scripts that cannot be driven as a unified pipeline. The user cannot point at a directory of images and get clusters out end-to-end without manually chaining disconnected scripts. The Streamlit app has no A-to-Z run capability. There is no tested path from raw images → clusters that is guaranteed to work. The immediate trigger was `export_clustering_data.py` failing with `core=0, holdout=1094` (see Immediate Failure below).

**Symptoms**:
1. `scripts/export_clustering_data.py` requires PRE-EXTRACTED embeddings as input; there is no single entrypoint that accepts an image directory
2. Quality gating silently rejects ALL faces when SixDRepNet unavailable and landmark-based pose thresholds are strict — no fallback, no warning to user
3. `face_cluster/crops.py` and `face_cluster/export.py` referenced in RECOVERY_PLAN.md as required stages do NOT EXIST
4. No `face_cluster/pipeline.py` — no `FaceClusteringPipeline` class with a clean `run(image_dir, output_dir)` API
5. Streamlit apps (`face_clustering_labeling.py`, `face_clustering_debug/`) are read-only viewers with no ability to trigger the pipeline or see live progress
6. Zero A-to-Z tests covering the full detect → embed → quality-gate → cluster → export cycle on real test images
7. Immediate failure log:
   ```
   Computed pose for 0/1094 faces   ← SixDRepNet not available
   Quality gating results: core=0, holdout=1094
   Export failed: No faces passed quality gating!
   ```

**Immediate Failure Root Cause**:
`export_clustering_data.py --embeddings results/Google_Germany_FULL/embeddings.npy`:
- SixDRepNet not available → pose estimation returns 0 results for all 1094 faces
- Landmark-based heuristic poses already stored in FaceRecord; strict thresholds (`yaw_max=30°`, `pitch_max=25°`) filter out all faces in this large, varied dataset
- No fallback when external pose estimator unavailable: quality gate should gracefully degrade to blur+area only
- `blur_min=50.0` may also be too strict for faces loaded via the pre-computed embeddings path

**Structural Root Cause**:
`RECOVERY_PLAN.md` defines the correct architecture but it has NOT been implemented. Specifically missing:
- `face_cluster/crops.py` (Stage 3 in spec)
- `face_cluster/export.py` (Stage 5 in spec)
- `face_cluster/pipeline.py` (orchestrator with clean API)
- Tests in `tests/face_clustering/` for each stage

**Steps to Reproduce**:
```bash
python scripts/export_clustering_data.py \
  --embeddings results/Google_Germany_FULL/embeddings.npy \
  --output results/Google_Germany_FULL/clustering_export
# → Export failed: No faces passed quality gating!
```

**Resolution**:
See plan below — full re-implementation of face_cluster as cohesive sub-package.

**Findings**:
(to be filled after resolution)

---

### SIGHTING-007: Ground Truth Face Crop Mapping Mismatch [OPEN]
**Status**: 🔴 OPEN
**Severity**: Critical
**Reported**: 2026-03-31
**Persona**: ML/Testing Team

**Problem Description**:
The ground truth face crops in `test_data/face_crops_ground_truth/` do not correspond to the faces detected at the specified indices in `test_data/ground_truth_mapping.csv`. This breaks the full pipeline test and makes ground truth data unreliable for validation.

**Symptoms**:
1. **Full pipeline test fails with very low embedding similarity**:
   - Expected: >0.95 similarity between pipeline and ground truth embeddings
   - Actual: Many faces show <0.3 similarity, some negative
   - Examples: Face 558: -0.002, Face 557: 0.015, Face 546: 0.295

2. **Distance matrix correlation near zero**:
   - Expected: >0.90 correlation between pipeline and ground truth distance matrices
   - Actual: 0.0064 correlation (essentially random - no relationship)

3. **Visual verification confirms wrong faces**:
   ```
   Face 545: similarity 0.715 (moderate - possible match)
   Face 546: similarity 0.295 (low - wrong face)
   Face 550: similarity 0.612 (moderate)
   Face 551: similarity 0.623 (moderate)
   Face 557: similarity 0.015 (very low - completely different face)
   Face 558: similarity -0.002 (negative - completely different person)
   Face 562: similarity -0.012 (negative - completely different person)
   Face 569: similarity -0.027 (negative - completely different person)
   Face 573: similarity -0.003 (negative - completely different person)
   Face 580: similarity 0.670 (moderate - borderline)
   Face 584: similarity 0.393 (low - possibly wrong)
   Face 587: similarity 0.005 (very low - wrong face)
   Face 589: similarity 0.745 (moderate - borderline)
   Face 634: similarity 0.102 (very low - wrong face)
   Face 637: similarity 0.075 (very low - wrong face)
   ```

4. **Test results**:
   - ✓ `test_pipeline_extracts_all_faces`: PASSED (15/15 faces extracted with HEIC support)
   - ✗ `test_pipeline_embeddings_match_ground_truth`: FAILED (15/15 faces mismatched)
   - ✗ `test_pipeline_preserves_identity_structure`: FAILED (45 distance violations)
   - ✗ `test_distance_matrix_correlation`: FAILED (correlation 0.0064 instead of >0.90)

**Suspicion**:
1. **Different face detection run**: Ground truth crops may have been extracted from a different face detection run with different results (missed/extra faces causing index shift)

2. **Face ordering mismatch**: Mapping CSV may assume one ordering (e.g., confidence), but crops were created with different ordering or detection parameters

3. **Manual intervention**: Crops may have been manually created/edited/renamed without updating mapping CSV

4. **Source image version mismatch**: Source images in `test_data/source_images_ground_truth/` may differ from images used to create crops (different EXIF, different resolution, etc.)

5. **Database extraction bug**: The script (`scripts/query_face_lineage.py`) that created `ground_truth_mapping.csv` from database may have had a bug in index assignment

6. **SIGHTING-006 related**: Similar filename offset bug as SIGHTING-006 may have occurred during ground truth creation

**Steps to Reproduce**:
1. Run diagnostic script:
   ```bash
   python scripts/verify_face_index_mapping.py
   ```
   This creates side-by-side visual comparisons in `test_data/face_index_verification/`

2. Check visual comparisons - compare detected crops (left) vs ground truth crops (right)
   - Many show completely different faces or different orientations

3. Run full pipeline test:
   ```bash
   python -m pytest tests/test_face_pipeline_full.py -v -s
   ```
   - First test passes (15/15 faces extracted successfully with HEIC support)
   - Remaining 3 tests fail due to embedding mismatch

**Evidence Files**:
- `HEIC_AND_FACE_ORDERING_SUMMARY.md` - Full analysis and implementation summary
- `scripts/verify_face_index_mapping.py` - Diagnostic tool (creates visual comparisons)
- `test_data/face_index_verification/` - Visual comparisons (15 comparison images)
- `tests/test_face_pipeline_full.py` - Failing test suite
- `test_data/ground_truth_mapping.csv` - Suspect mapping file

**Impact**:

**BLOCKS**:
- Full pipeline end-to-end validation (cannot verify detect → align → embed pipeline works correctly)
- Ground truth expansion (cannot add more faces with confidence in mapping accuracy)
- Face ordering validation (cannot verify reading_order vs confidence_order produces correct results)
- Regression testing for face pipeline changes

**STILL WORKING**:
- ✅ Isolated crop test (`test_face_embeddings_ground_truth.py`) - All tests pass
- ✅ HEIC loading - All 15/15 faces extract successfully from HEIC files
- ✅ Face ordering utilities - Created and ready to use (`sim_bench/utils/face_ordering.py`)

**Resolution Options**:

**Option 1: Regenerate Ground Truth** (RECOMMENDED - cleanest solution)
1. Choose ordering convention: `reading_order` (user-requested, spatially intuitive) or `detection_order` (backward compatible)
2. Run face detection on all 15 source images with chosen ordering
3. Save aligned crops with deterministic filenames based on `(image_name, face_index)`
4. Manually label person identities (4 people, 15 faces)
5. Generate new `ground_truth_mapping.csv` with verified indices
6. Run diagnostic script to verify correspondence before committing
7. Document which ordering was used and creation timestamp

**Option 2: Reverse-Engineer Mapping** (preserves existing labels but risky)
1. For each of 15 ground truth crops, extract embedding
2. For each source image, detect all faces, extract all face embeddings
3. Find best match using cosine similarity (require threshold >0.95 to avoid ambiguity)
4. Update `ground_truth_mapping.csv` with discovered indices
5. Verify no ambiguous matches (multiple faces with high similarity)
6. Run full diagnostic to confirm correspondence

**Option 3: Investigate Original Creation Method** (forensic approach)
1. Check database `people.face_instances` for original creation metadata
2. Review `scripts/query_face_lineage.py` and `scripts/verify_ground_truth_traceability.py` for export logic
3. Check git history for when crops were created and what script was used
4. Determine if there's a systematic bug in database export or crop creation
5. Fix bug and regenerate if found

**Option 4: Manual Verification** (labor-intensive but thorough)
1. For each of 15 faces, manually open source image
2. Visually identify which detected face corresponds to ground truth crop
3. Update mapping CSV with correct indices
4. Re-run tests to verify

**Recommended Approach**: **Option 1** (Regenerate)
- Clean slate, no forensics needed
- Use deterministic `reading_order` as requested by user
- Documented creation process prevents future issues
- Can be completed in <1 hour

**Next Steps** (awaiting user decision):
1. User chooses resolution approach
2. If Option 1: Decide on ordering convention (reading_order vs detection_order)
3. Execute chosen approach
4. Verify with diagnostic script
5. Update tests and documentation
6. Add creation metadata to prevent recurrence

**Related Issues**:
- SIGHTING-006: Face Crop Filenames Don't Match Face IDs (+2 Offset) - similar filename mapping bug
- EMBEDDING_EXTRACTION_MYSTERY.md - unsolved 727-face corruption issue
- EMBEDDING_CORRUPTION_ROOT_CAUSE_ANALYSIS.md - caching and lineage issues

**Findings**:
(to be filled after resolution)

---

### SIGHTING-006: Face Crop Filenames Don't Match Face IDs (+2 Offset)
**Status**: ✅ RESOLVED
**Severity**: Critical
**Reported**: 2026-03-20
**Updated**: 2026-03-23
**Resolved**: 2026-03-23
**Persona**: Senior SW Engineer

**Problem Description**:
Regenerated embeddings from face crops have systematic +2 offset - `stored[N]` contains embedding for `face_{N+2}_aligned.jpg` instead of `face_{N}_aligned.jpg`.

**Symptoms**:
- Fresh vs stored embeddings show massive distances (0.78+)
- Comparing fresh embeddings: `stored[569]` matches `fresh[571]` (distance 0.0)
- Pattern confirmed for all test faces: `stored[N]` = `fresh[N+2]`
- FRESH metadata claims face_ids = [0,1,2,...] but crops don't match

**ROOT CAUSE IDENTIFIED** (2026-03-23):

**Location**: `scripts/benchmark_face_clustering.py:328-347`

**The Bug**:
```python
def save_face_crops(metadata: List[Dict[str, Any]], config: CropConfig) -> List[int]:
    saved_count = 0  # Counter for filenames
    for i, face_meta in enumerate(metadata):  # i = metadata index (0,1,2,...)
        if save_single_face_crop(face_meta, saved_count, config):  # ← PASSES saved_count
            saved_indices.append(i)
            saved_count += 1  # Only increments on SUCCESS
```

**Line 222**: `prefix = f'face_{index:04d}'` (uses `saved_count`, not metadata index)

**What Happens**:
1. Loop iterates metadata with index `i` (0, 1, 2, 3, ...)
2. But passes `saved_count` to `save_single_face_crop()` for the filename
3. `saved_count` only increments when crop saves successfully
4. If faces 0 and 1 fail validation (invalid bbox, missing landmarks) → NOT saved
5. `saved_count` stays at 0 when processing face 2
6. Face 2 (metadata index 2) gets saved as `face_0000.jpg` (should be `face_0002.jpg`)
7. All subsequent faces shifted: metadata[N] → saved as `face_{N-2}.jpg`

**Why +2 Specifically**:
First 2 faces in metadata failed validation checks:
- Face 0: Invalid bbox OR missing landmarks
- Face 1: Invalid bbox OR missing landmarks
- Face 2+: Valid, but saved with shifted filenames

**Impact**:
- `stored[569]` = embedding from `face_0569.jpg` = metadata[569]'s data
- But `face_0569.jpg` actually contains face from metadata[571] (because 2 faces were skipped)
- `fresh[571]` = embedding from `face_0571.jpg` = same physical face
- Distance(stored[569], fresh[571]) = 0.0 ← Perfect match!

**Verification Tools Created**:
1. `scripts/debug_sighting_006/run_debug.py` - Systematic hypothesis testing
2. `scripts/debug_sighting_006/trace_crop_save_logic.py` - Simulates save logic to find skipped faces
3. `scripts/debug_sighting_006/verify_crop_metadata_consistency.py` - Checks alignment

**Debugging Steps Completed**:
1. Created `notebooks/debug_embeddings_comparison.ipynb`
2. Compared 3 extraction methods → proved extraction code correct
3. Searched all 727 faces → found systematic +2 offset
4. Created modular debug framework with 6 hypothesis tests
5. Deep code analysis → pinpointed exact bug location (line 340)
6. Created trace script to identify which faces were skipped

**Steps to Reproduce**:
1. Run `benchmark_face_clustering.py` on dataset where first 2 faces fail validation
2. Observe: `face_0000.jpg` exists but corresponds to metadata[2], not metadata[0]
3. Run trace script: `python scripts/debug_sighting_006/trace_crop_save_logic.py --metadata <path> --crops <dir>`

**Resolution**:
✅ **FIXED** - Changed line 341 to use metadata index `i` instead of `saved_count`

**Fix Implemented** (Option 3 - Use metadata index):
```python
# Line 341: Changed from
if save_single_face_crop(face_meta, saved_count, config):

# To
if save_single_face_crop(face_meta, i, config):  # Use metadata index i
```

**Result**:
- Crop filename now matches metadata index
- If faces 0,1 fail → `face_0000.jpg` and `face_0001.jpg` don't exist (gaps OK)
- Face 2 saves as `face_0002.jpg` ✅ (not `face_0000.jpg`)
- Alignment preserved: `metadata[N]` ←→ `embeddings[N]` ←→ `face_{N:04d}.jpg`

**Validation Added**:
- `validate_crop_filenames()` function catches this bug immediately
- Runs after crop saving, before data is written to files
- Three test cases verify detection and prevention
- Tests pass: `pytest tests/test_crop_validation.py`

**Prevention**:
1. Add assertion after `save_face_crops()`:
   ```python
   assert all(crop_files[i] == f"face_{metadata[saved_indices[i]]['face_id']:04d}_aligned.jpg")
   ```
2. Use face_id from metadata for ALL filenames (never use loop counters)
3. Add test: `test_crop_filename_matches_metadata_face_id()`

**Testing Method Created**:
`tests/test_face_crop_integrity.py` - Verifies crop filenames match embeddings

---

### SIGHTING-005: Pre-clusters Contain Mixed People (Transitive Closure Problem)
**Status**: OPEN
**Severity**: Critical
**Reported**: 2026-03-06
**Persona**: ML Engineer

**Problem Description**:
Pre-merge clusters (from initial kNN + connected components clustering) already contain mixed people before any merging happens. Merge stage analysis revealed the real problem is in initial clustering, not merging.

**Specific Examples**:
- Pre-cluster 3: Contains multiple different people
- Pre-cluster 6: Contains multiple different people

**Symptoms**:
- User examining merge history to understand over-merged clusters
- Discovered that even PRE-MERGE clusters have wrong photos
- Individual faces in pre-clusters are far from exemplars (dist > 0.6 when threshold is 0.4)
- Some faces closer to OTHER clusters' exemplars than their own

**Root Cause Hypothesis**:
**Transitive Closure Problem** in kNN graph:
```
Face A → Face B → Face C
  │                  │
Person 1          Person 2

- Face A and B are similar (same person)
- Face B and C are similar (different person, but B is "bridge")
- A and C are NOT similar (different people)
- But connected components groups them together!
```

**Current Parameters**:
```python
K = 5                      # kNN neighbors
distance_threshold = 0.35  # Edge creation
min_cluster_size = 2

Process:
1. Build mutual kNN graph
2. Prune edges > 0.35
3. Connected components → clusters
```

**Why It Fails**:
1. **Threshold too loose**: 0.35 allows some cross-person connections
2. **Transitive connections**: A→B→C even if A≠C
3. **No outlier removal**: Bridge faces create wrong clusters
4. **No coherence check**: Cluster can have high diameter (0.6+)

**Investigation Needed**:
1. **Why are incorrect faces connected?**
   - What are their kNN neighbors?
   - Which edges connect them to the cluster?
   - Are they bridges between two identity groups?

2. **What metrics predict incorrect clustering?**
   - Distance to nearest exemplar?
   - Fraction of kNN neighbors outside cluster?
   - Alternative cluster assignments?

**Proposed Diagnostics**:
1. Pre-Cluster Explorer UI
2. Face-level kNN neighbor analysis
3. Graph path visualization
4. Alternative assignment analysis
5. Outlier detection metrics

**Resolution**:
Phase 1: Build diagnostics to understand problem (in progress)
Phase 2: Implement solution (TBD after analysis)

**Findings**:
(to be filled after diagnostic analysis)

---

### SIGHTING-004: Same Cluster Pair Merging Multiple Times in Merge Log
**Status**: RESOLVED
**Severity**: Medium (Design Confusion)
**Reported**: 2026-03-06
**Resolved**: 2026-03-06
**Persona**: ML Engineer

**Problem Description**:
Merge decisions log shows the same cluster pair (e.g., cluster 0 + 6) merging multiple times across different iterations. Once two clusters merge, one disappears and they become a single entity - they should never merge again. Either the merge logic has a bug (not actually merging), or the decision log is recording duplicates incorrectly.

**Symptoms**:
- User observing: "Pre-cluster 6 → Cluster 0" appearing multiple times in merge history
- Merge decisions CSV has 6,196 total decisions (1,454 merged, 4,742 rejected)
- For 727 faces going from 75→27 clusters (48 merges), why are there 1,454 "merged" decisions?
- Math doesn't add up: 48 merges should = 48 decisions, not 1,454

**Investigation Results** (via `debug_merge_duplicates.py`):
```
✅ actually_merged=True: 48 entries (correct!)
❌ actually_merged=False: 1,406 entries (valid candidates not chosen)
❌ 96 unique pairs across all iterations
❌ Example: Cluster 3 + 43 appeared 44 times (iterations 1-44)
   - Iterations 1-43: actually_merged=False (valid but not best)
   - Iteration 44: actually_merged=True (finally chosen)
```

**Root Cause**:
**Confusing terminology in decision logging.**

`action='merged'` means "this pair COULD merge" (passed validation checks), NOT "this pair DID merge".

In each iteration:
- 30-60 pairs are valid merge candidates → logged with `action='merged'`
- Only 1 pair actually executes → marked with `actually_merged=True`
- Same pair can be valid candidate in multiple iterations before finally being chosen

Example: Cluster 3 + 43
- Iterations 1-43: Valid candidate but another pair was better → `action='merged'`, `actually_merged=False`
- Iteration 44: Best candidate, actually executed → `action='merged'`, `actually_merged=True`

**Resolution**:
✅ Merge logic is CORRECT - clusters properly merge and disappear
✅ Decision logging is CORRECT - tracks all valid candidates for analysis
❌ UI was WRONG - filtered by `action='merged'` instead of `actually_merged=True`

**Fixes Applied**:
1. Updated UI to filter by `actually_merged=True` (shows only 48 actual merges)
2. Deduplicated rejected attempts (show last attempt per unique pair)
3. Added caption explaining duplicate candidates

**Findings**:
- `action='merged'` should be renamed to `action='valid_candidate'` for clarity
- Logging ALL candidates is useful for analysis (why wasn't this pair chosen?)
- UI must distinguish between "could merge" vs "did merge"
- This explains why merge analysis was unusable - showing 1,454 candidates instead of 48 merges!

---

### SIGHTING-003: pre_merge_cluster_id Column Not Being Added to faces.csv
**Status**: RESOLVED
**Severity**: High (User Error)
**Reported**: 2026-03-06
**Resolved**: 2026-03-06
**Persona**: Senior SW Engineer

**Problem Description**:
Export script runs without errors but fails to add `pre_merge_cluster_id` column to faces.csv, preventing diagnostic tabs from working in labeling app.

**Symptoms**:
- Export script completes successfully
- All diagnostic files created (merge_decisions.csv, pre_merge_clusters.csv, cluster_lineage.json)
- BUT faces.csv missing `pre_merge_cluster_id` column
- Labeling app shows: "Missing: pre_merge_cluster_id column in faces.csv"
- User has re-run export 3+ times with same result

**Suspicion**:
1. Code adds column during faces_data construction, but something prevents it from being included
2. Maybe pre_merge_result is None despite merge being enabled?
3. Maybe column is being added but saved to wrong file?
4. Maybe there's a conditional that's preventing the column from being added?

**Steps to Reproduce**:
1. Run: `python scripts/export_clustering_data.py --embeddings <path> --output <dir>`
2. Check faces.csv columns
3. Column `pre_merge_cluster_id` is missing

**Debug Process**:
Created `scripts/debug_export_issue.py` to systematically check files, columns, and logs.

**Resolution**:
**Root Cause**: User was loading OLD export directory in labeling app, not the NEW export directory.
- Export worked correctly: `results\Google_Germany\clustering_export` ✅ (column exists)
- App was loading: `results/face_clustering_training/diagnostic_test` ❌ (old data)

Debug script revealed:
```
✅ pre_merge_cluster_id column EXISTS
Sample values: [-1, -1, -1, -1, 0, -1, 1, 2, 0, -1]
Unique values: 76
```

**Solution**: User changed app to load correct directory, all tabs work.

**Findings**:
- Export code working correctly from the start
- Debug script invaluable for systematic diagnosis
- UI should show current loaded directory path more prominently
- Consider adding validation: if export_summary.json shows diagnostic files in metadata, but columns missing, show clear error

---

### SIGHTING-001: Face Alignment Not Working - Upside Down Faces Not Corrected
**Status**: IN PROGRESS
**Severity**: Critical
**Reported**: 2026-02-19
**Persona**: Senior SW Engineer
**Implementation Started**: 2026-02-19

**Problem Description**:
Face alignment is failing to properly orient faces. An upside-down face (requiring ~180° rotation) is only being rotated 6° instead of the correct angle. The entire face processing pipeline in `extract_face_embeddings.py` has severe architectural issues with mixed responsibilities.

**Symptoms**:
- Face #118 (20250822_194950.jpg) is upside down - eyes at bottom, mouth/nose at top
- System identifies landmarks correctly (knows eyes are down) but rotates only 6° instead of ~180°
- Aligned face crops show incorrectly oriented faces
- Landmarks on debug panel don't match actual facial features

**Suspicion**:
1. `roll_angle` calculation only considers eye-line angle (±90° max), not full orientation
2. 5-point alignment uses `estimateAffinePartial2D` which may not handle 180° flips
3. Mixed code paths: MediaPipe vs InsightFace detection handled in same step
4. No validation that alignment actually corrected orientation

**Architectural Issues Identified**:
1. `extract_face_embeddings.py` combines:
   - Face source detection (MediaPipe vs InsightFace)
   - Face cropping
   - Face alignment (5-point or roll-based)
   - Embedding extraction
   - Caching logic
2. No separation of concerns - can't test alignment independently
3. No unit tests for alignment correctness
4. Violates single responsibility principle
5. Multiple code paths for different face detection backends

**Steps to Reproduce**:
1. Run face clustering on album containing upside-down faces
2. Open Face Clustering Debug app
3. Go to Overview → Gallery
4. Click 🔍 on an upside-down face
5. Observe: landmarks don't match, face orientation is wrong

**Root Cause Analysis (2026-02-19)**:

1. **`compute_roll_angle()` only measures eye-line tilt, NOT face orientation**
   - Uses `atan2(dy, dx)` on eye positions → returns -180° to +180° but just for eye tilt
   - For upside-down face: eyes at bottom, but roll_angle ≈ 0° (eyes level, just upside down)
   - Missing: check if nose is BELOW eyes, mouth is BELOW nose

2. **5-point alignment can't fix upside-down faces**
   - `estimateAffinePartial2D` computes similarity transform (rotation + scale + translation)
   - For upside-down landmarks → tries to match to upright template → finds small rotation with high error
   - Would need 180° pre-rotation BEFORE 5-point alignment

3. **No validation that alignment worked**
   - No check that transformed landmarks are close to reference positions
   - Should fail loudly if alignment error is high

**Required Architecture Changes**:

```
CURRENT (broken):
insightface_detect_faces → filter_faces → score_face_frontal → extract_face_embeddings
                                              ↑                        ↑
                                         roll_angle only         alignment + crop + embed
                                         (no orientation)        (too many responsibilities)

PROPOSED (clean):
insightface_detect_faces
    → filter_faces
    → detect_face_orientation   [NEW - determines 0°/90°/180°/270° rotation needed]
    → align_faces               [NEW - applies rotation + 5-point alignment]
    → validate_alignment        [NEW - checks alignment quality, flags bad ones]
    → crop_faces                [NEW - just cropping, separate from alignment]
    → extract_face_embeddings   [SIMPLIFIED - just embedding extraction]
```

**Unit Tests Needed**:
1. `test_face_orientation_detection.py` - test with 0°, 90°, 180°, 270° rotated faces
2. `test_face_alignment.py` - test that alignment produces expected landmark positions
3. `test_alignment_validation.py` - test that bad alignments are detected

**Specific Test Case (Face #118)**:
```python
def test_upside_down_face_orientation():
    """Face with eyes at bottom, mouth at top should detect 180° rotation needed."""
    landmarks = [
        [100, 150],  # left_eye (at BOTTOM)
        [200, 150],  # right_eye (at BOTTOM)
        [150, 100],  # nose (ABOVE eyes - wrong!)
        [110, 50],   # left_mouth (at TOP - wrong!)
        [190, 50],   # right_mouth (at TOP - wrong!)
    ]
    orientation = detect_face_orientation(landmarks)
    assert orientation == 180, f"Expected 180° rotation, got {orientation}°"
```

**Resolution**:
Implementation in progress. New architecture implemented:

1. **Created `detect_face_orientation` step** (`sim_bench/pipeline/steps/detect_face_orientation.py`)
   - Analyzes 5-point landmarks to detect 0°/90°/180°/270° rotation
   - Checks vertical relationships (eyes above nose, nose above mouth)
   - Stores `orientation_angle` and `orientation_confidence` in face_info

2. **Created `align_faces` step** (`sim_bench/pipeline/steps/align_faces.py`)
   - Pre-rotates image by detected orientation angle
   - Transforms landmarks to rotated coordinates
   - Applies 5-point affine alignment to ArcFace template
   - Stores aligned crops in `context.aligned_faces`

3. **Created `validate_alignment` step** (`sim_bench/pipeline/steps/validate_alignment.py`)
   - Runs face detection on aligned crops
   - Verifies landmarks are near expected reference positions
   - Flags faces with high alignment error

4. **Created `crop_faces` step** (`sim_bench/pipeline/steps/crop_faces.py`)
   - Simple bbox cropping without alignment (for debug)

5. **Updated `extract_face_embeddings`**
   - Fixed imports (moved to top of file)
   - Now uses pre-aligned faces from `align_faces` step when available
   - Falls back to inline alignment for backward compatibility

6. **Added unit tests**
   - `tests/pipeline/test_face_orientation_detection.py` (15 tests)
   - `tests/pipeline/test_face_alignment.py` (16 tests)
   - Includes regression test for Face #118 (upside-down face)

7. **Updated pipeline.yaml**
   - Added `detect_face_orientation` and `align_faces` to default_pipeline
   - Added configuration sections for new steps

**Findings**:
- `compute_roll_angle()` measures eye-line tilt, NOT face orientation
- Face orientation requires checking spatial relationships (is nose below eyes?)
- 5-point alignment must be preceded by orientation correction for rotated faces
- Single-responsibility steps are testable and debuggable
- See `docs/LEARNINGS.md` for prevention guidelines

---

### SIGHTING-002: ML Cluster Merging Scripts Produce Different Results Than Working Notebook
**Status**: RESOLVED
**Severity**: High
**Reported**: 2026-02-28
**Resolved**: 2026-02-28
**Persona**: ML Engineer

**Problem Description**:
The ML cluster merging implementation (Phase 1 of PLAN_ML_CLUSTER_MERGING.md) has working code in `notebooks/debug_knn_graph_clustering.ipynb`, but the production scripts built to replicate this behavior produce completely different clustering results. A comparison script was also built but crashes when comparing the outputs.

**Symptoms**:
1. **Notebook works correctly**: Produces 274 faces, 25 clusters (11 after merge), 140 noise faces
2. **Export script produces wrong results**: Only 254 faces (missing 20), 24 clusters, 137 noise
3. **45.3% of faces have different cluster assignments** between notebook and script
4. **Comparison script crashes** with IndexError when trying to analyze differences (face IDs missing from one dataset)
5. Example: Notebook cluster 3 has 6 faces [12, 13, 14, 29, 35, 258], export cluster 3 has only 4 faces [12, 13, 14, 238]

**Suspicion**:
1. Export script may be using different quality gating parameters (why 20 fewer faces?)
2. Clustering pipeline may have subtle differences (order of operations, random seeds, tie-breaking)
3. ConservativeMerger may produce different results due to threshold calculation differences
4. Scripts were "reinvented" instead of extracting working code from notebook

**Steps to Reproduce**:
1. Run `notebooks/debug_knn_graph_clustering.ipynb` → exports `notebook_clustering_results.csv` (274 faces, 11 clusters after merge)
2. Run `python scripts/export_clustering_data.py --embeddings results/face_clustering_benchmark/embeddings_*.npy` → exports `faces.csv` (254 faces, different clustering)
3. Run `python scripts/compare_clustering_results.py` → crashes with IndexError

**Error Output**:
```
IndexError: single positional indexer is out-of-bounds
  File "D:\sim-bench\scripts\compare_clustering_results.py", line 68
    export_cluster = export_df[export_df['face_id'] == face_id]['cluster_id'].iloc[0]
```

**Root Cause**:
**False alarm - scripts were being compared on DIFFERENT input data.**

Investigation revealed:
1. Original comparison used `embeddings_2026-02-16_00-44-28.npy` (254 faces) for export script
2. Notebook automatically uses most recent embeddings: `embeddings_2026-02-20_10-19-23.npy` (274 faces)
3. When export script run on SAME embeddings as notebook: **0 mismatches (100% match)**

Secondary issue: Comparison script crashed with IndexError when face IDs were missing (fixed)

**Resolution**:
1. ✅ **Created `scripts/export_from_notebook_logic.py`** - Extracted exact notebook code for validation
2. ✅ **Fixed `scripts/compare_clustering_results.py`** - Handle missing face IDs gracefully
3. ✅ **Verified `scripts/export_clustering_data.py` is correct** - Produces identical results to notebook when given same input
4. ✅ **Created `scripts/compare_notebook_vs_export.py`** - Clean comparison on same embeddings

**Verification**:
```bash
# Run export on same embeddings as notebook
python scripts/export_clustering_data.py --embeddings results/face_clustering_benchmark/embeddings_2026-02-20_10-19-23.npy

# Compare results
python scripts/compare_notebook_vs_export.py
# Output: Mismatched assignments: 0 / 274 faces (0.0%)
```

**Findings**:
- Export script implementation is CORRECT - exact match with notebook
- Always compare on SAME input data when validating implementations
- Comparison scripts should handle edge cases (missing IDs, different row counts)
- See `docs/LEARNINGS.md` for prevention guidelines

**Prevention**:
When comparing notebook vs script outputs:
1. **Always use the same input data** - explicitly specify paths, don't rely on "most recent" logic
2. Document which embeddings file was used in comparison
3. Build comparison scripts that handle different-sized datasets gracefully
4. Compare metadata (file sizes, row counts) FIRST before diving into details
