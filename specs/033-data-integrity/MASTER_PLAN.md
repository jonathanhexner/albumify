# Master Plan — Data Integrity Cleanup (spec-033)

**Triggered by**: SIGHTING-059
**Status**: Implemented (P-A through P-H landed 2026-05-15 in a single session per user instruction "run straight through all 7 remaining phases").
**Date**: 2026-05-15

## What this is

One document with: the problem, the plan, the decisions. Companion reference: [`pipeline_map.html`](pipeline_map.html) for deep current-state tables (DB schema, full context↔storage mapping, etc.).

---

## 1. Premise

- **No logic changes.** Same clustering, same merge, same thresholds.
- **No new ML.** No new gates, no new filters.
- The pipeline is producing less-good results on Albumify than on FC App standalone, and per-image inspection is fragmented. Goal: **make order**.

## 2. Root cause in one paragraph

Face clustering is not a separate pipeline step — it runs **inline inside `cluster_people`** via `face_cluster_bridge.py`. That bridge does two destructive things: (1) it drops five fields when converting `FaceForClustering` → `FaceRecord` (blur, pose, det_score, landmarks, aligned_face), and (2) it force-disables the five quality gates that would have caught the consequences (`yaw_max=999`, `pitch_max=999`, `roll_max=999`, `blur_min=0.0`, `det_score_min=None` — hardcoded literals at `bridge.py:66-71`). Every SIGHTING-059 symptom — cluster 6 with no crops, blur=0 for 428 faces, det/yaw/pitch/roll=NULL, empty `filter_decisions` table — is downstream of that one bridge.

The FC App standalone pipeline doesn't have this bridge, which is why it produces good results.

## 3. The six bridge field-drops (and what they cost)

| FaceRecord field | bridge.py:48-52 | Today's value | Consequence |
|---|---|---|---|
| `blur_score` | `getattr(face, "blur_score", 0.0)` | 0.0 for every face | blur gate disabled; all faces "pass" |
| `pose` | `getattr(face, "pose", None)` | None for every face | yaw/pitch/roll gates disabled |
| `det_score` | `getattr(face, "det_score", None)` | usually None | det gate disabled |
| `landmarks` | not extracted | None | can't re-align downstream |
| `aligned_face` | not extracted | None | unalignable faces become exemplars (cluster 6 / face_46) |
| `area` | `w * h` of normalized bbox | fraction² (~0.001…0.45) | "Area 0 px²" in UI; area gate compares fraction to px |

## 4. The gap table — UI vs reality

| UI control | Today | Reality |
|---|---|---|
| "Min Face Size (px)" `config_min_face_size` | binds to detect_faces + 3 scoring steps | does NOT reach the face-clustering filter |
| (no UI) `min_face_ratio = 0.005` | hidden | the actual face-size rejector |
| FC App "min_face_area px" | binds to FC standalone gate | no effect on Albumify runs |
| Min IQA / Min Sharpness | binds to filter_quality | works (image-level) |
| Blur / yaw / pitch / roll knobs (FC App) | binds to face_cluster gates | force-disabled by bridge.py:66-71 on Albumify |
| `aligned_face is None` rejection | no UI, no gate | missing — unalignable faces reach exemplar status |

---

## 5. The six phases

Each phase = one PR. Each PR ships its own test.

### P-A · UI label honesty (no logic change)

**Scope**: every widget in `pipeline_runner.py`, `run_tab.py`, `recluster_tab.py`, `merge_controls.py`.

- Add `help=` text to every face-clustering control declaring (a) which step config it writes to, (b) which filter consumes it, (c) units.
- Surface hidden `min_face_ratio` as a slider (default 0.005 — unchanged). **[locked decision]**
- FC App `min_face_area` gets relabel + "no effect on Albumify runs" note.
- Disabled-by-bridge knobs get a "no effect on Albumify runs" note.

**Acceptance**: every face-clustering control has an honest label + unit + scope note. `test_ui_aligns_with_filters.py` still green.

### P-B · Context contract (NEW spec-034)

**Scope**: write `specs/034-pipeline-context-contract/spec.md` — a table of every field on `PipelineContext` and `_RunContext` with: type, unit, producer step, consumer step(s), persisted? (yes→DB column / no→reason), lifecycle.

**Acceptance**: every dataclass field has exactly one row in the spec. Architecture test `test_pipeline_context_contract.py` fails CI on drift.

### P-C · Storage plumbing + Pydantic FaceRecord

This is the load-bearing phase. Three nested deliverables:

**C-1 · Plumb fields through the bridge.** Stop dropping blur/pose/det/landmarks/aligned_face in `face_cluster_bridge.faces_to_face_records()`. Bridge reads from `context.insightface_faces[path][i]` to recover what FaceForClustering doesn't carry.

**C-2 · Pydantic FaceRecord.** Convert `face_cluster/types.py::FaceRecord` from dataclass to `pydantic.BaseModel`:
- Required fields (bbox, embedding, face_id, image_id) have no default — caller must supply.
- Stage-dependent fields (aligned_face, d10_score) stay `Optional`.
- `@field_validator` per field expresses invariants declaratively: bbox is 4-tuple of positives, area > 0, if pose is set then all 3 dims present.
- `model_config = ConfigDict(extra="forbid")` — typo'd field names fail at construction.
- Pydantic is already a project dep (`sim_bench/api/schemas/`); no new dependency.
- Bridge fixture test `test_bridge_completeness.py` — 5-image real fixture through `faces_to_face_records`, asserts every Pydantic validator fires when given a bad input AND that no field is at sentinel-zero on a real run.

**C-3 · Persist what's now plumbed.** Schema extension (additive columns, no version bump per locked decision — v5 cutover lands in a separate PR):
- Add to `faces`: `iqa_score`, `ava_score`, `sharpness_score`, `scene_cluster_id` (all nullable, populated from existing context fields).
- Populate currently-NULL columns in `faces` / `face_scores`.
- Wire `filters=context.filters` at `face_cluster_export.py:119` (closes spec-032 P1 on Albumify path).
- Schema v5 bump in a separate cutover PR. **[locked decision]**

**Acceptance**:
- `SELECT COUNT(*) FROM faces WHERE blur_score = 0.0` returns 0 on a fresh Albumify run (assuming blur was computed).
- `SELECT COUNT(*) FROM filter_decisions` > 0 on a fresh Albumify run.
- Pydantic FaceRecord rejects `bbox=(0,0,0,0)`, `area=0`, `pose=(1.0,)` with explicit ValidationError messages.

### P-D · "Click an image, see everything" (extends spec-023)

**Scope**: add `RunStore.image_detail(image_path) -> ImageDetail` in `face_cluster/run_store.py`. Returns a single typed object containing image-level scores, scene assignment, every face on this image, every face's per-filter decisions, cluster assignments. One SQL `JOIN ... WHERE image_path = ?` — no new computation.

Spec-023 already declares this as US1 (Part A). P-D closes spec-023's "data source = `image_metrics` JSON" gap by routing both the main-app popup and the FC App popup through this one reader. **[locked decision: extend spec-023, no new spec]**

**Acceptance**: on a post-P-C run, `RunStore.image_detail("20250822_122626.jpg")` returns image scores + all faces + per-face filter_decisions, all non-NULL.

### P-E · Drift-prevention tests

Three architecture tests, landed alongside their respective phases:
- `test_pipeline_context_contract.py` (P-B) — every dataclass field in spec-034.
- `test_storage_covers_context.py` (P-C) — every "Persisted: yes" field has a DB column.
- `test_image_detail_completeness.py` (P-D) — image_detail returns non-NULL for every always-populated column.

### P-F · Config parity — FC App ↔ Albumify

**Why**: today you can't run identical face-clustering configs on the two pipelines because the bridge force-overrides values (`bridge.py:66-71`). Even if you could, you can't compare them because they have different shapes (FC App = `PipelineConfig` dataclass; Albumify = nested `step_configs` dict).

Four steps:

- **F-1 · Single canonical config**. `face_cluster.PipelineConfig` is the one source of truth. Albumify builds it via `build_fc_config_from_albumify(step_configs, ui_inputs)` with **zero hardcoded overrides**. Missing required fields raise loudly. (Pairs with P-C — once fields are populated, gates don't need defensive disable.)
- **F-2 · Effective config dump**. Every run writes `effective_config.json` to its run dir in identical shape regardless of producer. Already 80% done via `pipeline_run.json:config` — needs symmetric serialization.
- **F-3 · Config diff CLI**. `python -m face_cluster.config_diff <run_a> <run_b>` printing field-level differences with units. Trivial once F-2 lands.
- **F-4 · Round-trip parity test**. Load an FC App profile; run Albumify with the same source dir + that profile; assert effective configs are identical (modulo timestamps). CI-gated.

**Acceptance**: after F-4, "why did Albumify give worse results?" has a deterministic answer — diff tool prints which fields differ, OR (after P-C) data differs but configs don't.

### P-G · Typed step configs (Pydantic)

**Why**: today every step does `config.get("key", default)` on a plain dict. A UI widget that fans out to a misspelled key, or a step that asks for a key the UI never wrote, fails silently with the default. This is the runtime value-flow gap the audit identified.

**Scope** (face-clustering-related steps only — full pipeline migration is out of scope):
- One `BaseModel` per face-clustering step: `FilterFacesConfig`, `FilterQualityConfig`, `FilterPortraitsConfig`, `FilterBestFacesConfig`, `ClusterPeopleConfig`, `ScoreFace*Config`, `InsightFaceDetectConfig`.
- Each model declares its fields with types, defaults, and `Field(description=...)` (the same description the UI `help=` should use — single source of truth).
- `model_config = ConfigDict(extra="forbid")` — unknown keys raise.
- UI builds the Pydantic instance from widget values; pipeline runner passes the typed instance to `step.process(context, config: FilterQualityConfig)`.
- Existing dict-based callers keep working via `FilterQualityConfig.model_validate(some_dict)` at the boundary; internal access becomes `config.min_iqa_score` not `config["min_iqa_score"]`.

**Acceptance**:
- A misspelled config key (`mim_face_size`) at any boundary raises `ValidationError` instead of silently defaulting.
- UI `help=` strings are generated from the same `Field(description=...)` source (eliminating the "label vs reality" drift class that motivated P-A).
- All face-clustering steps' configs are Pydantic models; CI check rejects any new face-clustering step that uses `dict` config.

### P-H · Pandera schemas on DB I/O

**Why**: P-E's `test_storage_covers_context.py` proves the schema has the right columns but doesn't prove rows actually populate them. NULL columns on a fresh Albumify run today *would* pass that test. Pandera closes the gap by validating actual DataFrame content at write and read time.

**Scope**:
- `face_cluster/run_exporter.py`: `@check_io` decorator on `_write_faces_and_scores` with a `DataFrameSchema` declaring which columns are `nullable=False` (e.g., `det_score`, `blur_score`, `yaw`, `pitch`, `roll` after P-C). Run fails loudly at export time if any required column is null.
- `face_cluster/run_store.py`: same schemas on the read side. `RunStore.faces()` returns a Pandera-validated DataFrame; downstream UI can trust column non-nullness without re-checking.
- New dep: `pandera>=0.18` (pyproject.toml). Verify it imports cleanly on Windows + .venv before P-H lands.
- For RunExporter inputs that are lists of Pydantic FaceRecord (not DataFrames), the validation lives on FaceRecord itself (from P-C C-2) — Pandera is only needed where data is DataFrame-shaped.

**Acceptance**:
- A synthetic FaceRecord with `blur_score=None` reaches `_write_faces_and_scores`; Pandera fails the run with a message naming the column and row index.
- `RunStore.faces()` on a v4 run produced before P-C raises Pandera ValidationError on the NULL columns (so we know which old runs need re-export, instead of silently propagating NULLs into the UI).

---

## 6. Locked decisions

1. **P-A**: `min_face_ratio` surfaced as a slider, default 0.005 unchanged.
2. **P-C**: Schema rollout staged — additive columns first (no version bump per PR); v5 cutover in a separate PR.
3. **P-C C-2**: FaceRecord becomes a Pydantic BaseModel (replaces the earlier "no defaults + __post_init__" plan). Pydantic already a project dep.
4. **P-D**: Extends spec-023. No new spec-number.
5. **P-F**: Phase added (resolves user concern about FC App vs Albumify divergence).
6. **P-G/H**: Pydantic for object contracts (step configs, FaceRecord, ImageDetail) + Pandera for DataFrame I/O (DB writes/reads). Adds runtime contract enforcement at every boundary the audit flagged as weak.
7. **Spec numbering**: spec-033 is data-integrity (this); spec-034 is the new context-contract spec.

## 7. Sequencing

```
P-A  (label honesty)
 │
 ├── P-B  (context contract spec-034)
 │
 ├── P-G  (Pydantic step configs)       ─┐
 │                                        │
 ├── P-C  (plumbing + Pydantic FaceRecord)┤
 │                                        │
 ├── P-H  (Pandera DB I/O)               ─┤── builds on Pydantic from P-C/P-G
 │                                        │
 │                                        ├── P-D  (image_detail reader)
 │                                        │
 └── P-F  (config parity) ───────────────┘   uses P-G's typed config + P-C's plumbing

P-E drift tests land alongside their phases.
```

Each phase = one PR. P-A is cheapest and most visible (label-only). P-G ideally lands before or alongside P-C so the new FaceRecord, step configs, and Pandera schemas all use the same Pydantic v2 patterns.

## 8. What "done" looks like for SIGHTING-059

- Every UI control either does what its label says, or its `help=` text (sourced from the Pydantic `Field(description=...)`) explains the scope.
- No DB column is NULL on a fresh Albumify run except by design — and Pandera schemas reject the run at write time if one slips through.
- Typo'd config keys raise `ValidationError` at the UI→step boundary, not silently default.
- FaceRecord cannot be constructed with missing or invalid fields — Pydantic `extra="forbid"` and `@field_validator` enforce.
- Clicking any face shows: which filters ran, with what measured values, with what thresholds, which one (if any) rejected it.
- Clicking any image shows: image-level scores, scene assignment, every face with full per-face state.
- A diff tool answers "why did these two runs differ" in one command.
- A new pipeline step author cannot ship a step that writes to context without spec-034 entry — CI rejects. Same author cannot ship a `dict` config — typed model required.

---

## 9. References

- **Current-state map**: [`pipeline_map.html`](pipeline_map.html) — pipeline diagram, full context↔storage tables, gap table, drift points. Read this when you want the deep snapshot.
- **Sighting**: [`docs/project/SIGHTINGS.md`](../../docs/project/SIGHTINGS.md) → SIGHTING-059 (open) and SIGHTING-060 (open, area unit drift).
- **Related specs**:
  - spec-023 — Image Detail View (P-D extends this)
  - spec-030 — Storage Ownership Refactor (RunExporter/RunStore; P-C extends schema)
  - spec-032 — Filter Context (P-C completes Albumify-path forwarding)
  - spec-034 — Pipeline Context Contract (NEW, written in P-B)
- **Historical research** (preserved in [`_archive/`](_archive/), referenced for context only):
  - diagnostic_report.html, diagnostic_round2.html — investigation that led here
  - design.html, filter_context.html — earlier design iterations
