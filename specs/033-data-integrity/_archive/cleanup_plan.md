# SIGHTING-059 Cleanup Plan — Making Order

**Status**: APPROVED IN PRINCIPLE — three locked decisions recorded below; phase PRs pending explicit go-ahead
**Companion**: [`pipeline_map.html`](pipeline_map.html) (current-state map)
**Premise**: No logic changes. Fix labels, plug context↔storage gaps, make per-image inspection a single query.

## Locked decisions (2026-05-13)

1. **P-A**: `min_face_ratio` IS surfaced as a user-visible slider. Default 0.005 unchanged. Honesty over knob-minimization.
2. **P-C**: Schema rollout is STAGED. Additive columns first (no version bump per PR). A separate cutover PR bumps `PRAGMA user_version` 4→5 once all columns are plumbed and read paths updated.
3. **P-D**: EXTENDS spec-023 (Image Detail View). No new spec-034. Adds a new Part D to spec-023: canonical `RunStore.image_detail(image_path)` reader as the single source of truth for both the main-app popup (spec-023 Part A) and the FC App image inspection. Closes spec-023's "data source = `image_metrics` JSON" gap by routing through the v4 DB.

---

## What we are not doing

- **No algorithm change.** Quality gates stay disabled where they are today. Area stays fractional in Albumify. Merge thresholds unchanged. Whatever runs today still runs tomorrow.
- **No new ML.** No new filters, no new gates, no new clustering.
- **No new sightings.** This is scoped strictly to SIGHTING-059 (data integrity + UI/pipeline coherence).

The goal is **legibility and robustness**, not behavior change.

---

## Spec landscape (context for the plan)

| Concern | Existing spec | Status | Gap this plan addresses |
|---|---|---|---|
| Storage / DB schema | spec-030 (storage-ownership-refactor) | In Progress (Phase 3 landed) | No formal context→DB mapping appendix |
| Filter decisions | spec-032 (filter-context) | P0–P5 landed | P1 incomplete on Albumify path |
| Step observability | spec-012 (pipeline-observability) | Implemented | StepDecision only; not full context contract |
| Diameter cap | spec-031 | P1 landed | n/a — not relevant to cleanup |
| **PipelineContext field contract** | **none** | — | **P-B writes one** |
| **Image-level "show me everything"** | partial (spec-015 face popup, spec-023 image-detail-and-fc-integration) | spec-015 draft | **P-D extends spec-023 or adds new** |
| **UI ↔ config binding contract** | spec-032 P4 (UI alignment test) | landed | Static check passes; labels still lie. **P-A fixes labels.** |

---

## The five phases (user said "similar priority"; this is recommended sequence, not strict)

### P-A · UI label honesty (1 PR, ~150 LoC, no logic change)

**Problem**: "Min Face Size (px)" doesn't drive face clustering. "Min Face Area" defaults to None and would compare px to fraction anyway. `min_face_ratio = 0.005` is the actual face-size rejector and has no UI.

**Scope** — every widget in:
- `app/streamlit/components/pipeline_runner.py`
- `app/face_clustering/tabs/run_tab.py`
- `app/face_clustering/tabs/recluster_tab.py`
- `app/shared/merge_controls.py`

**Per-widget audit** (label-only changes; keep all defaults and bindings as-is):

| Widget | Today | After P-A |
|---|---|---|
| `config_min_face_size` "Min Face Size (px)" | binds to detect_faces + 3 scoring steps' `min_face_size` | Label unchanged; `help=` reads: "Minimum face bbox size (pixels) for detection and per-face scoring. **Does not** filter for clustering — see 'Min Face Area' below." |
| (currently hidden) `min_face_ratio` | hidden, default 0.005 | **Surface** as new slider "Min Face Area (fraction of image)" — same 0.005 default, writes to existing `detect_faces.min_face_ratio` config slot. No new logic. |
| `run_min_face_area` FC App "min_face_area px (0=off)" | FC standalone only; bridge area is fraction² | Relabel "Min Face Area (FC standalone only) — pixels²". Add `help=`: "Has no effect on Albumify-produced runs (area is fractional on that path)." |
| `run_blur_min`, `run_yaw_max` etc. | bound to FC App gates | Add `help=` note: "Inactive on Albumify-produced runs (bridge force-disables these gates)." |
| Cluster cap controls (spec-031) | new, fine | Confirm `help=` mentions `enabled=False` default. |

**Deliverable**: PR with widget label diffs. No `KNOWN_FILTERS` changes needed (P4 test still passes). One screenshot per UI surface attached to the PR showing every face-clustering control has an honest label + unit declaration.

**Acceptance**:
- `test_ui_aligns_with_filters.py` still green.
- No widget in scope has a label whose unit contradicts the value flowing through it.
- `min_face_ratio` is now user-tunable.

---

### P-B · PipelineContext field contract (NEW spec-034, ~250 LoC test + 1 spec.md)

**Deliverable**: `specs/034-pipeline-context-contract/spec.md` — one table, every field on `PipelineContext` AND `face_cluster.pipeline._RunContext`:

| Field | Type | Unit | Producer step | Consumer step(s) | Persisted? | Status |
|---|---|---|---|---|---|---|

Each row also gets a **lifecycle** note: when populated, when cleared, when reset on recluster.

**Plus**: `tests/architecture/test_pipeline_context_contract.py` — a single test that introspects the two dataclasses and asserts every field has exactly one row in the spec. Drift fails CI.

**Acceptance**:
- Every field accounted for, including deprecated (`quality_passed`, `portrait_passed`).
- A reader can answer "where does `context.iqa_scores` come from and where does it go?" by reading one table.

---

### P-C · Context→Storage mapping (extends spec-030, ~60 LoC schema + 100 LoC test)

**Deliverable**: New section in `specs/030-storage-ownership-refactor/spec.md`: **"Context-to-Storage Mapping"**.

Two tables:

**Table 1: DB column → context field**

| DB table.column | Type | Context field producer | Transformation | Today's reality on Albumify |
|---|---|---|---|---|

**Table 2: Context fields NOT persisted**

| Context field | Reason | Status |
|---|---|---|
| `iqa_scores` | none (bug) | LOST IN TRANSIT — image-level scoring discarded |
| `ava_scores` | none (bug) | LOST IN TRANSIT |
| `scene_cluster_labels` | none (bug) | LOST IN TRANSIT |
| `aligned_faces` | by design — content lives in `crops/` | ephemeral |
| `face_embeddings` | persisted as `embeddings.npy` | OK |
| `insightface_faces[*].yaw/pitch/roll` | bug — bridge doesn't carry | LOST IN TRANSIT |

**Code change**: extend RunExporter to persist the LOST IN TRANSIT entries. **No new computation** — just plumb existing context values into existing-or-new columns.

Schema bump v4 → v5 in `face_cluster/run_exporter.py`:
- Add to `faces` table: `iqa_score`, `ava_score`, `sharpness_score` (new columns, nullable)
- Add `scene_cluster_id` to `faces` table (nullable)
- Populate existing NULL columns (`yaw`, `pitch`, `roll`, `blur_score`, `det_score`, `pose_score`, `eyes_score`, `expression_score`, `frontal_score`) by having `face_cluster_export.py` read `context.insightface_faces` directly instead of relying on FaceRecord
- Wire `filters=context.filters` at `face_cluster_export.py:119` (closes spec-032 P1 on Albumify path)

**Acceptance**:
- `SELECT COUNT(*) FROM faces WHERE iqa_score IS NULL` returns 0 on a fresh Albumify run (assuming `score_iqa` ran).
- `SELECT COUNT(*) FROM filter_decisions` > 0 on a fresh Albumify run.
- Architecture test `test_storage_covers_context.py`: every context field marked `Persisted: yes` in spec-034 has a corresponding column in v5 schema. Failures = drift.

---

### P-D · "Click an image, see everything" — single canonical join (EXTENDS spec-023)

**Decision (locked)**: spec-023 already covers "click any image, see everything" (US1, Part A). Its current data source is `image_metrics` JSON in `PipelineResult` — fragmented across pipeline_service / api / cache layers. P-D adds **Part D** to spec-023: a canonical `RunStore.image_detail(image_path)` reader in `face_cluster/run_store.py`, sourced from v4 DB only. Both the main-app popup (spec-023 Part A) and the FC App image popup wire to this reader. This closes the duplication spec-023 Part B describes from the other direction.

**Deliverable**: one new method on `RunStore`:

```python
class RunStore:
    def image_detail(self, image_path: str) -> ImageDetail:
        """All persisted facts about one image, joined across tables."""
```

`ImageDetail` dataclass returns:
- **Image-level**: path, iqa, ava, sharpness, scene_cluster_id, image-level `filter_decisions` rows.
- **Faces**: list of `FaceDetail`, one per face on this image, each containing every column from `faces` + `face_scores` + per-face `filter_decisions` rows + cluster assignment(s) per iteration + crop path.

Implementation is one SQL view or a `JOIN ... WHERE image_path = ?` query — no new computation, no new fields. Just compose what P-C persisted.

**UI**: image-detail popup in FC App (and Albumify if appropriate) wires to this method. Single source of truth — no more per-tab `pd.read_csv` calls.

**Acceptance**:
- On the reference Albumify run (post-P-C), `RunStore.image_detail("20250822_122626.jpg")` returns: image-level scores present, all faces in the image, every face has non-NULL det_score / yaw / pitch / roll / iqa, plus its filter_decisions rows.
- For `20250822_123354.jpg` (face_46), the FaceDetail includes a `face_crop` rejection with reason `landmarks_missing`.

---

### P-E · Drift-prevention contracts (across phases)

Three tests, each cheap, all CI-gated:

1. **`test_pipeline_context_contract.py`** (P-B) — every dataclass field has a spec row.
2. **`test_storage_covers_context.py`** (P-C) — every "Persisted: yes" context field has a v5 DB column.
3. **`test_image_detail_completeness.py`** (P-D) — on a 3-image fixture, `RunStore.image_detail()` returns non-NULL for every column the spec marks as "always populated".

These are the contracts that keep the data integrity from rotting again.

---

## Sequencing

User said "similar priority", so phases are independent where possible. Practical order:

```
P-A  ── label honesty (smallest, most visible, validates the audit) ──┐
                                                                       │
P-B  ── context contract spec ─────┐                                  │
                                    ├── P-C ── context↔storage ──┐    │
                       (parallel)   │                              │    │
                                    │                              └────┴── P-D ── image_detail()
                                                                              │
                                                                              └── P-E tests landed in each
```

Each phase = one PR. Each PR ships its own architecture test.

---

## What "done" looks like for SIGHTING-059

- The pipeline_map.html gap table has every row labelled `ok` or explicitly accepted as "out of scope, no fix".
- Clicking any face in the FC App shows: which filters ran on it, with what measured values, with what thresholds, and which one (if any) rejected it.
- Clicking any image shows: image scores, scene assignment, every face detected on it with full per-face state.
- No DB column on a fresh Albumify run is NULL by accident — only by design (and the design is in the spec).
- A new pipeline step author cannot ship a step that writes to context but doesn't appear in spec-034 — CI rejects the PR.

---

## Open questions — resolved

All three decisions locked above. See "Locked decisions" at the top of this file.

## Ready for execution

Each phase is a separate PR. Pending explicit "start P-A" (or other phase) instruction from the user — per CLAUDE.md, no code changes without explicit request.
