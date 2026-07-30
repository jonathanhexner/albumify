# spec-069 — Manual face disqualification (low-quality / wrong-person exclusion)

**Created**: 2026-06-05
**Status**: Draft
**Priority**: P3 (quality-of-life; not on the parity critical path)
**Predecessors**: spec-066 (Gallery — surfaces the action + the `⚠` quality flag), spec-045 (Cluster Analysis service/repo), spec-040 (per-run `face_clustering.db`)
**Carved out from**: spec-066 review Q&A (2026-06-05) — disqualify is a **mutation**, so it cannot live in the read-only Gallery tab. See `specs/066-.../ARCHITECTURE_PLAN.md §3c` (row G5-action).

---

## Problem

Gallery (spec-066) can *flag* low-quality faces (blurry / outlier) but cannot *act* on them.
Today the only quality control is **automatic, at pipeline time** (`sim_bench/pipeline/steps/quality_gate.py`
— a blur threshold drops faces before clustering). There is **no way for a human to say
"this specific face is junk / is the wrong person — exclude it"** after a run completes.

The user wants to use the Gallery's visual review to disqualify faces, and have that decision
**persist and be honored** by downstream re-clustering and export.

## Why this is a separate spec (not a Gallery feature)

| Read-only (spec-066) | Mutation (this spec) |
|---|---|
| reads `FaceRow.is_outlier`, `blur_score` | **writes** an exclusion record |
| no persistence | needs a storage model + audit trail |
| no downstream effect | re-cluster / export must honor exclusions |
| no reversibility concern | must be un-doable |

Putting a write path in Gallery would break the arch contract (`test_gallery_tab.py`: tabs do
no SQL/FS) and the read-only invariant. This spec owns the data model; Gallery only renders the button.

## What we build (4-layer, per framework standards)

```
COMPONENT  disqualify button (in cluster_strip / face_detail_panel)
   ▼
SERVICE    DisqualificationService  (face_cluster/views/, Streamlit-free, typed)
             disqualify(face_id, reason) -> ExclusionResult
             restore(face_id)            -> ExclusionResult
             list_exclusions()           -> List[FaceExclusion]
             suggest()                   -> List[FaceExclusion]   # auto from blur/outlier
   ▼
REPOSITORY new table `face_exclusions` in the per-run face_clustering.db
             + an action_log audit row (producer=fc_app_v2, action_type=face_disqualify)
```

### Data contract (typed)

```python
@dataclass(frozen=True, slots=True)
class FaceExclusion:
    face_id: int
    reason: str                 # enum-ish: "blurry" | "wrong_person" | "occluded" | "other"
    source: str                 # "manual" | "suggested"
    created_at: str             # iso
    active: bool                # restore() flips to False (soft-delete; keeps audit)

@dataclass(frozen=True, slots=True)
class ExclusionResult:
    face_id: int
    now_excluded: bool
    n_active_exclusions: int
```

### New DB table (per-run `face_clustering.db`)

```
face_exclusions(
  face_id     INTEGER PRIMARY KEY,
  reason      TEXT NOT NULL,
  source      TEXT NOT NULL DEFAULT 'manual',
  created_at  TEXT NOT NULL,
  active      INTEGER NOT NULL DEFAULT 1
)
```
Alembic migration + ORM model + Pandera schema (per the doc-update mandate). Arch test
`test_orm_models_in_sync_with_alembic.py` must stay green.

### Downstream honoring (the part that makes it real)

- **Re-cluster** (spec-063 Recluster tab): the recluster input filters out `active=1` exclusions
  before building the kNN graph. One hook in the recluster step's `Inputs` assembly.
- **Export**: excluded faces dropped from exported clusters/labels.
- **Gallery / Cluster Analysis**: excluded faces render dimmed with a "restore" affordance,
  or hidden behind a "show excluded" toggle.

## Locked decisions

1. **Soft-delete, never hard.** `restore()` flips `active=0`→ keeps the audit trail. A face is
   never physically removed from the run DB.
2. **Per-run scope.** An exclusion lives in that run's `face_clustering.db`. Re-running from
   scratch is a clean slate (exclusions don't leak across runs). A *recluster snapshot* inherits
   its parent's exclusions (it forks the DB).
3. **Service owns the rule** `blur_score < thr OR is_outlier` for `suggest()` — threshold comes
   from run metadata, not a UI literal (spec-053 convention).

## Open decisions (need ✓ before implementation)

| # | Question | Lean |
|---|---|---|
| **Q1** | Does disqualify auto-trigger a recluster, or just mark + let the user recluster manually? | **mark only** — explicit recluster keeps it predictable + cheap |
| **Q2** | Bulk "disqualify all suggested" one-click, or per-face only? | offer both; bulk is a thin loop over `suggest()` |
| **Q3** | Should excluded faces stay visible (dimmed) or hidden by default? | dimmed + "show/hide excluded" toggle |
| **Q4** | Where does the button live — Gallery strip, Face Analysis panel, or both? | **both** (Gallery for sweep, Face Analysis for the considered single-face call) |

## Acceptance criteria

| # | Criterion | Verified by |
|---|---|---|
| AC1 | `face_exclusions` table + ORM + Alembic + Pandera, all in sync | arch tests |
| AC2 | `DisqualificationService.disqualify/restore/list/suggest` return typed dataclasses; Streamlit-free | grep + unit tests |
| AC3 | Recluster honors `active` exclusions (excluded face absent from output) | integration test on **labeled golden set** (`test_data/face_clustering/`) |
| AC4 | Disqualify writes an `action_log` audit row (`action_type=face_disqualify`) | test |
| AC5 | Gallery + Face Analysis expose disqualify/restore; tabs stay ≤80 LOC, no SQL | arch test |
| AC6 | Budapest e2e: disqualify a face → recluster → that face_id is gone | new `slow` scenario |
| **AC7** | **Quality, not vibes**: on the labeled set, disqualifying an intentionally-degraded face keeps **purity** (no mixed-person cluster) and **completeness** (no person split) — reuse `test_pipeline_e2e.py` helpers | integration test |

### Why the labeled set matters here

The 9-image `ground_truth.csv` dataset is what makes disqualify *verifiable*: with per-face
person labels, AC3/AC7 assert the **clustering stays correct** after an exclusion, not merely
that a row vanished. Budapest (AC6) covers the UI path; the golden set covers the semantics.

## Effort estimate

**~5-7 h** — DB/ORM/migration ~1.5h, service + tests ~2h, recluster/export honoring ~1.5h,
UI wiring (2 tabs) ~1h, e2e ~1h. Larger than a Gallery sub-feature precisely because it crosses
the schema + re-cluster + export boundaries — which is why it's its own spec.

## Non-goals

- Auto-disqualification policy changes (the pipeline `quality_gate` stays as-is).
- Cross-run / global exclusion lists ("never show this person again").
- ML-assisted "is this the wrong person" detection — manual reason only.
