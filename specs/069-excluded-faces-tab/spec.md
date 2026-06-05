# spec-069 — Face Metrics & Exclusions tab (sortable per-face table)

**Created**: 2026-06-05
**Status**: Draft
**Priority**: P2
**Predecessors**: spec-065 (Quality tab), spec-067 (exposed SIGHTING-092)
**Sightings**: SIGHTING-092, **SIGHTING-093** (persistence gaps — gates the
  gate-reason breakdown, NOT the metrics table)
**Investigation**: [INVESTIGATION.md](INVESTIGATION.md) — where the 233
  "missing" faces go (answer: unassigned-after-clustering, not gate-rejected).
**Design artifact**: [DESIGN.html](DESIGN.html) — Mock · FE/BE · contracts ·
  interfaces · test plan.

## Re-scope (2026-06-05, after the investigation run)

The investigation showed the gate **rejects ~0 faces** with profile_4, and the
233 "missing" faces are simply **unassigned after clustering** — and the
pipeline persists neither gate verdicts nor noise rows (SIGHTING-093). So the
work splits:

- **Primary deliverable (this spec, buildable NOW):** a **sortable per-face
  metrics table tab** — every face with a thumbnail + blur / area / det_score
  (+ pose when available) + status (`cluster N` / `unassigned`). Status is
  derived as `all_faces − assigned`; **no pipeline change needed.** This is the
  artifact validated as useful via `verify_run.py` /
  `VERIFY_v2_budapest_20260605.html`.
- **Deferred (behind SIGHTING-093):** the gate-reason breakdown (which gate
  failed, per-gate counts) — needs the pipeline to persist `filter_decisions`
  and gates that actually fire. NOT in this spec.

---

## Problem

"Why isn't this face in an album?" has **two** distinct answers the app
currently can't show together:

1. **Gate-rejected** — the face failed a quality gate (blur / pose / area /
   det-score) and never entered clustering. Source: `filter_decisions`.
2. **Unassigned (noise)** — the face passed gating, went into clustering, but
   matched no one. Source: cluster assignment where `cluster_id == NOISE_LABEL`.

The existing Quality tab (spec-065) shows only #1. SIGHTING-092 found Scenario
F asserting `total − assigned` (which is #2) against the Quality tab (which
reports #1) — and the legacy reference run has zero `filter_decisions`, so #1
is empty there. The two notions must be split and shown explicitly.

## Decision: one tab, unified funnel (not two tabs)

Both populations answer the same user question at different pipeline stages,
so they belong in one **funnel** view rather than separate tabs the user must
stitch together. Extend/rename the **Quality** tab → **"Excluded Faces"**:

```
detected ─► ✗ gate-rejected (blur/pose/area/det_score)   [#1]
         └► passed ─► assigned to clusters
                   └► ✗ unassigned / noise               [#2]
```

- Top: funnel counts (detected → gate-rejected → unassigned → assigned).
- One face grid with a `reason` filter; each excluded face shows a thumbnail
  + its reason (a gate name, or `unassigned`).
- Split into two tabs later ONLY if the drill-down *actions* diverge
  (e.g. "lower threshold" for gate vs "force-merge" for noise).

Full design, mock, contracts, interfaces: **DESIGN.html**.

## Test strategy: human-validated golden baseline (deferred)

We do NOT guess count bands up front — that mistake produced the unvalidated
12–18 and 220–240 bands that bit spec-067. Instead:

- **Phase 1 (this spec):** build the feature; run the v2 pipeline on a real
  album (`scripts/run_v2.py` on Budapest); **user reviews the rendered tab**.
- **Phase 2 (follow-up, after user sign-off):** freeze the observed counts as
  the expected criteria → a headless `ExcludedFacesService` endpoint test
  (mirrors spec-067 C) asserting the funnel counts ≈ baseline and that the
  **funnel closes**: `n_gate_rejected + n_assigned + n_unassigned == n_detected`.

So Phase-1 acceptance is visual; Phase-2 locks it in.

## Acceptance criteria

| # | Criterion | Phase | Verified by |
|---|-----------|-------|-------------|
| AC1 | `ExcludedFacesService.summary()` returns the funnel counts + per-gate breakdown | 1 | unit (synthetic) |
| AC2 | `list_excluded(criteria)` returns typed `ExcludedFace` rows for both stages, with crop paths | 1 | unit (synthetic) |
| AC3 | Tab renders funnel strip + reason filter + face grid; spec-068 telemetry `tab.done name=excluded_faces …` | 1 | AppTest |
| AC4 | Empty/legacy run (no filter_decisions) renders the unassigned side without error | 1 | AppTest (reference run) |
| AC5 | Run on a real album; user visually confirms the tab is correct | 1 | **user sign-off** |
| AC6 | Frozen-baseline endpoint test: funnel closes + counts ≈ baseline | 2 | endpoint test (deferred) |

## Out of scope

- Per-gate "re-run with lower threshold" action (future).
- Force-merge from the noise list (Cluster Analysis already owns merge).
- Changing the gate algorithms themselves.

## Effort estimate

~half day: service (~80 LOC) + tab (~60 LOC) + synthetic unit tests +
AppTest. Phase-2 endpoint test is a separate ~1 h after sign-off.
