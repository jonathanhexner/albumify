# spec-071 — Align Merged Clusters tab with V1 Merge Analysis

**Created**: 2026-06-05
**Status**: Code Review (A+B implemented + verified 2026-06-05; user sign-off + /code-review pending)
**Priority**: P2 (V1→V2 parity)
**Predecessors**: spec-065 (the V2 Merged Clusters tab), spec-042 (parity program)
**Design artifact**: [DESIGN.html](DESIGN.html) — Mock · FE/BE · Data contracts ·
  Interfaces · Test plan.

---

## Problem

V1's **Merge Analysis** tab is a rich review surface; V2's **Merged Clusters**
tab (spec-065) is a bare table + raw-JSON detail. The two should be aligned —
bring V1's production review value into V2 — so an operator can see *why* each
merge was made or rejected, and *which faces* were involved.

Investigation (V1 `merge_analysis_tab.py`) grouped V1's features:

| Group | V1 feature | This spec |
|---|---|---|
| **A. Heuristic review** | per-pair gate badges (cross / exemplar / support / margin / diameter), thresholds (T_a/T_b/T_global), margins, diameters, rejection reason, summary | ✅ in scope |
| **B. Visual pair galleries** | side-by-side **face crops of the two clusters** in each merge pair | ✅ in scope |
| C. Approval controls (human accept/reject) | tied to the labeling workflow | ❌ **out (user decision 2026-06-05)** |
| D. ML-model merge mode | research-only | ❌ deferred (spec-042) |
| E. Remerge / recompute | recompute merges with new params | ❌ V2 **Recluster** tab owns this |

## Key finding: the data is already there

`merge_decisions` persists the full per-pair breakdown and
`MergeDecisionRow` already exposes it (verified on `v2_budapest_20260605b`):
`iteration, cluster_a, cluster_b, cluster_a_size, cluster_b_size,
exemplar_dist, threshold_used, T_a, T_b, T_global, support, unique_support,
required_support, post_diameter, max_allowed_diameter, margin_gap,
margin_competitor_id, passes_cross, passes_exemplar, passes_support,
passes_margin, passes_diameter, action, actually_merged, rejection_reason`.

So **Group A is UI-only** (surface fields the service already returns).
**Group B** reuses existing reads: `find_assignments(iteration=…)` →
the two clusters' face_ids at the pair's iteration → `crop_path`. **No
pipeline or DB-schema change.**

## What we build

1. **Enrich the Merged Clusters table + detail** (Group A): replace the raw
   `st.json` with a structured per-pair panel — a **gate-badges strip**
   (✓/✗ per gate, with the measured value vs threshold), plus the
   distance / support / margin / diameter numbers. Keep the existing filter
   (all / merged / rejected).
2. **Visual pair view** (Group B): on selecting a pair, render the two
   clusters' faces side by side (thumbnails via `crop_path`), labelled
   `cluster_a (n)` / `cluster_b (n)`, so the operator sees the people the
   gate decision was about.
3. A thin **summary strip** (n merges, n rejected, top rejection gate) at the
   top, mirroring V1's summary.

## Acceptance criteria

| # | Criterion | Verified by |
|---|-----------|-------------|
| AC1 | Selecting a pair shows a gate-badges strip (cross/exemplar/support/margin/diameter ✓/✗ + value vs threshold) | AppTest + unit |
| AC2 | Selecting a pair shows cluster_a + cluster_b face crops side by side | AppTest |
| AC3 | Summary strip: n_merged, n_rejected, top_rejection_gate | unit |
| AC4 | Filter (all/merged/rejected) preserved | AppTest |
| AC5 | No pipeline/DB change — reads only via existing repo methods | git diff + arch test |
| AC6 | spec-068 telemetry `tab.done name=merged_clusters …`; renders without error on a run with merges | AppTest (real run) |
| AC7 | User visual sign-off | you |

## Out of scope
- Approval/accept-reject controls (C), ML mode (D), remerge (E).
- Retiring V1's Merge Analysis tab (separate burn-in decision).

## Test strategy
Same as spec-069/070: service unit tests on synthetic merge_decisions (gate
flags, summary), AppTest on a real run with merges (`v2_budapest_*`), then
**your visual sign-off** as the golden baseline. No guessed thresholds.

## Effort estimate
~half day: a `MergeReviewService` enrichment (or extend `MergedClustersService`)
+ gate-badge / pair-crop components + tests. UI-heavy, data already present.
