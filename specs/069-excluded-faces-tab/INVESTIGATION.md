# spec-069 — Investigation: where do the 233 "missing" faces go?

**Date**: 2026-06-05
**Trigger**: a fresh v2 run on Budapest (`v2_budapest_20260605`, profile_4)
recorded the disposition of only 107 / 340 faces.

## Method

Ran `scripts/run_v2.py` on `D:\Budapest2025_Google` with profile_4, then read
the run log + `face_clustering.db` directly.

## Findings

### 1. The quality gate does NOT reject (with profile_4)

Run log:
```
Quality gating: 186 core, 154 holdout faces
quality_gate: 185 core / 154 holdout / 340 total   (1 dropped: None embedding)
```
- `select_core_set` splits faces into **core** (top-K per image, passed gates,
  seed clustering) and **holdout** (beyond top-K per image — deferred for
  later attachment, NOT a gate failure).
- For profile_4 the gates exclude ~0 faces: blur is bypassed (NOT WIRED —
  `blur_score` 0.0 → `_effective_blur_min` forced to 0), pose off
  (`use_pose_estimation=False`), area/det lenient. So core vs holdout is a
  ranking, not a rejection.

### 2. The 233 are unassigned-after-clustering, and not persisted

DB:
```
faces:               340
cluster_assignments: 107   (one iteration; cluster_id >= 0; NO noise rows)
filter_decisions:      0   (empty)
```
- Pipeline: cluster 185 core → merge → attach holdout → diameter cap →
  **107 assigned to 15 clusters**.
- `340 − 107 = 233` faces end in **no final cluster** (core that went to noise
  + holdout never attached). They get **no `cluster_assignments` row** — only
  assigned faces are written. So they're derivable as `all_faces − assigned`,
  but not explicitly stored.

### 3. Two persistence gaps (root cause of "no data")

- **G1 — verdicts dropped (v2):** `QualityGateStep.process` computes
  `result.verdicts` (per-face, per-gate decisions) but writes only
  `core_indices` / `holdout_indices` to context. The verdicts are never
  persisted to `filter_decisions`. (The Albumify export path *does* write
  them — see `face_cluster_export.py:146` — so this is v2-specific.)
- **G2 — unassigned not stored:** noise / unattached faces get no
  `cluster_assignments` row, so "faces in no cluster" must be derived.

### 4. Config-surface gap (pose)

`use_pose_estimation` is read by the gate step (`quality_gate.py:82`) and
SixDRepNet is installed, but `FCParams` exposes only `require_pose`, not
`use_pose_estimation`. So no profile / `run_v2.py` flag can turn pose
*computation* on — it defaults False. `require_pose=True` alone would gate on
a pose that's never computed.

## Consequences

- The **per-face metrics table** (blur / area / det_score + assigned vs
  unassigned status) is buildable TODAY from existing data — status derived as
  `all_faces − assigned`. → spec-069 primary deliverable.
- A **gate-rejection breakdown** (which gate failed, counts) needs G1 fixed
  (persist verdicts) AND gates that actually fire (wire blur/pose). → deferred
  predecessor, tracked as a sighting.
- F (budapest Quality scenario) failing is explained: `filter_decisions` is
  empty for *every* v2 run, not just the legacy reference.
