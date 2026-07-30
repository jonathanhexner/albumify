# spec-079 — Albumify onto the Face-Clustering-v2 Shared Core

Status: **In Progress**
Owner: Jonathan Hexner
Created: 2026-06-05 · Refocused 2026-06-19 onto Track A (execution/config)

> **Active track = A (execution + config unification).** Track B (read/serialization
> layer) is preserved, deferred, at the end of `tasks.md`. The plan-of-record for
> Track A is `UNIFICATION_PLAN.html` (+ `UNIFICATION_EXPLAINED.html` for architecture).

## Problem

Albumify and Face-Clustering v2 already share the pipeline **executor**
(`execute_spec`), but the same configuration run through the two apps produces
**different people** (Budapest + `profile_5`: FC v2 = **8 identities**, Albumify
= **20**). Root cause is *not* the clustering algorithm — Stage 0b proved the
bridge and the 8 unified steps agree sub-stage for sub-stage on identical input.
The divergence is the **face set** that reaches the identical clusterer:

- **Pose gap (primary).** On the Albumify path `FaceRecord.pose is None` for every
  face, so the pose gate (yaw≤30 / pitch≤25) rejects **0** off-angle faces; FC v2
  rejects **53** (pose_yaw 46 + pose_pitch 7). 53 = 186−133 core = the whole gap.
- **Blur gap (secondary).** `blur_min` is pinned to `0.0` on the Albumify path
  because the InsightFace producer chain has no blur scorer.
- **Config divergence.** FC v2 sends typed `FCParams`; Albumify sends untyped dicts
  validated against `ClusterPeopleConfig` — a *subset* of `FCParams`. Two languages.
- **Dual clustering path.** Albumify reaches clustering via the deprecated
  monolithic `cluster_people` → `face_cluster_bridge`; FC v2 via 8 unified steps.

## Goal

**Same `FCParams` through either app → identical clusters**, verified by an
executable test, while the FC v2 baselines stay byte-for-byte unchanged. Then
collapse the two config systems into one (`FCParams`, hierarchical) and delete
the divergent clustering path.

## Equivalence anchors (binding)

Two albums, two purposes — both must hold at the end:

| Anchor | Album + profile | Asserts | Reference run |
|---|---|---|---|
| **Equivalence target** | Budapest + `profile_5` | FC v2 == Albumify: **8 identities**, sizes `[26,20,12,7,3,2,2]`, 72 assigned | FC v2 `a588521b…` |
| **No-regression guard** | Budapest + `profile_4` | FC v2 baseline UNCHANGED: **15 clusters · 340 faces** | `6437d335…` |

Constants live in `tests/_budapest_baseline.py`. The `profile_4` guard is the
binding v2 E2E gate (`tests/face_clustering/e2e_budapest/`, CLAUDE.md).

### Facts established in Stage 0 (do not re-litigate)

- Clustering **code** is equivalent (bridge ≡ 8 unified steps). The fix is the
  producer chain + config, not an algorithm swap. (`LOCALIZE_GAP.html`)
- `cluster_scenes` / `cluster_by_identity` are image-level steps, unrelated to the
  identity count. `cluster_by_identity`'s design flaw is parked in PRD-080.

## Approach — staged; each stage gated by test → /code-review → commit

| Stage | Outcome | Module(s) | Destructive? |
|---|---|---|---|
| 1 | Commit **RED** cross-app equivalence test; retire false-confidence parity test | `tests/architecture/` | no |
| 2 | Blur-scoring step in the shared producer chain; un-pin `blur_min=0.0` | `steps/`, `fc_params.py` | no |
| 3 | Populate `FaceRecord.pose` **at the detector** (both apps) | `steps/insightface_detect_faces.py` | no |
| 4 | `FCParams` = single typed contract, made **hierarchical** over `configs/*.py`; retire `ClusterPeopleConfig` | `fc_params.py`, `steps/configs/` | no (config reshape) |
| 5 | Delete `face_cluster_bridge.py` + monolithic `cluster_people`; Albumify `default_pipeline` → 8 unified steps | `steps/`, `configs/pipeline.yaml` | **yes** (deletes dead path) |
| R | Empty `steps/configs/__init__.py`; move `STEP_CONFIG_MODELS` → `configs/registry.py` | `steps/configs/` | no (isolated refactor, SIGHTING) |

## Design decisions

1. **Pose fix lands at the producer, not the bridge.** The bridge is deleted in
   Stage 5; fixing it there is throwaway. `insightface_detect_faces` is the one
   place both apps get pose. (The in-tree bridge patch is a *diagnostic* only.)
2. **`FCParams` becomes hierarchical** (`gating` / `graph` / `exemplars` / `split`
   / `merge` sub-models), composed of the existing `steps/configs/*.py` models.
   This collapses the two competing config systems: today `FCParams` is a flat
   blob broadcast identically to all 8 steps (so `extra="forbid"` is impossible),
   while `configs/*.py` are per-step `extra="forbid"` models that sit orphaned.
   `to_step_configs()` then *projects* each sub-model to its owning step.

## No-harm test strategy

Two guards stay **green after every stage**; one target test flips **RED→GREEN once**.

- **Guard A** — `tests/face_clustering/e2e_budapest/` (`profile_4` → 15/340). Any
  movement = harm; stop and diagnose.
- **Guard B** — validator contract: `tests/pipeline/test_pipeline_spec.py` +
  `test_albumify_default_spec.py`. The spec stays valid.
- **Guard C** — FC v2 `profile_5` anchor stays at 8 (the fixes change *Albumify*).
- **Target** — `tests/architecture/test_app_cluster_equivalence.py`: RED (8≠20)
  from Stage 1; GREEN (8==8) is the definition of done.
- **Hard rule:** the WIP-edited goldens (`tests/_fixtures/budapest_golden/*.json`,
  currently 13) are **NOT committed** until the target is GREEN at 8==8.

## Non-goals

- Collapsing the HTTP boundary (Albumify stays a web/API client).
- Removing the central catalog DB (albums, `universal_cache`, overrides, events).
- Track B (read/serialization layer) — deferred; see `tasks.md`.

## Open decisions (for Head of Engineering)

1. Commit the RED equivalence test now (visibly failing)? — **Recommend yes.**
2. Fold hierarchical `FCParams` into Stage 4 vs defer? — **Recommend fold in.**
3. Run Refactor R now as a sighting? — your call; isolated, zero-behavior.

## Exit / Code Review gate

`/code-review` after each stage on changed modules; no open High-severity finding
before commit. Final `REVIEW.md` before flipping to **Implemented** — which
requires the equivalence target GREEN **and** Guard A unchanged.
