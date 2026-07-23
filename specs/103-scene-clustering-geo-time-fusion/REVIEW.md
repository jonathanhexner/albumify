# REVIEW.md — spec-103 `build_scene_distance` (Path A) pipeline wiring

**Scope reviewed**: the pipeline integration only — the `build_scene_distance` step, the
`SceneDistanceBuilder` (Path A) helper, the two new `PipelineContext` fields, and the `cluster_scenes`
precomputed-metric branch. NOT the full experiment validation (labels T1 / ARI T4 remain open — spec stays
**In Progress**, not Implemented).

**Overall verdict**: `pass-with-followup`. No high-severity findings. No previously-green test is left
failing by this change (the one contract test my new fields tripped was fixed in the same change). Three
unrelated pre-existing failures surfaced during the full run and are filed as **SIGHTING-118** (none touch
scene clustering).

Date: 2026-07-22 · Reviewer: automated walk of CODE_REVIEW_CHECKLIST.md (8 sections).

---

## 1 · Structure — `pass`
- `build_scene_distance.py` step is 71 LOC (≤80, spec-053). `SceneDistanceBuilder` added to
  `geo_time_fusion.py` (domain module, not the step) — a closed type family with `SceneDistanceInputs`/
  `SceneDistanceResult`, reused from the existing fuser. No module >300 LOC.
- Step reads context → builds `Inputs` → calls `helper.calc()` → writes `Result` (the spec-053 shape).
- `calc()` takes one typed `Inputs` (not >4 params). No dead code; the superseded `SceneDistanceFuser`
  is explicitly labeled and still used by the experiment sweep (not orphaned).

## 2 · Code quality — `pass`
- No try/except on the new path (pure numpy). No bare excepts.
- `if/else` nesting ≤2 (the precomputed-vs-embedding branch in `cluster_scenes`).
- `config.get("boost", 0.6)` / `get("tau_sec", 60.0)` are genuinely-optional tuning knobs with a typed
  `BuildSceneDistanceConfig` backing them — not load-bearing silent defaults.
- Comments explain *why* (one-sided boost = the over-merge fix), not what.

## 3 · Naming / package — `pass`
- `build_scene_distance` step name matches its file and produces (`scene_distance`/`scene_distance_signal`).
- Helper lives in `sim_bench/scene_cluster/` alongside `two_stage.py` (≥3 files now share the concern —
  correctly a subpackage). Config registered in `STEP_CONFIG_MODELS` + `__all__`.

## 4 · Layering / coupling — `pass`
- `scene_cluster/` (domain) imports only numpy/stdlib; the step imports the helper (correct direction, no
  reverse import). `context.py` types the field as `Optional[Any]` to avoid importing the helper dataclass
  (no new coupling).
- **Single writer**: `scene_distance` is written only by `build_scene_distance`; read only by
  `cluster_scenes`. `scene_clusters` still has its single writer (`cluster_scenes`) — unchanged.

## 5 · Testability — `pass`
- **Test inventory**: unit = 7 (`test_scene_distance_builder.py`) + 4 (`test_build_scene_distance.py`,
  step+wiring) + 6 (`test_two_stage.py`, the A3 probe) = 17 new. Static/architecture = the spec-034
  contract test (now covers the 2 new fields). E2E on real data = the 3-trip report runs
  (`report_path_a.py`, Budapest/Austria/Germany) — the feature exercised on production photos.
- **Failure-mode walk-through**:
  - *over-merge* (visually-distinct shots minutes apart glued): `ut_SceneDistance_never_pushes_apart` +
    `ut_SceneDistance_far_apart_is_pure_visual` guard the one-sidedness that prevents it.
  - *near-dup miss* (rescue): covered end-to-end by the Austria case in the report; unit-guarded by
    `ut_SceneDistance_short_range_pull_shrinks_distance`.
  - *byte-identical-when-absent* (A5, binding): `ut_A5_boost_zero_equals_embedding_path` +
    `ut_cluster_scenes_unchanged_when_distance_absent`.
  - *no imputation* (missing time never fabricated): `ut_SceneDistance_missing_time_is_pure_visual`.
- Mock usage: none (toy vectors + filename timestamps, real code paths).
- Placement correct: unit in `tests/scene_cluster` + `tests/pipeline`; contract in `tests/architecture`.

## 6 · Boundary contracts — `pass`
- `BuildSceneDistanceConfig(extra="forbid")` on the new config (bounds: `boost∈[0,1]`, `tau_sec>0`).
- The `scene_distance` field is consumed at a real boundary (`cluster_scenes` clusters it) — not a
  defined-but-unused contract.
- **Config knob → producer check**: `cluster_scenes` only reads `scene_distance` when it is not None, and
  the only producer (`build_scene_distance`) sets it — no gated-value-without-producer (SIGHTING-061 class).

## 7 · Documentation — `pass`
- `spec.md` (design-update note added, supersedes D3), `tasks.md` (T2/T3 checked), this `REVIEW.md`.
- `classes.html` (helper + Inputs/Result row), `data_flow.html` (opt-in step node), `context.py`
  docstring, `configs/pipeline.yaml` block — all updated. No DB column changed (in-run only) → no
  `db_schemas.html` change needed. `CHANGES_LOG.md` entry added. spec-034 contract updated for the 2 fields.
- LEARNINGS: candidate entry — "additive metadata fusion over-merges; one-sided short-range boost is the
  safe shape" — see follow-up below.

## 8 · Risk register — `pass-with-followup`
- **Backwards-compat**: zero. Step is gated by presence (not in any default pipeline); absent → clustering
  byte-identical (A5 test). No legacy data breaks.
- **Hot-path perf**: `SceneDistanceBuilder.calc` has an O(N²) Python loop for the time discount. On the
  largest trip (Germany, 797 imgs → ~317k pair-iterations) it is sub-second and dwarfed by DINOv2
  extraction; acceptable. **Follow-up**: vectorize the discount (broadcast the pairwise `dt` matrix) if it
  is ever run on >few-thousand-image sets. Filed as a note in tasks.md.
- **Deferred**: labels (T1) + ARI scoring (T4) still open — spec stays In Progress. Face-style cluster-merge
  idea parked in tasks.md. Geo term unused in Path A (visual+time only).

---

## Findings
| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | none (fixed) | New context fields tripped `test_pipeline_context_fields_covered` | Fixed: spec-034 rows added; test green |
| 2 | low | O(N²) Python discount loop | Follow-up: vectorize if run on >few-thousand imgs (tasks.md) |
| 3 | n/a (pre-existing) | 3 unrelated test failures + 1 stale-import test | Filed **SIGHTING-118**; none touch scene clustering |

**No high-severity findings. Handoff of the wiring is OK. Spec remains In Progress pending the
experiment-validation tasks (T1 labels, T4 ARI) before any flip to Implemented.**
