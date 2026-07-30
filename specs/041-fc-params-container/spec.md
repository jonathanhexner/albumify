# spec-041 — FC App v2 Configuration Container & Run Tab Parity

Status: **Code Review** (pending `/code-review` slash command — user-triggered)
Author: Jonathan Hexner
Created: 2026-05-22
Predecessor: [spec-040](../040-unified-pipeline-framework/spec.md) (Unified Pipeline Framework)

---

## 1. Objective

Make the FC App v2 (`app/face_clustering_v2/`) drivable end-to-end with the same configurability as the original FC App (`app/face_clustering/`) — without re-introducing the per-knob, per-widget threading that spec-040 was designed to eliminate.

Concrete goals:

1. Introduce **`FCParams`** — a single Pydantic v2 container holding every clustering knob (35+ fields). This becomes the contract between the UI, the test suite, profile JSONs, and the algorithm layer.
2. Migrate `run_v2_pipeline`, the Run tab, the e2e tests, and the equivalence sweep to consume `FCParams` instead of free-form dicts.
3. Delete `_build_fc_config` from `face_clustering_steps.py` — the 8 per-step `.get(..., default)` blocks become obsolete because the container guarantees full population.
4. Collapse `scripts/migrate_fc_profiles.py` to a 5-line wrapper around `FCParams.model_validate(flat).model_dump_json()`.

Non-goal:

- Re-port the other 5 tabs from the original FC App (Recluster, Merge Analysis, Merge ML, Quality, Gallery). Tracked separately; this spec only touches the **Run** and **Clusters** tabs already shipped in spec-040 T4.
- Replacing the `face_cluster.config.PipelineConfig` dataclass (`FCConfig`). It remains the algorithm-layer contract; `FCParams.to_fc_config()` is the one boundary translator.

---

## 2. Why now

### 2.1 The trigger

User feedback during spec-040 burn-in:

> "Run failed: Clustering chain failed: Step 'build_face_knn_graph' failed: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (186,) + inhomogeneous part. I don't quite understand the approach. I need to be able to run the app in my way to verify it works."

Two distinct complaints, same root cause: v2's config surface is too narrow and too untyped to let the user drive it. The numpy crash was a path-key mismatch in the A1 dual-write (fixed in this branch on 2026-05-21), but the underlying frustration — "I can't tune this like the original" — is a real product gap.

### 2.2 The structural cost of the current shape

`run_v2_pipeline` currently accepts `step_configs: dict[str, dict]`. Today the UI builds **one** dict and broadcasts it to every step. Each step then calls `_build_fc_config(config)` which does ~25 `.get("...", default)` lookups. Defaults live in two places (the UI's literal and the helper's defaults) and drift silently.

If we added the 25 missing knobs by my original plan (3-commit Streamlit-widget-per-field), we'd be:

- adding 25 `st.session_state` keys,
- adding 25 widget bindings,
- adding 25 dict-key insertions,
- relying on `_build_fc_config` to know all 25 names by name.

That's exactly the per-knob threading spec-040 set out to eliminate. The locked decision was "no translator-in-disguise"; this would re-introduce one.

### 2.3 The structural payoff of `FCParams`

| Concern | Today (post-spec-040) | After spec-041 |
|---|---|---|
| Default values | In `_build_fc_config` AND UI literals | One source: `FCParams` field defaults |
| Adding a knob | Edit UI + helper + (sometimes) step | Add field to `FCParams`; bind widget |
| Profile save/load | `scripts/migrate_fc_profiles.py` (115 LOC, 6 tests) | `params.model_dump_json()` |
| Validation | None (silent `.get(default)`) | Pydantic `Field(ge=..., le=...)` + validators |
| Test config construction | Untyped dicts; equivalence sweep duplicates field names | `FCParams(...)` literal shared by both sides |
| Drift between UI & algorithm | Detected at run time as crashes or silent wrong-defaults | Detected at import time by drift test |

---

## 3. Scope

### In scope

| Area | Change |
|---|---|
| **New module** | `face_cluster/fc_params.py` (Pydantic v2 model) |
| **Pipeline entry** | `run_v2_pipeline` gains a `params: FCParams` kwarg (preferred); legacy `step_configs` kwarg kept for one release for backward-compat |
| **Steps** | `face_clustering_steps.py` — drop `_build_fc_config`; each step does `FCConfig(**config)` directly |
| **Run tab** | Tier 1–4 knobs (11 + 3 + 3 + 18 ≈ 35) bound via `FCParams` constructor at click time |
| **Tests** | New `tests/face_clustering/test_fcparams.py`; migrate `test_fc_app_v2_e2e.py` and `test_legacy_vs_v2_equivalence.py` to FCParams literals |
| **Profile script** | `scripts/migrate_fc_profiles.py` collapsed to a 5-line wrapper; existing 6 tests trimmed/repointed |
| **CLI** | New `scripts/run_v2.py` for headless A/B runs (`--src --out --profile`) |
| **Drift guard** | `tests/architecture/test_fcparams_fcconfig_parity.py` — asserts the two field sets match |

### Out of scope

- The 5 remaining FC App tabs (Recluster, Merge Analysis, Merge ML, Quality, Gallery).
- Async worker + live-log machinery in the v2 Run tab (the original's `_AsyncState`).
- Replacing `FCConfig` dataclass with `FCParams` algorithm-side.
- The Albumify-side `cluster_people` step still calling the deprecated `face_cluster_bridge` (that's spec-040 Phase 7).

---

## 4. Design

### 4.1 The `FCParams` contract

```python
# face_cluster/fc_params.py
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


class FCParams(BaseModel):
    """User-facing configuration container for FC App v2.

    Single source of truth for every clustering knob. Bound to UI widgets
    in app/face_clustering_v2/tabs/run_tab.py and to profile JSON files.
    The algorithm layer continues to consume face_cluster.config.PipelineConfig
    (`FCConfig`); FCParams.to_fc_config() is the one boundary translator.
    """
    model_config = ConfigDict(extra='forbid')

    # --- Cluster ---
    K: int = Field(5, ge=1, le=100)
    distance_threshold: float = Field(0.35, ge=0.01, le=1.0)
    min_cluster_size: int = Field(2, ge=1, le=50)

    # --- Quality gates ---
    blur_min: float = Field(50.0, ge=0.0, le=500.0)
    max_faces_per_image_core: int = Field(3, ge=1, le=50)
    min_face_area: Optional[int] = Field(None, ge=0)
    det_score_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    yaw_max: float = Field(30.0, ge=5.0, le=90.0)
    pitch_max: float = Field(25.0, ge=5.0, le=90.0)
    roll_max: float = Field(25.0, ge=5.0, le=90.0)
    require_pose: bool = False

    # --- Exemplars ---
    N_exemplars_max: int = Field(10, ge=1, le=100)
    exemplars_d10_threshold: float = Field(0.35, ge=0.01, le=1.0)
    exemplar_suppression_radius: float = Field(0.2, ge=0.01, le=1.0)

    # --- Optional stages ---
    split_enabled: bool = False
    merge_enabled: bool = False
    attach_enabled: bool = False

    # --- Merge ---
    merge_candidate_threshold: float = Field(0.45, ge=0.01, le=1.5)
    merge_exemplar_threshold: float = Field(0.45, ge=0.01, le=1.5)
    use_adaptive_merge_threshold: bool = True
    exemplar_percentile: int = Field(90, ge=0, le=100)
    global_percentile: int = Field(75, ge=0, le=100)
    alpha: float = Field(1.0, ge=0.0, le=3.0)
    beta: float = Field(0.5, ge=0.0, le=3.0)
    merge_use_cross_gate: bool = True
    merge_cross_threshold: float = Field(0.40, ge=0.01, le=1.5)
    merge_cross_max_size: int = Field(5, ge=1, le=20)
    merge_support_frac: float = Field(0.3, ge=0.0, le=1.0)
    merge_support_min: int = Field(2, ge=1, le=20)
    merge_support_unique: bool = False
    merge_margin: float = Field(0.0, ge=0.0, le=0.5)
    merge_diameter_expansion_factor: float = Field(1.5, ge=1.0, le=5.0)

    # --- Diameter cap (spec-031) ---
    cluster_diameter_cap_enabled: bool = False
    max_full_diameter: float = Field(1.2, ge=0.3, le=2.0)
    max_exemplar_diameter: float = Field(0.8, ge=0.2, le=2.0)

    # --- Translations ---
    def to_fc_config(self) -> "FCConfig":
        """Boundary translator: FCParams → algorithm-layer FCConfig."""
        from face_cluster.config import PipelineConfig as FCConfig
        return FCConfig(**self.model_dump())

    def to_step_configs(self) -> dict[str, dict]:
        """Broadcast: every unified step gets the full param dict."""
        from face_cluster.fc_app_runner import UNIFIED_CLUSTERING_STEPS
        d = self.model_dump()
        return {name: d for name in UNIFIED_CLUSTERING_STEPS}

    @classmethod
    def load(cls, path) -> "FCParams":
        from pathlib import Path
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")
```

### 4.2 Call sites that change

| File | Today | After |
|---|---|---|
| `app/face_clustering_v2/pipeline.py` | `run_v2_pipeline(..., step_configs: dict = None)` | `run_v2_pipeline(..., params: FCParams = None, step_configs: dict = None)`. If `params` given, derive `step_configs = params.to_step_configs()`. |
| `app/face_clustering_v2/tabs/run_tab.py` | Builds `cfg = {...}` dict, broadcasts to every step name | Builds `params = FCParams(**widget_values)`, calls `run_v2_pipeline(params=params)` |
| `sim_bench/pipeline/steps/face_clustering_steps.py` | Each step: `fc_cfg = _build_fc_config(config)` | Each step: `fc_cfg = FCConfig(**config)`. Helper deleted. |
| `tests/face_clustering/test_fc_app_v2_e2e.py` | Builds `cfg` dict in fixture | Builds `FCParams(...)` literal in fixture |
| `tests/face_clustering/test_legacy_vs_v2_equivalence.py` | 4 dict literals parametrized | 4 `FCParams(...)` literals parametrized |
| `scripts/migrate_fc_profiles.py` | 115 LOC, custom `is_v2_shape`/`reshape_v1_to_v2` | 5-line wrapper: `FCParams(**flat).save(path)` |

### 4.3 Drift guard

Single test in `tests/architecture/test_fcparams_fcconfig_parity.py`:

```python
def test_fcparams_fields_match_fcconfig_fields():
    fc_fields = {f.name for f in dataclasses.fields(FCConfig)}
    p_fields = set(FCParams.model_fields)
    assert fc_fields == p_fields, (
        f"missing in FCParams: {fc_fields - p_fields}\n"
        f"missing in FCConfig:  {p_fields - fc_fields}"
    )
```

If anyone adds a knob to one without the other, CI fails.

### 4.4 Backward compatibility

- `run_v2_pipeline(step_configs=...)` continues to work for one release.
- Existing v2 profiles (output of `migrate_fc_profiles.py` from spec-040) load via `FCParams.model_validate({**legacy_flat})` because field names match.
- `_build_fc_config` deletion is internal; no external caller imports it.

---

## 5. Testing strategy

### 5.1 Test pyramid

| Level | File | What it asserts |
|---|---|---|
| Unit (FCParams) | `tests/face_clustering/test_fcparams.py` | Defaults, range validation, `extra='forbid'`, JSON round-trip, `to_fc_config()`, `to_step_configs()` |
| Architecture | `tests/architecture/test_fcparams_fcconfig_parity.py` | Field-set equality with `FCConfig` |
| Integration | `tests/face_clustering/test_fc_app_v2_e2e.py` | End-to-end run via `FCParams` literal |
| Equivalence | `tests/face_clustering/test_legacy_vs_v2_equivalence.py` (slow, 50 imgs) | 4-config sweep ≥ 95% agreement, parametrized over `FCParams` literals |
| Profile | `tests/face_clustering/test_profile_migration.py` | Updated for the collapsed migration |
| CLI smoke | `tests/face_clustering/test_run_v2_script.py` | New: `scripts/run_v2.py --profile` runs against the 6-jpg fixture |

### 5.2 Non-UI verification path

Critically, **every test path runs without Streamlit**. The Streamlit Run tab is verified only by manual smoke + the existing Playwright check (FCParams adds no new UI logic to test — widgets are just bindings).

A typical headless A/B looks like:

```bash
# Save a profile from a known-good run
.venv/Scripts/python scripts/run_v2.py --src D:/album --out runs/baseline \
    --K 5 --distance_threshold 0.35 --save-profile profiles/baseline.json

# Re-run from profile, compare
.venv/Scripts/python scripts/run_v2.py --src D:/album --out runs/repro \
    --profile profiles/baseline.json
```

---

## 6. Plan — execution order

One commit per step. Each step is independently reviewable and leaves the tree green.

### Step 1 — `FCParams` container + unit tests (~150 LOC + 80 LOC tests)

- Add `face_cluster/fc_params.py` (the model in §4.1).
- Add `tests/face_clustering/test_fcparams.py`:
  - default values
  - `extra='forbid'` rejects unknown keys
  - `Field(ge=..., le=...)` rejects out-of-range
  - `model_dump_json()` round-trip preserves equality
  - `to_fc_config()` returns an `FCConfig` with all fields populated
  - `to_step_configs()` returns a dict keyed by every entry in `UNIFIED_CLUSTERING_STEPS`
  - `load()` / `save()` round-trip through a `tmp_path`
- Add `tests/architecture/test_fcparams_fcconfig_parity.py` (drift guard).
- Run: `pytest tests/face_clustering/test_fcparams.py tests/architecture/test_fcparams_fcconfig_parity.py`.

**Acceptance:** All new tests green. `FCParams.to_fc_config()` produces an `FCConfig` identical to today's `_build_fc_config({...defaults})`.

### Step 2 — `run_v2_pipeline` accepts `params` (~30 LOC + 20 LOC tests)

- Add `params: Optional[FCParams] = None` kwarg to `run_v2_pipeline` in `app/face_clustering_v2/pipeline.py`.
- Internal: if `params` provided, `step_configs = params.to_step_configs()`; else preserve current behaviour.
- Update `test_fc_app_v2_e2e.py` fixture to build an `FCParams` literal and call `run_v2_pipeline(params=params)`.
- Run: `pytest tests/face_clustering/test_fc_app_v2_e2e.py`.

**Acceptance:** Existing 4 e2e tests pass via the new param path; old `step_configs` path still works (we don't remove it yet).

### Step 3 — Drop `_build_fc_config` (~60 LOC removed, no tests change)

- Replace `fc_cfg = _build_fc_config(config)` with `fc_cfg = FCConfig(**config)` in all 8 step classes of `face_clustering_steps.py`.
- Delete the `_build_fc_config` helper.
- Run: full clustering test suite — `pytest tests/face_clustering/ -k "not slow"`.

**Acceptance:** No test regressions. `face_clustering_steps.py` LOC drops by ~60.

### Step 4 — Equivalence sweep on `FCParams` (~40 LOC, tests only)

- Migrate `test_legacy_vs_v2_equivalence.py` parametrization to `FCParams` literals. The legacy side calls `params.to_fc_config()`; the v2 side calls `params.to_step_configs()`.
- Run: `pytest -m slow tests/face_clustering/test_legacy_vs_v2_equivalence.py`.

**Acceptance:** All 4 configs still ≥ 95% agreement. (If a config drifts: investigate before proceeding — it means a knob defaults differ between sides.)

### Step 5 — Run tab parity (~200 LOC widget block + tests)

- Replace the 7-widget block in `app/face_clustering_v2/tabs/run_tab.py` with a full block covering all 35 fields:
  - Group with `st.expander` sections matching the original's stage layout (Quality / Cluster / Exemplars / Optional / Merge).
  - At click time: `params = FCParams(**widget_values)` — Pydantic validates; `st.error(e)` on `ValidationError`.
  - Replace today's `step_configs=...` call with `params=params`.
- Reuse `app/shared/merge_controls.render_merge_params` for the Merge sub-panel (keys prefixed `v2_run_`).
- Manual Playwright smoke: confirm form renders, default run succeeds.

**Acceptance:** Run tab exposes the same 35 knobs as the original; the user can drive a run end-to-end with non-default settings; widget validation surfaces on invalid combinations.

### Step 6 — Collapse profile migration script (~110 LOC removed, ~5 LOC added)

- Rewrite `scripts/migrate_fc_profiles.py` to use `FCParams`:
  ```python
  def migrate_one(p: Path, dry_run=False) -> str:
      try:
          flat = json.loads(p.read_text(encoding="utf-8"))
      except Exception:
          return "invalid"
      try:
          params = FCParams.model_validate(flat)
      except ValidationError:
          return "invalid"
      if "K" not in flat and "step_configs" in flat:
          return "already_v2"
      if dry_run:
          return "migrated"
      p.with_suffix(".v1.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
      params.save(p)
      return "migrated"
  ```
- Trim `test_profile_migration.py` to match (some tests collapse — e.g., the v2-shape detector goes away).
- Run: `pytest tests/face_clustering/test_profile_migration.py`.

**Acceptance:** Migration is still idempotent, still writes `.v1.json` backup, still handles invalid JSON.

### Step 7 — Headless CLI runner (~80 LOC + 30 LOC tests)

- Add `scripts/run_v2.py`:
  - `--src`, `--out`, `--profile path/to.json` (alternative: `--K`, `--distance_threshold`, ...).
  - Constructs `FCParams`, calls `run_v2_pipeline(params=params)`.
  - Logs result counts to stdout; exit 0 on success, 1 on failure.
  - `--save-profile <path>` writes the final params as a profile.
- Add `tests/face_clustering/test_run_v2_script.py`: subprocess-call against the 6-jpg fixture.

**Acceptance:** Headless run produces the same v5 DB shape as the UI path. Profile round-trip works.

### Step 8 — Code review + docs

- Run `/code-review` slash command to produce `specs/041-fc-params-container/REVIEW.md`.
- Update `docs/architecture/classes.html` with `FCParams` (Pydantic) added.
- Append `CHANGES_LOG.md` entry per step (CLAUDE.md mandate is per-change, not per-spec).
- Flip spec status `Draft` → `Code Review` → `Implemented`.

**Acceptance:** Code review checklist green; high-severity findings closed.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Equivalence sweep drifts** after FCParams migration (defaults differ from `_build_fc_config`'s defaults) | Step 1 unit test compares `FCParams().to_fc_config()` against a known-good `FCConfig()` literal; Step 4 runs the slow sweep as gate. |
| **Profile JSON breakage** for users with v2-shape profiles from spec-040 | `FCParams.model_validate(flat)` works for both legacy-flat and v2 shapes because field names are identical; backup `.v1.json` is preserved. |
| **Pydantic range constraints reject historically-used values** (e.g., someone has `K=150` in a profile) | First-pass ranges chosen to match the original Streamlit slider min/max. Edge cases handled by a "loose" mode (`Field(..., le=None)`) — defer unless a real user hits it. |
| **`FCConfig` algorithm-layer dataclass adds a new field** without a corresponding `FCParams` field | Drift test in Step 1 catches it at import time / CI; fails loud. |
| **`step_configs=` kwarg deprecation timing** | Keep both signatures for one release; emit `DeprecationWarning` on `step_configs=` after spec-041 ships. Remove in spec-042 or later. |

---

## 8. Open questions

1. **`FCConfig` retirement?** Long-term, should the algorithm layer consume `FCParams` directly (Pydantic everywhere) and `FCConfig` go away? This spec says **no, defer** — keep `FCConfig` as the algorithm-layer dataclass and `FCParams.to_fc_config()` as the one boundary. Revisit in a future spec if the duality becomes a maintenance burden.
2. **Validator for merge-gated fields?** E.g., when `merge_enabled=False`, should setting `merge_candidate_threshold=0.99` be a warning, an error, or silently ignored? Recommended: silently ignored, matching today's behaviour. Document explicitly in the `merge_enabled` field's docstring.
3. **Should `FCParams` live in `face_cluster/` or in `sim_bench/`?** Recommended: `face_cluster/fc_params.py` — it mirrors `FCConfig` location and the FC App imports it. No `sim_bench/` dependency.

---

## 9. Definition of Done

- [ ] `face_cluster/fc_params.py` exists with 35 fields matching `FCConfig`.
- [ ] `tests/face_clustering/test_fcparams.py` passes (≥ 8 tests).
- [ ] `tests/architecture/test_fcparams_fcconfig_parity.py` passes.
- [ ] `_build_fc_config` deleted; all 8 steps use `FCConfig(**config)` directly.
- [ ] `run_v2_pipeline(params=FCParams(...))` is the documented call shape.
- [ ] Run tab exposes all 35 knobs; manual run with non-defaults succeeds.
- [ ] Slow equivalence sweep ≥ 95% agreement across all 4 configs.
- [ ] `scripts/run_v2.py` runs the v2 pipeline headless from a profile JSON.
- [ ] `scripts/migrate_fc_profiles.py` collapsed to a 5-line wrapper; tests still pass.
- [ ] `REVIEW.md` produced; high-severity findings closed.
- [ ] `CHANGES_LOG.md` entries per commit.
- [ ] Spec status flipped to **Implemented**.
