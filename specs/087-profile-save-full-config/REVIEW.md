# spec-087 — Code Review

**Reviewed**: 2026-06-25 · **Reviewer**: Claude (code-review) · **Base**: working tree
**Scope**: spec-087 (profiles save/load the full config; SIGHTING-106).

## Part 1 — How it works

**Goal:** the Configure & Run profile bar saved only the `rc_*` clustering subset, dropping
all `config_*` quality/detection/selection params (sharpness, IQA, ...). Fix = a profile now
stores BOTH the flat `rc_*` keys (cross-app interop) AND the full nested `config` blob.

**Module / data flow (`app/streamlit/components/pipeline_runner.py`):**
```
config built (~:467) ──► session_state["_last_built_config"]   (stash, one render fresh)
profile bar Save ──► _build_profile_payload(ss) = {rc_*…, "config": _last_built_config} ──► ProfileStore
profile bar Load ──► _apply_profile_to_session(profile, ss):
                       set rc_* directly; stage nested as _pending_profile_config; clear config_*
render (:188)    ──► saved_config = _resolve_saved_config(ss, api_config)  (pending one-shot)
                       → existing `value=saved_*.get(...)` widget init shows the profile values
```
Three pure helpers (`_build_profile_payload`, `_apply_profile_to_session`,
`_resolve_saved_config`) hold the logic; the UI just calls them.

## Part 2 — Findings

### §1 Structure — pass-with-followup
Change is +33 LOC of small, single-purpose helpers. **Pre-existing:** `pipeline_runner.py`
is ~500 LOC (>300) — not introduced here; consider extracting the profile bar to its own
module later (TODO). Helpers ≤2 params.

### §2 Code quality — pass
No try/except added. `ss.get("_last_built_config", {})` / `_resolve` fallback are genuinely
optional values, not load-bearing silent defaults. Prefix-clear loop materialises keys before
deleting (no mutate-during-iterate).

### §3 Naming — pass
`_build_profile_payload` / `_apply_profile_to_session` / `_resolve_saved_config` say what they
do; new session keys (`_last_built_config`, `_pending_profile_config`) use `_` prefix so the
`config_*` clear can't hit them.

### §4 Layering & coupling — pass
Single profile save/load path (was duplicated inline ×3). The two new session keys are
two-way (writer `:467` / reader Save; writer Load / reader resolver) and are documented in
the docstrings — §4 "two-way writers documented" satisfied.

### §5 Testability — pass-with-followup
- `tests/streamlit/test_profile_save_load.py` — 5 unit tests on the pure helpers: payload has
  both stores (AC1/2), load clears `config_*` + stages pending (AC3), back-compat for old
  profiles (AC4), round-trip preserves sharpness (AC5), resolver one-shot. The bug surface
  (what is saved/restored) is covered.
- **Finding (low/medium):** no automated Streamlit `AppTest` exercises the button-handler +
  rerun + widget-reinit wiring; AC6 (real app) is manual. → TODO: AppTest for the profile bar.

### §6 Boundary contracts — pass-with-followup
The profile payload is a schemaless dict via `ProfileStore` (pre-existing design; no Pydantic
contract / `extra="forbid"`). Not introduced by this spec. → optional TODO if profiles become
a typed contract.

### §7 Documentation — pass
spec.md / tasks.md / REVIEW.md / CHANGES_LOG / SIGHTINGS(-106 → FIXED) all present. No class /
Pydantic / DB / pipeline-step change → no `docs/architecture` HTML update required.

### §8 Risk — pass
- Back-compat: old profiles (flat `rc_*` only) load with no error, leave `config_*` on API
  settings — tested (`test_old_profile_without_config_loads_clean`).
- Cross-app: face_clustering Recluster tab reads top-level `rc_*` and ignores the extra
  `"config"` key — interop preserved (decision b).
- Hot-path: stashing a dict reference per render — negligible.
- Low risk: the `config_*` prefix-clear is blanket; safe today (all config widgets use that
  prefix; bookkeeping keys use `_`). Noted.

## Part 3 — Verdict

| Area | Verdict |
|---|---|
| §1 Structure | accept (pre-existing module size — TODO) |
| §2 Code quality | accept |
| §3 Naming | accept |
| §4 Layering | accept |
| §5 Testability | accept w/ follow-up (AppTest for UI wiring) |
| §6 Boundary contracts | accept w/ follow-up (typed profile payload, optional) |
| §7 Documentation | accept |
| §8 Risk | accept |

**No blockers. Handoff ACCEPTED** — spec-087 may flip to Implemented.
`tests/streamlit` + `tests/architecture` = 135 passed; `tests/api` = 22 passed.

### Follow-up tickets (TODO, non-blocking)
1. AppTest for the profile bar save/load/rerun wiring (covers what unit tests can't).
2. (Optional) split the profile bar out of `pipeline_runner.py` (module >300 LOC).
3. (Optional) typed profile payload (Pydantic) if profiles become a hardened contract.

### Note (not a finding)
`pipeline_runner.py` also contains a pre-existing, already-logged SIGHTING-102 change (pose
step `min_face_size` removal, CHANGES_LOG 2026-06-22) that predates this session — it will be
included when spec-087 is committed.
