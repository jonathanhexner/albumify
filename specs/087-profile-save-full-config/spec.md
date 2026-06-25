# spec-087 — Profiles save/load the full config (not just clustering params)

**Created**: 2026-06-25 · **Status**: Implemented · **Priority**: P1
**Source**: SIGHTING-106 — Configure & Run profile Save dropped `min_sharpness` (and all
`config_*` quality/detection/selection params). Decision: **option (b)** — store both.

## Problem
The profile bar Save/Load (`pipeline_runner.py`) only persists the `_RC_PARAM_KEYS` (`rc_*`
clustering) subset of `session_state`. Every `config_*` widget value (sharpness, IQA, face
size, det conf, max-per-cluster, dup threshold, …) is silently dropped on Save and never
restored on Load. The authoritative full settings live in the `config` dict (`:375-438`),
used by Run and the working "Save Settings" button, but the profile bar never sees it.

## What we build (option b — store both)
A profile now stores **both**:
- the flat `rc_*` keys (unchanged — keeps cross-app interop with the face_clustering
  Recluster tab, which reads top-level `rc_*`), and
- a nested `"config"` blob = the exact dict Run/Save-Settings build.

No new hand-maintained field list — reuse what already exists:
1. **Stash the built config.** After `config` is assembled (`:375`), write it to
   `session_state["_last_built_config"]`. The profile bar renders earlier in the same run,
   so on the Save-click rerun this stash holds the live config (one render fresh).
2. **Save** → `store.save(name, {**flat_rc, "config": _last_built_config})`.
3. **Load** → set `rc_*` from the flat keys (existing); if the profile has `"config"`, stash
   it as `_pending_profile_config` and **clear the `config_*` session keys** so the widgets
   re-initialise from it.
4. **Consume on render** → `saved_config = _pending_profile_config (one-shot) else
   user_settings.config`. This reuses the existing `value=saved_*.get(...)` widget-init path,
   so there is no nested-config → widget inverse mapping to drift.

Logic extracted to 3 pure helpers (`_build_profile_payload`, `_apply_profile_to_session`,
`_resolve_saved_config`) so it's unit-testable without Streamlit.

## AC
| # | Criterion | Verified |
|---|---|---|
| 1 | Saved profile contains the nested `config` incl. `filter_quality.min_sharpness` | unit test |
| 2 | Saved profile still contains flat `rc_*` keys (cross-app interop intact) | unit test |
| 3 | Loading clears stale `config_*` keys + seeds `saved_config` so widgets show profile values | unit test |
| 4 | Old profiles (no `"config"`) load without error (back-compat) | unit test |
| 5 | Round-trip: save sharpness=0.1 → load → resolved `saved_config` yields 0.1 | unit test |
| 6 | Manual: Configure & Run save "default8" w/ sharpness 0.1, reload → shows 0.1 | manual |

## Notes
- Back-compat: old profiles have only flat `rc_*`; Load leaves `config_*` widgets on the API
  settings (today's behaviour). New saves always add `"config"`.
- Cross-app: the face_clustering app ignores the extra `"config"` key (reads `rc_*`).
