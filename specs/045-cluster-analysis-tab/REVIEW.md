# Code Review — spec-045 Cluster Analysis Tab

**Reviewer**: Claude (Opus 4.7, /code-review pattern)
**Date**: 2026-05-29
**Spec status proposed**: `Draft` → `Code Review`
**Checklist**: [`docs/guides/CODE_REVIEW_CHECKLIST.md`](../../docs/guides/CODE_REVIEW_CHECKLIST.md)

---

## Verdict per area

| § | Area | Verdict | Notes |
|---|---|---|---|
| 1 | Structure | **pass** | Largest new file (`cluster_analysis.py`, ~245 LOC) has one named responsibility (the Service + its typed results); under the 300 LOC bar. Tab orchestrator 73 LOC; each component 28–62 LOC. |
| 2 | Code quality | **pass** | No bare `except:`; only one broad `except Exception` (force-merge UI catches errors to surface via `st.error` rather than the Streamlit traceback overlay — boundary-appropriate). No silent defaults on load-bearing paths. |
| 3 | Naming + package | **pass** | Service / Repository / Component file names mirror their roles. `__all__` declared in every new module. |
| 4 | Layering + coupling | **pass** | Forbidden patterns guarded by `tests/architecture/test_cluster_analysis_tab.py` (5 cases). No reverse imports detected. |
| 5 | Testability | **pass** | 36 new tests across 4 files (12 + 3 Repository; 12 + 4 Service; 5 architecture). Synthetic-data layer is the load-bearing signal; real-fixture layer is opt-in smoke (skips on CI). |
| 6 | Boundary contracts | **pass-with-followup** | All Service returns are typed dataclasses; Repository config is frozen-slotted with `__init__` validation. *Follow-up:* `ForceMergePreview` is a `@dataclass`, not a Pydantic model — if downstream callers feed it to FastAPI, promote it. |
| 7 | Documentation | **pass-with-followup** | `spec.md` / `tasks.md` / `REVIEW.md` present. `classes.html` + `data_flow.html` + `architecture_standards.md` updated. `LEGACY_VS_V2_CLUSTER_TAB.html` shipped. *Follow-up:* `db_schemas.html` not updated — no new columns were added, but a "Read-side consumers" note would help future readers. |
| 8 | Risk register | **pass** | All deferrals named in `tasks.md` §"Out of scope" with cross-references. |

**Overall**: **pass-with-followup**. No high-severity findings. Status can flip to `Code Review` → `Implemented` once the two follow-ups are filed.

---

## Findings

### F-1 (Minor) — Components LOC over budget by ~20%

`force_merge.py` (62 LOC) and `cluster_debug.py` (49 LOC) push the components total to 240 LOC vs spec.§Phase 6 target of ≤ 200. Each component is under 100 LOC and has a single render function; the overage is in inline-state-management for the preview / heatmap rather than scope creep. **Recommendation**: accept; revisit if a future component crosses 100 LOC alone.

### F-2 (Minor) — spec.§"Repository contract" sketches `__init__(Session)`; tasks.md / implementation use `__init__(Config)`

The PRD's "Repository contract" code block says the constructor takes a SQLAlchemy `Session`. Tasks.md T004 says `__init__(config)`. The implementation follows tasks.md (correct per spec D3 — per-run DBs aren't Alembic-managed, no Session to inject). **Action**: fix the spec.md text in a follow-up commit; arch test `test_repository_takes_typed_config` already enforces the correct signature, so this is documentation-only drift.

### F-3 (Minor) — spec.§8.2 references fixture `v2_pilot_run_dir`; actual name is `v2_budapest_run_dir`

Trivial rename in `spec.md`. No code change.

### F-4 (Minor) — Embedding dim hard-coded to 512 in the legacy snapshot writer leaked into the test fixture

The synthetic fixture had to bump `_EMBED_DIM` from 16 → 512 because `face_cluster.manual_merge_snapshot` hard-codes `EMB_DIM = 512`. If the production embedding dim ever changes, the snapshot writer (and our fixture) will need a coordinated update. **Action**: file a sighting against `manual_merge_snapshot` to read the dim from the input embeddings instead of hard-coding.

### F-5 (Minor — informational) — Manual smoke (T054) + Playwright (T063) deferred to user

Both require a live `streamlit run` + a loaded fixture run; can't be exercised inside the agent session. Playwright script is shipped at `tests/manual/_v2_cluster_analysis_smoke.py` and `tasks.md` flags both as `[~] deferred to user`.

---

## Test inventory

| Layer | Path | Count | Speed |
|---|---|---|---|
| Repository synthetic | `tests/face_clustering/repositories/test_cluster_analysis_repo_synthetic.py` | 12 | ms |
| Repository real | `tests/face_clustering/repositories/test_cluster_analysis_repo_real.py` | 3 (skipped on CI) | sec |
| Service synthetic | `tests/face_clustering/views/test_cluster_analysis_service_synthetic.py` | 12 | ms-sec |
| Service real | `tests/face_clustering/views/test_cluster_analysis_service_real.py` | 4 (skipped on CI) | sec |
| Architecture (drift guards) | `tests/architecture/test_cluster_analysis_tab.py` | 5 | ms |
| Manual smoke | `tests/manual/_v2_cluster_analysis_smoke.py` | 1 | opt-in |
| **Total automated** | | **36** | |

Full-suite verification (latest run): **197 / 197 pass** across `tests/face_clustering/{repositories,views} tests/architecture` — no regressions in spec-043 / spec-044 / spec-046 / spec-048 / spec-050 baselines.

---

## Failure mode walk-through

| Failure class | Test that would catch a regression |
|---|---|
| Tab reaches into DB directly (re-opens the legacy SQL-in-Streamlit antipattern) | `test_tab_has_no_direct_db_or_filesystem_access` |
| `cfg.get("field", "?")` literal sneaks back in | `test_tab_has_no_cfg_get_literals` |
| Service returns a bare dict | `test_service_returns_typed_objects` |
| Repository constructor takes raw `**kwargs` | `test_repository_takes_typed_config` |
| Force-merge result drifts away from preview's gate fields | `test_force_merge_preview_fields_match_writer` |
| Bare `-1` for noise re-enters the Repository (spec-045 NOISE_LABEL contract violated) | Service synthetic #5 (unknown cluster → handle.failed) catches the symptom; the absence of `-1` is grep-checked via the wider `tests/architecture/test_no_bare_noise_label.py` proposed as a future spec-045 follow-up arch test |
| ClusterView/Debug crash on a run with a noise bucket | Service synthetic #4 / #6 (the bug we hit and fixed during Phase 4) |
| Force-merge mutates parent run dir | Repository synthetic #11 (sha256 parent-dir "unchanged" guard) |
| Read-only mode bypassed | Repository synthetic #12 (ValidationError + no sibling dir) |
| Async handle's "cancelled" state silently consumed by a polling consumer | Service synthetic #12 (cancel-on-new-call) |

---

## Architecture-doc updates landed in this spec

| File | Change |
|---|---|
| `docs/architecture/classes.html` | +7 rows in §4 Internal types (Repo config, criteria, Assignment, ForceMergePreview, ForceMergeResult, AsyncHandle) and §5 Writers/readers (Repository + Service). |
| `docs/architecture/data_flow.html` | New §"spec-045 — Cluster Analysis read path" with ASCII layering + invariants paragraph. |
| `docs/architecture/architecture_standards.md` | New §B0.2.1 "Schema-owning (B0a) vs query-shape (B0b) Repositories" with comparison table + distinguishing rule. |
| `specs/045-cluster-analysis-tab/LEGACY_VS_V2_CLUSTER_TAB.html` | Already landed in commit `41e0893` (PRD review session). |

---

## Sign-off

No high-severity findings. F-1 through F-5 are minor; F-2 and F-3 are documentation-only and tracked here. **Spec status proposed: `Code Review` → `Implemented`** after F-2 / F-3 are folded into spec.md in a follow-up commit (or accepted as documented drift if the maintainer prefers to leave the spec frozen).
