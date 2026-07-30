# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## On Init Checklist
Upon starting a new session:
1. Scan `TODO.md` for open tasks (`[ ]` or `[>]`)
2. Scan `docs/project/FEATURE_REQUESTS.md` for open requests
3. Check `docs/project/SIGHTINGS.md` for open/in-progress issues
4. Review recent entries in `docs/project/LEARNINGS.md` (last 5-10) and skim `CHANGES_LOG.md` (last 2 weeks)

## MANDATORY: Feature Development Workflow

**Any new feature MUST have a spec before implementation. See `WORKFLOW.md` for details.**

Gates: `specs/NNN-<name>/spec.md` + `specs/NNN-<name>/tasks.md` must exist.
Exemptions: bug fixes (file sighting instead), isolated refactors, doc-only changes.
Spec lifecycle: `Draft` → `In Progress` → `Code Review` → `Implemented`.

**Code Review gate is MANDATORY.** Before flipping a spec to `Implemented`, run the `/code-review` slash command to produce `REVIEW.md` against `docs/guides/CODE_REVIEW_CHECKLIST.md`. High-severity findings block handoff. New feature requests do NOT go in `docs/project/FEATURE_REQUESTS.md` (deprecated); they become a proper spec dir via WORKFLOW.md.

---

## Communication style (mandatory)
- **Responses ≤ 150 lines.** If a thorough answer would exceed that, write the long version to an artifact (HTML / MD under `specs/` or `docs/`) and link to it.
- **Prefer diagrams** (ASCII / mermaid / image) over prose for anything structural (data flow, module relationships, decision trees).
- **No jargon without a 1-clause inline definition.** Examples of terms to define-or-skip: monkeypatch, fileConfig, autouse fixture, stamp head, Greenspun's tenth rule.
- Lead with the direct answer. No recap of what the user just said. No "let me explain". No symmetric "what was supposed / what actually" templates.
- If a follow-up clarification might be needed, let the user ask. Don't preempt.

## Implementation gate (mandatory — strengthens §MANDATORY: Feature Development Workflow)
- **No code without a spec.** Any non-trivial change requires `specs/NNN-<name>/spec.md` + `specs/NNN-<name>/tasks.md` BEFORE the first edit. If asked to implement without one, propose the spec first; skip only on explicit "just do it" from the user.
- **No "Implemented" without REVIEW.md.** Run `/code-review` at the end of every implementation spec. Walk all 8 sections of `docs/guides/CODE_REVIEW_CHECKLIST.md`. Resolve high-severity findings before flipping the spec status.
- **For spec-implementer subagent users**: it will refuse to start without a spec dir and refuse to finish without REVIEW.md. Drive it via `Agent(subagent_type="spec-implementer", ...)`.
- Exemptions unchanged: bug fixes → file a sighting in `docs/project/SIGHTINGS.md`; pure refactor with no behavior change → CHANGES_LOG only; docs-only → CHANGES_LOG only.

## Pipeline step file convention (mandatory — spec-053)
- **One Step class per file** under `sim_bench/pipeline/steps/`. Files that bundle multiple steps (e.g. `face_clustering_steps.py`) are anti-pattern; split when touched.
- **Step files are thin (≤80 LOC).** A step reads context, builds a helper `Inputs` dataclass, calls `helper.calc(inputs)`, writes the `Result` back to context. Domain logic belongs in `face_cluster/`, not in the step.
- **Every domain helper exposes `calc(inputs) -> result`** as its primary entry. Config goes in `__init__`; per-call data goes in a typed `Inputs` dataclass; output is a typed `Result` dataclass. The individual helper methods (`compute_blur_scores`, `build_graph`, …) stay public for notebook callers; **pipeline steps must use `calc()`** so orchestration constraints (e.g. "compute blur before selecting core set") are enforced by construction.
- **`Inputs` / `Result` are NOT `PipelineContext`.** The step is the translator; helpers stay framework-agnostic so notebooks can use them directly.
- Rationale: spec-053 codified this after two quality-gate steps drifted (one forgot to call `compute_blur_scores`) and the blur threshold silently self-disabled in production. The convention makes that class of bug impossible by construction.

---

## General
- **Never implement code changes without an explicit request.** Question → answer. Problem report → debug first. Propose changes, don't make them without approval.
- If solution is obvious and quick, go for it. Else file a sighting in `docs/project/SIGHTINGS.md`.
- Log learnings from failures in `docs/project/LEARNINGS.md` (3-5 lines, date, newest first).
- Produce plans for approval before non-trivial changes. Small isolated fixes may be implemented directly.
- No feature is complete without a test (or explicit justification). See `docs/guides/TESTING_RULES.md` for conventions and anti-patterns.
- Break down work into `TODO.md` items. Update status when starting (`[>]`) and completing (`[x]`).
- Log new user requests in `docs/project/FEATURE_REQUESTS.md`.
- **Documentation structure**: Follow `docs/README.md` for placement rules.

## Delivery Quality — Complete Before Returning to User

**A fix or feature is NOT done until tested end-to-end and verified visually.**

1. **Fix ALL reported issues in one pass** — not one at a time.
2. **Restart the app yourself** — verify the fix is live.
3. **Run Playwright E2E against the real app** with real data. Check screenshots.
4. **Check what the user will actually see** — no images = not done, button does nothing = not done.

### V2 baseline gate (binding)

**Any change to a v2 tab or any new v2 feature MUST run `tests/face_clustering/e2e_budapest/` before being marked Implemented.** The functionality matrix lives in [`tests/face_clustering/e2e_budapest/README.md`](../tests/face_clustering/e2e_budapest/README.md) — every scenario lists its click sequence, its concrete assertions, and what regression class it catches.

  - Source: `D:\Budapest2025_Google`
  - Profile: `profile_4.json`
  - Expected: `n_clusters == 15`, `n_faces == 340`, cluster sizes `[35, 24, 14, 7, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2]`
  - Reference run id (Scenario B): `6437d335de914755bc3edb825c9591c0`
  - Run: `.venv/Scripts/python -m pytest -m budapest tests/face_clustering/e2e_budapest/ -v`

**When adding a new tab**: (1) add one row to the README matrix, (2) add one `test_scenario_<letter>_<short_desc>.py` file in that dir, (3) list concrete assertions beyond "no exception." Specs 063-066 own scenarios C-H respectively.

**A v2 commit without this suite green is not Implemented**, regardless of unit-test status.
5. **Trace data end-to-end** — pipeline step → context → DB → API schema → API response → frontend model → UI display. Missing any link = silently broken.
6. **Test the full data path** — `py_compile` passing means nothing without real data.
7. **Do not report a feature as complete when you know parts are unimplemented.** If it needs to be in 4 places and you did 1, that is 25% done. Say so, or finish the job.

## Experiment reports (mandatory)
Every exploration/experiment (model comparison, detector research, validation study, threshold sweep) MUST leave an inspectable trail in `reports/`:
1. `reports/<YYYY-MM-DD>_<slug>/report.html` — experiment goal, method, **a few sample images inline** (downscaled copies inside the folder), how to access the full image set (path on data drive), and outcomes.
2. `reports/<YYYY-MM-DD>_<slug>/summary.md` — short markdown summary of the same.
3. Append a **2-line entry** to `reports/EXPERIMENTS.md` (the index): line 1 = name + goal, line 2 = outcome + link.
- Transient artifacts (e.g. generated blur variants) must NOT be score-and-delete: save at least a sample set into the report folder so results can be eyeballed later.
- Large image sets stay on the data drive (e.g. `D:\occlusion_dataset\...`); the report links to them.

## Coding
- Python 3.10+. Type hints. Full imports (no relative). Logging over prints.
- `__init__.py` kept empty except `face_cluster/__init__.py`.
- **Protobuf**: `protobuf>=3.20,<4` (MediaPipe).
- **Metadata Storage**: Always save source path, timestamp, version when exporting.

## Testing
- See `docs/guides/TESTING_RULES.md` for naming conventions (`ut_<Subject>`, `test_*`), anti-patterns, and requirements.
- Tests must run on Windows. ASCII-only in CLI output.
- Tests must use production-default config, not relaxed values.

## Windows Development
- Use `.venv/Scripts/python`, `.venv/Scripts/streamlit`. Never bare `python`.
- After adding a new package directory, re-run `.venv/Scripts/pip install -e .`.
- Forward slashes in code paths. DB at `~/.sim_bench/`.

## Sightings & Bug Discipline
- File briefly: problem, symptoms, suspicion, reproduction steps, responsible persona.
- On resolution: verify tested, log learning in `docs/project/LEARNINGS.md`.
- For every non-trivial bug: root cause → prevention mechanism → learning. Never fix symptoms only.

## Change Tracking
After EVERY code change, append to `CHANGES_LOG.md`: date, category tag ([FEATURE]/[BUGFIX]/[REFACTOR]/[DOCS]/[TEST]/[CONFIG]/[PERF]), files, change, reason.

## Architecture & Design Rules
- See `docs/architecture/index.html` for the central documentation index (live HTMLs: db_schemas, classes, data_flow).
- See `docs/architecture/overview.md` for system architecture.
- See `RECOVERY_PLAN.md` for face clustering module responsibilities.
- See `docs/guides/TROUBLESHOOTING.md` for debugging tips.
- **Documentation update mandate**: any change to a class, Pydantic model, Pandera schema, DB column, pipeline step, or boundary contract MUST update the corresponding HTML in `docs/architecture/` in the same PR. The Code Review gate (§7 of `docs/guides/CODE_REVIEW_CHECKLIST.md`) checks this.
- Architecture changes → update `docs/architecture/overview.md`. Wait for approval on non-trivial changes.

## Key Entry Points
| Entry Point | Purpose |
|---|---|
| `sim_bench/api/main.py` | FastAPI server |
| `app/streamlit/main.py` | Albumify frontend (7 pages) |
| `app/face_clustering/main.py` | Face clustering app (11 tabs) |
| `app/image_studio/main.py` | Image Analysis Studio (spec-094): Configure/Browse, quality + geo multi-method comparison |
| `face_cluster/pipeline.py` | Standalone clustering API |
| `configs/pipeline.yaml` | Pipeline steps and parameters |

## Common Commands
```bash
# Apps (always use .venv/Scripts/)
.venv/Scripts/python -m uvicorn sim_bench.api.main:app --reload --port 8000
.venv/Scripts/streamlit run app/streamlit/main.py
.venv/Scripts/streamlit run app/face_clustering/main.py
scripts/restart_apps.bat   # Kill + restart all 3

# Tests
.venv/Scripts/python -m pytest tests/
.venv/Scripts/python -m pytest tests/test_file.py -k "test_name"
.venv/Scripts/python tests/test_full_e2e_flow.py  # Full Playwright E2E

# Database
sqlite3 ~/.sim_bench/sim_bench.db "DELETE FROM universal_cache WHERE feature_type = 'face_embedding'"
```

---

## Change Log Location
**Always maintain**: `CHANGES_LOG.md` at project root
