# GEMINI.md

This file provides foundational mandates for the Gemini CLI agent when working in this repository. These instructions take absolute precedence over general workflows.

## On Init Checklist
Upon starting a new session:
1. Scan `TODO.md` for open tasks (status `[ ]` or `[>]`)
2. Scan `docs/FEATURE_REQUESTS.md` for open requests
3. Check `docs/SIGHTINGS.md` for `OPEN` or `IN PROGRESS` issues
4. Review recent entries in `docs/LEARNINGS.md` and `CHANGES_LOG.md` (last 2 weeks)
5. Propose codifying recurring patterns into reusable skills if found.

## MANDATORY: Feature Development via spec-kit
**Any new feature request MUST follow the spec-kit workflow in `WORKFLOW.md`. No exceptions.**
1. `specs/NNN-<name>/spec.md` (via `templates/commands/specify.md`)
2. `specs/NNN-<name>/plan.md` (via `templates/commands/plan.md`)
3. `specs/NNN-<name>/tasks.md` (via `templates/commands/tasks.md`)
4. All `checklists/` items marked `[x]` (via `templates/commands/checklist.md`)

**Exemptions**: Bug fixes (file a sighting in `docs/SIGHTINGS.md`), isolated refactors, doc-only changes.

## General Guidelines
- **No Unrequested Changes**: Never implement code changes without an explicit Directive. Inquiries only require analysis/proposals.
- **Debug First**: When a problem is reported, perform first-level debugging before proposing code changes.
- **Failure Reports**: Append failures/learnings (3-5 lines) to `docs/LEARNINGS.md`, newest first.
- **Approval Required**: For non-trivial changes (architecture, data flow, config), produce a plan for approval first.
- **Testing Standards**:
    - No feature is complete without unit tests in `tests/`.
    - Tests must use **production-default config**.
    - **Windows Verification**: Tests must be verified on Windows. Use ASCII-only characters in CLI output to avoid `charmap` errors.
    - **Clustering**: Assert both purity AND completeness.
    - **Contracts**: Test writer-reader pairs end-to-end.
- **Architecure Discipline**: Update `docs/architecture.md` immediately after approved changes are implemented.

## ML & Face Clustering Rules
- **Benchmark everything**: Save inputs, outputs, configs, and metrics.
- **Standalone Face Clustering**: Use `face_cluster/pipeline.py` as the API. Follow the responsibility table in `RECOVERY_PLAN.md`.
- **Non-Blocking UI**: In Streamlit, dispatch slow operations to background threads via the `_AsyncState` pattern.

## Coding Standards
- **Python 3.10+**: Use type hints consistently.
- **Imports**: Always use full imports (e.g., `from sim_bench.pipeline.base import BaseStep`). No local/relative imports.
- **Windows venv**: Always use `.venv\Scripts\python` or `.venv\Scripts\streamlit`.
- **Editable Install**: Re-run `.venv\Scripts\pip install -e .` after adding new top-level packages.
- **Logging**: Use proper logging; avoid `print`.

## Change Tracking
**After EVERY code change, append to `CHANGES_LOG.md`**:
- ISO 8601 Timestamp
- **Category**: [FEATURE], [BUGFIX], [REFACTOR], [DOCS], [TEST], [CONFIG], [PERF]
- Files modified, brief description, and reasoning.

## Sub-Agent Usage
- Use `codebase_investigator` for complex architectural analysis or bug root-cause investigations.
- Use `generalist` for repetitive batch tasks (e.g., fixing lint across multiple files, high-volume data processing).
- Use `cli_help` for questions about Gemini CLI features or configuration.

## Common Commands (Windows)
- **Backend**: `.venv\Scripts\python -m uvicorn sim_bench.api.main:app --reload --port 8000`
- **Frontend**: `.venv\Scripts\streamlit run app\streamlit\main.py`
- **Face Clustering App**: `.venv\Scripts\streamlit run app\face_clustering.py`
- **Tests**: `.venv\Scripts\python -m pytest tests\`
- **CLI Benchmarking**: `.venv\Scripts\python -m sim_bench.cli --methods chi_square --datasets ukbench`
