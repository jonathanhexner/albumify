# spec-024: Pipeline Run Observability

**Status**: Implemented
**Date**: 2026-05-03

## Problem

When running the pipeline, the frontend provides poor feedback:
1. Long-running steps show no progress — the user stares at a blank screen not knowing if it's working or stuck
2. If a step fails, the error isn't surfaced clearly — user must dig through backend logs
3. The frontend crashes during long runs due to infinite `st.rerun()` polling loops that exhaust DB connections and accumulate session state
4. No per-step timing or completion status visible during execution
5. After failure, no indication which step failed or what the partial results are

## Root Cause Analysis

**Crash mechanism** (configure.py lines 46-73):
```
_render_running_pipeline() → poll_pipeline_status() → time.sleep(1) → st.rerun()
```
Every rerun triggers a full page re-render + HTTP poll + DB query. Over a 2-minute run = 120 reruns. Over 10 minutes = 600 reruns. Session state accumulates, DB connections exhaust, Streamlit crashes.

**Missing data**: The API returns `(status, current_step, progress)` but NOT:
- List of completed steps with individual timings
- Error details per step (which step failed, traceback)
- Step-level progress messages

**No error recovery**: `executor.py` has no try/catch around step execution. If a step crashes, the entire pipeline fails with no partial results saved.

## User Stories

1. As a user running a pipeline, I see a live step-by-step progress display showing: which step is running, which completed (with duration), estimated time remaining.
2. As a user, if a step fails, I immediately see which step failed and the error message — without needing to check backend logs.
3. As a user, the frontend never crashes during a pipeline run regardless of how long it takes.
4. As a user, I can cancel a running pipeline and keep partial results.

## Design

### Backend Changes

**executor.py** — Add per-step error handling:
```python
def _execute_step(self, step, context, config) -> StepResult:
    try:
        step.process(context, step_config)
        return StepResult(step_name=name, success=True, duration_ms=dur)
    except Exception as e:
        logger.error(f"Step {name} failed: {e}", exc_info=True)
        return StepResult(step_name=name, success=False, error_message=str(e), duration_ms=dur)
```

**pipeline_service.py** — Track completed steps in PipelineRun:
- Add `completed_steps` JSON field to PipelineRun model (list of `{step, duration_ms, status}`)
- Update after each step completes (commit per step, not just at end)
- On step failure: save partial results, set `error_message` with step name + error

**API response** — Enrich GET /pipeline/{job_id}:
```json
{
  "status": "running",
  "current_step": "cluster_people",
  "progress": 0.7,
  "completed_steps": [
    {"step": "discover_images", "duration_ms": 200, "status": "completed"},
    {"step": "detect_persons", "duration_ms": 8500, "status": "completed"},
    {"step": "insightface_detect_faces", "duration_ms": 12000, "status": "completed"}
  ],
  "total_steps": 15,
  "elapsed_ms": 25000,
  "error_message": null
}
```

### Frontend Changes

**configure.py** — Replace infinite rerun loop with controlled polling:
```python
# Instead of: time.sleep(1); st.rerun()
# Use: st.empty() container with auto-refresh fragment

@st.fragment(run_every=2)  # Poll every 2 seconds, only refreshes this fragment
def _pipeline_progress_fragment():
    progress = poll_pipeline_status(job_id)
    _render_step_progress(progress)
    if progress.status in ("completed", "failed"):
        st.rerun(scope="app")  # Full rerun only on completion
```

**Progress display** — Visual step-by-step list:
```
[OK] discover_images          0.2s
[OK] detect_persons           8.5s
[OK] insightface_detect_faces 12.0s
[>>] cluster_people           ...running (15s)
[ ] identity_refinement
[ ] select_best
```

With progress bar, elapsed time, and estimated remaining time.

**Error display** — On failure:
```
[FAIL] cluster_people — RuntimeError: Numpy is not available
       ↳ Check logs/2026-05-03_01-27-18/ for full traceback
```

## Acceptance Criteria

1. Pipeline progress shows completed steps with individual durations
2. Current step shows elapsed time
3. On step failure: error displayed immediately with step name + message
4. Frontend does NOT crash during 10-minute pipeline runs
5. Cancel button stops the pipeline and saves partial results
6. Page doesn't rerun more than once every 2 seconds during pipeline execution

## Implementation Notes

- Use `@st.fragment(run_every=2)` for polling — this is Streamlit's built-in auto-refresh that only refreshes the fragment, not the full page
- `completed_steps` should be committed to DB after each step (not batched)
- Consider using the existing WebSocket endpoint instead of HTTP polling (already implemented but unused)
