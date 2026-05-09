# spec-024 Tasks

## Backend
- [x] T1: Add try/catch per step in executor.py
- [x] T2: Add `completed_steps` JSON field to PipelineRun model + migration
- [x] T3: Commit step completion to DB after each step (with flag_modified for JSON)
- [x] T4: On step failure: StepResult with error_message, pipeline continues if not fail_fast
- [x] T5: Enrich GET /pipeline/{job_id} response with completed_steps + total_steps

## Frontend
- [x] T6: Replace infinite st.rerun() loop with @st.fragment(run_every=2) polling
- [x] T7: Render step-by-step progress list (completed with duration, running, pending)
- [x] T8: Show error message on step failure (step name + error text)
- [ ] T9: Cancel button stops pipeline and saves partial results
- [ ] T10: Test: 10-minute simulated run without frontend crash

## Verification
- [x] T11: API test — 20/20 steps tracked with per-step durations
- [ ] T12: Playwright E2E — verify progress updates during pipeline run
