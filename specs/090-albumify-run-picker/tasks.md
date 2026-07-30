# spec-090 tasks

- [x] T1 `session.py`: `current_run_id` field + `get/set_current_run_id`; reset on album change.
- [x] T2 NEW `components/run_selector.py`: pure `resolve_run_id` + `render_run_selector`.
- [x] T3 `pages/results.py`: render selector once; `_pick_run()` resolves job_id in every tab
      (replaced 5 hardcoded `results[0]`); people summary uses the picked run.
- [x] T4 `pages/people.py`: render selector + pass `run_id` to `get_people`.
- [x] T5 `pages/face_management.py`: `_get_active_run_id` honours the picked run.
- [x] T6 `tests/streamlit/test_run_selector.py` — 5 tests (resolve logic). Green.
- [ ] T7 CHANGES_LOG ✓; `/code-review`; flip status. Manual: pick an older run → its data shows.
