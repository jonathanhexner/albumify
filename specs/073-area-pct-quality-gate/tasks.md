# Tasks — Area-% quality gate (073)

- [ ] T001 `fc_params.py`: add `min_face_area_pct: Optional[float]` Field (0–100, None=off).
- [ ] T002 `config.py` PipelineConfig: add `min_face_area_pct: Optional[float] = None`.
- [ ] T003 `ui_spec.py`: add `min_face_area_pct` to the "quality" group (number_input, zero_is_none).
- [ ] T004 `quality.py`: `_add_area_pct_gate` + call in `_top_k_verdict`/`_evaluate_gates`; add "area_pct" to rejection priority.
- [ ] T005 unit test: reject below thr / pass above / off when None / permissive when area_ratio None.
- [ ] T006 parity + ui_spec arch tests green; AppTest Run tab shows the knob; Scenario D green.
- [ ] T007 REVIEW.md, CHANGES_LOG, status → Implemented.
