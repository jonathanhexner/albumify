# spec-083 — tasks

- [x] T1 `ImageRow.n_passed` + `list_images()` grouped count (faces w/ rejection_reason IS NULL).
- [x] T2 `ColumnSpec("n_passed","Passed")` in image_metrics; add to defaults.
- [x] T3 NEW `components/face_pick_grid.py` — clickable face grid (Open → Face Analysis).
- [x] T4 NEW `components/image_pick_grid.py` — clickable image grid (Open → in-tab analysis).
- [x] T5 `face_metrics_tab.py` — layout toggle Grid(default)/Table; grid path.
- [x] T6 `images_tab.py` — layout toggle Grid(default)/Table; master-detail (selected_image_path + Back).
- [x] T7 `image_analysis.py` — default passing-only boxes + "show filtered" toggle + identity list + legend.
- [x] T8 Unit: n_passed count (186==186 real run); faces_to_box pass-filter (7 tests).
- [x] T9 AppTest Face Metrics grid click (→ Face Analysis) + Images grid click (→ detail + Back).
- [x] T10 Budapest gate: NEW scenarios J (FM click) + K (Images click) + D — all green; real-browser screenshots.
- [x] T11 REVIEW.md + CHANGES_LOG; README matrix rows J/K; SIGHTING-095 (pre-existing run_metadata golden); status Implemented.
