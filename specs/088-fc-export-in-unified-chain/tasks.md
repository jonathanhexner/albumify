# spec-088 tasks

- [x] T1 Route the flag in `PipelineService._broadcast_clustering_config` (NOT FCParams —
      it's `extra="forbid"`/parity-tested). Copies flag + FCParams dump to the export step cfg.
- [x] T2 NEW `sim_bench/pipeline/steps/face_cluster_analysis_export.py` (≤80 LOC) +
      registered in `all_steps.py`.
- [x] T3 `configs/pipeline.yaml`: added after `assign_people_clusters` in `default_pipeline`.
- [x] T4 `pipeline_service.start_pipeline`: `context.album_name = album.name`.
- [x] T5 `tests/pipeline/test_fc_analysis_export.py` — 5 tests (routing on/off, step on/off,
      no-clusters skip). default-spec validation still green.
- [~] T6 Regression: architecture suite 130 passed; validate_spec OK. Full budapest e2e
      (clustering-unchanged) NOT run (heavy) — relies on read-only placement + user run.
- [x] T7 CHANGES_LOG ✓; SIGHTING-107 → FIXED ✓; `/code-review` → REVIEW.md (no blockers) ✓;
      status → Implemented ✓. Caught + fixed a self-introduced regression
      (`test_produces_not_empty` — step now declares `produces={"fc_export_dir"}`). 2 non-blocking
      TODOs filed (real-run integration test; data_flow.html). AC1/AC3 = user's real-run check.
