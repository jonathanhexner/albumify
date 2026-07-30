# spec-088 — Code Review

**Reviewed**: 2026-06-25 · **Reviewer**: Claude (code-review) · **Base**: working tree
**Scope**: spec-088 (re-enable "Export for analysis" in the unified clustering chain; SIGHTING-107).

## Part 1 — How it works
spec-079 removed the only step (`cluster_people`) that called `export_for_analysis()`, so the
toggle was dead. This adds a thin step at the end of the unified chain that reuses that helper.

```
UI "Export for analysis" ✓  → step_configs.cluster_people.export_for_analysis
  → PipelineService._broadcast_clustering_config  (routes flag + FCParams dump)
      → step_configs["face_cluster_analysis_export"]
  → FaceClusterAnalysisExportStep.process()  (after assign_people_clusters)
      → export_for_analysis(face_records, cluster_result, merged_cluster_result,
                            core_indices, fc_cfg, merge_log, merge_metadata, context)
      → results/<album>/face_clustering_<ts>/_v4/face_clustering.db ; sets context.fc_export_dir
```
Files: NEW `steps/face_cluster_analysis_export.py`; `steps/all_steps.py` (register);
`services/pipeline_service.py` (`_broadcast` routing + `start_pipeline` sets `album_name`);
`configs/pipeline.yaml` (add step); NEW `tests/pipeline/test_fc_analysis_export.py`.

## Part 2 — Findings

### §1 Structure — pass
New step is ~60 LOC, single responsibility (write the FC export), thin (reads context → calls
helper), per spec-053.

### §2 Code quality — pass
Guards: flag off → return; no `cluster_result`/`face_records` → skip with a log. No bare
except. Routing is a small conditional.

### §3 Naming — pass
`face_cluster_analysis_export` / `FaceClusterAnalysisExportStep` consistent with siblings.

### §4 Layering & coupling — pass
Reuses the existing `export_for_analysis()` (no duplicated export logic). `fc_export_dir` has a
single writer (the export helper). Flag routed as an IO concern, kept OUT of `FCParams`
(`extra="forbid"` + parity-tested) — correct boundary.

### §5 Testability — pass (1 self-introduced blocker FIXED) + follow-up
- `tests/pipeline/test_fc_analysis_export.py` — 5 tests: flag routes on/off; step fires/skips;
  no-cluster skip. Green.
- **Regression caught & FIXED:** `test_steps.py::test_produces_not_empty` failed because the new
  step declared `produces=set()`. Fixed → `produces={"fc_export_dir"}` (the step does set that
  context field). Re-run green.
- **Pre-existing failures (NOT spec-088, already tracked):** `test_person_penalty_strategy`
  (SIGHTING-083), `test_intra_person_similarity` (SIGHTING-081), `test_face_pipeline_e2e`
  (SIGHTING-088), `test_face_embedding_validation` collection error (SIGHTING-084 / stale
  `filter_quality_gate` import). Verified each is unrelated to this change (scoring-expectation
  drift, real-embedding benchmark < threshold, real-model E2E setup).
- **Follow-up (medium):** AC1/AC3 — that a real export dir is written AND opens in
  `face_cluster.loader.load_pipeline_result` — are NOT automatically tested; they need a live
  pipeline run (the user's re-run, which also yields the over-merge diagnosis data). → TODO.

### §6 Boundary contracts — pass
New step uses a permissive `config_schema={"type":"object"}` (like the other unified steps);
no new Pydantic contract introduced. The step rebuilds `FCParams` (filtering its own keys) →
`to_fc_config()` for the export helper's serialization.

### §7 Documentation — pass + follow-up
spec.md / tasks.md / REVIEW.md / CHANGES_LOG / SIGHTINGS(-107 → FIXED) present.
- **Follow-up (low):** a pipeline step was added → `docs/architecture/data_flow.html` should note
  the export step (documentation mandate). → TODO.

### §8 Risk — pass
- **Clustering output unchanged (AC5):** the step is read-only; placed after
  `assign_people_clusters`; architecture suite 130 passed; `validate_spec(default_pipeline)` OK.
- Back-compat: toggle off → no step work, `fc_export_dir` stays None.
- Hot-path: writes only when the toggle is on (opt-in).

## Part 3 — Verdict

| Area | Verdict |
|---|---|
| §1–§4, §6, §8 | accept |
| §5 Testability | accept (blocker fixed) + follow-up (real-run AC1/AC3 test) |
| §7 Documentation | accept + follow-up (data_flow.html) |

**No outstanding blockers** (the one self-introduced regression is fixed). spec-088 may flip to
Implemented. The 7 pre-existing tests/pipeline failures are tracked and unrelated.

### Follow-up tickets (TODO, non-blocking)
1. Integration test: run a small pipeline with the toggle on, assert an export dir is written
   and loads via `face_cluster.loader.load_pipeline_result` (covers AC1/AC3).
2. Update `docs/architecture/data_flow.html` for the new export step.
