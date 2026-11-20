# spec-116 — Tasks (One run path)

**Status:** In Progress 2026-10-30
**Spec:** [spec.md](spec.html) · [updates](updates.html) · roadmap stage **C** · [ROADMAP.md](../../ROADMAP.html)

Design lives in the spec (§3 plan, §4 class diagram, §5-§6 file tables). This file orders the work.

## W1 — One spec, one output, one writer
- [x] T1.1 `RunSpec.build`: steps + configs + `people_clustering` spread (moved from `PipelineService._broadcast_clustering_config`). Test `tests/pipeline/test_run_spec.py`. (R2)
- [x] T1.2 `RunOutput.from_context` (moves `_build_image_metrics`, face subclusters, people payload), `to_json`/`from_json`. (R4)
- [x] T1.3 `RunPersister.persist` (moves the `pipeline_results`/`people`/metric-table writes). `PipelineService` uses it; rows identical to before on a fixture context. Test `tests/serving/test_run_output.py`. (R4, R12)

## W2 — Regular runs go through the job service
- [x] T2.1 Manifest records key → original path; `PathMap.from_job`, `original`, `scale`; `RunOutput.remap`. Test `tests/serving/test_path_map.py`. (R5)
- [x] T2.2 `JobCurator` runs a given `RunSpec`; writes `results/run_output.json` and `progress.json`; `ModalVolumeRemote.run_job` / `_curate_job` carry the spec. (R2, R3, R7)
- [x] T2.3 `PipelineService.start_pipeline(..., runner)` creates the run and a job with the same id; `execute_pipeline` stages, runs via `JobService`, remaps, persists, then loads the job for Compare. Cancel route. Tests `tests/api/test_pipeline_service_runners.py`. (R8, R11)
- [x] T2.4 Face-distance reads the run's embeddings (`results/embeddings.npy`), falls back to the cache. (R14)

## W3 — The app
- [x] T3.1 Configure & Run: Run on, estimate + confirm, progress (steps on this PC, stage on the L4), Cancel. `tests/test_configure_run_on_e2e.py`. (R1, R6, R7)
- [x] T3.2 Albums page: runs table, open in Results, tick two → Compare. `tests/test_albums_table_e2e.py`. (R10)
- [x] T3.3 Jobs page: run form removed, link to the run in Albums; update `tests/test_jobs_page_run_e2e.py`. (R15)

## W4 — Prove on real data
- [>] T4.1 (measured, verdict NOT close enough; `report.html`/`summary.md` not written, see updates.md) Budapest (122) full-size (old path) vs 768 copies on this PC → Compare runs; report in `reports/`. (R13)
- [>] T4.2 (two attempts, neither got a GPU; stopped pending the owner) Slovakia26 on the L4 from Configure & Run (owner OK first), then the re-run; check every page in Playwright. (R5, R6, R9)
- [>] T4.3 (full suite, gate and REVIEW.md done; R5 R6 R9 R13 open) Full suite, `/code-review` → REVIEW.md, architecture HTMLs, updates.md evidence.
