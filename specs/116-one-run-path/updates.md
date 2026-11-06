# spec-116 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress 2026-10-30

## 2026-10-31 — W1-W3 built; R13 fails its check; both L4 attempts ended in Modal's queue

Every album run now goes Configure & Run -> `JobService` -> this PC or the L4 -> `run_output.json` -> `RunPersister`. **Not Implemented:** R5, R6, R9 (L4) and R13 are open.

**How it was verified**

| Claim | Evidence |
|---|---|
| R1 R2 R3 R4 R8 R11 R14 unit level | `tests/pipeline/test_run_spec.py`, `tests/serving/test_run_output.py`, `tests/serving/test_path_map.py`, `tests/api/test_pipeline_service_runners.py`, `tests/api/test_pipeline_router.py`, `tests/api/test_results_face_distance.py`: part of 255 passed on the affected suites |
| R1 R6 R7 R8 R10 R15 in the app | `tests/test_configure_run_on_e2e.py`, `tests/test_albums_table_e2e.py`, `tests/test_jobs_page_run_e2e.py`: 11 passed (Modal faked) |
| Full suite | 1,739 passed, 2 skipped, 145 deselected, 0 failed |
| V2 gate | `tests/face_clustering/e2e_budapest/`: 11 passed |
| R5 stored paths, real | 768 run, Budapest, this PC: 0 staged paths in `image_metrics`, `selected_images`, `scene_clusters`, `step_decisions`, `siamese_comparisons`, `face_subclusters`, `people` (16 people, 487 face rows). L4 crops: open |
| R8, real | Both L4 attempts lost DNS on the PC; each run ended `failed` with the message and no result rows |
| R9 this PC, real | 48 photos: 593.3 s cold, 14.9 s warm, ratio 0.025 |
| R12, real | Budapest run15 `f782ad87`: Results, People & Faces, Explore: 0 exceptions, 0 broken images. Same for the real 768 run `a40333fd` |
| R13 | **Fails:** see below. `reports/2026-10-30_spec116_r13_fullsize_vs_768/results.json`, `diagnostic_no_sharpness_gate.json` |

**R13, Budapest 122, this PC (full-size vs 768).** Production config: NOT close enough: 8 vs 96 photos reach the album, 5 vs 49 picks, 0 vs 13 scenes, 11 vs 8 people, precision 0.27. Cause: the sharpness score depends on resolution (median 0.05 vs 0.31) and the gate is one fixed 0.2, so it removes 114 vs 26 photos (SIGHTING-144). Diagnostic with the gate off on both arms (not production): 122 each, 61 vs 58 picks, 15 vs 15 scenes, recall and precision 1.00, but still NOT close enough on people: 11 vs 8, largest person 19 vs 11 faces. Owner decision needed. The full-size arm reproduces the old path's inputs, not its code: the in-API runner is gone.

**L4 (T4.2), both attempts never got a GPU.** The owner's OK reached this agent as a relayed coordinator message.

| Attempt | Run | App | What happened |
|---|---|---|---|
| 1, 2026-10-30 | `b4180532` | `ap-0s9GGjOWWYUyS7Lgy3MBG7` | Estimate 715 photos, 9.7 min, $0.13. Queued 34 min "waiting to be scheduled on a GPU_L4 worker", then DNS failed on the PC |
| 2, 2026-10-31 | `683cf93f` | `ap-veL55C9lBBArN1JMHMkXQU` | Queued about 10 min, then DNS failed; Modal: "Stopping app - local client disconnected" |

No curator app is live and neither run left a folder on `albumify-jobs`. No pipeline ran on the L4, so Slovakia26 has no L4 evidence yet. Modal billing itself was not checked. Stages reached on the real L4 path: staging, uploading, running; "copying back" only with the faked Modal. Re-runs were not attempted. SIGHTING-143 (a queued run reads "Running") and SIGHTING-145 (the PC must stay connected for a whole L4 run).

**Decisions / deviations:** spec §5-§6 gained `RunEmbeddings`, `FaceService`, `PeopleService`, `SIM_BENCH_DB` (People & Faces would have been empty for staged runs). Rows are written after the job loads. Straightened picks show as the original. Staging is non-recursive. The Jobs page also lost Re-run (one way to start a run). The websocket no longer pushes step progress.

**Outstanding artefacts:** `report.html` and `summary.md` for R13 were refused by the agent harness (subagents may not write report files); the owner or parent must add them.

## 2026-10-25 — Drafted

Owner, after looking for the spec-115 L4 run in the app: "this should be in albums … the execution engine
shouldn't matter to our UI. we should be able to choose where to run at the beginning of the run."
Code map: both paths already call `execute_spec`, but the job path skips the `people_clustering` settings, keys
the cache on staged copies, and saves its results to a separate database (`control.db`) that the Albums,
Results and People pages never read. This spec makes Configure & Run the only way to start a run, on either
machine. Separately filed SIGHTING-142 (near-duplicate picks).
