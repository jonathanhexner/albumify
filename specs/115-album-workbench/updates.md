# spec-115 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** Implemented 2026-10-23 — W1-W5 done; review ACCEPT after fixes.

## 2026-10-23 — Implemented: W3-W5, real L4 runs from the app, review fixes

Owner: "run all the way through the spec; verify the results show up correctly in the app" (L4 runs approved
for W5). Built W3 (Modal: `albumify-inputs` + `albumify-cache` Volumes, results-only download, retention + daily
purge, deployed) and W4 (estimate + spend cap, cancel, one Modal job at a time, re-run by job, `RunSettings`,
Jobs page run form / live status / Cancel / Re-run → Compare, "Settings that differ", delete-photos button).

**Real runs (Slovakia26, from the Jobs page with Playwright):**

| Run | Photos | Uploaded | Pipeline | Start → album | Copied back | Result |
|---|---|---|---|---|---|---|
| L4, full trip | 715 | 715 | 437 s | 10.7 min (2.4 staging) | 5 files | 251 picks, 80 people; 251 displays + 987 crops rendered on the PC in 2 s |
| L4 re-run, same photos | 715 | **0** | **46 s** | **1 min 51 s** | 5 files | Compare: close enough (250 same, 1 swapped, 3 also kept) |
| local, 48-photo slice | 48 | 0 (already staged) | 609 s (CPU) | 10.6 min | n/a | 19 picks, 7 people |
| L4 re-run of the slice | 48 | 0 | 7.6 s | 63 s | 5 files | Compare local vs L4: close enough (18 same, 1 swapped, 0 missing/extra); settings diff = runner, device, torch/onnxruntime, step devices |
| L4, 6 new Budapest photos, cancelled | 6 | 6 | n/a | n/a | n/a | cancel took 5.2 s; "Delete photos" removed all 6 from Modal (721 → 715) |

**Found and fixed during the runs:** a `run.json` read while it was being copied back failed `GET /jobs/{id}` (copy-back
now atomic, read tolerant); a cancel during upload would have run the GPU anyway (now stops before any call); the
re-run button could not select the new job (Streamlit widget already drawn: deferred); a duplicate widget key on the
run form (`jobs_run_start`) blanked the page; the Compare page warned about a preset selector.

**Code review (REVIEW.md):** 0 high, 7 medium, 8 low; all medium and the actionable low findings fixed with tests;
the remaining ones are recorded risks.

| Claim | Evidence |
|---|---|
| R1-R5 in the real app | `tests/test_jobs_page_run_e2e.py` 4 passed (real API + Streamlit, local `minimal`); real L4 runs above |
| R2 spend cap on every L4 path | `tests/control/test_job_service.py::ut_Estimate` (3) |
| R4 cancel: queued, waiting its turn, running, before the child starts; loading refused | `ut_Cancel` (3), `ut_ReviewFixes` (3); real L4 cancel 5.2 s |
| R8 results only; R9 cache on the L4 | real runs above; `tests/serving/test_modal_staging.py` 13 passed |
| R13 retention, atomic claims, upload claim | `ut_InputRetention` (6); real upload wrote `.used.json`; real delete 6/6; purge deployed |
| R12 settings | `tests/serving/test_run_settings.py` 2; real Compare settings diff |
| Review fixes M1-M7, L1-L5, L7 | `ut_ReviewFixes`, `test_local_renders.py::..._bad_box...`, `ut_ModalVolumeRemoteDownload`, router `test_negative_start_is_refused` |
| What the owner sees | final Playwright check: L4 album picks / dropped 96/96 / removed 48/48 / People 410/410 images load, no page errors; Compare local vs L4 with settings table (screenshots on disk) |
| Nothing else broke | full local suite 1,680 passed, 2 skipped, 0 failed (7.8 min; one test importing the real `modal` broke 12 Playwright tests in the same run, fixed, see LEARNINGS); Budapest V2 suite 11 passed (W2); no V2 tab changed in W3-W5 |

## 2026-10-23 — W2 done: the worker returns results only; the server renders displays and crops

- **T2.1 orientation gate** on 20 real slovakia26 faces (5 per EXIF rotation): every PC crop is the right face;
  ArcFace scored 18/20 against the pre-set 19/20 (a tilted head, an 11 px face); cut without the rotation, rotated
  photos score ~0. **Owner accepted.** Found **SIGHTING-141**: the worker's crops had red/blue swapped
  (`_crop_from_aligned` flipped RGB it took for BGR). Fixed (one line + a real-chain test); Budapest suite 11/11.
- **`LocalRenders`** (`sim_bench/control/local_renders.py`): `display/` for picks and `faces/<id>.jpg` (box + 25%
  margin, read EXIF-upright like the detector) from the server's own copies; existing files kept (older jobs);
  unreadable photos skipped and reported. `ResultsLoader` runs it before loading rows.
- `JobCurator` writes no displays or crops by default (`write_renders=False`); `JobStore.results_files()` lists what a
  worker returns (5 files for slovakia26); `JobStore.resolve` serves an `inputs/` key only if the job's manifest lists it (R7).

| Claim | Evidence |
|---|---|
| PC crops are the right faces, incl. rotated photos (R10) | `reports/2026-10-23_spec115_crop_orientation/report.html` (crops local only); owner accepted |
| Crops cut from the box, upright on EXIF-rotated photos; displays for picks only; old files kept; unreadable skipped (R10, R14) | `tests/control/test_local_renders.py` 6 passed (the rotation test fails without the transpose) |
| Worker returns no pictures; `results_files` = album.json, run.json, results/* (R8 part) | `tests/serving/test_curate_job.py::test_by_default_the_worker_returns_no_pictures` |
| Local and results-only (Modal-style) jobs leave identical folders (R10) | `tests/control/test_job_service.py::..._leave_the_same_files` |
| Another job's photo is 404 through this job's route (R7) | `tests/api/test_jobs_router.py::test_other_jobs_photo_is_404` |
| Real: render from results only | slovakia26 results without worker pictures: 147 displays, 987 crops (worker made 842; gate-rejected faces now get one too) in 5.1 s; sheet across all 4 rotations upright, natural colour |
| SIGHTING-141 fixed, no V2 regression | `tests/pipeline/test_export_crop_reuse.py::test_a_real_alignment_of_a_red_photo_gives_a_red_crop` (fails on the old code: red photo -> blue crop); Budapest suite 11 passed; fresh run crops natural skin tones |

## 2026-10-23 — Content fixes from the PRD review; requirements renumbered

Owner approved the review's content fixes. Old R6 (staged once), R7 (cache) and R15 (model version) are now one
"guarantees" row, **R6**; the freed **R7** is file serving per job (found: `JobStore.resolve` serves any photo in the
shared store to any job's route if the key is known; task T2.6). Cancel now has a stated end state (R4), the L4 a
spend cap (R2: 1,000 photos / 30 GPU-min), crop orientation a pass gate (R10: IoU ≥ 0.5 on ≥ 19/20), old per-job
photo copies on Modal are purged (R13), and §2 has a definition of done. Every check names its test.

## 2026-10-23 — Split out of spec-112 and rewritten in the new PRD format

Owner set a new PRD format (objective, requirements table with checks, plan flowchart, class diagram,
file tables) and asked for this work as its own spec in it. Content is spec-112's Phase A2 ("album
workbench", drafted and approved 2026-10-23: photos kept 7 days on Modal, re-run by pipeline name).
spec-112 keeps Phase A (built) and Phase B (cloud DB + storage) and links here.

## 2026-10-23 — W1 done: inputs staged once, job runs use the feature cache (commit de83348)

- `InputStore`: each photo resized once to `<jobs_root>/_inputs/<sha16>/<name>.jpg`; a job's `input/` holds
  `manifest.json`. Older jobs with `input/*.jpg` keep working. The API's image route resolves keys inside the
  job folder or the store only.
- The pipeline gets the job's paths explicitly (`discover_images` accepts given images); the per-run temp copy is gone.
- **Found: job runs never had a feature cache.** `run_pipeline` attached no cache handler, so every local and
  L4 job recomputed everything. Now attached (`ALBUMIFY_FEATURE_CACHE`, default `~/.sim_bench/sim_bench.db`).
- The spec-079 model-version check existed but was untested; test added.

| Claim | Evidence |
|---|---|
| Stale-version rows are misses; same version hits; unversioned steps unchanged (R6) | `tests/pipeline/test_cache_model_version.py` 4 passed |
| Content keys, no rewrite (mtime fixed), bad keys refused, manifest, resolve stays inside (R6, R14) | `tests/serving/test_input_store.py` 13 passed |
| Second job stages 0 photos, same paths, same mtimes (R6) | `tests/control/test_job_service.py` (`..._stages_nothing_new`) |
| Pipeline gets exactly the job's inputs, same paths across jobs; older jobs still run (R6, R14) | `tests/serving/test_curate_job.py` (+3), `tests/pipeline/test_discover_given_images.py` 3 |
| Real: the cache carries across jobs (R6) | 12 Budapest photos, `minimal`, two local jobs, fresh cache DB: job 2 staged 0 new; score_iqa 1017->26 ms, score_ava 1980->0, extract_scene_embedding 7719->10; identical picks |
| Nothing else broke | full local suite 1,627 passed, 2 skipped, 0 failed (10.7 min) |
