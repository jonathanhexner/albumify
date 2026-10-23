# spec-115 — Progress & verification updates

**Spec:** [spec.md](spec.html) · [tasks.md](tasks.html) · roadmap stage: **C** · see [ROADMAP.md](../../ROADMAP.html)

**Status:** In Progress — W1 done 2026-10-23; W2 next.

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
